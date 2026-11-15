import warnings
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.optim import lr_scheduler

import pytorch_lightning as pl
import faiss

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, AutoProcessor, CLIPVisionModel

import utils
from model.blip_model import BlipForImageTextRetrievalWrapper
from model.weighted_ms_loss import WeightedMultiSimilarityLossCM
from model.pooling_cm import CLSReweightingPooler


_TEXT_ENCODER_DIM = {
    'bge-l': 1024,
    'bge-b':  768,
    'bge-s':  384,
}


def _dinov2_hidden_dim(model_name: str) -> int:
    name = model_name.lower()
    if 'vitl' in name: return 1024
    if 'vitg' in name: return 1536
    return 768


def _text_hidden_dim(model_name: str) -> int:
    name = model_name.lower()
    for key, dim in _TEXT_ENCODER_DIM.items():
        if key in name:
            return dim
    raise ValueError(
        f"Unknown text model '{model_name}'. "
        f"Add its hidden dim to _TEXT_ENCODER_DIM or pass text_hidden_dim explicitly."
    )

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LaVPR_cross(pl.LightningModule):

    def __init__(
        self,
        image_model_name: str = 'dinov2_vitb14',
        text_model_name:  str = 'BAAI/bge-large-en-v1.5',
        embeds_dim: int = 512,
        vlm_image_only: bool = False,
        lr: float = 2e-5,
        optimizer: str = 'adamw',
        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        warmpup_steps: int = 500,
        milestones: list = None,
        lr_mult: float = 0.3,
        epochs: int = 10,
        loss_name: str = 'MultiSimilarityLoss',
        miner_name: str = 'MultiSimilarityMiner',
        miner_margin: float = 0.1,
        faiss_gpu: bool = False,
        freeze_image: bool = False,
        freeze_text: bool = False,
        train_text: int = 1,
        train_image: int = 1,
        lora_all_linear: bool = False,
        lora_target_modules: list = None,
        lora_r: int = 64,
        unimodal_loss: float = 0.0,
        pos_loss: int = 0,
        is_llp: int = 0,
        text_max_tokens_len: int = 512,
        train_proj_only=False,
    ):
        super().__init__()

        if milestones is None:
            milestones = [5, 10, 15]

        self.image_model_name    = image_model_name
        self.image_model_name_lc = image_model_name.lower()
        self.text_model_name     = text_model_name.lower()
        self.embeds_dim          = embeds_dim
        self.vlm_image_only      = vlm_image_only
        self.lr                  = lr
        self.optimizer           = optimizer
        self.weight_decay        = weight_decay
        self.momentum            = momentum
        self.warmpup_steps       = warmpup_steps
        self.milestones          = milestones
        self.lr_mult             = lr_mult
        self.epochs              = epochs
        self.loss_name           = loss_name
        self.miner_name          = miner_name
        self.miner_margin        = miner_margin
        self.faiss_gpu           = faiss_gpu
        self.train_text          = train_text
        self.train_image         = train_image
        self.lora_all_linear     = lora_all_linear
        self.lora_target_modules = lora_target_modules
        self.lora_r              = lora_r
        self.unimodal_loss       = unimodal_loss
        self.pos_loss            = pos_loss
        self.is_llp              = is_llp
        self.text_max_tokens_len = text_max_tokens_len        

        n = self.image_model_name_lc
        self._is_longclip  = 'longclip' in n or 'long-clip' in n
        self._is_clip      = 'clip' in n and not self._is_longclip
        self._is_blip      = 'blip' in n
        self._is_dinov2    = 'dinov2' in n
        self._is_sela      = 'sela' in n
        _is_clip_family    = self._is_longclip or self._is_clip

        self._is_joint_vlm    = _is_clip_family and not vlm_image_only
        self._is_vlm_img_only = _is_clip_family and vlm_image_only

        self.save_hyperparameters()

        if 'WeightedMultiSimilarityLoss' in loss_name:
            self.loss_fn = WeightedMultiSimilarityLossCM()
        else:
            self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)

        self.batch_acc = []
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Encoder loading ------------------------------------------------
        if self._is_joint_vlm:
            self.max_text_length = 248 if self._is_longclip else 77
            self.vlm_encoder  = AutoModel.from_pretrained(image_model_name, trust_remote_code=True)
            self.processor    = AutoProcessor.from_pretrained(image_model_name, trust_remote_code=True)
            self.text_encoder = None
            self.tokenizer    = None
            img_hidden_dim = self.vlm_encoder.config.vision_config.hidden_size
            txt_hidden_dim = self.vlm_encoder.config.text_config.hidden_size

        elif self._is_vlm_img_only:
            self.image_encoder = CLIPVisionModel.from_pretrained(image_model_name, trust_remote_code=True)
            img_hidden_dim     = self.image_encoder.config.hidden_size
            txt_hidden_dim     = _text_hidden_dim(text_model_name)
            self.tokenizer     = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder  = AutoModel.from_pretrained(text_model_name, attn_implementation="sdpa")

        elif self._is_blip:
            self.image_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(image_model_name)
            img_hidden_dim     = self.image_encoder.config.vision_config.hidden_size
            txt_hidden_dim     = _text_hidden_dim(text_model_name)
            self.tokenizer     = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder  = AutoModel.from_pretrained(text_model_name, attn_implementation="sdpa")

        elif self._is_dinov2:
            hub_model = (image_model_name
                         if image_model_name.startswith('dinov2_vit')
                         else 'dinov2_vitb14')
            self.image_encoder = torch.hub.load('facebookresearch/dinov2', hub_model)
            img_hidden_dim     = _dinov2_hidden_dim(image_model_name)
            txt_hidden_dim     = _text_hidden_dim(text_model_name)
            self.tokenizer     = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder  = AutoModel.from_pretrained(text_model_name, attn_implementation="sdpa")

        elif self._is_sela:
            self.image_encoder = torch.hub.load(
                'Lu-Feng/SelaVPRplusplus', 'SelaVPRplusplus',
                backbone='dinov2-base', aggregation='gem', hashing=False, rerank=False)
            img_hidden_dim    = 2048
            txt_hidden_dim    = _text_hidden_dim(text_model_name)
            self.tokenizer    = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder = AutoModel.from_pretrained(text_model_name, attn_implementation="sdpa")

        else:
            raise ValueError(
                f"Cannot determine encoder type from image_model_name='{image_model_name}'.")

        self.image_proj = nn.Linear(img_hidden_dim, embeds_dim, bias=False)
        self.text_proj  = nn.Linear(txt_hidden_dim, embeds_dim, bias=False)

        if is_llp and not self._is_joint_vlm:
            self.llp = CLSReweightingPooler(txt_hidden_dim)

        # ---- Freeze / LoRA --------------------------------------------------
        if self._is_joint_vlm:
            if freeze_image:
                for p in self.vlm_encoder.parameters():
                    p.requires_grad = False
            if train_image == 1:
                self.vlm_encoder = self._init_lora(
                    self.vlm_encoder, lora_target_modules, lora_all_linear, lora_r)
            elif train_image == 2:
                for p in self.vlm_encoder.parameters():
                    p.requires_grad = True
            elif freeze_image:
                self.vlm_encoder.eval()
        else:
            if freeze_image:
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
            if freeze_text:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

            _lora_image_capable = not self._is_sela
            if train_image == 1:
                if _lora_image_capable:
                    self.image_encoder = self._init_lora(
                        self.image_encoder, lora_target_modules, lora_all_linear, lora_r)
                else:
                    for p in self.image_encoder.parameters():
                        p.requires_grad = True
            elif train_image == 2:
                for p in self.image_encoder.parameters():
                    p.requires_grad = True
            elif freeze_image:
                self.image_encoder.eval()

            if train_text == 1:
                self.text_encoder = self._init_lora(
                    self.text_encoder, lora_target_modules, lora_all_linear, lora_r)
            elif train_text == 2:
                for p in self.text_encoder.parameters():
                    p.requires_grad = True
            elif freeze_text:
                self.text_encoder.eval()

        nn.init.kaiming_uniform_(self.image_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.text_proj.weight,  mode='fan_in', nonlinearity='relu')
        
    # -------------------------------------------------------------------------
    # LoRA helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _init_lora(encoder, lora_targets, lora_all_linear, lora_r):
        if lora_all_linear:
            lora_targets = "all-linear"
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.1,
            target_modules=lora_targets, task_type=TaskType.FEATURE_EXTRACTION,
            use_rslora=True, bias="none")
        return get_peft_model(encoder, lora_config)

    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------

    def encode_image(self, img):
        if self._is_dinov2:
            out       = self.image_encoder.forward_features(img)
            img_local = out['x_norm_patchtokens']
            img_raw   = out['x_norm_clstoken']

        elif self._is_blip:
            encoder    = (self.image_encoder.base_model
                          if hasattr(self.image_encoder, 'base_model')
                          else self.image_encoder)
            vision_out = encoder.vision_model(pixel_values=img)
            tokens     = vision_out[0]
            img_raw    = tokens[:, 0, :]
            img_local  = tokens[:, 1:, :]

        elif self._is_joint_vlm:
            encoder    = (self.vlm_encoder.base_model
                          if hasattr(self.vlm_encoder, 'base_model')
                          else self.vlm_encoder)
            vision_out = encoder.vision_model(pixel_values=img)
            tokens     = vision_out.last_hidden_state
            img_raw    = tokens[:, 0, :]
            img_local  = tokens[:, 1:, :]

        elif self._is_vlm_img_only:
            encoder    = (self.image_encoder.base_model
                          if hasattr(self.image_encoder, 'base_model')
                          else self.image_encoder)
            vision_out = encoder(pixel_values=img)
            tokens     = vision_out.last_hidden_state
            img_raw    = tokens[:, 0, :]
            img_local  = tokens[:, 1:, :]

        elif self._is_sela:
            encoder   = (self.image_encoder.base_model
                         if hasattr(self.image_encoder, 'base_model')
                         else self.image_encoder)
            img_raw   = encoder(img)
            img_local = None

        img_embeds     = F.normalize(self.image_proj(img_raw), p=2, dim=1)
        img_local_proj = img_local
        
        return img_embeds, img_local_proj

    def encode_text(self, text):
        if self._is_joint_vlm:
            text_inputs = self.processor(
                text=text, return_tensors="pt",
                padding="max_length", truncation=True,
                max_length=self.max_text_length,
            ).to(self.device)
            attention_mask = text_inputs.get('attention_mask', None)
            encoder        = (self.vlm_encoder.base_model
                              if hasattr(self.vlm_encoder, 'base_model')
                              else self.vlm_encoder)
            text_outputs   = encoder.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=attention_mask,
                output_hidden_states=False, return_dict=True)
            text_local  = text_outputs.last_hidden_state
            text_raw    = text_outputs.pooler_output
            text_tokens = text_inputs
        else:
            text_tokens = self.tokenizer(
                text, padding=True, truncation=True,
                return_tensors='pt').to(self.device)
            attention_mask = text_tokens['attention_mask']
            model_output   = self.text_encoder(
                **text_tokens, output_hidden_states=False, return_dict=True)
            text_local = model_output.last_hidden_state
            if self.is_llp and hasattr(self, 'llp'):
                text_raw = self.llp(text_local, mask=attention_mask)
            else:
                text_raw = text_local[:, 0]

        text_embeds     = F.normalize(self.text_proj(text_raw), p=2, dim=1)
        text_local_proj = text_local        

        return text_embeds, text_local_proj, attention_mask, text_tokens

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, img, text, labels=None):
        img_embeds,  img_local_proj  = self.encode_image(img)
        text_embeds, text_local_proj, attention_mask, text_tokens = self.encode_text(text)        

        return (img_embeds, text_embeds, 
                img_local_proj, text_local_proj, attention_mask, text_tokens)

    # -------------------------------------------------------------------------
    # Optimiser
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(
                self.parameters(), lr=self.lr,
                weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer.lower() in ('adamw', 'adam'):
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        scheduler = lr_scheduler.MultiStepLR(
            opt, milestones=self.milestones, gamma=self.lr_mult)
        return [opt], [scheduler]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def loss_function(
        self,
        img_embeds: torch.Tensor,
        labels: torch.Tensor,
        text_embeds: torch.Tensor,        
        img_local_proj: torch.Tensor | None = None,
        text_local_proj: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        text_labels     = labels.clone()
        text_embeds_all = text_embeds
        loss = 0
        batch_acc = 0.0                    
        
        if self.miner is not None:
            miner_outputs = self.miner(
                img_embeds, labels,
                ref_emb=text_embeds_all, ref_labels=text_labels)
            loss = self.loss_fn(
                img_embeds, labels,
                indices_tuple=miner_outputs,
                ref_emb=text_embeds_all, ref_labels=text_labels)
            nb_samples = img_embeds.shape[0]

            if self.unimodal_loss:
                miner_txt = self.miner(text_embeds_all, text_labels)
                txt_loss  = self.loss_fn(text_embeds_all, text_labels,
                                        indices_tuple=miner_txt)
                loss = loss + self.unimodal_loss * txt_loss

            nb_mined  = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(img_embeds, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss        

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc),
                 prog_bar=True, logger=True)
        
        return loss

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):       
        
        places, labels, texts = batch

        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        flat_texts = [texts[j][i] for i in range(BS) for j in range(N)]        

        (img_embeds, text_embeds, 
         img_local_proj, text_local_proj,
         attention_mask, _) = self(images, flat_texts, labels)

        loss = self.loss_function(
            img_embeds, labels, text_embeds, 
            img_local_proj, text_local_proj, attention_mask)
        self.log('loss', loss.item(), logger=True)
        return loss
    
    
    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        (img_embeds, text_embeds, _,
        img_local, text_local, attn_mask, _) = self(places, texts)

        out = {
            'descriptors': img_embeds.detach().cpu(),
            'text_embeds': text_embeds.detach().cpu(),
        }
        
        return out


    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        k_values = [1, 5, 10, 15, 20, 50, 100]

        for val_set_name, val_dataset, step_outputs in zip(
                dm.val_set_names, dm.val_datasets, val_step_outputs):

            img_feats  = torch.cat([d['descriptors'] for d in step_outputs], dim=0)
            text_feats = torch.cat([d['text_embeds']  for d in step_outputs], dim=0)

            if 'pitts' in val_set_name:
                num_references = val_dataset.num_db
                positives      = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives      = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f'Please implement validation_epoch_end for {val_set_name}')

            r_list = img_feats[:num_references]
            q_list = text_feats[num_references:]

            # ── Global retrieval ──────────────────────────────────────────────────
            recalls = utils.get_validation_recalls(
                r_list=r_list, q_list=q_list, k_values=k_values,
                gt=positives, print_results=True,
                dataset_name=val_set_name, faiss_gpu=self.faiss_gpu)

            self.log(f'{val_set_name}/R1',  recalls[1],  prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5',  recalls[5],  prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls[10], prog_bar=False, logger=True)

    
    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        ckpt_cb = next(
            (cb for cb in self.trainer.checkpoint_callbacks
             if isinstance(cb, pl.callbacks.ModelCheckpoint)), None)
        if ckpt_cb is None:
            return
        ckpt_dir = os.path.dirname(ckpt_cb.dirpath)

        if self._is_joint_vlm:
            if self.train_image == 1:
                save_path = os.path.join(ckpt_dir, 'vlm_lora')
                self.vlm_encoder.save_pretrained(save_path)
                print(f"Saved joint VLM LoRA to: {save_path}")
        