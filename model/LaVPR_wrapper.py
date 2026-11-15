import torch
import numpy as np
import os
import peft
from model.LaVPR_cross import LaVPR_cross
from model.LaVPR_reranker import LaVPR_reranker
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor
import open_clip


class LaVPR_wrapper():
    def __init__(self, args):
        self.model_name  = args.model_name.lower()
        self.device      = args.device
        self.embeds_dim  = args.embeds_dim
        self.encoder_dim = args.embeds_dim
        self.reranker    = args.reranker
        self.cross_modal = args.cross_modal

        
        # ---- LaVPR_cross or CrossModalRerankModule checkpoint ----
        # If --reranker_lavpr is set, model_path points to the reranker ckpt
        # which contains the frozen backbone inside it. Load the backbone
        # from there. Otherwise load LaVPR_cross directly.
        if getattr(args, 'reranker', False):
            from model.reranker_module import CrossModalRerankModule
            # Detect score head depth from state dict before loading
            # score_head.1.weight shape: (1, d)=shallow  (d/2, d)=deep
            _ckpt = torch.load(args.model_path, map_location='cpu',
                                weights_only=False)
            _sd   = _ckpt.get('state_dict', {})
            _sh_w = _sd.get('reranker.score_head.1.weight')
            _deep = (_sh_w is not None and _sh_w.shape[0] > 1)
            _reranker_module = CrossModalRerankModule.load_from_checkpoint(
                args.model_path,
                map_location    = args.device,
                deep_score_head = _deep,
                freeze_bn = args.rerank_freeze_bn
            )
            self.vlm_encoder = _reranker_module.backbone.eval().to(args.device)
            # Keep reranker network for later use in rerank_by_lavpr
            self.reranker_network = _reranker_module.reranker.eval().to(args.device)
            del _reranker_module
        else:
            self.vlm_encoder = LaVPR_cross.load_from_checkpoint(
                args.model_path,
                map_location=args.device,
                strict=True,
            )
            self.vlm_encoder.text_max_tokens_len = args.max_tokens_len
            self.vlm_encoder = self.vlm_encoder.eval().to(args.device)
            self.reranker_network = None

        # Expose local token dims for eval buffer allocation
        #from model.LaVPR_cross import _IMAGE_ENCODER_DIM, _text_hidden_dim
        # img_key               = self.vlm_encoder.image_model_name
        # self.img_local_dim    = _IMAGE_ENCODER_DIM.get(img_key, 768)
        self.img_local_tokens = 256   # DINOv2-B at 224px (14x14 patches)
        # self.txt_local_dim    = _text_hidden_dim(self.vlm_encoder.text_model_name)
        self.txt_local_tokens = 512   # BGE max sequence length
        self.img_local_dim    = self.vlm_encoder.image_proj.in_features
        self.txt_local_dim    = self.vlm_encoder.text_proj.in_features

        
         
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings    = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask       = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    # Encode image
    # ------------------------------------------------------------------

    def encode_image(self, images):    
        with torch.no_grad():
            img_embeds, _ = self.vlm_encoder.encode_image(images)
        return img_embeds

        

    # ------------------------------------------------------------------
    # Encode text
    # ------------------------------------------------------------------

    def encode_text(self, texts):        
        with torch.no_grad():
            text_embeds, _, _, _ = self.vlm_encoder.encode_text(texts)
        return text_embeds

        

    # ------------------------------------------------------------------
    # Encode dual  (non-cross-modal path)
    # ------------------------------------------------------------------

    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vpr_encoder(images)
            text_features  = self.encode_text(texts)
        return image_features, text_features

    # ------------------------------------------------------------------
    # Encode single — returns global embeds + local tokens
    # ------------------------------------------------------------------

    def encode_single(self, images, texts, no_grad=True):        
        if no_grad:
            with torch.no_grad():                                
                (img_embeds, text_embeds, 
                img_local, text_local,
                attention_mask, text_tokens) = self.vlm_encoder(images, texts)                
        else:                
            (img_embeds, text_embeds, 
                img_local, text_local,
                attention_mask, text_tokens) = self.vlm_encoder(images, texts)
            
        return (img_embeds, text_embeds,
                img_local, text_local,
                text_tokens, attention_mask)
                

        