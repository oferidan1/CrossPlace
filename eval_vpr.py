import eval_parser
import sys
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from model.LaVPR_wrapper import LaVPR_wrapper
import os
from dataloaders.test_dataset import TestDataset
from dataloaders.MapillaryTestDataset import MSLSTest
import utils.visualizations as visualizations
import csv


def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _maybe(mod, name, x):
    layer = getattr(mod, name, None)
    return layer(x) if layer is not None else x


def _z(x):
    return (x - x.mean()) / x.std().clamp(min=1e-6)


def _zn_np(x: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D numpy array; return x unchanged if std is tiny."""
    std = x.std()
    return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else x


# ---------------------------------------------------------------------------
# Timing stats accumulator
# ---------------------------------------------------------------------------

class TimingStats:
    """Accumulate wall-clock timings and print a summary table."""

    def __init__(self):
        self._data = {}   # name -> list of (n_items, elapsed_s)

    def record(self, name: str, n_items: int, elapsed_s: float):
        self._data.setdefault(name, []).append((n_items, elapsed_s))

    def summary(self) -> str:
        if not self._data:
            return "(no timing data)"
        lines = [
            f"\n{'='*72}",
            f"  {'Stage':<30}  {'Calls':>5}  {'Items':>8}  "
            f"{'Total(s)':>9}  {'ms/item':>8}",
            f"  {'(items = images for encode, queries for rerank)':<68}",
            f"  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}",
        ]
        for name, records in self._data.items():
            n_calls = len(records)
            n_items = sum(r[0] for r in records)
            total_s = sum(r[1] for r in records)
            ms_per  = (total_s / n_items * 1000) if n_items > 0 else 0
            lines.append(
                f"  {name:<30}  {n_calls:>5}  {n_items:>8}  "
                f"{total_s:>9.2f}  {ms_per:>8.2f}")
        lines.append(f"{'='*72}\n")
        return "\n".join(lines)


TIMING = TimingStats()


# ---------------------------------------------------------------------------
# Stage-1 retrieval cache: save / load top-K results as CSV
# ---------------------------------------------------------------------------
STAGE1_TOPK_SAVE = 50


def save_stage1_cache(save_path, predictions, scores,
                      query_filenames, db_filenames,
                      topk=STAGE1_TOPK_SAVE):
    k      = min(topk, predictions.shape[1])
    db_arr = np.array(db_filenames)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['query_path']
        for r in range(1, k + 1):
            header += [f'db_path_{r}', f'score_{r}']
        writer.writerow(header)
        for q_idx, q_path in enumerate(query_filenames):
            row = [q_path]
            for r in range(k):
                db_idx = int(predictions[q_idx, r])
                row.append(db_arr[db_idx])
                row.append(f'{scores[q_idx, r]:.6f}')
            writer.writerow(row)
    logger.info(
        f"Stage-1 cache saved → {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")


def load_stage1_cache(save_path, db_filenames):
    db_path_to_idx = {p: i for i, p in enumerate(db_filenames)}
    query_filenames  = []
    top_db_filenames = []
    predictions_list = []
    scores_list      = []
    with open(save_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        k      = (len(header) - 1) // 2
        for row in reader:
            query_filenames.append(row[0])
            db_paths, scs, idxs = [], [], []
            for r in range(k):
                db_path = row[1 + r * 2]
                sc      = float(row[2 + r * 2])
                db_idx  = db_path_to_idx.get(db_path, -1)
                db_paths.append(db_path)
                scs.append(sc)
                idxs.append(db_idx)
            top_db_filenames.append(db_paths)
            predictions_list.append(idxs)
            scores_list.append(scs)
    predictions = np.array(predictions_list, dtype=np.int64)
    scores      = np.array(scores_list,      dtype=np.float32)
    n_missing   = (predictions == -1).sum()
    if n_missing:
        logger.warning(
            f"Stage-1 cache: {n_missing} db paths not found in current dataset.")
    logger.info(
        f"Stage-1 cache loaded ← {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")
    return {
        'query_filenames':  query_filenames,
        'top_db_filenames': top_db_filenames,
        'predictions':      predictions,
        'scores':           scores,
    }


def save_rerank_cache(save_path, predictions, scores,
                      query_filenames, db_filenames,
                      topk=STAGE1_TOPK_SAVE):
    k      = min(topk, predictions.shape[1])
    db_arr = np.array(db_filenames)
    if scores is None:
        scores = np.tile(
            np.arange(k, 0, -1, dtype=np.float32), (predictions.shape[0], 1))
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['query_path']
        for r in range(1, k + 1):
            header += [f'db_path_{r}', f'score_{r}']
        writer.writerow(header)
        for q_idx, q_path in enumerate(query_filenames):
            row = [q_path]
            for r in range(k):
                db_idx = int(predictions[q_idx, r])
                sc_val = float(scores[q_idx, r])
                row.append(db_arr[db_idx])
                row.append('' if np.isnan(sc_val) else f'{sc_val:.6f}')
            writer.writerow(row)
    logger.info(
        f"Stage-2 cache saved → {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")


# ---------------------------------------------------------------------------
# Combined cache: retrieval + rerank scores side-by-side for alpha ablation
# ---------------------------------------------------------------------------

def save_combined_cache(save_path, retrieval_predictions, retrieval_scores,
                        rerank_predictions, rerank_scores,
                        query_filenames, db_filenames,
                        topk=STAGE1_TOPK_SAVE):
    """
    Saves one row per query in RETRIEVAL order so retrieval scores are
    monotone descending.  For each candidate both ret_score and rerank_score
    are recorded so offline alpha ablation needs only this one file.

    rerank_predictions / rerank_scores may be None (pure retrieval run).
    Candidates outside the reranked window get an empty rerank_score field
    (reads back as NaN).

    Columns:
        query_path,
        db_path_1, ret_score_1, rerank_score_1,
        db_path_2, ret_score_2, rerank_score_2,
        ...
    """
    k      = min(topk, retrieval_predictions.shape[1])
    db_arr = np.array(db_filenames)

    # Build per-query {db_idx: rerank_score} lookup from rerank results.
    rerank_lookup = {}   # q_idx -> {db_idx: score}
    if rerank_predictions is not None and rerank_scores is not None:
        for q_idx in range(rerank_predictions.shape[0]):
            rerank_lookup[q_idx] = {
                int(rerank_predictions[q_idx, r]): float(rerank_scores[q_idx, r])
                for r in range(rerank_predictions.shape[1])
                if not np.isnan(rerank_scores[q_idx, r])
            }

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['query_path']
        for r in range(1, k + 1):
            header += [f'db_path_{r}', f'ret_score_{r}', f'rerank_score_{r}']
        writer.writerow(header)

        for q_idx, q_path in enumerate(query_filenames):
            row   = [q_path]
            q_lut = rerank_lookup.get(q_idx, {})
            for r in range(k):
                db_idx    = int(retrieval_predictions[q_idx, r])
                ret_sc    = float(retrieval_scores[q_idx, r])
                rerank_sc = q_lut.get(db_idx, float('nan'))
                row += [
                    db_arr[db_idx],
                    f'{ret_sc:.6f}',
                    '' if np.isnan(rerank_sc) else f'{rerank_sc:.6f}',
                ]
            writer.writerow(row)

    logger.info(
        f"Combined cache saved → {save_path}  "
        f"(queries={len(query_filenames)}, topk={k}, "
        f"has_rerank={'yes' if rerank_predictions is not None else 'no'})")


def load_combined_cache(save_path, db_filenames):
    """
    Returns a dict with arrays shaped (n_queries, topk):
        query_filenames   list[str]
        db_paths          list[list[str]]
        predictions       int64   (db indices, -1 if path not found)
        ret_scores        float32
        rerank_scores     float32 (nan where not reranked)
    """
    db_path_to_idx = {p: i for i, p in enumerate(db_filenames)}
    query_filenames = []
    db_paths_rows, ret_rows, rerank_rows, pred_rows = [], [], [], []

    with open(save_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        k      = (len(header) - 1) // 3
        for row in reader:
            query_filenames.append(row[0])
            db_ps, rets, reranks, idxs = [], [], [], []
            for r in range(k):
                db_path  = row[1 + r * 3]
                ret_sc   = float(row[2 + r * 3])
                rr_field = row[3 + r * 3]
                rr_sc    = float(rr_field) if rr_field else float('nan')
                db_ps.append(db_path)
                rets.append(ret_sc)
                reranks.append(rr_sc)
                idxs.append(db_path_to_idx.get(db_path, -1))
            db_paths_rows.append(db_ps)
            ret_rows.append(rets)
            rerank_rows.append(reranks)
            pred_rows.append(idxs)

    n_missing = sum(1 for row in pred_rows for idx in row if idx == -1)
    if n_missing:
        logger.warning(
            f"Combined cache: {n_missing} db paths not found in current dataset.")
    logger.info(
        f"Combined cache loaded ← {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")
    return {
        'query_filenames': query_filenames,
        'db_paths':        db_paths_rows,
        'predictions':     np.array(pred_rows,   dtype=np.int64),
        'ret_scores':      np.array(ret_rows,    dtype=np.float32),
        'rerank_scores':   np.array(rerank_rows, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Offline alpha ablation over a combined cache
# ---------------------------------------------------------------------------

def ablate_alpha(combined, positives_per_query, recall_values,
                 alphas=None, normalize='zscore'):
    """
    Sweep alpha values using pre-computed retrieval + rerank scores.

    combined            — dict returned by load_combined_cache()
    positives_per_query — list[set[int]]  (db indices that are positives)
    recall_values       — e.g. [1, 5, 10, 25]
    alphas              — list of floats in [0,1]; None → 11 values 0.0..1.0
    normalize           — 'zscore' | 'minmax' | 'none'

    Returns:
        results  dict  alpha -> np.array of recall@k values (%)
    """
    if alphas is None:
        alphas = [round(a / 10, 1) for a in range(11)]

    ret_scores    = combined['ret_scores'].copy()     # (Q, K)
    rerank_scores = combined['rerank_scores'].copy()  # (Q, K), NaN where missing
    predictions   = combined['predictions']           # (Q, K)
    Q, K          = ret_scores.shape

    # Queries where rerank scores are entirely NaN can only use retrieval.
    has_rerank = ~np.all(np.isnan(rerank_scores), axis=1)   # (Q,)

    def _norm_rows(mat):
        out = mat.copy()
        for q in range(Q):
            row   = out[q]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                out[q] = np.where(valid, 0.0, np.nan)
                continue
            if normalize == 'zscore':
                std    = row[valid].std()
                out[q] = np.where(valid,
                                  (row - row[valid].mean()) / max(std, 1e-8),
                                  np.nan)
            elif normalize == 'minmax':
                lo, hi = row[valid].min(), row[valid].max()
                out[q] = np.where(valid,
                                  (row - lo) / max(hi - lo, 1e-8),
                                  np.nan)
            # 'none' → leave as-is
        return out

    ret_n    = _norm_rows(ret_scores)
    rerank_n = _norm_rows(rerank_scores)

    def compute_recalls(preds_matrix):
        rec = np.zeros(len(recall_values))
        for q_idx, preds in enumerate(preds_matrix):
            pos = positives_per_query[q_idx]
            for i, n in enumerate(recall_values):
                if np.any(np.isin(preds[:n], list(pos))):
                    rec[i:] += 1
                    break
        return rec / Q * 100

    results = {}
    for alpha in alphas:
        blended = np.full_like(ret_n, np.nan)
        for q in range(Q):
            if not has_rerank[q] or alpha == 1.0:
                # Pure retrieval
                blended[q] = ret_n[q]
            elif alpha == 0.0:
                # Pure rerank; fall back to retrieval where rerank is missing
                blended[q] = np.where(np.isnan(rerank_n[q]),
                                      ret_n[q], rerank_n[q])
            else:
                rr         = np.where(np.isnan(rerank_n[q]), 0.0, rerank_n[q])
                blended[q] = alpha * ret_n[q] + (1.0 - alpha) * rr

        final_preds = np.zeros_like(predictions)
        for q in range(Q):
            order           = np.argsort(-blended[q])
            final_preds[q]  = predictions[q][order]

        results[alpha] = compute_recalls(final_preds)

    # Pretty-print summary
    header = (f"\n{'='*70}\n  Alpha ablation  (normalize={normalize})\n"
              f"{'='*70}")
    logger.info(header); print(header)
    rv_str = "  ".join([f"R@{v:>2}" for v in recall_values])
    logger.info(f"  {'alpha':>6}  {rv_str}")
    print(     f"  {'alpha':>6}  {rv_str}")
    for alpha, rec in results.items():
        row = "  ".join([f"{r:5.1f}" for r in rec])
        logger.info(f"  {alpha:>6.2f}  {row}")
        print(     f"  {alpha:>6.2f}  {row}")
    logger.info('='*70); print('='*70)

    return results


def verify_cache_query_order(cache, test_ds):
    cached  = cache['query_filenames']
    current = test_ds.images_paths[test_ds.num_database:]
    if len(cached) != len(current):
        logger.warning(
            f"Cache has {len(cached)} queries but dataset has {len(current)} "
            "— predictions may be misaligned!")
        return
    n_bad = sum(c != d for c, d in zip(cached, current))
    if n_bad:
        logger.warning(
            f"Cache query filenames: {n_bad}/{len(cached)} mismatches.")
    else:
        logger.info("Cache query filenames match current dataset ✓")


# ---------------------------------------------------------------------------

def encode_batch(model, args, images, texts, indices, all_descriptors,
                 vision_descriptors, text_descriptors, img_local_descs,
                 text_local_descs, text_tokens_all, text_attentions):
    if args.bfloat16:
        images = images.bfloat16()

    t0 = time.perf_counter()

    if args.cross_modal <= 1:
        if args.is_image:
            image_features = model.encode_image(images.to(args.device))
            vision_descriptors[indices.numpy(), :] = image_features.cpu().float().numpy()
        text_features = model.encode_text(texts)
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
    else:
        descriptors, text_features, img_local, text_local, text_tokens, text_attention_mask = \
            model.encode_single(images.to(args.device), texts)
        if args.is_image and descriptors is not None:
            vision_descriptors[indices.numpy(), :] = descriptors.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
        if img_local is not None:
            N = min(img_local.shape[1], img_local_descs.shape[1])
            img_local_descs[indices.numpy(), :N, :] = img_local.cpu().float().numpy()[:, :N, :]
        if text_local is not None:
            L = min(text_local.shape[1], text_local_descs.shape[1])
            text_local_descs[indices.numpy(), :L, :] = text_local.cpu().float().numpy()[:, :L, :]
        if text_attention_mask is not None:
            L = min(text_attention_mask.shape[1], text_attentions.shape[1])
            text_attentions[indices.numpy(), :L] = text_attention_mask.cpu().long().numpy()[:, :L]

    elapsed = time.perf_counter() - t0
    TIMING.record('encode_batch', len(indices), elapsed)


def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors,
                            queries_descriptors, max_results):
    t0 = time.perf_counter()
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    database_descriptors = normlize_features(database_descriptors)
    queries_descriptors  = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    elapsed = time.perf_counter() - t0
    TIMING.record('faiss_stage1', len(queries_descriptors), elapsed)
    return scores, predictions


@torch.no_grad()
def rerank_predictions(
    model, test_ds, predictions, img_local_descs, text_local_desc,
    text_attentions, vision_descs=None, text_descs=None,
    max_rerank_k=20, rerank_batch=64, alpha=0.0, device='cuda',
):
    """LaVPR reranker. Returns (reranked_predictions, reranked_scores)."""
    reranker = model.reranker_network
    if reranker is None:
        raise RuntimeError("reranker_network is None.")
    reranker.eval()
    w_dtype = next(reranker.parameters()).dtype
    num_db  = test_ds.num_database
    num_q   = test_ds.num_queries
    n_cand  = min(max_rerank_k, predictions.shape[1])

    img_proj_in = reranker.img_proj.weight.shape[1]
    txt_proj_in = reranker.txt_proj.weight.shape[1]
    got_img_dim = img_local_descs.shape[-1]
    got_txt_dim = text_local_desc.shape[-1]

    logger.info(f"LaVPR reranker  img_proj expects {img_proj_in}  "
                f"got {got_img_dim}  |  "
                f"txt_proj expects {txt_proj_in}  got {got_txt_dim}")

    if got_img_dim != img_proj_in:
        raise RuntimeError(
            f"rerank_by_lavpr: img local dim mismatch — "
            f"reranker expects {img_proj_in}, buffer has {got_img_dim}.")
    if got_txt_dim != txt_proj_in:
        raise RuntimeError(
            f"rerank_by_lavpr: text local dim mismatch — "
            f"reranker expects {txt_proj_in}, buffer has {got_txt_dim}.")

    logger.info(f"LaVPR reranking top-{n_cand} for {num_q} queries  "
                f"(batch={rerank_batch}  alpha={alpha}  dtype={w_dtype})")
    reranked        = predictions.copy()
    reranked_scores = np.full((num_q, predictions.shape[1]), np.nan, dtype=np.float32)

    for q_start in tqdm(range(0, num_q, rerank_batch), desc="LaVPR rerank"):
        t0    = time.perf_counter()
        q_end = min(q_start + rerank_batch, num_q)
        bsz   = q_end - q_start
        q_ds_idxs = [num_db + q for q in range(q_start, q_end)]

        txt_locals, txt_masks = [], []
        for i in q_ds_idxs:
            full_mask = text_attentions[i]
            real_len  = max(int(full_mask.sum()), 1)
            txt_locals.append(text_local_desc[i][:real_len])
            txt_masks .append(full_mask         [:real_len])

        max_L = max(t.shape[0] for t in txt_locals)
        D_txt = txt_locals[0].shape[1]
        txt_np = np.zeros((bsz, max_L, D_txt), dtype=np.float32)
        msk_np = np.zeros((bsz, max_L),        dtype=np.float32)
        for b, (tl, tm) in enumerate(zip(txt_locals, txt_masks)):
            L = tl.shape[0]; txt_np[b, :L] = tl; msk_np[b, :L] = tm

        txt_t = torch.from_numpy(txt_np).to(device)
        msk_t = torch.from_numpy(msk_np).bool().to(device)

        cand_idx = predictions[q_start:q_end, :n_cand]
        img_np   = np.stack([img_local_descs[cand_idx[b]]
                             for b in range(bsz)], axis=0)
        img_t    = torch.from_numpy(img_np).to(device)

        K, N_p, D_img = n_cand, img_t.shape[2], img_t.shape[3]
        L_   = txt_t.shape[1]
        D_t_ = txt_t.shape[2]

        imgs_flat = img_t.reshape(bsz * K, N_p, D_img).to(w_dtype)
        txt_flat  = (txt_t.unsqueeze(1)
                         .expand(-1, K, -1, -1)
                         .reshape(bsz * K, L_, D_t_)
                         .contiguous().to(w_dtype))
        msk_flat  = (msk_t.unsqueeze(1)
                          .expand(-1, K, -1)
                          .reshape(bsz * K, L_)
                          .contiguous())

        _, scores_flat = reranker(imgs_flat, txt_flat, msk_flat)
        local_scores   = scores_flat.float().view(bsz, K).cpu().numpy()

        if alpha > 0.0 and vision_descs is not None and text_descs is not None:
            for b in range(bsz):
                q_ds_idx   = num_db + q_start + b
                q_global   = text_descs[q_ds_idx]
                db_globals = vision_descs[cand_idx[b]]
                q_norm     = q_global   / (np.linalg.norm(q_global) + 1e-8)
                db_norm    = db_globals / (np.linalg.norm(
                    db_globals, axis=1, keepdims=True) + 1e-8)
                global_sim = db_norm @ q_norm
                local_scores[b] = (alpha * _zn_np(global_sim)
                                   + (1.0 - alpha) * _zn_np(local_scores[b]))

        for b in range(bsz):
            order = np.argsort(-local_scores[b])
            reranked[q_start + b, :n_cand]        = cand_idx[b][order]
            reranked_scores[q_start + b, :n_cand] = local_scores[b][order]

        TIMING.record('rerank_lavpr_batch', bsz * n_cand, time.perf_counter() - t0)

    logger.info("LaVPR reranking complete.")
    return reranked, reranked_scores



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    start_time = datetime.now()

    logger.remove()
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log",
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD     = {'mean': [0.48145466, 0.4578275, 0.40821073],
                         'std':  [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD   = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    
    model = LaVPR_wrapper(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")

    
    num_img_tokens  = model.img_local_tokens
    num_text_tokens = model.txt_local_tokens
    txt_local_dim   = model.txt_local_dim        
    img_local_dim   = model.img_local_dim

    if 'eval_conditions' not in locals():
        eval_conditions = [(None, getattr(args, 'queries_folder', None),
                                  getattr(args, 'queries_csv', None))]

    max_results       = max(args.recall_values)
    need_local        = getattr(args, 'reranker',       False)                         
    stage1_cache_path = getattr(args, 'stage1_cache', None)

    db_vision_descriptors = None
    db_text_descriptors   = None
    db_all_descriptors    = None
    db_img_local_descs    = None
    db_text_local_descs   = None
    db_text_tokens_all    = None
    db_text_attentions    = None

    all_cond_results      = []
    all_retrieval_recalls = []
    all_reranking_recalls = []
    is_msls_challenge     = False
    reranking_enabled     = False

    for cond_idx, (cond_name, q_folder, q_csv) in enumerate(eval_conditions):
        args.queries_folder = q_folder
        args.queries_csv    = q_csv

        if cond_name:
            banner = (f"\n{'='*75}\n  Evaluating SVOX condition: {cond_name} "
                      f"({cond_idx+1}/{len(eval_conditions)})\n"
                      f"  Queries: {q_folder}\n  CSV: {q_csv}\n{'='*75}")
            logger.info(banner); print(banner)

        if 'msls_challenge' in getattr(args, 'image_root', ''):
            test_ds = MSLSTest(
                dataset_root=args.database_folder, image_root=args.image_root,
                csv_path=args.queries_csv, mean_std=dataset_mean_std,
                image_size=args.image_size)
            is_msls_challenge = True
        else:
            test_ds = TestDataset(
                args.database_folder, args.queries_folder, args.queries_csv,
                args.image_root, mean_std=dataset_mean_std,
                positive_dist_threshold=args.positive_dist_threshold,
                image_size=args.image_size, use_labels=args.use_labels)
        logger.info(f"Testing on {test_ds}")

        num_db    = test_ds.num_database
        num_q     = test_ds.num_queries
        total_len = len(test_ds)

        vision_descriptors = np.zeros((total_len, model.encoder_dim), dtype="float32")
        text_descriptors   = np.zeros((total_len, model.encoder_dim), dtype="float32")
        all_descriptors    = np.zeros((total_len, model.encoder_dim), dtype="float32")
        img_local_descs    = np.zeros((total_len, num_img_tokens,  img_local_dim), dtype="float32")
        text_local_descs   = np.zeros((total_len, num_text_tokens, txt_local_dim), dtype="float32")
        text_tokens_all    = np.zeros((total_len, num_text_tokens), dtype=np.int64)
        text_attentions    = np.zeros((total_len, num_text_tokens), dtype=np.int64)

        cache_suffix  = f"_{cond_name}" if cond_name else ""
        load_cache    = (Path(stage1_cache_path).with_suffix(f"{cache_suffix}.csv")
                         if stage1_cache_path else None)
        cache_loaded  = load_cache is not None and load_cache.exists()
        save_cache    = log_dir / f"stage1_cache{cache_suffix}.csv"
        skip_encoding = cache_loaded and not need_local

        if skip_encoding:
            logger.info("Stage-1 cache found — skipping gallery and query encoding.")

        with torch.inference_mode():
            if skip_encoding:
                pass
            elif db_vision_descriptors is None:
                logger.info(f"Extracting database descriptors ({num_db} images)")
                t_gallery_start = time.perf_counter()
                database_subset_ds  = Subset(test_ds, list(range(num_db)))
                database_dataloader = DataLoader(
                    dataset=database_subset_ds,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)
                for images, indices, texts in tqdm(database_dataloader,
                                                   desc="Encoding gallery"):
                    encode_batch(model, args, images, texts, indices,
                                 all_descriptors, vision_descriptors,
                                 text_descriptors, img_local_descs,
                                 text_local_descs, text_tokens_all,
                                 text_attentions)
                TIMING.record('encode_gallery_total', num_db,
                              time.perf_counter() - t_gallery_start)
                db_vision_descriptors = vision_descriptors[:num_db].copy()
                db_text_descriptors   = text_descriptors[:num_db].copy()
                db_all_descriptors    = all_descriptors[:num_db].copy()
                db_img_local_descs    = img_local_descs[:num_db].copy()
                db_text_local_descs   = text_local_descs[:num_db].copy()
                db_text_tokens_all    = text_tokens_all[:num_db].copy()
                db_text_attentions    = text_attentions[:num_db].copy()
            else:
                logger.info(f"Reusing pre-encoded gallery ({num_db} images)")
                vision_descriptors[:num_db] = db_vision_descriptors
                text_descriptors[:num_db]   = db_text_descriptors
                all_descriptors[:num_db]    = db_all_descriptors
                img_local_descs[:num_db]    = db_img_local_descs
                text_local_descs[:num_db]   = db_text_local_descs
                text_tokens_all[:num_db]    = db_text_tokens_all
                text_attentions[:num_db]    = db_text_attentions

            if not skip_encoding:
                desc_label = (f"Encoding {cond_name} queries"
                              if cond_name else "Extracting query descriptors")
                t_query_start = time.perf_counter()
                queries_subset_ds  = Subset(test_ds, list(range(num_db, num_db + num_q)))
                queries_dataloader = DataLoader(
                    dataset=queries_subset_ds,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)
                for images, indices, texts in tqdm(queries_dataloader,
                                                   desc=desc_label):
                    encode_batch(model, args, images, texts, indices,
                                 all_descriptors, vision_descriptors,
                                 text_descriptors, img_local_descs,
                                 text_local_descs, text_tokens_all,
                                 text_attentions)
                TIMING.record('encode_queries_total', num_q,
                              time.perf_counter() - t_query_start)

        if need_local:
            logger.info(f"local buffers | img {np.abs(img_local_descs).sum():.1f}  "
                        f"text {np.abs(text_local_descs).sum():.1f}")    
            
        database_descriptors = vision_descriptors[:num_db]
        queries_descriptors  = text_descriptors[num_db:]        
        
        scores, retrieval_predictions = get_queries_predictions(
                model.encoder_dim, database_descriptors, all_descriptors,
                queries_descriptors, max_results)
        
        reranking_predictions = retrieval_predictions.copy()
        reranking_scores      = None
        reranking_enabled     = False

        if getattr(args, 'reranker', False) and args.cross_modal == 3 \
                and args.max_sim == 0:
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions, reranking_scores = rerank_predictions(
                model, test_ds, reranking_predictions,
                img_local_descs, text_local_descs, text_attentions,
                vision_descs=vision_descriptors, text_descs=text_descriptors,
                max_rerank_k=max_rerank_k,
                rerank_batch=getattr(args, 'rerank_batch', 64),
                alpha=getattr(args, 'rerank_alpha', 0.0),
                device=args.device)

        if not reranking_enabled and (
            bool(getattr(args, 'reranker',       False))
            
        ):
            reranking_enabled = True

        if reranking_enabled:
            # ── Stage-2 cache (rerank order, rerank scores) ──────────────────
            save_rerank_cache(
                log_dir / f"stage2_cache{cache_suffix}.csv",
                reranking_predictions, reranking_scores,
                test_ds.images_paths[num_db:],
                test_ds.images_paths[:num_db],
                topk=STAGE1_TOPK_SAVE)

            # ── Combined cache (retrieval order, both scores side-by-side) ───
            save_combined_cache(
                log_dir / f"combined_cache{cache_suffix}.csv",
                retrieval_predictions, scores,
                reranking_predictions, reranking_scores,
                test_ds.images_paths[num_db:],
                test_ds.images_paths[:num_db],
                topk=STAGE1_TOPK_SAVE)

        if is_msls_challenge:
            final_predictions = (reranking_predictions
                                 if reranking_enabled else retrieval_predictions)
            test_ds.save_predictions(
                final_predictions, log_dir / "msls_challenge_predictions.txt", k=25)
        else:
            if args.use_labels:
                positives_per_query = test_ds.get_positives()

                def compute_recalls(preds_matrix):
                    rec = np.zeros(len(args.recall_values))
                    for query_index, preds in enumerate(preds_matrix):
                        for i, n in enumerate(args.recall_values):
                            if np.any(np.isin(preds[:n],
                                              positives_per_query[query_index])):
                                rec[i:] += 1; break
                    rec   = rec / test_ds.num_queries * 100
                    r_str = ", ".join([f"R@{val}: {r:.1f}"
                                      for val, r in zip(args.recall_values, rec)])
                    return rec, r_str

                retrieval_recalls, retrieval_recalls_str = compute_recalls(
                    retrieval_predictions)
                all_retrieval_recalls.append(retrieval_recalls)

                prefix = f"[SVOX {cond_name}] " if cond_name else ""
                if reranking_enabled:
                    reranking_recalls, reranking_recalls_str = compute_recalls(
                        reranking_predictions)
                    all_reranking_recalls.append(reranking_recalls)
                    logger.info(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    logger.info(f"{prefix}Reranking Recall: {reranking_recalls_str}")
                    all_cond_results.append(
                        (cond_name or "Overall",
                         retrieval_recalls_str, reranking_recalls_str))

                    # ── Offline alpha ablation ────────────────────────────────
                    if getattr(args, 'ablate_alpha', False):
                        combined = load_combined_cache(
                            log_dir / f"combined_cache{cache_suffix}.csv",
                            test_ds.images_paths[:num_db])
                        ablate_alpha(
                            combined,
                            positives_per_query=positives_per_query,
                            recall_values=args.recall_values,
                            alphas=getattr(args, 'ablate_alphas', None),
                            normalize=getattr(args, 'ablate_norm', 'zscore'))
                else:
                    logger.info(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    print(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    all_cond_results.append(
                        (cond_name or "Overall", retrieval_recalls_str, None))

                model_path    = (args.lora_path if args.lora_path is not None
                                 else args.model_path)
                csv_model_tag = (f"{args.model_name} [SVOX {cond_name}]"
                                 if cond_name else args.model_name)
                with open("eval_vpr_results.csv", "a") as f:
                    if reranking_enabled:
                        f.write(f"{model_path},{csv_model_tag},"
                                f"Retrieval: {retrieval_recalls_str} | "
                                f"Reranking: {reranking_recalls_str}\n")
                    else:
                        f.write(f"{model_path},{csv_model_tag},"
                                f"{retrieval_recalls_str}\n")

        if args.num_preds_to_save != 0:
            logger.info("Saving final predictions")
            final_predictions = (reranking_predictions
                                 if reranking_enabled else retrieval_predictions)
            save_dir = log_dir / cond_name if cond_name else log_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            visualizations.save_preds(
                final_predictions[:, :args.num_preds_to_save], test_ds,
                save_dir, args.save_only_wrong_preds, args.use_labels,
                test_ds.images_paths_csv, texts=test_ds.descriptions)

    if len(eval_conditions) > 1 and all_retrieval_recalls:
        mean_retrieval     = np.mean(all_retrieval_recalls, axis=0)
        mean_retrieval_str = ", ".join([f"R@{val}: {r:.1f}"
                                        for val, r in zip(args.recall_values,
                                                          mean_retrieval)])
        header = (f"\n{'='*75}\n  SVOX Evaluation Summary "
                  f"(All {len(eval_conditions)} Conditions)\n{'='*75}")
        logger.info(header); print(header)
        for c_name, r_str, rr_str in all_cond_results:
            line = (f"  {c_name:<10} | Retrieval: {r_str}"
                    + (f" | Reranking: {rr_str}" if rr_str else ""))
            logger.info(line); print(line)
        sep = f"{'-'*75}"
        logger.info(sep); print(sep)
        model_path = (args.lora_path if args.lora_path is not None
                      else args.model_path)
        if reranking_enabled and all_reranking_recalls:
            mean_reranking     = np.mean(all_reranking_recalls, axis=0)
            mean_reranking_str = ", ".join([f"R@{val}: {r:.1f}"
                                            for val, r in zip(args.recall_values,
                                                              mean_reranking)])
            mean_line = (f"  {'MEAN':<10} | Retrieval: {mean_retrieval_str} "
                         f"| Reranking: {mean_reranking_str}")
            logger.info(mean_line); print(mean_line)
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name} [SVOX Mean],"
                        f"Retrieval: {mean_retrieval_str} | "
                        f"Reranking: {mean_reranking_str}\n")
        else:
            mean_line = f"  {'MEAN':<10} | Retrieval: {mean_retrieval_str}"
            logger.info(mean_line); print(mean_line)
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name} [SVOX Mean],"
                        f"{mean_retrieval_str}\n")
        footer = f"{'='*75}\n"
        logger.info(footer); print(footer)

    # ── Timing summary ────────────────────────────────────────────────────────
    # timing_str = TIMING.summary()
    # logger.info(timing_str)
    # print(timing_str)


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)