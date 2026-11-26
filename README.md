<div align="center">

# CrossPlace

### Towards Closing the Performance Gap in Cross-Modal Place Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2611.34877-b31b1b.svg)](https://arxiv.org/abs/2611.34877)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/oferidan1/CrossPlace)

[Overview](#overview) •
[Setup](#setup) •
[Datasets](#datasets) •
[Training](#training) •
[Evaluation](#evaluation) •
[Results](#results) •
[Citation](#citation) •
[License](#license-and-data-usage)

</div>

---

Official implementation of **CrossPlace**, a new method for Cross-Modal Place Recognition (CMPR): given a natural-language description of a place, retrieve the matching image from a large geo-tagged database.


## Overview

Text-to-image place recognition still lags far behind its image-to-image counterpart. CrossPlace narrows that gap with two contributions:

- **A two-stage CMPR framework.** CrossPlace first retrieves candidates with a global image-text embedding, then reranks them using fine-grained image-text correspondences.
- **A broader evaluation protocol.** We extend CMPR evaluation to city-scale retrieval (SF-XL) and to severe weather and illumination changes (SVOX).

## Setup

The codebase has been tested with **PyTorch 2.9.0**, **CUDA 12.6**, and **xFormers**.

```bash
cd CrossPlace

# Create and activate an environment (optional but recommended)
conda create -n crossplace python=3.12
conda activate crossplace

# Install dependencies
pip install -r requirements.txt
```

## Datasets

### Prerequisite: LaVPR

CrossPlace builds on [LaVPR](https://github.com/oferidan1/LaVPR), which provides the text descriptions for the major VPR datasets. **Set it up first** by following the instructions in the LaVPR repository.

### Additional datasets

| Dataset | Purpose | How to get it |
| --- | --- | --- |
| **GSV-Cities** | Training | [gsv-cities](https://github.com/amaralibey/gsv-cities) |
| **SVOX** | Evaluation under weather and illumination changes | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) |
| **SF-XL** | City-scale evaluation | [CosPlace](https://github.com/gmberton/CosPlace) (dataset request form) |
| **Extended descriptions** | Text descriptions added by this work | Included in this repository (see below) |

Extract the extended descriptions before training or evaluating:

```bash
unzip datasets/descriptions_extended.zip -d datasets/descriptions
```

## Training

Training on GSV-Cities for 10 epochs takes about **10 hours** on a single NVIDIA RTX PRO 6000 Blackwell.

### Stage 1: Image-Text Alignment (Cross-Modal)

```bash
python train_cross.py \
    --train_csv=datasets/descriptions/gsv_cities_descriptions.csv \
    --image_root=PATH_TO_GSV_CITIES \
    --val_csv=datasets/descriptions/pitts30k_val_800_queries.csv \
    --val_image_root=PATH_TO_PITTS30K_VAL
```

### Stage 2: Reranker

```bash
python train_reranker.py \
    --model_type=cross_attn \
    --n_layers=1 \
    --d_modal=128 \
    --n_heads=4 \
    --loss_name=listwise \
    --max_cands=50
```

Checkpoints and logs are saved automatically to the `logs/` directory.

## Evaluation

### Pretrained model

| Model | Backbone | Embedding dim | Download |
| --- | --- | --- | --- |
| CrossPlace | DINOv2 + BGE-L | 768 | [🤗 cross_place.ckpt](https://huggingface.co/oferidan1/CrossPlace/blob/main/cross_place.ckpt) |

Download the checkpoint into `checkpoints/`:

```bash
hf download oferidan1/CrossPlace cross_place.ckpt --local-dir checkpoints
```

### Running the evaluation

```bash
python eval_vpr.py \
    --database_folder=PATH_TO_DB_IMAGES \
    --queries_folder=PATH_TO_QUERY_IMAGES \
    --image_root=PATH_TO_IMAGE_ROOT \
    --queries_csv=PATH_TO_DESCRIPTION_CSV \
    --model_name=dinov2 \
    --embeds_dim=768 \
    --model_path=checkpoints/cross_place.ckpt \
    --cross_modal=3 \
    --reranker=1 \
    --rerank_alpha=0.2
```

| Argument | Description |
| --- | --- |
| `--database_folder` | Folder containing the database (reference) images |
| `--queries_folder` | Folder containing the query images |
| `--image_root` | Root directory of the dataset images |
| `--queries_csv` | CSV file with the text descriptions of the queries |
| `--model_name` | Backbone architecture |
| `--embeds_dim` | Dimension of the global embedding |
| `--model_path` | Path to the trained checkpoint |
| `--reranker` | Enables the second-stage reranker (`1`) or disables it (`0`) |
| `--rerank_alpha` | Weight between first-stage and reranking scores |


## Results

Recall@1 (%) on the standard cross-modal benchmarks, the condition-specific SVOX splits, and the city-scale SF-XL splits. Best results in **bold**; the last row is the gain of the full pipeline over LaVPR.

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">Standard</th>
      <th colspan="5" align="center">SVOX</th>
      <th colspan="2" align="center">SF-XL</th>
    </tr>
    <tr>
      <th align="center">AmsterTime</th>
      <th align="center">MSLS</th>
      <th align="center">Pitts30k</th>
      <th align="center">Night</th>
      <th align="center">Sun</th>
      <th align="center">Rain</th>
      <th align="center">Snow</th>
      <th align="center">Overcast</th>
      <th align="center">v1</th>
      <th align="center">v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BLIP (zero-shot)</td>
      <td align="center">1.3</td>
      <td align="center">1.6</td>
      <td align="center">8.1</td>
      <td align="center">1.5</td>
      <td align="center">0.8</td>
      <td align="center">0.5</td>
      <td align="center">1.6</td>
      <td align="center">1.6</td>
      <td align="center">0.1</td>
      <td align="center">1.3</td>
    </tr>
    <tr>
      <td>LaVPR</td>
      <td align="center">13.2</td>
      <td align="center">38.0</td>
      <td align="center">50.2</td>
      <td align="center">15.7</td>
      <td align="center">30.9</td>
      <td align="center">30.7</td>
      <td align="center">29.5</td>
      <td align="center">36.8</td>
      <td align="center">9.5</td>
      <td align="center">19.6</td>
    </tr>
    <tr>
      <td>CrossPlace (Retrieval)</td>
      <td align="center">21.0</td>
      <td align="center">53.8</td>
      <td align="center">61.2</td>
      <td align="center">20.9</td>
      <td align="center">45.1</td>
      <td align="center">46.7</td>
      <td align="center">49.5</td>
      <td align="center">57.1</td>
      <td align="center">16.3</td>
      <td align="center">34.6</td>
    </tr>
    <tr>
      <td><b>CrossPlace (Reranking)</b></td>
      <td align="center"><b>24.0</b></td>
      <td align="center"><b>60.1</b></td>
      <td align="center"><b>66.4</b></td>
      <td align="center"><b>25.4</b></td>
      <td align="center"><b>52.0</b></td>
      <td align="center"><b>59.7</b></td>
      <td align="center"><b>61.5</b></td>
      <td align="center"><b>69.2</b></td>
      <td align="center"><b>22.7</b></td>
      <td align="center"><b>41.6</b></td>
    </tr>
    <tr>
      <td><i>Δ vs. LaVPR</i></td>
      <td align="center"><i>+10.8</i></td>
      <td align="center"><i>+22.1</i></td>
      <td align="center"><i>+16.2</i></td>
      <td align="center"><i>+9.7</i></td>
      <td align="center"><i>+21.1</i></td>
      <td align="center"><i>+29.0</i></td>
      <td align="center"><i>+32.0</i></td>
      <td align="center"><i>+32.4</i></td>
      <td align="center"><i>+13.2</i></td>
      <td align="center"><i>+22.0</i></td>
    </tr>
  </tbody>
</table>

See the [paper](https://arxiv.org/abs/2611.34877) for the full comparison.

## Citation

If you find this work useful, please cite:

```bibtex
@article{idan2026crossplace,
  title={Towards Closing the Performance Gap in Cross-Modal Place Recognition},
  author={Idan, Ofer and Badur, Dan and Shavit, Yoli},
  journal={arXiv preprint arXiv:2611.34877},
  year={2026}
}
```

## License and Data Usage

This repository contains two kinds of assets: source code and original text annotations. It does **not** redistribute any images.

| Component | License | Notes |
| --- | --- | --- |
| **Source code** | [LICENSE](LICENSE) | |
| **Text annotations** | [CC BY 4.0](LICENSE) | Original work by the authors (`datasets/descriptions_extended.zip`) |

### Third-party datasets

This repository provides text annotations for the datasets below but does not redistribute their images. Users must obtain the images from the original sources and comply with their licenses.

| Dataset | Original license | Status |
| --- | --- | --- |
| **SF-XL** | [MIT](https://github.com/gmberton/CosPlace/blob/main/LICENSE) | Referenced |
| **SVOX** | Queries: [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); database: [Google Street View terms](https://www.google.com/permissions/geoguidelines/) | Referenced |

## Acknowledgements

This repository builds on several excellent open-source projects:

- [LaVPR](https://github.com/oferidan1/LaVPR): text descriptions for the major VPR datasets, which this work builds on.
- [DINOv2](https://github.com/facebookresearch/dinov2): pretrained vision backbone.
- [BGE-L](https://huggingface.co/BAAI/bge-large-en): pretrained text encoder.
- [GSV-Cities](https://github.com/amaralibey/gsv-cities): large-scale VPR training dataset.
- [CosPlace](https://github.com/gmberton/CosPlace): the SF-XL dataset used for city-scale evaluation.
- [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader): download and formatting of the SVOX dataset.
- [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation): standardized VPR evaluation framework.