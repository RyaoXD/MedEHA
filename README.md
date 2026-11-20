# MedEHA

Official PyTorch implementation of **MedEHA: Entity-aware hierarchical alignment for medical vision-language representation learning**.

> Xiangmin Kong, Xibin Jia, Dawei Yang, Luo Wang, Kelei Jin, Zhenghan Yang

MedEHA is a medical vision–language pre-training (MedVLP) framework that represents each radiology report as medical entity triples and aligns them with the image from coarse to fine:

- **Hierarchical Entity Modeling** — encode each triple (finding, anatomical location, existence status) and then the dependencies among triples in the same report
- **Global Image–Report Alignment (GA)** — image–report contrastive alignment
- **Prototype-based Semantic Alignment (PSA)** — shared prototype assignments for image and report embeddings
- **Language-Guided Alignment (LGA)** — group image patches by entity semantics and align each entity with its group
- **Pathology Fine-grained Alignment (PFA)** — align each entity with an attention-refined local visual feature

## Requirements

```bash
conda create -n medeha python=3.8 -y
conda activate medeha
# Install a CUDA-compatible torch / torchvision build first, then:
pip install -r requirements.txt
pip install -e .
```

Place [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) under `model_files/Bio_ClinicalBERT/` (or set `MEDEHA_BERT_DIR`).

## Datasets

| Dataset | Usage | Download |
| --- | --- | --- |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | Pre-training | PhysioNet (credentialed access) |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Classification / zero-shot | Stanford ML Group |
| [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Classification / segmentation / detection | Kaggle |
| [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | Segmentation | Kaggle |
| [COVIDx](https://github.com/lindawangg/COVID-Net) | Classification / zero-shot | COVID-Net |
| [Object-CXR](https://jfhealthcare.github.io/object-CXR/) | Detection | JF Healthcare |

Set the dataset root, then preprocess:

```bash
export MEDEHA_DATA_DIR=/path/to/dataset
# optional: export MEDEHA_OUTPUT_DIR=/path/to/outputs
bash preprocess.sh
```

Entity triples follow the [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP) / [RadGraph](https://arxiv.org/abs/2106.14463) release. Place them under `data/entity/` (or set `MEDEHA_ENTITY_DIR`):

```text
landmark_observation_adj_mtx.npy
observation_explanation.json
anaomy_explanation.json
```

## Pre-training

```bash
bash run_pretrain.sh
```

## Downstream Evaluation

Set `CKPT=/path/to/medeha.ckpt`, then run:

```bash
# Classification (CheXpert / RSNA / COVIDx)
CKPT=/path/to/medeha.ckpt bash run_downstream_classifier.sh

# Segmentation (RSNA / SIIM)
CKPT=/path/to/medeha.ckpt bash run_downstream_segmenter.sh

# Detection (RSNA / Object-CXR)
CKPT=/path/to/medeha.ckpt bash run_downstream_detector.sh
```

Zero-shot classification:

```bash
python ./medeha/models/downstream/zero_shot_classifier.py --dataset chexpert --ckpt_path /path/to/medeha.ckpt
```

## Citation

If you use this code, please cite:

```bibtex
@article{kong2026medeha,
  title   = {MedEHA: Entity-aware hierarchical alignment for medical vision-language representation learning},
  author  = {Kong, Xiangmin and Jia, Xibin and Yang, Dawei and Wang, Luo and Jin, Kelei and Yang, Zhenghan},
  year    = {2026}
}
```

## Acknowledgements

This implementation builds on the medical VLP community (e.g., ConVIRT, GLoRIA, MGCA, MedKLIP). Entity triples follow the MedKLIP / RadGraph preprocessing. Please obtain datasets from their original providers under the corresponding licenses and data-use agreements.

## License

This repository is released under the [MIT License](LICENSE).
