"""
Zero-shot image classification using frozen MedEHA image + text encoders.

Image and text features are L2-normalized projected embeddings (global_embed),
then combined CLIP-style: multiclass uses softmax over class prompts; multilabel
uses paired positive/negative prompts per label and sigmoid logits.
"""
from __future__ import annotations

import json
from argparse import ArgumentParser
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics import AUROC, Accuracy

from medeha.constants import CHEXPERT_COMPETITION_TASKS
from medeha.datasets.classification_dataset import (
    CheXpertImageDataset,
    COVIDXImageDataset,
    RSNAImageDataset,
)
from medeha.datasets.data_module import DataModule
from medeha.datasets.transforms import DataTransforms, Moco2Transform
from medeha.models.backbones.encoder import BertEncoder, ImageEncoder
from medeha.pretrain.module import MedEHA


def _encode_text_global(
    text_encoder: BertEncoder,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode short prompts to L2-normalized global text embeddings [K, D]."""
    tokenizer = text_encoder.tokenizer
    enc = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    token_type = torch.zeros_like(ids)
    report_feat, _, _, _ = text_encoder(ids, attn, token_type)
    emb = text_encoder.global_embed(report_feat)
    return F.normalize(emb.float(), dim=-1)


def _encode_image_global(
    img_encoder: ImageEncoder,
    images: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        feats, _, _ = img_encoder(images, None)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        emb = img_encoder.global_embed(feats)
    return F.normalize(emb.float(), dim=-1)


def default_prompts_for_dataset(
    dataset: str,
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Returns (positive_prompts, negative_prompts).
    negative_prompts is None for multiclass (single prompt per class).
    """
    if dataset == "chexpert":
        pos, neg = [], []
        for name in CHEXPERT_COMPETITION_TASKS:
            pos.append(
                f"a frontal chest X-ray with radiographic signs of {name.lower()}."
            )
            neg.append(
                f"a frontal chest X-ray without radiographic signs of {name.lower()}."
            )
        return pos, neg
    if dataset == "rsna":
        return (
            ["a chest X-ray showing pneumonia."],
            ["a chest X-ray without pneumonia."],
        )
    if dataset == "covidx":
        return (
            [
                "a normal chest X-ray without acute lung pathology.",
                "a chest X-ray with COVID-19 lung involvement.",
                "a chest X-ray with non-COVID pneumonia.",
            ],
            None,
        )
    raise ValueError(f"Unknown dataset for default prompts: {dataset}")


class MedEHAZeroShotClassifier(LightningModule):
    def __init__(
        self,
        img_encoder: ImageEncoder,
        text_encoder: BertEncoder,
        prompts_positive: List[str],
        prompts_negative: Optional[List[str]],
        multilabel: bool,
        num_classes: int,
        logit_scale: float = 100.0,
        model_name: str = "",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["img_encoder", "text_encoder"])

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        for m in (self.img_encoder, self.text_encoder):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        self.prompts_positive = list(prompts_positive)
        self.prompts_negative = (
            list(prompts_negative) if prompts_negative is not None else None
        )
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.register_buffer(
            "logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32)
        )

        if multilabel:
            if self.prompts_negative is None or len(self.prompts_negative) != len(
                self.prompts_positive
            ):
                raise ValueError(
                    "multilabel zero-shot requires prompts_negative of same length "
                    "as prompts_positive."
                )
            if len(self.prompts_positive) != num_classes:
                raise ValueError(
                    f"Expected {num_classes} prompt pairs, got {len(self.prompts_positive)}."
                )
            if num_classes == 1:
                self.test_auc = AUROC(task="binary", num_classes=num_classes)
            else:
                self.test_auc = AUROC(task="multilabel", num_classes=num_classes)
        else:
            if len(self.prompts_positive) != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class prompts, got {len(self.prompts_positive)}."
                )
            if num_classes == 1:
                self.test_acc = Accuracy(
                    task="binary", num_classes=num_classes, top_k=1
                )
                self.test_auc = AUROC(task="binary", num_classes=num_classes)
            else:
                self.test_acc = Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=1
                )
                self.test_auc = AUROC(task="multiclass", num_classes=num_classes)

        self.register_parameter("_pl_opt_holder", nn.Parameter(torch.zeros(1)))

        self._text_pos: Optional[torch.Tensor] = None
        self._text_neg: Optional[torch.Tensor] = None
        self._text_cache_device: Optional[torch.device] = None

    def _ensure_text_cache(self) -> None:
        dev = self.device
        if (
            self._text_pos is not None
            and self._text_cache_device == dev
            and self._text_pos.device == dev
        ):
            return
        with torch.no_grad():
            self._text_pos = _encode_text_global(
                self.text_encoder, self.prompts_positive, dev
            )
            if self.prompts_negative is not None:
                self._text_neg = _encode_text_global(
                    self.text_encoder, self.prompts_negative, dev
                )
            else:
                self._text_neg = None
        self._text_cache_device = dev

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._ensure_text_cache()
        img_e = _encode_image_global(self.img_encoder, images)
        scale = self.logit_scale.to(img_e.dtype)
        if self.multilabel:
            assert self._text_neg is not None
            pos = img_e @ self._text_pos.T
            neg = img_e @ self._text_neg.T
            return scale * (pos - neg)
        logits = scale * (img_e @ self._text_pos.T)
        return logits

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        if self.multilabel:
            probs = torch.sigmoid(logits)
            self.test_auc.update(probs, y.long())
            self.log("test_auc", self.test_auc, on_epoch=True)
        else:
            if y.dim() > 1 and y.size(-1) == 1:
                y = y.squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            pred_cls = probs.argmax(dim=-1)
            self.test_acc(pred_cls, y.long())
            self.test_auc.update(probs, y.long())
            self.log("test_acc", self.test_acc, on_epoch=True)
            self.log("test_auc", self.test_auc, on_epoch=True)

    def configure_optimizers(self):
        # No training; placeholder for Lightning when only running test/validate.
        return torch.optim.AdamW([self._pl_opt_holder], lr=1e-10)


def load_prompts_from_json(path: str) -> Tuple[List[str], Optional[List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        pos = data.get("positive") or data.get("prompts_positive")
        neg = data.get("negative") or data.get("prompts_negative")
        if pos is None:
            raise ValueError("JSON must contain 'positive' or 'prompts_positive' list.")
        return list(pos), (list(neg) if neg is not None else None)
    if isinstance(data, list):
        return [str(x) for x in data], None
    raise ValueError("JSON must be a list of prompts or an object with positive/negative.")


def cli_main():
    parser = ArgumentParser(
        description="Zero-shot classification with a frozen MedEHA checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to MedEHA Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chexpert",
        choices=("chexpert", "rsna", "covidx"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument("--valid_mode", type=str, default="valid")
    parser.add_argument(
        "--logit_scale",
        type=float,
        default=100.0,
        help="Scalar multiplier on image-text cosine similarities (CLIP-style).",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="",
        help="Optional JSON: list of class prompts, or {positive: [...], negative: [...]}.",
    )
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        dm = DataModule(
            CheXpertImageDataset,
            None,
            Moco2Transform,
            args.data_pct,
            args.batch_size,
            args.num_workers,
            valid_mode=args.valid_mode,
            random_state=args.seed,
        )
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        dm = DataModule(
            RSNAImageDataset,
            None,
            DataTransforms,
            args.data_pct,
            args.batch_size,
            args.num_workers,
            valid_mode=args.valid_mode,
            random_state=args.seed,
        )
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        dm = DataModule(
            COVIDXImageDataset,
            None,
            DataTransforms,
            args.data_pct,
            args.batch_size,
            args.num_workers,
            valid_mode=args.valid_mode,
            random_state=args.seed,
        )
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"Unsupported dataset: {args.dataset}")

    medeha = MedEHA.load_from_checkpoint(args.ckpt_path, strict=False)

    if args.prompts_json:
        pos, neg = load_prompts_from_json(args.prompts_json)
    else:
        pos, neg = default_prompts_for_dataset(args.dataset)

    model = MedEHAZeroShotClassifier(
        img_encoder=medeha.img_encoder_q,
        text_encoder=medeha.text_encoder_q,
        prompts_positive=pos,
        prompts_negative=neg,
        multilabel=multilabel,
        num_classes=num_classes,
        logit_scale=args.logit_scale,
        model_name=medeha.hparams.img_encoder,
    )

    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
