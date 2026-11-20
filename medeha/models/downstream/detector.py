import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from medeha.constants import OUTPUT_DIR
from medeha.datasets.data_module import DataModule
from medeha.datasets.detection_dataset import (OBJCXRDetectionDataset,
                                               RSNADetectionDataset)
from medeha.datasets.transforms import DetectionDataTransforms
from medeha.models.backbones.detector_backbone import ResNetDetector
from medeha.models.backbones.transformer_seg import SETRModel
from medeha.pretrain.module import MedEHA
from medeha.models.downstream.ssl_detector import SSLDetector

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cudnn.allow_tf32 = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVEPATH = str(OUTPUT_DIR)

def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for MedEHA")
    parser.add_argument("--base_model", type=str, default="vit")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Path to a MedEHA Lightning checkpoint")
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--experiment_name", type=str, default="medeha_baseline")
    parser.add_argument("--valid_mode", type=str, default="valid")
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if not args.ckpt_path:
        raise RuntimeError("--ckpt_path is required (path to a MedEHA checkpoint).")

    # args.deterministic = True
    args.max_epochs = 50
    # args.accelerator = "gpu"
    args.gpus = 1

    seed_everything(args.seed)

    if args.dataset == "rsna":
        datamodule = DataModule(RSNADetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers, args.valid_mode)
    elif args.dataset == "object_cxr":
        datamodule = DataModule(OBJCXRDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers, args.valid_mode)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")

    if args.base_model == "resnet_50":
        args.img_encoder = ResNetDetector("resnet_50")
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            if not isinstance(state_dict, dict):
                state_dict = {}
            ckpt_dict = dict()
            for k, v in state_dict.items():
                if k.startswith("img_encoder_q.model"):
                    new_k = ".".join(k.split(".")[2:])
                    ckpt_dict[new_k] = v

            args.img_encoder.model.load_state_dict(ckpt_dict)

        # Freeze encoder
        for param in args.img_encoder.parameters():
            param.requires_grad = False
    elif args.base_model == "vit":
        medeha = MedEHA.load_from_checkpoint(args.ckpt_path, strict=False)
        args.img_encoder = medeha.img_encoder_q.model

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"{SAVEPATH}/medeha_ckpts/medeha2_detection/{args.experiment_name}/{args.dataset}_{args.data_pct}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=20, verbose=False, mode="max"),
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"{SAVEPATH}/medeha_data/medeha2_detection/{args.dataset}_{args.data_pct}/{args.experiment_name}")
    os.makedirs(logger_dir, exist_ok=True)
    # wandb_logger = WandbLogger(
    #     project="MedEHA2_detection", save_dir=logger_dir,
    #     name=f"{args.dataset}_{args.data_pct}_{args.experiment_name}_{extension}")
    tensorboard_logger = TensorBoardLogger(
        save_dir=logger_dir,
        name=f"{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=tensorboard_logger
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
