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
from medeha.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from medeha.datasets.data_module import DataModule
from medeha.datasets.transforms import DataTransforms, Moco2Transform
from medeha.pretrain.module import MedEHA
from medeha.models.downstream.ssl_classifier import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVEPATH = str(OUTPUT_DIR)

def cli_main():
    parser = ArgumentParser("Finetuning of classifier task for MedEHA")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Path to MedEHA Lightning checkpoint")
    parser.add_argument("--dataset", type=str,
                        default="chexpert", help="chexpert, rsna or covidx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
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

    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers,
                                valid_mode=args.valid_mode, random_state=args.seed)
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers,
                                valid_mode=args.valid_mode, random_state=args.seed)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers,
                                valid_mode=args.valid_mode, random_state=args.seed)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    medeha = MedEHA.load_from_checkpoint(args.ckpt_path, strict=False)

    args.model_name = medeha.hparams.img_encoder
    args.backbone = medeha.img_encoder_q
    args.in_features = args.backbone.feature_dim
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    model = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"{SAVEPATH}/medeha_ckpts/medeha2_classifier/{args.experiment_name}/{args.dataset}_{args.data_pct}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"{SAVEPATH}/medeha_data/medeha2_classifier/{args.dataset}_{args.data_pct}/{args.experiment_name}")
    os.makedirs(logger_dir, exist_ok=True)
    # wandb_logger = WandbLogger(
    #     project="MedEHA2_classifier",
    #     save_dir=logger_dir,
    #     name=f"MedEHA2_classifier_{args.dataset}_{args.data_pct}_{args.experiment_name}_{extension}")
    tensorboard_logger = TensorBoardLogger(
        save_dir=logger_dir,
        name=f"{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=tensorboard_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
