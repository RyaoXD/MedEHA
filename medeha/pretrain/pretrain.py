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
from medeha.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                                         multimodal_collate_fn)
from medeha.datasets.transforms import DataTransforms
from medeha.pretrain.module import MedEHA

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVEPATH = str(OUTPUT_DIR)


def add_medeha_argparse_args(parent_parser):
    """CLI arguments for MedEHA pretraining (fed into MedEHA(**args.__dict__))."""
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # model parameters
    parser.add_argument("--img_encoder", type=str, default="vit_base")
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--emb_dim", type=int,
                        default=128, help="128, 256")
    # training parameters
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_pct", type=float, default=1.)
    # entity extraction parameters
    parser.add_argument("--entity_extraction", action="store_true", default=False)
    parser.add_argument(
        "--entity_fuse",
        action="store_true",
        default=False,
        help="Legacy fused entity-report alignment (used when --entity_align is off).",
    )
    parser.add_argument(
        "--entity_align",
        action="store_true",
        default=False,
        help="Decoupled report/entity/image triple alignment (overrides entity_fuse when both set).",
    )
    parser.add_argument(
        "--hyperparam_independ",
        action="store_true",
        default=False,
        help="With --entity_align: use lambda_ga_ri/re/ei instead of a single lambda_ga.",
    )
    parser.add_argument(
        "--projecter_independ",
        action="store_true",
        default=False,
        help="With --entity_align: separate global/local projection heads for entity branch.",
    )
    parser.add_argument("--lambda_ga_ri", type=float, default=1.0)
    parser.add_argument("--lambda_ga_re", type=float, default=1.0)
    parser.add_argument("--lambda_ga_ei", type=float, default=1.0)
    parser.add_argument("--entity_shape", type=tuple, default=(75, 51, 2))
    parser.add_argument("--max_n_entity", type=int, default=30)
    parser.add_argument("--entity_prompts", type=int, default=None)
    parser.add_argument("--with_report_feat", action="store_true", default=False)
    parser.add_argument("--with_token_feat", action="store_true", default=False,
                        help="If set (with --with_report_feat and decoder report layer), "
                             "cross-attend entities to report token/sentence memory instead of a single global vector.")
    parser.add_argument("--sentence_token", action="store_true", default=False,
                        help="With --with_token_feat: memory = ERMA-style per-short-sentence BERT features (batched sentences).")
    parser.add_argument("--max_report_sents", type=int, default=30,
                        help="Max short sentences per study when sentence_token (ERMA-style cap).")
    parser.add_argument("--max_words_sent", type=int, default=30,
                        help="Max BERT tokens per short sentence when sentence_token.")
    parser.add_argument("--twostep_attention", action="store_true", default=False)
    parser.add_argument("--entity_layer_type", type=str,
                        default="pool", help="pool, decoder")
    parser.add_argument("--n_entity_layers", type=int, default=4)
    parser.add_argument("--report_layer_type", type=str,
                        default="pool", help="pool, decoder")
    parser.add_argument("--n_report_layers", type=int, default=4)
    # Entity Slot Pred (ESP) / Pseudo Triplet Pred (PTP)
    parser.add_argument(
        "--enable_entity_pred",
        action="store_true",
        default=False,
        help="Enable Entity Slot Pred (ESP); requires --entity_extraction.",
    )
    parser.add_argument(
        "--enable_pseudo_triplet_pred",
        action="store_true",
        default=False,
        help="Enable Pseudo Triplet Pred (PTP); requires --entity_extraction and "
             "--language_group (fuse) or --visual_group (entity_align).",
    )
    # language group parameters
    parser.add_argument("--language_group", action="store_true", default=False)
    parser.add_argument("--visual_group", action="store_true", default=False)
    parser.add_argument("--group_threshold", type=float, default=0.0)
    parser.add_argument("--threshold_momentum", type=float, default=1.0)
    # prototype parameters
    parser.add_argument("--num_local_atten_layer_heads", type=int, default=1)
    parser.add_argument("--num_attn_weight_layer_heads", type=int, default=1)
    parser.add_argument("--freeze_prototypes_epochs", type=int or None, default=3)
    parser.add_argument("--instance_prototype", type=str or None,
                        default=None, help="random, mapping")
    parser.add_argument("--num_instance_prototypes", type=int, default=500)
    parser.add_argument("--token_prototype", type=str or None,
                        default=None, help="random")
    parser.add_argument("--num_token_prototypes", type=int, default=500)
    parser.add_argument("--pca_with_weight", action="store_true", default=False)
    parser.add_argument("--prototype_orth", action="store_true", default=False)
    # temperature parameters
    parser.add_argument("--learnable_temperature", action="store_true", default=False)
    parser.add_argument("--ga_temp", type=float, default=0.05)
    parser.add_argument("--pfa_temp", type=float, default=0.1)
    parser.add_argument("--ica_temp", type=float, default=0.2)
    parser.add_argument("--pca_temp", type=float, default=0.2)
    parser.add_argument("--group_temp", type=float, default=0.2)
    # loss parameters
    parser.add_argument("--lambda_ga", type=float, default=1.)
    parser.add_argument("--lambda_pfa", type=float, default=1.)
    parser.add_argument("--lambda_ica", type=float, default=1.)
    parser.add_argument("--lambda_pca", type=float, default=1.)
    parser.add_argument("--lambda_orth", type=float, default=1.)
    parser.add_argument("--lambda_group", type=float, default=1.)
    parser.add_argument("--lambda_esp", type=float, default=0.1,
                        help="Weight for Entity Slot Pred (ESP): image-global (a,s) on present slots.")
    parser.add_argument("--lambda_ptp", type=float, default=0.1,
                        help="Weight for Pseudo Triplet Pred (PTP): grouped-patch consistency.")
    parser.add_argument(
        "--enable_loss_pfa",
        action="store_true",
        default=False,
        help="Compute PFA and add it to the total loss. If omitted, the PFA forward is skipped.",
    )
    parser.add_argument(
        "--enable_loss_ica",
        action="store_true",
        default=False,
        help="Compute ICA and add it to the total loss. If omitted, ICA (and instance-prototype normalize) is skipped.",
    )
    parser.add_argument(
        "--enable_loss_lga",
        action="store_true",
        default=False,
        help="Compute LGA CE and add it to the total loss. Grouping still runs if --language_group (PFA/PTP). VGA unchanged.",
    )
    # MedKLIP-style comparison module (off by default; ERMA path unchanged)
    parser.add_argument(
        "--enable_medklip",
        action="store_true",
        default=False,
        help="Enable MedKLIP-style entity-query↔patch fusion with existence/anatomy losses. "
             "Requires --entity_extraction. Does not alter ERMA losses unless their λ/flags are set.",
    )
    parser.add_argument(
        "--n_medklip_layers",
        type=int,
        default=4,
        help="Transformer decoder layers in MedKLIP fusion module.",
    )
    parser.add_argument(
        "--medklip_temp",
        type=float,
        default=0.2,
        help="Temperature for MedKLIP anatomy (location) contrastive/CE loss.",
    )
    parser.add_argument(
        "--lambda_medklip_cls",
        type=float,
        default=1.0,
        help="Weight for MedKLIP existence (cls) loss.",
    )
    parser.add_argument(
        "--lambda_medklip_loc",
        type=float,
        default=1.0,
        help="Weight for MedKLIP anatomy/position (loc) loss.",
    )
    # experiment parameters
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--ckpt_path", type=str or None, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_medeha_argparse_args(parser)
    args = parser.parse_args()
    if args.experiment_name == "test":
        args.batch_size = 16
        
        args.entity_extraction = True
        args.entity_align = True
        args.max_n_entity = 30
        args.entity_prompts = 0
        args.with_report_feat = True
        args.with_token_feat = True
        # args.sentence_token = True
        args.twostep_attention = True
        args.entity_layer_type = "decoder"  # pool, decoder
        args.n_entity_layers = 4
        args.report_layer_type = "decoder"  # pool, decoder
        args.n_report_layers = 4
        
        args.language_group = True
        args.visual_group = True
        args.group_threshold = 0.2
        
        args.enable_loss_pfa = True
        args.enable_loss_ica = True
        args.enable_loss_lga = True
        
        args.enable_entity_pred = True
        args.enable_pseudo_triplet_pred = False
        
        args.threshold_momentum = 1.0
        # args.threshold_momentum = 0.99
        
        # args.num_local_atten_layer_heads = 1
        # args.num_attn_weight_layer_heads = 1
        args.instance_prototype = "random"
        args.num_instance_prototypes = 500
        # args.token_prototype = "random"
        # args.pca_with_weight = True
        # args.num_token_prototypes = 500
        args.learnable_temperature = True

        
    args.deterministic = True
    args.max_epochs = 50
    args.gpus = 1
    
    if not hasattr(args, 'accumulate_grad_batches') or args.accumulate_grad_batches is None:
        args.accumulate_grad_batches = 2
    
    if not hasattr(args, 'precision') or args.precision == 32:
        args.precision = 16

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(
        MultimodalPretrainingDataset, multimodal_collate_fn,
        DataTransforms, args.data_pct,
        args.batch_size, args.num_workers,
        entity_extraction=args.entity_extraction,
        sentence_token=args.with_token_feat and args.sentence_token,
        max_report_sents=args.max_report_sents,
        max_words_sent=args.max_words_sent,
    )

    # Add load from checkpoint
    model = MedEHA(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = f"{SAVEPATH}/erma_ckpts/erma/{args.experiment_name}/{extension}"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2),
        EarlyStopping(monitor="val/loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = f"{SAVEPATH}/erma_data/erma/{args.experiment_name}"
    os.makedirs(logger_dir, exist_ok=True)
    if args.experiment_name == "test":
        trainer = Trainer.from_argparse_args(
            args=args,
            precision=args.precision,
            amp_backend="native",
            callbacks=callbacks,
            )
    else:
        tensorboard_logger = TensorBoardLogger(
            save_dir=logger_dir, name=f"{extension}")
        trainer = Trainer.from_argparse_args(
            args=args,
            precision=args.precision,
            amp_backend="native",
            callbacks=callbacks,
            logger=tensorboard_logger,
            )

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule,
                ckpt_path=args.ckpt_path
                )

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    main()
