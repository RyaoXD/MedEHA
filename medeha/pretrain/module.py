import math
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from transformers import BertTokenizer
from medeha.constants import BERT_DIR, ENTITY_DIR
from medeha.models.backbones.encoder import (
    BertEncoder,
    GlobalEmbedding,
    ImageEncoder,
    LocalEmbedding,
    AttentionPool2d,
    TransformerFusionLayer,
)
from torch import distributed as dist


def _init_linear_normal_(module: nn.Module, std: float = 0.02) -> None:
    """Initialize Linear weights with normal; bias to zero (BERT-style small init)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MedEHA(LightningModule):
    '''Pytorch lightning implementation of MedEHA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 ga_temp: float = 0.1,
                 pfa_temp: float = 0.1,
                 ica_temp: float = 0.2,
                 pca_temp: float = 0.2,
                 group_temp: float = 0.2,
                 num_instance_prototypes: int = 500,
                 num_token_prototypes: int = 500,
                 entity_extraction: bool = False,
                 with_report_feat: bool = False,
                 twostep_attention: bool = False,
                 entity_layer_type: str = "pool",  # "pool" or "decoder"
                 n_entity_layers: int = 2,  # transformer decoder layers
                 report_layer_type: str = "pool",  # "pool" or "decoder"
                 n_report_layers: int = 2,  # transformer decoder layers
                 #  entity_shape
                 entity_prompts: int = 8,
                 max_n_entity: int = 30,
                 language_group: bool = False,
                 group_threshold: float = 0.0,
                 threshold_momentum: float = 1.0,
                 num_local_atten_layer_heads: int = 1,
                 num_attn_weight_layer_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_ga: float = 1,
                 lambda_pfa: float = 1,
                 lambda_ica: float = 1,
                 lambda_pca: float = 1,
                 lambda_group: float = 1,
                 lambda_esp: float = 0.1,
                 lambda_ptp: float = 0.1,
                 enable_entity_pred: bool = False,
                 enable_pseudo_triplet_pred: bool = False,
                 enable_loss_pfa: bool = True,
                 enable_loss_ica: bool = True,
                 enable_loss_lga: bool = True,
                 enable_medklip: bool = False,
                 n_medklip_layers: int = 4,
                 medklip_temp: float = 0.2,
                 lambda_medklip_cls: float = 1.0,
                 lambda_medklip_loc: float = 1.0,
                 freeze_prototypes_epochs: int = 5,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 entity_shape: tuple = (75, 51, 2),
                 learnable_temperature: bool = False,
                 pca_with_weight: bool = False,
                 with_token_feat: bool = False,
                 sentence_token: bool = False,
                 max_report_sents: int = 30,
                 max_words_sent: int = 30,
                 entity_fuse: bool = True,
                 entity_align: bool = False,
                 hyperparam_independ: bool = False,
                 projecter_independ: bool = False,
                 lambda_ga_re: float = 1.0,
                 lambda_ga_ri: float = 1.0,
                 lambda_ga_ei: float = 1.0,
                 compare_protocol: str = "none",
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        if not hasattr(self.hparams, "enable_entity_pred"):
            setattr(self.hparams, "enable_entity_pred", False)
        if not hasattr(self.hparams, "enable_pseudo_triplet_pred"):
            setattr(self.hparams, "enable_pseudo_triplet_pred", False)
        if not hasattr(self.hparams, "enable_loss_pfa"):
            setattr(self.hparams, "enable_loss_pfa", True)
        if not hasattr(self.hparams, "enable_loss_ica"):
            setattr(self.hparams, "enable_loss_ica", True)
        if not hasattr(self.hparams, "enable_loss_lga"):
            setattr(self.hparams, "enable_loss_lga", True)
        if not hasattr(self.hparams, "enable_medklip"):
            setattr(self.hparams, "enable_medklip", False)
        if not hasattr(self.hparams, "n_medklip_layers"):
            setattr(self.hparams, "n_medklip_layers", 4)
        if not hasattr(self.hparams, "medklip_temp"):
            setattr(self.hparams, "medklip_temp", 0.2)
        if not hasattr(self.hparams, "lambda_medklip_cls"):
            setattr(self.hparams, "lambda_medklip_cls", 1.0)
        if not hasattr(self.hparams, "lambda_medklip_loc"):
            setattr(self.hparams, "lambda_medklip_loc", 1.0)
        if not hasattr(self.hparams, "with_token_feat"):
            setattr(self.hparams, "with_token_feat", False)
        if not hasattr(self.hparams, "sentence_token"):
            setattr(self.hparams, "sentence_token", False)
        if not hasattr(self.hparams, "max_report_sents"):
            setattr(self.hparams, "max_report_sents", 30)
        if not hasattr(self.hparams, "max_words_sent"):
            setattr(self.hparams, "max_words_sent", 30)
        if not hasattr(self.hparams, "entity_fuse"):
            setattr(self.hparams, "entity_fuse", True)
        if not hasattr(self.hparams, "entity_align"):
            setattr(self.hparams, "entity_align", False)
        if not hasattr(self.hparams, "hyperparam_independ"):
            setattr(self.hparams, "hyperparam_independ", False)
        if not hasattr(self.hparams, "projecter_independ"):
            setattr(self.hparams, "projecter_independ", False)
        if not hasattr(self.hparams, "lambda_ga_re"):
            setattr(self.hparams, "lambda_ga_re", 1.0)
        if not hasattr(self.hparams, "lambda_ga_ri"):
            setattr(self.hparams, "lambda_ga_ri", 1.0)
        if not hasattr(self.hparams, "lambda_ga_ei"):
            setattr(self.hparams, "lambda_ga_ei", 1.0)
        if not hasattr(self.hparams, "compare_protocol"):
            setattr(self.hparams, "compare_protocol", "none")
        if self.hparams.entity_align and not self.hparams.entity_extraction:
            raise ValueError("entity_align requires --entity_extraction.")
        if self.hparams.enable_medklip and not self.hparams.entity_extraction:
            raise ValueError("enable_medklip requires --entity_extraction.")
        if self.hparams.entity_fuse and self.hparams.entity_align:
            print(
                "WARNING: entity_fuse and entity_align are both enabled; "
                "using entity_align (decoupled report/entity/image alignment)."
            )

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        
        self.hparams.max_n_entity = 30 \
            if self.hparams.max_n_entity is None else self.hparams.max_n_entity
        self.hparams.entity_prompts = 0 \
            if self.hparams.entity_prompts is None else self.hparams.entity_prompts
        if self.hparams.entity_extraction:
            print("Preprocessing entities explanations...")
            tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR))
            self.obs_book = json.load(open(
                ENTITY_DIR / "observation_explanation.json", "r"))
            self.ana_book = json.load(open(
                ENTITY_DIR / "anaomy_explanation.json", "r"))
            
            self.obs_tokens = tokenizer(
                list(self.obs_book.values()),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=60,
                )
            self.ana_tokens = tokenizer(
                list(self.ana_book.values()),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=30,
                )
            self.status_embedding = nn.Embedding(2, self.text_encoder_q.embedding_dim)
            nn.init.normal_(self.status_embedding.weight, std=0.02)
            if self.hparams.twostep_attention:
                if self.hparams.entity_layer_type == "pool":
                    self.token_entity_layer = AttentionPool2d(
                        self.hparams.entity_prompts+3, self.text_encoder_q.embedding_dim, 64)
                else:  # decoder
                    self.token_entity_layer = TransformerFusionLayer(
                        embed_dim=self.text_encoder_q.embedding_dim,
                        num_heads=8,
                        num_layers=self.hparams.n_entity_layers,
                        max_seq_len=self.hparams.entity_prompts+3,
                        fusion_type="global"
                    )
            else:
                self.token_entity_layer = None
                
            report_n_entity = self.hparams.max_n_entity if self.hparams.twostep_attention\
                else self.hparams.max_n_entity*(self.hparams.entity_prompts+4)
            self.cls_token = nn.Parameter(
                torch.empty(self.text_encoder_q.embedding_dim, dtype=torch.float32))
            nn.init.normal_(self.cls_token, std=0.02)
            if self.hparams.report_layer_type == "pool":
                self.report_entity_layer = AttentionPool2d(
                    report_n_entity, self.text_encoder_q.embedding_dim, 64)
            else:  # decoder
                self.report_entity_layer = TransformerFusionLayer(
                    embed_dim=self.text_encoder_q.embedding_dim,
                    num_heads=8,
                    num_layers=self.hparams.n_report_layers,
                    max_seq_len=report_n_entity+1 if self.hparams.with_report_feat else report_n_entity,
                    fusion_type="split",
                    cross_attention=self.hparams.with_report_feat
                )
            n_o = int(self.hparams.entity_shape[0])
            n_a = int(self.hparams.entity_shape[1])
            self.num_entity_slots = n_o * n_a
            self.entity_slot_embedding = nn.Parameter(
                torch.empty(self.num_entity_slots, self.hparams.emb_dim))
            nn.init.normal_(self.entity_slot_embedding, std=0.02)
            self.entity_pred_ana = nn.Linear(self.hparams.emb_dim, n_a)
            self.entity_pred_sta = nn.Linear(self.hparams.emb_dim, 2)
            _init_linear_normal_(self.entity_pred_ana)
            _init_linear_normal_(self.entity_pred_sta)
            d_txt = self.text_encoder_q.embedding_dim
            self.pseudo_ana_feat = nn.Parameter(torch.empty(1, d_txt))
            self.pseudo_sta_feat = nn.Parameter(torch.empty(1, d_txt))
            nn.init.normal_(self.pseudo_ana_feat, std=0.02)
            nn.init.normal_(self.pseudo_sta_feat, std=0.02)
            _proj_hidden = 2048
            if self.hparams.projecter_independ:
                _d_in = self.text_encoder_q.embedding_dim
                _d_out = self.hparams.emb_dim
                self.entity_global_embed = GlobalEmbedding(
                    _d_in, _proj_hidden, _d_out)
                self.entity_local_embed = LocalEmbedding(
                    _d_in, _proj_hidden, _d_out)
            else:
                self.entity_global_embed = None
                self.entity_local_embed = None
        else:
            self.status_embedding = None
            self.token_entity_layer = None
            self.report_entity_layer = None
            self.num_entity_slots = 0
            self.entity_slot_embedding = None
            self.entity_pred_ana = None
            self.entity_pred_sta = None
            self.pseudo_ana_feat = None
            self.pseudo_sta_feat = None
            self.entity_global_embed = None
            self.entity_local_embed = None
        if self.hparams.entity_prompts != 0:
            _ep_shape = (
                self.hparams.entity_shape[0],
                self.hparams.entity_shape[1],
                self.hparams.entity_shape[2],
                self.hparams.entity_prompts,
                self.text_encoder_q.embedding_dim,
            )
            self.entity_prompts_vector = nn.Parameter(torch.empty(_ep_shape))
            nn.init.normal_(self.entity_prompts_vector, std=0.02)
        else:
            self.entity_prompts_vector = None

        # MedKLIP-style entity-query ↔ patch fusion (opt-in; off by default)
        if self.hparams.enable_medklip:
            d_txt = self.text_encoder_q.embedding_dim
            self.medklip_fusion = TransformerFusionLayer(
                embed_dim=d_txt,
                num_heads=8,
                num_layers=int(self.hparams.n_medklip_layers),
                max_seq_len=self.hparams.max_n_entity,
                fusion_type="local",
                cross_attention=True,
            )
            self.medklip_exist_head = nn.Linear(d_txt, 1)
            _init_linear_normal_(self.medklip_exist_head)
            # ViT-B / BioClinicalBERT are both 768-d; keep Identity unless dims diverge.
            if self.img_encoder_q.feature_dim != d_txt:
                self.medklip_patch_proj = nn.Linear(
                    self.img_encoder_q.feature_dim, d_txt, bias=False
                )
                nn.init.normal_(self.medklip_patch_proj.weight, std=0.02)
            else:
                self.medklip_patch_proj = nn.Identity()
            self.medklip_temp = nn.Parameter(
                torch.log(torch.tensor(float(self.hparams.medklip_temp)))
            )
        else:
            self.medklip_fusion = None
            self.medklip_exist_head = None
            self.medklip_patch_proj = None
            self.medklip_temp = None

        # set default group threshold to patch number
        if self.hparams.group_threshold <= 0:
            if self.hparams.language_group:
                self.language_group_threshold = 1 / self.img_encoder_q.model.patch_embed.num_patches
            else:
                self.language_group_threshold = 0.
            if self.hparams.visual_group:
                self.token_group_threshold = 1 / self.hparams.max_n_entity
            else:
                self.token_group_threshold = 0.
        else:
            self.language_group_threshold = (
                self.hparams.group_threshold if self.hparams.language_group else 0.)
            self.token_group_threshold = (
                self.hparams.group_threshold if self.hparams.visual_group else 0.)

        self.patch_cur_threshold = 0.
        self.token_cur_threshold = 0.
            
        
        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_local_atten_layer_heads, batch_first=True)
        # token local attention layer
        self.token_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_local_atten_layer_heads, batch_first=True)

        self.patch_attention_weights_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_attn_weight_layer_heads, batch_first=True)

        self.hparams.freeze_prototypes_epochs = 5 if self.hparams.freeze_prototypes_epochs is None \
            else self.hparams.freeze_prototypes_epochs
        
        # define token prototype
        if self.hparams.token_prototype is None:
            self.token_prototype_layer = None
        else:
            self.token_prototype_layer = nn.Linear(emb_dim, self.hparams.num_token_prototypes, bias=False)
            nn.init.normal_(self.token_prototype_layer.weight, std=0.02)
        
        # define instance prototype
        if self.hparams.instance_prototype is None:
            self.instance_prototype_layer = None
        elif self.hparams.instance_prototype == "mapping":
            assert not self.token_prototype_layer is None, "No token_prototype_layer"
            self.instance_prototype_layer = nn.Sequential(
                nn.Linear(num_instance_prototypes, emb_dim, bias=False),
                nn.Linear(emb_dim, num_instance_prototypes, bias=False))
            for _m in self.instance_prototype_layer:
                _init_linear_normal_(_m)
        else:
            self.instance_prototype_layer = nn.Linear(emb_dim, num_instance_prototypes, bias=False)
            _init_linear_normal_(self.instance_prototype_layer)

        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn
        
        _log_temp = torch.log
        if self.hparams.learnable_temperature:
            self.ga_temp = nn.Parameter(
                _log_temp(torch.tensor(self.hparams.ga_temp, dtype=torch.float32)),
                requires_grad=True,
            )
            self.pfa_temp = nn.Parameter(
                _log_temp(torch.tensor(self.hparams.pfa_temp, dtype=torch.float32)),
                requires_grad=True,
            )
            self.ica_temp = nn.Parameter(
                _log_temp(torch.tensor(self.hparams.ica_temp, dtype=torch.float32)),
                requires_grad=True,
            )
            self.pca_temp = nn.Parameter(
                _log_temp(torch.tensor(self.hparams.pca_temp, dtype=torch.float32)),
                requires_grad=True,
            )
            self.gro_temp = nn.Parameter(
                _log_temp(torch.tensor(self.hparams.group_temp, dtype=torch.float32)),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "ga_temp",
                _log_temp(torch.tensor(self.hparams.ga_temp, dtype=torch.float32)),
                persistent=True,
            )
            self.register_buffer(
                "pfa_temp",
                _log_temp(torch.tensor(self.hparams.pfa_temp, dtype=torch.float32)),
                persistent=True,
            )
            self.register_buffer(
                "ica_temp",
                _log_temp(torch.tensor(self.hparams.ica_temp, dtype=torch.float32)),
                persistent=True,
            )
            self.register_buffer(
                "pca_temp",
                _log_temp(torch.tensor(self.hparams.pca_temp, dtype=torch.float32)),
                persistent=True,
            )
            self.register_buffer(
                "gro_temp",
                _log_temp(torch.tensor(self.hparams.group_temp, dtype=torch.float32)),
                persistent=True,
            )
        self.min_temp, self.max_temp = 0.01, 0.5

            
    @staticmethod
    def _scatter_sentence_report_feats(feat, report_idx, batch_size, max_sents):
        """CoMA-style: per-short-sentence BERT vectors [N,D] -> padded [B,max_sents,D]."""
        device, dtype = feat.device, feat.dtype
        d = feat.size(-1)
        out = torch.zeros(batch_size, max_sents, d, device=device, dtype=dtype)
        pad = torch.ones(batch_size, max_sents, device=device, dtype=torch.bool)
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(feat.size(0)):
            b = int(report_idx[i].item())
            k = int(counts[b].item())
            if k < max_sents:
                out[b, k] = feat[i]
                pad[b, k] = False
                counts[b] = k + 1
        return out, pad

    @staticmethod
    def _word_memory_padding_mask(attn_mask, word_len):
        """Approximate padding on aggregated word_feat length word_len from BERT attn_mask."""
        valid = attn_mask.sum(dim=1).long().clamp(min=2) - 2
        valid = torch.clamp(valid, max=word_len)
        idx = torch.arange(word_len, device=attn_mask.device).unsqueeze(0).expand(
            attn_mask.size(0), -1)
        return idx >= valid.unsqueeze(1)

    # merge decomposed concept clause for alignment
    def merge_entity(self, observations_feat_q, anaomies_feat_q, states_feat_q,
                           entity_ids, entity_pos, num_entity_group=3, rep_feat=None,
                           rep_memory_mask=None,
                           pseudo_replace: bool = False):
        l_entity = self.hparams.entity_prompts + num_entity_group
        n_entity, n_dim = observations_feat_q.shape
        _device = observations_feat_q.device
        batch_size = entity_ids.max().item() + 1
        if pseudo_replace:
            anaomies_feat_q = self.pseudo_ana_feat.expand(n_entity, -1)
            states_feat_q = self.pseudo_sta_feat.expand(n_entity, -1)
        
        # integrate batch entity feature vectors (n_entity, l_entity, n_dim)
        entity_features = torch.zeros(n_entity, l_entity, n_dim, device=_device)
        entity_features[:, 0] = observations_feat_q
        entity_features[:, 1] = anaomies_feat_q
        entity_features[:, 2] = states_feat_q
        if self.hparams.entity_prompts > 0:
            entity_features[:, -self.hparams.entity_prompts:] = self.entity_prompts_vector[
                entity_pos[:, 0].long(),
                entity_pos[:, 1].long(),
                entity_pos[:, 2].long()
            ]
        
        # integrate batch sample entity feature vectors (n_batch, max_n_entity, l_entity, n_dim)
        unique_vals, counts = torch.unique(entity_ids, return_counts=True)
        sample_count = torch.zeros(max(entity_ids)+1, dtype=torch.long, device=entity_ids.device)
        sample_count[unique_vals] = counts
        
        token_l = self.hparams.max_n_entity if self.hparams.twostep_attention else \
                                        self.hparams.max_n_entity * (l_entity + 1)

        tokens_feat_q = torch.zeros(batch_size,
                                    token_l,
                                    n_dim, device=_device)
        reports_entity = torch.zeros(batch_size, self.hparams.max_n_entity, device=_device)

        start_idx = 0
        for batch_idx in range(batch_size):
            count = min(sample_count[batch_idx].item(), self.hparams.max_n_entity)
            if count == 0:
                continue

            end_idx = start_idx + count
            sample_entity = entity_features[start_idx:end_idx]

            if self.hparams.twostep_attention:
                pooled_features, _ = self.token_entity_layer(sample_entity)
                tokens_feat_q[batch_idx, :pooled_features.size(0)] = pooled_features
            else:
                cls_token = self.cls_token.expand(sample_entity.size(0), 1, -1)
                sample_entity = torch.cat([cls_token, sample_entity], dim=1).view(-1, n_dim)
                tokens_feat_q[batch_idx, :sample_entity.size(0)] = sample_entity

            reports_entity[batch_idx, :count] = torch.ones(count, device=_device)
            start_idx = end_idx

        if self.hparams.report_layer_type == "pool":
            report_feat_q, pool_attn = self.report_entity_layer(tokens_feat_q)
        else:
            if self.hparams.with_report_feat:
                if (
                    getattr(self.hparams, "with_token_feat", False)
                    and rep_feat is not None
                    and rep_feat.dim() == 3
                ):
                    if rep_memory_mask is None:
                        rep_memory_mask = torch.zeros(
                            rep_feat.size(0),
                            rep_feat.size(1),
                            device=rep_feat.device,
                            dtype=torch.bool,
                        )
                    report_feat_q, tokens_feat_q, pool_attn = self.report_entity_layer(
                        tokens_feat_q,
                        rep_feat,
                        attention_mask=reports_entity.clone().detach().bool(),
                        memory_key_padding_mask=rep_memory_mask,
                    )
                elif rep_feat is not None:
                    report_feat_q, tokens_feat_q, pool_attn = self.report_entity_layer(
                        tokens_feat_q,
                        rep_feat.unsqueeze(1),
                        attention_mask=reports_entity.clone().detach().bool(),
                    )
                else:
                    report_feat_q, tokens_feat_q, pool_attn = self.report_entity_layer(
                        tokens_feat_q,
                        attention_mask=reports_entity.clone().detach().bool(),
                    )
            else:
                report_feat_q, tokens_feat_q, pool_attn = self.report_entity_layer(
                    tokens_feat_q, attention_mask=reports_entity.clone().detach().bool())

        if not self.hparams.twostep_attention:
            tokens_feat_q = tokens_feat_q[:, ::(l_entity+1)]
            pool_attn = pool_attn[:, ::(l_entity+1)]
        return report_feat_q, tokens_feat_q, pool_attn, reports_entity

    def _use_entity_align(self):
        return bool(
            self.hparams.entity_extraction
            and self.hparams.entity_align
        )

    def _compute_medklip_losses(
        self,
        obs_feat_q,
        ana_ids,
        sta_ids,
        entindex,
        patch_feat_q,
        batch_size,
    ):
        """MedKLIP-style entity-query fusion losses (existence + anatomy).

        Uses knowledge-encoded observation embeddings as queries that cross-attend
        image patches, then predicts existence (BCE) and anatomy (softmax CE over
        the anatomy book — full-softmax contrastive position supervision).
        """
        device = obs_feat_q.device
        dtype = obs_feat_q.dtype
        zero = obs_feat_q.new_tensor(0.0)
        if (
            self.medklip_fusion is None
            or obs_feat_q.numel() == 0
            or batch_size <= 0
        ):
            return zero, zero

        max_n = int(self.hparams.max_n_entity)
        d_txt = obs_feat_q.size(-1)
        queries = torch.zeros(batch_size, max_n, d_txt, device=device, dtype=dtype)
        q_mask = torch.zeros(batch_size, max_n, device=device, dtype=torch.bool)
        tgt_ana = torch.full((batch_size, max_n), -100, dtype=torch.long, device=device)
        tgt_sta = torch.zeros(batch_size, max_n, device=device, dtype=dtype)

        rid = entindex.long()
        ana_ids = ana_ids.long()
        sta_ids = sta_ids.long()
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(obs_feat_q.size(0)):
            b = int(rid[i].item())
            if b < 0 or b >= batch_size:
                continue
            slot = int(counts[b].item())
            if slot >= max_n:
                continue
            queries[b, slot] = obs_feat_q[i]
            q_mask[b, slot] = True
            tgt_ana[b, slot] = ana_ids[i]
            tgt_sta[b, slot] = sta_ids[i].to(dtype)
            counts[b] += 1

        if not q_mask.any():
            return zero, zero

        # Ensure patch memory matches entity embedding dim
        patch_mem = self.medklip_patch_proj(patch_feat_q)

        fused, _ = self.medklip_fusion(
            queries,
            report_feat=patch_mem,
            attention_mask=q_mask,
        )
        # fused: [B, max_n, D]
        valid = q_mask.view(-1)
        fused_flat = fused.reshape(-1, d_txt)[valid]
        sta_flat = tgt_sta.view(-1)[valid]
        ana_flat = tgt_ana.view(-1)[valid]

        exist_logit = self.medklip_exist_head(fused_flat).squeeze(-1)
        loss_cls = F.binary_cross_entropy_with_logits(exist_logit, sta_flat)

        # Anatomy book embeddings for position contrastive / CE
        self.ana_tokens = {k: v.to(device) for k, v in self.ana_tokens.items()}
        ana_tt = self.ana_tokens.get("token_type_ids")
        if ana_tt is None:
            ana_tt = torch.zeros_like(self.ana_tokens["input_ids"])
        ana_book, _, _, _ = self.text_encoder_q(
            self.ana_tokens["input_ids"],
            self.ana_tokens["attention_mask"],
            ana_tt,
        )
        ana_book = F.normalize(ana_book, dim=-1)
        fused_n = F.normalize(fused_flat, dim=-1)
        temp = torch.exp(self.medklip_temp).clamp(self.min_temp, self.max_temp)
        logits = fused_n.mm(ana_book.t()) / temp
        # Drop entities whose anatomy id is out of book range
        n_a = ana_book.size(0)
        loc_mask = (ana_flat >= 0) & (ana_flat < n_a)
        if loc_mask.any():
            loss_loc = F.cross_entropy(logits[loc_mask], ana_flat[loc_mask])
        else:
            loss_loc = zero
        return loss_cls, loss_loc

    def _entity_global_project(self, x):
        if self.entity_global_embed is not None:
            return self.entity_global_embed(x)
        return self.text_encoder_q.global_embed(x)

    def _entity_local_project(self, x):
        if self.entity_local_embed is not None:
            return self.entity_local_embed(x)
        return self.text_encoder_q.local_embed(x)

    def _encode_caption_report(self, batch):
        """Encode full medical report (global + word/sentence tokens) for alignment."""
        if (
            self.hparams.with_token_feat
            and self.hparams.sentence_token
            and batch.get("caption_is_sentence", False)
        ):
            rs, _, _, cap_sents = self.text_encoder_q(
                batch["caption_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
            )
            bz = batch["imgs"].size(0)
            cap_word_feat, _ = self._scatter_sentence_report_feats(
                rs,
                batch["reportindex"],
                bz,
                self.hparams.max_report_sents,
            )
            cap_report_feat = cap_word_feat.mean(dim=1)
        else:
            cap_report_feat, cap_word_feat, _, cap_sents = self.text_encoder_q(
                batch["caption_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
            )
        return cap_report_feat, cap_word_feat, cap_sents

    @staticmethod
    def _bilateral_infonce(emb_a, emb_b, temp):
        bz = emb_a.size(0)
        labels = torch.arange(bz).type_as(emb_a).long()
        scores = emb_a.mm(emb_b.t()) / temp
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores.transpose(0, 1), labels)
        return loss0 + loss1

    def _instance_proto_pair_loss(self, emb_a, emb_b, ica_temp):
        if self.hparams.instance_prototype == "mapping":
            proto_a = self.token_prototype_layer(emb_a)
            proto_a = self.instance_prototype_layer(proto_a)
            proto_b = self.token_prototype_layer(emb_b)
            proto_b = self.instance_prototype_layer(proto_b)
        else:
            proto_a = self.instance_prototype_layer(emb_a)
            proto_b = self.instance_prototype_layer(emb_b)

        with torch.no_grad():
            code_a = torch.exp(proto_a / self.hparams.epsilon).t()
            code_a = self.get_assignments(
                code_a, self.hparams.sinkhorn_iterations)
            code_b = torch.exp(proto_b / self.hparams.epsilon).t()
            code_b = self.get_assignments(
                code_b, self.hparams.sinkhorn_iterations)

        prob_a = F.softmax(proto_a / ica_temp, dim=1)
        prob_b = F.softmax(proto_b / ica_temp, dim=1)
        loss_a2b = -torch.mean(
            torch.sum(code_a * torch.log(prob_b), dim=1))
        loss_b2a = -torch.mean(
            torch.sum(code_b * torch.log(prob_a), dim=1))
        return (loss_a2b + loss_b2a) / 2.0

    def _token_proto_pair_loss(
        self, emb_a, emb_b, pca_temp, weight_a=None, weight_b=None):
        proto_a = self.token_prototype_layer(emb_a)
        proto_b = self.token_prototype_layer(emb_b)
        proto_a = rearrange(proto_a, "b n d -> (b n) d")
        proto_b = rearrange(proto_b, "b n d -> (b n) d")

        with torch.no_grad():
            code_a = torch.exp(proto_a / self.hparams.epsilon).t()
            code_a = self.get_assignments(
                code_a, self.hparams.sinkhorn_iterations)
            code_b = torch.exp(proto_b / self.hparams.epsilon).t()
            code_b = self.get_assignments(
                code_b, self.hparams.sinkhorn_iterations)

        prob_a = F.softmax(proto_a / pca_temp, dim=-1)
        prob_b = F.softmax(proto_b / pca_temp, dim=-1)
        if weight_a is not None:
            prob_a = prob_a * weight_a.view(-1).unsqueeze(-1)
        if weight_b is not None:
            prob_b = prob_b * weight_b.view(-1).unsqueeze(-1)

        loss_a2b = -torch.mean(
            torch.sum(code_a * torch.log(prob_b), dim=1))
        loss_b2a = -torch.mean(
            torch.sum(code_b * torch.log(prob_a), dim=1))
        return (loss_a2b + loss_b2a) / 2.0

    def group_embedings(self, embed_1, embed_2, similarity_threshold=0.1):
        """Group keys onto queries: embed_1 is q [B,Tq,D], embed_2 is k [B,Tk,D]; output [B,Tq,D]."""
        sim_matrix = torch.bmm(embed_1, embed_2.permute(0, 2, 1))
        min_vals = torch.min(sim_matrix, dim=-1, keepdim=True)[0]
        max_vals = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        range_vals = torch.clamp(max_vals - min_vals, min=1e-8)
        sim_matrix = (sim_matrix - min_vals) / range_vals
        current_thresh = sim_matrix.mean()
        
        sim_matrix = torch.where(sim_matrix < similarity_threshold, 0.0, sim_matrix)
        sum_sim = torch.clamp(torch.sum(sim_matrix, dim=-1, keepdim=True), min=1e-8)
        align_weights = sim_matrix / sum_sim

        grouped_embed = torch.einsum('btp,bpd->btd', align_weights, embed_2)
        grouped_embed = F.normalize(grouped_embed, p=2, dim=-1)
        return grouped_embed, current_thresh

    @staticmethod
    def _flat_attn_weights(weights, bz, n, device, dtype):
        """Flatten attention weights to [B*n]; uniform if shape mismatches."""
        need = bz * n
        if weights is None:
            return torch.full((need,), 1.0 / max(n, 1), device=device, dtype=dtype)
        flat = weights.reshape(-1)
        if flat.numel() != need:
            return torch.full((need,), 1.0 / max(n, 1), device=device, dtype=dtype)
        return flat

    def _report_token_attn_weights(self, batch, n_tokens, device, dtype):
        """Per-report-token weights from BERT attention_mask (aligned to word/sent length)."""
        am = batch["attention_mask"]
        start = 1
        end = min(start + n_tokens, am.size(1))
        valid = am[:, start:end].to(device=device, dtype=dtype)
        if valid.size(1) < n_tokens:
            pad = torch.zeros(
                valid.size(0), n_tokens - valid.size(1), device=device, dtype=dtype)
            valid = torch.cat([valid, pad], dim=1)
        return valid / valid.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _diagonal_local_ce(self, sim, n_q, n_kv, attn_weight, bz):
        """Diagonal local contrastive: query i matches key/value index i (needs n_q <= n_kv)."""
        if n_q > n_kv:
            return None
        device, dtype = sim.device, sim.dtype
        sim_flat = rearrange(sim, "b nq nk -> (b nq) nk")
        targets = torch.arange(n_q, device=device, dtype=torch.long).repeat(bz)
        w = self._flat_attn_weights(attn_weight, bz, n_q, device, dtype)
        return torch.sum(
            F.cross_entropy(sim_flat, targets, reduction="none") * w
        ) / bz

    def cross_alignment_loss(
        self,
        emb_q_1,
        emb_q_2,
        attn_layer,
        attn_weight,
        temp,
        mask=None,
        attn_weight_kv=None,
        mask_q=None,
    ):
        """Bidirectional local alignment: q1->q2 and q2->q1 cross-attention + diagonal CE."""
        bz = emb_q_1.size(0)
        n1, n2 = emb_q_1.size(1), emb_q_2.size(1)
        w_kv = attn_weight_kv if attn_weight_kv is not None else attn_weight

        with torch.cuda.amp.autocast(dtype=emb_q_1.dtype):
            atten_12, _ = attn_layer(
                emb_q_1, emb_q_2, emb_q_2, key_padding_mask=mask)
            atten_21, _ = attn_layer(
                emb_q_2, emb_q_1, emb_q_1, key_padding_mask=mask_q)
        atten_12 = F.normalize(atten_12, dim=-1)
        atten_21 = F.normalize(atten_21, dim=-1)

        sim_12 = torch.bmm(emb_q_1, atten_12.permute(0, 2, 1)) / temp
        sim_21 = torch.bmm(emb_q_2, atten_21.permute(0, 2, 1)) / temp

        losses = []
        loss_12 = self._diagonal_local_ce(sim_12, n1, n2, attn_weight, bz)
        if loss_12 is not None:
            losses.append(loss_12)
        loss_21 = self._diagonal_local_ce(sim_21, n2, n1, w_kv, bz)
        if loss_21 is not None:
            losses.append(loss_21)

        if not losses:
            return emb_q_1.new_tensor(0.0)
        return sum(losses) / len(losses)
    
    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Encode report text and token-level features
        if self.hparams.entity_extraction:
            obs_ids = batch["entity_ids"][0]
            ana_ids = batch["entity_ids"][1]
            
            # obtain the required encoded ids
            all_ids = torch.unique(torch.cat([obs_ids, ana_ids + obs_ids.max() + 1]))
            obs_mask = all_ids <= obs_ids.max()
            ana_mask = ~obs_mask
            
            # merge input
            _device = obs_ids.device
            self.obs_tokens = {k: v.to(_device) for k, v in self.obs_tokens.items()}
            self.ana_tokens = {k: v.to(_device) for k, v in self.ana_tokens.items()}
            max_len = max(self.obs_tokens["input_ids"].size(1), self.ana_tokens["input_ids"].size(1))
            
            all_input_ids = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=_device)
            all_attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=_device)
            all_token_type_ids = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=_device)
            
            obs_count = obs_mask.sum().item()
            if obs_count > 0:
                obs_indices = all_ids[obs_mask]
                obs_input = self.obs_tokens["input_ids"][obs_indices]
                obs_attn = self.obs_tokens["attention_mask"][obs_indices]
                all_input_ids[:obs_count, :obs_input.size(1)] = obs_input
                all_attention_mask[:obs_count, :obs_attn.size(1)] = obs_attn
            
            ana_count = ana_mask.sum().item()
            if ana_count > 0:
                ana_indices = all_ids[ana_mask] - obs_ids.max() - 1
                ana_input = self.ana_tokens["input_ids"][ana_indices]
                ana_attn = self.ana_tokens["attention_mask"][ana_indices]
                all_input_ids[obs_count:obs_count+ana_count, :ana_input.size(1)] = ana_input
                all_attention_mask[obs_count:obs_count+ana_count, :ana_attn.size(1)] = ana_attn
            
            all_feat_q, _, _, _ = self.text_encoder_q(
                all_input_ids,
                all_attention_mask,
                all_token_type_ids
            )
            
            # split output
            obs_feat_q = all_feat_q[:obs_count]
            ana_feat_q = all_feat_q[obs_count:]
            
            _, obs_book_loc = torch.unique(obs_ids, return_inverse=True)
            _, ana_book_loc = torch.unique(ana_ids, return_inverse=True)
            
            obs_feat_q = obs_feat_q[obs_book_loc]
            ana_feat_q = ana_feat_q[ana_book_loc]
            
            sta_feat_q = self.status_embedding(batch["entity_ids"][2].long())
            
            rep_memory_mask = None
            report_enc_for_merge = None
            if self.hparams.with_report_feat:
                if (
                    self.hparams.with_token_feat
                    and self.hparams.sentence_token
                    and batch.get("caption_is_sentence", False)
                ):
                    rs, _, _, _ = self.text_encoder_q(
                        batch["caption_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                    )
                    bz = batch["imgs"].size(0)
                    report_enc_for_merge, rep_memory_mask = self._scatter_sentence_report_feats(
                        rs,
                        batch["reportindex"],
                        bz,
                        self.hparams.max_report_sents,
                    )
                elif self.hparams.with_token_feat:
                    _, word_f, _, _ = self.text_encoder_q(
                        batch["caption_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                    )
                    report_enc_for_merge = word_f
                    rep_memory_mask = self._word_memory_padding_mask(
                        batch["attention_mask"], word_f.size(1)
                    )
                else:
                    report_enc_for_merge, _, _, _ = self.text_encoder_q(
                        batch["caption_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                    )
            else:
                report_enc_for_merge = None
            
            # merge_entity
            report_feat_q, token_feat_q, token_attn_q, sents = self.merge_entity(
                obs_feat_q, ana_feat_q, sta_feat_q,
                batch["entindex"], batch["entity_ids"].t(),
                rep_feat=report_enc_for_merge,
                rep_memory_mask=rep_memory_mask,
            )
            pseudo_token_feat_q = None
            if self.hparams.enable_pseudo_triplet_pred:
                _, pseudo_token_feat_q, _, _ = self.merge_entity(
                    obs_feat_q, ana_feat_q, sta_feat_q,
                    batch["entindex"], batch["entity_ids"].t(),
                    rep_feat=report_enc_for_merge,
                    rep_memory_mask=rep_memory_mask,
                    pseudo_replace=True,
                )
            medklip_obs_feat = obs_feat_q
            medklip_ana_ids = batch["entity_ids"][1]
            medklip_sta_ids = batch["entity_ids"][2]
            medklip_entindex = batch["entindex"]
        else:
            report_feat_q, token_feat_q, token_attn_q, sents = self.text_encoder_q(
                batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
            pseudo_token_feat_q = None
            medklip_obs_feat = None
            medklip_ana_ids = None
            medklip_sta_ids = None
            medklip_entindex = None
        entity_emb_q = None
        report_token_emb_q = None
        if self._use_entity_align():
            cap_report_feat, cap_word_feat, _ = self._encode_caption_report(batch)
            report_emb_q = self.text_encoder_q.global_embed(cap_report_feat)
            report_token_emb_q = self.text_encoder_q.local_embed(cap_word_feat)
            entity_emb_q = self._entity_global_project(report_feat_q)
            token_emb_q = self._entity_local_project(token_feat_q)
            if pseudo_token_feat_q is not None:
                pseudo_token_emb_q = self._entity_local_project(pseudo_token_feat_q)
            else:
                pseudo_token_emb_q = None
        else:
            report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
            token_emb_q = self.text_encoder_q.local_embed(token_feat_q)
            if pseudo_token_feat_q is not None:
                pseudo_token_emb_q = self.text_encoder_q.local_embed(
                    pseudo_token_feat_q)
            else:
                pseudo_token_emb_q = None

        # Encode image global / patch features (no visual prompt: 2nd & 3rd ViT outputs coincide)
        img_feat_q, patch_feat_q, _ = self.img_encoder_q(
            batch["imgs"], None)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)

        loss_medklip_cls = img_feat_q.new_tensor(0.0)
        loss_medklip_loc = img_feat_q.new_tensor(0.0)
        if (
            self.hparams.enable_medklip
            and medklip_obs_feat is not None
        ):
            loss_medklip_cls, loss_medklip_loc = self._compute_medklip_losses(
                medklip_obs_feat,
                medklip_ana_ids,
                medklip_sta_ids,
                medklip_entindex,
                patch_feat_q,
                batch["imgs"].size(0),
            )

        # Grouped embeddings are shared by LGA, PFA (entity_fuse + language_group),
        # and PTC. Skip the grouping forward when none of those paths will use G^p.
        need_grouped_patches = self.hparams.language_group and (
            self.hparams.enable_loss_lga
            or self.hparams.enable_loss_pfa
            or self.hparams.enable_pseudo_triplet_pred
        )
        pseudo_g_patch_emb_q = None
        g_patch_emb_q = None
        g_token_emb_q = None
        if need_grouped_patches:
            if self._use_entity_align():
                g_patch_emb_q, patch_cur_threshold = self.group_embedings(
                    token_emb_q,
                    report_token_emb_q,
                    self.language_group_threshold,
                )
            else:
                g_patch_emb_q, patch_cur_threshold = self.group_embedings(
                    token_emb_q, patch_emb_q, self.language_group_threshold)
            g_patch_emb_q = F.normalize(g_patch_emb_q, dim=-1)
            if (
                self.hparams.enable_pseudo_triplet_pred
                and pseudo_token_emb_q is not None
                and (
                    not self._use_entity_align()
                    or self.hparams.visual_group
                )
            ):
                if self._use_entity_align():
                    pseudo_g_patch_emb_q, _ = self.group_embedings(
                        pseudo_token_emb_q,
                        patch_emb_q,
                        self.token_group_threshold,
                    )
                else:
                    pseudo_g_patch_emb_q, _ = self.group_embedings(
                        pseudo_token_emb_q,
                        patch_emb_q,
                        self.language_group_threshold,
                    )
                pseudo_g_patch_emb_q = F.normalize(
                    pseudo_g_patch_emb_q, dim=-1)
            if 0 < self.hparams.threshold_momentum < 1.0:
                if torch.isnan(patch_cur_threshold) or torch.isinf(patch_cur_threshold):
                    if self._use_entity_align():
                        patch_cur_threshold = 1 / max(
                            report_token_emb_q.size(1), 1)
                    else:
                        patch_cur_threshold = (
                            1 / self.img_encoder_q.model.patch_embed.num_patches
                        )
                self.patch_cur_threshold = (
                    self.patch_cur_threshold * batch_idx + patch_cur_threshold
                ) / (batch_idx + 1)

        if self.hparams.visual_group:
            if self._use_entity_align():
                g_token_emb_q, token_cur_threshold = self.group_embedings(
                    token_emb_q, patch_emb_q, self.token_group_threshold)
            else:
                g_token_emb_q, token_cur_threshold = self.group_embedings(
                    patch_emb_q, token_emb_q, self.token_group_threshold)
            g_token_emb_q = F.normalize(g_token_emb_q, dim=-1)
            if 0 < self.hparams.threshold_momentum < 1.0:
                if torch.isnan(token_cur_threshold) or torch.isinf(token_cur_threshold):
                    token_cur_threshold = 1 / self.hparams.max_n_entity
                self.token_cur_threshold = (
                    self.token_cur_threshold * batch_idx + token_cur_threshold
                ) / (batch_idx + 1)

        token_emb_q = F.normalize(token_emb_q, dim=-1)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        if entity_emb_q is not None:
            entity_emb_q = F.normalize(entity_emb_q, dim=-1)
        if report_token_emb_q is not None:
            report_token_emb_q = F.normalize(report_token_emb_q, dim=-1)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        loss_entity = img_emb_q.new_tensor(0.0)
        if (
            self.hparams.entity_extraction
            and self.entity_slot_embedding is not None
            and self.hparams.enable_entity_pred
        ):
            n_a = int(self.hparams.entity_shape[1])
            B = img_emb_q.size(0)
            E = self.num_entity_slots
            slot_e = self.entity_slot_embedding.unsqueeze(0).expand(B, -1, -1)
            h = F.normalize(img_emb_q.unsqueeze(1) + slot_e, dim=-1)
            logits_ana = self.entity_pred_ana(h)
            logits_sta = self.entity_pred_sta(h)
            obs = batch["entity_ids"][0].long()
            ana = batch["entity_ids"][1].long()
            sta = batch["entity_ids"][2].long()
            rid = batch["entindex"].long()
            slot = obs * n_a + ana
            tgt_ana = torch.full((B, E), -100, dtype=torch.long, device=img_emb_q.device)
            tgt_sta = torch.full((B, E), -100, dtype=torch.long, device=img_emb_q.device)
            m = (slot >= 0) & (slot < E) & (rid >= 0) & (rid < B)
            tgt_ana[rid[m], slot[m]] = ana[m]
            tgt_sta[rid[m], slot[m]] = sta[m]
            la = F.cross_entropy(
                logits_ana.view(-1, n_a), tgt_ana.view(-1), ignore_index=-100)
            ls = F.cross_entropy(
                logits_sta.view(-1, 2), tgt_sta.view(-1), ignore_index=-100)
            loss_entity = (la + ls) * 0.5

        loss_pseudo = img_emb_q.new_tensor(0.0)
        if (
            pseudo_g_patch_emb_q is not None
            and self.hparams.entity_extraction
            and self.hparams.enable_pseudo_triplet_pred
            and (
                not self._use_entity_align()
                or (
                    self.hparams.visual_group
                    and g_token_emb_q is not None
                )
            )
        ):
            if self._use_entity_align():
                cos = (g_token_emb_q * pseudo_g_patch_emb_q).sum(dim=-1)
            else:
                cos = (g_patch_emb_q * pseudo_g_patch_emb_q).sum(dim=-1)
            mask = sents.bool()
            if mask.any():
                loss_pseudo = (1.0 - cos[mask]).mean()

        # Token / patch attention weights only if a later loss will use them.
        need_token_attn = (
            self.hparams.enable_loss_pfa
            or self.hparams.visual_group
            or (
                self.hparams.language_group
                and self.hparams.enable_loss_lga
            )
        )
        need_patch_attn = (
            self.hparams.enable_loss_pfa
            or self.hparams.visual_group
            or self.hparams.token_prototype is not None
        )
        token_atten_weights = None
        patch_atten_weights = None
        bz = img_emb_q.size(0)
        if need_token_attn:
            with torch.no_grad():
                token_atten_weights = token_attn_q.clone()
                nonzero_mask = token_atten_weights != 0
                for i in range(bz):
                    if nonzero_mask[i].any():
                        vals = token_atten_weights[i][nonzero_mask[i]]
                        low = torch.quantile(vals, 0.1)
                        high = torch.quantile(vals, 0.9)
                        token_atten_weights[i][nonzero_mask[i]] = torch.clamp(
                            token_atten_weights[i][nonzero_mask[i]], low, high)
                token_atten_weights = token_atten_weights / token_atten_weights.sum(
                    dim=1, keepdim=True).clamp(min=1e-8)

        if need_patch_attn:
            with torch.cuda.amp.autocast(dtype=patch_emb_q.dtype):
                _, attn_weights = self.patch_attention_weights_layer(
                    patch_emb_q, patch_emb_q, patch_emb_q
                )
            patch_atten_weights = attn_weights.mean(dim=1)
            patch_atten_weights /= patch_atten_weights.sum(
                dim=1, keepdims=True)

        _ga_temp = torch.exp(self.ga_temp).clamp(self.min_temp, self.max_temp)
        _pfa_temp = torch.exp(self.pfa_temp).clamp(self.min_temp, self.max_temp)
        _ica_temp = torch.exp(self.ica_temp).clamp(self.min_temp, self.max_temp)
        _pca_temp = torch.exp(self.pca_temp).clamp(self.min_temp, self.max_temp)
        _gro_temp = torch.exp(self.gro_temp).clamp(self.min_temp, self.max_temp)
        
        # Language Group Alignment (LGA). Grouping itself still runs when
        # language_group is on (PFA / PTC may need G^p); only the LGA CE is gated.
        loss_lga = img_emb_q.new_tensor(0.0)
        if self.hparams.language_group and self.hparams.enable_loss_lga:
            bz = token_emb_q.size(0)
            labels = torch.arange(bz).type_as(token_emb_q).long()

            group_sim_patch = torch.bmm(
                token_emb_q, g_patch_emb_q.permute(0, 2, 1)) / _gro_temp

            group_num = group_sim_patch.size(1)
            group_sim_patch = rearrange(group_sim_patch, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(group_num).type_as(
                token_emb_q).long().repeat(bz)
            loss_lga = torch.sum(F.cross_entropy(
                group_sim_patch, targets, reduction="none") * token_atten_weights.view(-1)) / bz

        # Visual Group Alignment (VGA)
        loss_vga = img_emb_q.new_tensor(0.0)
        if self.hparams.visual_group:
            bz = token_emb_q.size(0) if self._use_entity_align() else patch_emb_q.size(0)
            if self._use_entity_align():
                group_sim_token = torch.bmm(
                    token_emb_q, g_token_emb_q.permute(0, 2, 1)) / _gro_temp
                group_num = group_sim_token.size(1)
                group_sim_token = rearrange(
                    group_sim_token, "b n1 n2 -> (b n1) n2")
                targets = torch.arange(group_num).type_as(
                    token_emb_q).long().repeat(bz)
                loss_vga = torch.sum(
                    F.cross_entropy(
                        group_sim_token, targets, reduction="none")
                    * token_atten_weights.view(-1)
                ) / bz
            else:
                group_sim_token = torch.bmm(
                    patch_emb_q, g_token_emb_q.permute(0, 2, 1)) / _gro_temp
                group_num = group_sim_token.size(1)
                group_sim_token = rearrange(
                    group_sim_token, "b n1 n2 -> (b n1) n2")
                targets = torch.arange(group_num).type_as(
                    patch_emb_q).long().repeat(bz)
                loss_vga = torch.sum(
                    F.cross_entropy(
                        group_sim_token, targets, reduction="none")
                    * patch_atten_weights.view(-1)
                ) / bz

        loss_group = loss_lga + loss_vga

        # Global Alignment (GA)
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        if self._use_entity_align():
            loss_ga_ri = self._bilateral_infonce(
                img_emb_q, report_emb_q, _ga_temp)
            loss_ga_re = self._bilateral_infonce(
                report_emb_q, entity_emb_q, _ga_temp)
            loss_ga_ei = self._bilateral_infonce(
                entity_emb_q, img_emb_q, _ga_temp)
            if self.hparams.hyperparam_independ:
                loss_ga = (
                    self.hparams.lambda_ga_ri * loss_ga_ri
                    + self.hparams.lambda_ga_re * loss_ga_re
                    + self.hparams.lambda_ga_ei * loss_ga_ei
                )
            else:
                loss_ga = (loss_ga_ri + loss_ga_re + loss_ga_ei) / 3.0
        else:
            loss_ga = self._bilateral_infonce(
                img_emb_q, report_emb_q, _ga_temp)

        # compute retrieval accuracy (report <-> image)
        with torch.no_grad():
            scores_acc = img_emb_q.mm(report_emb_q.t()) / _ga_temp
            scores1_acc = scores_acc.transpose(0, 1)
            i2t_acc1, i2t_acc5 = self.precision_at_k(
                scores_acc, labels, top_k=(1, 5))
            t2i_acc1, t2i_acc5 = self.precision_at_k(
                scores1_acc, labels, top_k=(1, 5))
            acc1 = (i2t_acc1 + t2i_acc1) / 2.
            acc5 = (i2t_acc5 + t2i_acc5) / 2.

        # Pathology Fine-grained Alignment (PFA).
        # entity_fuse pairs: language_group only -> G^p <-> patch;
        # both groups -> G^p <-> G^t; visual_group only -> G^t <-> patch.
        # Skip the entire PFA forward when the loss is disabled (not just omit
        # it from the total).
        loss_pfa = img_emb_q.new_tensor(0.0)
        if self.hparams.enable_loss_pfa:
            if self.hparams.entity_extraction:
                mask = (1-sents).clone().detach().type_as(batch["imgs"]).bool()
            else:
                mask = torch.from_numpy(np.array(sents, dtype=object)[:, 1:] == "[PAD]").type_as(
                    batch["imgs"]).bool()

            if self._use_entity_align():
                report_atten_weights = self._report_token_attn_weights(
                    batch,
                    report_token_emb_q.size(1),
                    report_token_emb_q.device,
                    report_token_emb_q.dtype,
                )
                loss_pfa_ri = self.cross_alignment_loss(
                    report_token_emb_q,
                    patch_emb_q,
                    self.patch_local_atten_layer,
                    report_atten_weights,
                    temp=_pfa_temp,
                    attn_weight_kv=patch_atten_weights,
                )
                if self.hparams.language_group:
                    loss_pfa_re = self.cross_alignment_loss(
                        g_patch_emb_q,
                        token_emb_q,
                        self.token_local_atten_layer,
                        token_atten_weights,
                        temp=_pfa_temp,
                        mask=mask,
                    )
                else:
                    loss_pfa_re = self.cross_alignment_loss(
                        report_token_emb_q,
                        token_emb_q,
                        self.token_local_atten_layer,
                        report_atten_weights,
                        temp=_pfa_temp,
                        mask=mask,
                        attn_weight_kv=token_atten_weights,
                    )
                if self.hparams.visual_group:
                    loss_pfa_ei = self.cross_alignment_loss(
                        token_emb_q,
                        g_token_emb_q,
                        self.token_local_atten_layer,
                        token_atten_weights,
                        temp=_pfa_temp,
                        mask=mask,
                    )
                else:
                    loss_pfa_ei = self.cross_alignment_loss(
                        token_emb_q,
                        patch_emb_q,
                        self.token_local_atten_layer,
                        token_atten_weights,
                        temp=_pfa_temp,
                        mask=mask,
                        attn_weight_kv=patch_atten_weights,
                    )
                loss_pfa = (loss_pfa_ri + loss_pfa_re + loss_pfa_ei) / 3.0
            elif self.hparams.language_group:
                if self.hparams.visual_group:
                    loss_token = self.cross_alignment_loss(
                        g_patch_emb_q, g_token_emb_q,
                        self.token_local_atten_layer, token_atten_weights,
                        temp=_pfa_temp)
                    loss_patch = self.cross_alignment_loss(
                        g_token_emb_q, g_patch_emb_q,
                        self.patch_local_atten_layer, patch_atten_weights,
                        temp=_pfa_temp, mask=mask,
                        attn_weight_kv=token_atten_weights)
                else:
                    loss_token = self.cross_alignment_loss(
                        g_patch_emb_q, patch_emb_q,
                        self.token_local_atten_layer, token_atten_weights,
                        temp=_pfa_temp)
                    loss_patch = self.cross_alignment_loss(
                        patch_emb_q, g_patch_emb_q,
                        self.patch_local_atten_layer, patch_atten_weights,
                        temp=_pfa_temp, mask=mask,
                        attn_weight_kv=token_atten_weights)
                loss_pfa = (loss_patch + loss_token) / 2
            else:
                if self.hparams.visual_group:
                    loss_token = self.cross_alignment_loss(
                        patch_emb_q, g_token_emb_q,
                        self.patch_local_atten_layer, patch_atten_weights,
                        temp=_pfa_temp, mask=mask,
                        attn_weight_kv=patch_atten_weights)
                    loss_patch = self.cross_alignment_loss(
                        g_token_emb_q, patch_emb_q,
                        self.patch_local_atten_layer, patch_atten_weights,
                        temp=_pfa_temp, mask=mask,
                        attn_weight_kv=patch_atten_weights)
                else:
                    loss_token = self.cross_alignment_loss(
                        patch_emb_q, token_emb_q,
                        self.patch_local_atten_layer, patch_atten_weights,
                        temp=_pfa_temp, mask=mask,
                        attn_weight_kv=token_atten_weights)
                    loss_patch = self.cross_alignment_loss(
                        token_emb_q, patch_emb_q,
                        self.patch_local_atten_layer, token_atten_weights,
                        temp=_pfa_temp,
                        attn_weight_kv=patch_atten_weights)
                loss_pfa = (loss_patch + loss_token) / 2

        # Instance-level Codebook Alignment (ICA). Skip prototype pair loss
        # and instance-codebook renormalize when ICA is disabled. Token
        # prototypes are independent (PCA) and still normalized if present.
        with torch.no_grad():
            if (
                self.hparams.enable_loss_ica
                and self.instance_prototype_layer is not None
                and self.hparams.instance_prototype != "mapping"
            ):
                w = self.instance_prototype_layer.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                self.instance_prototype_layer.weight.copy_(w)
            if self.token_prototype_layer is not None:
                w = self.token_prototype_layer.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                self.token_prototype_layer.weight.copy_(w)

        loss_ica = img_emb_q.new_tensor(0.0)
        if (
            self.hparams.enable_loss_ica
            and self.hparams.instance_prototype is not None
        ):
            if self._use_entity_align():
                loss_ica_ri = self._instance_proto_pair_loss(
                    img_emb_q, report_emb_q, _ica_temp)
                loss_ica_re = self._instance_proto_pair_loss(
                    report_emb_q, entity_emb_q, _ica_temp)
                loss_ica_ei = self._instance_proto_pair_loss(
                    entity_emb_q, img_emb_q, _ica_temp)
                loss_ica = (loss_ica_ri + loss_ica_re + loss_ica_ei) / 3.0
            else:
                loss_ica = self._instance_proto_pair_loss(
                    img_emb_q, report_emb_q, _ica_temp)

        loss_pca = 0
        if not self.hparams.token_prototype is None:
            use_pca_weight = getattr(self.hparams, "pca_with_weight", False) or getattr(
                self.hparams, "tpa_with_weight", False)
            patch_w = patch_atten_weights if use_pca_weight else None
            token_w = token_atten_weights if use_pca_weight else None
            if self._use_entity_align():
                loss_pca_pi = self._token_proto_pair_loss(
                    patch_emb_q, report_token_emb_q, _pca_temp,
                    weight_a=patch_w, weight_b=None)
                loss_pca_pe = self._token_proto_pair_loss(
                    patch_emb_q, token_emb_q, _pca_temp,
                    weight_a=patch_w, weight_b=token_w)
                loss_pca_re = self._token_proto_pair_loss(
                    report_token_emb_q, token_emb_q, _pca_temp,
                    weight_a=None, weight_b=token_w)
                loss_pca = (loss_pca_pi + loss_pca_pe + loss_pca_re) / 3.0
            else:
                loss_pca = self._token_proto_pair_loss(
                    patch_emb_q, token_emb_q, _pca_temp,
                    weight_a=patch_w, weight_b=token_w)

            if self.hparams.prototype_orth:
                if not self.token_prototype_layer is None:
                    prototype_vector = torch.squeeze(self.token_prototype_layer.weight)
                    prototype_vector_T = torch.transpose(prototype_vector,0,1)
                    orth_operator = torch.matmul(prototype_vector,prototype_vector_T)
                    I_operator = torch.eye(prototype_vector.size(0),prototype_vector.size(0)).to(orth_operator.device)
                    difference_value = orth_operator - I_operator
                    loss_orth = (torch.norm(difference_value,p=1) ** 0.5)/difference_value.size(0)
                loss_pca += self.hparams.lambda_orth * loss_orth

        return (
            loss_ga,
            loss_pfa,
            loss_ica,
            loss_pca,
            loss_lga,
            loss_vga,
            loss_group,
            loss_entity,
            loss_pseudo,
            loss_medklip_cls,
            loss_medklip_loc,
            acc1,
            acc5,
        )

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def on_load_checkpoint(self, checkpoint):
        """Rename legacy keys in the checkpoint so they match current `state_dict` names."""
        sd = checkpoint.get("state_dict")
        if not sd:
            return
        # Older runs saved entity heads as esp_*; current module uses entity_pred_*.
        for ckpt_prefix, model_prefix in (
            ("esp_ana.", "entity_pred_ana."),
            ("esp_sta.", "entity_pred_sta."),
        ):
            for key in list(sd.keys()):
                if key.startswith(ckpt_prefix):
                    new_key = model_prefix + key[len(ckpt_prefix):]
                    if new_key not in sd:
                        sd[new_key] = sd.pop(key)
                    else:
                        del sd[key]
        for old_suffix, new_suffix in (
            ("gia_temp", "ga_temp"),
            ("tca_temp", "pfa_temp"),
            ("ipa_temp", "ica_temp"),
            ("tpa_temp", "pca_temp"),
        ):
            for key in list(sd.keys()):
                if key == old_suffix or key.endswith("." + old_suffix):
                    new_key = key.replace(old_suffix, new_suffix, 1)
                    if new_key not in sd:
                        sd[new_key] = sd.pop(key)
                    elif key != new_key:
                        del sd[key]

    def _group_loss_for_total(self, loss_lga, loss_vga):
        group = loss_vga
        if self.hparams.enable_loss_lga:
            group = group + loss_lga
        return group

    def _build_train_log_dict(
        self,
        total_loss,
        loss_ga,
        loss_pfa,
        loss_ica,
        loss_pca,
        loss_lga,
        loss_vga,
        loss_group,
        loss_entity,
        loss_pseudo,
        loss_medklip_cls,
        loss_medklip_loc,
        acc1,
        acc5,
    ):
        hp = self.hparams
        log = {
            "tra/loss": total_loss,
            "tra/acc1": acc1,
            "tra/acc5": acc5,
        }
        ga_log_on = hp.lambda_ga != 0 or (
            hp.entity_align and hp.entity_extraction and hp.hyperparam_independ
        )
        if ga_log_on:
            if hp.entity_align and hp.entity_extraction and hp.hyperparam_independ:
                log["tra/lga"] = loss_ga
            else:
                log["tra/lga"] = hp.lambda_ga * loss_ga
        if hp.lambda_pfa != 0 and hp.enable_loss_pfa:
            log["tra/lpfa"] = hp.lambda_pfa * loss_pfa
        ica_on = (
            hp.lambda_ica != 0
            and hp.instance_prototype is not None
            and hp.enable_loss_ica
        )
        if ica_on:
            log["tra/lica"] = hp.lambda_ica * loss_ica
        pca_on = hp.lambda_pca != 0 and hp.token_prototype is not None
        if pca_on:
            log["tra/lpca"] = hp.lambda_pca * loss_pca
        if hp.lambda_group != 0 and hp.language_group and hp.enable_loss_lga:
            log["tra/llga"] = hp.lambda_group * loss_lga
        if hp.lambda_group != 0 and hp.visual_group:
            log["tra/lvga"] = hp.lambda_group * loss_vga
        if hp.lambda_group != 0 and (hp.language_group or hp.visual_group):
            log["tra/lgro"] = hp.lambda_group * self._group_loss_for_total(
                loss_lga, loss_vga
            )
        if (
            hp.enable_entity_pred
            and hp.entity_extraction
            and hp.lambda_esp != 0
        ):
            log["tra/lesp"] = hp.lambda_esp * loss_entity
        ptp_group_on = hp.language_group
        if hp.entity_align and hp.entity_extraction:
            ptp_group_on = hp.visual_group
        if (
            hp.enable_pseudo_triplet_pred
            and hp.entity_extraction
            and ptp_group_on
            and hp.lambda_ptp != 0
        ):
            log["tra/lptp"] = hp.lambda_ptp * loss_pseudo
        if hp.enable_medklip and hp.lambda_medklip_cls != 0:
            log["tra/lmk_cls"] = hp.lambda_medklip_cls * loss_medklip_cls
        if hp.enable_medklip and hp.lambda_medklip_loc != 0:
            log["tra/lmk_loc"] = hp.lambda_medklip_loc * loss_medklip_loc

        # detach: when learnable_temperature=True these are Parameters
        if ga_log_on:
            log["temp/ga"] = (
                torch.exp(self.ga_temp).clamp(self.min_temp, self.max_temp).detach()
            )
        if hp.lambda_pfa != 0 and hp.enable_loss_pfa:
            log["temp/pfa"] = (
                torch.exp(self.pfa_temp).clamp(self.min_temp, self.max_temp).detach()
            )
        if ica_on:
            log["temp/ica"] = (
                torch.exp(self.ica_temp).clamp(self.min_temp, self.max_temp).detach()
            )
        if pca_on:
            log["temp/pca"] = (
                torch.exp(self.pca_temp).clamp(self.min_temp, self.max_temp).detach()
            )
        if hp.lambda_group != 0 and (hp.language_group or hp.visual_group):
            log["temp/group"] = (
                torch.exp(self.gro_temp).clamp(self.min_temp, self.max_temp).detach()
            )
        if hp.enable_medklip and self.medklip_temp is not None:
            log["temp/medklip"] = (
                torch.exp(self.medklip_temp)
                .clamp(self.min_temp, self.max_temp)
                .detach()
            )
        if hp.language_group:
            log["temp/language_group_threshold"] = self.language_group_threshold
        if hp.visual_group:
            log["temp/token_group_threshold"] = self.token_group_threshold
        return log

    def _build_val_log_dict(
        self,
        loss,
        loss_ga,
        loss_pfa,
        loss_ica,
        loss_pca,
        loss_lga,
        loss_vga,
        loss_group,
        loss_entity,
        loss_pseudo,
        loss_medklip_cls,
        loss_medklip_loc,
        acc1,
        acc5,
    ):
        hp = self.hparams
        log = {
            "val/loss": loss,
            "val/acc1": acc1,
            "val/acc5": acc5,
        }
        ga_log_on = hp.lambda_ga != 0 or (
            hp.entity_align and hp.entity_extraction and hp.hyperparam_independ
        )
        if ga_log_on:
            if hp.entity_align and hp.entity_extraction and hp.hyperparam_independ:
                log["val/lga"] = loss_ga
            else:
                log["val/lga"] = hp.lambda_ga * loss_ga
        if hp.lambda_pfa != 0 and hp.enable_loss_pfa:
            log["val/lpfa"] = hp.lambda_pfa * loss_pfa
        ica_on = (
            hp.lambda_ica != 0
            and hp.instance_prototype is not None
            and hp.enable_loss_ica
        )
        if ica_on:
            log["val/lica"] = hp.lambda_ica * loss_ica
        pca_on = hp.lambda_pca != 0 and hp.token_prototype is not None
        if pca_on:
            log["val/lpca"] = hp.lambda_pca * loss_pca
        if hp.lambda_group != 0 and hp.language_group and hp.enable_loss_lga:
            log["val/llga"] = hp.lambda_group * loss_lga
        if hp.lambda_group != 0 and hp.visual_group:
            log["val/lvga"] = hp.lambda_group * loss_vga
        if hp.lambda_group != 0 and (hp.language_group or hp.visual_group):
            log["val/lgro"] = hp.lambda_group * self._group_loss_for_total(
                loss_lga, loss_vga
            )
        if (
            hp.enable_entity_pred
            and hp.entity_extraction
            and hp.lambda_esp != 0
        ):
            log["val/lesp"] = hp.lambda_esp * loss_entity
        ptp_group_on = hp.language_group
        if hp.entity_align and hp.entity_extraction:
            ptp_group_on = hp.visual_group
        if (
            hp.enable_pseudo_triplet_pred
            and hp.entity_extraction
            and ptp_group_on
            and hp.lambda_ptp != 0
        ):
            log["val/lptp"] = hp.lambda_ptp * loss_pseudo
        if hp.enable_medklip and hp.lambda_medklip_cls != 0:
            log["val/lmk_cls"] = hp.lambda_medklip_cls * loss_medklip_cls
        if hp.enable_medklip and hp.lambda_medklip_loc != 0:
            log["val/lmk_loc"] = hp.lambda_medklip_loc * loss_medklip_loc
        return log

    def training_step(self, batch, batch_idx):
        (
            loss_ga,
            loss_pfa,
            loss_ica,
            loss_pca,
            loss_lga,
            loss_vga,
            loss_group,
            loss_entity,
            loss_pseudo,
            loss_medklip_cls,
            loss_medklip_loc,
            acc1,
            acc5,
        ) = self(batch, batch_idx, "train")
        if self._use_entity_align() and self.hparams.hyperparam_independ:
            ga_term = loss_ga
        else:
            ga_term = self.hparams.lambda_ga * loss_ga
        hp = self.hparams
        total_loss = ga_term + self.hparams.lambda_pca * loss_pca
        if hp.enable_loss_pfa:
            total_loss = total_loss + hp.lambda_pfa * loss_pfa
        if hp.enable_loss_ica:
            total_loss = total_loss + hp.lambda_ica * loss_ica
        if hp.language_group or hp.visual_group:
            total_loss = (
                total_loss
                + hp.lambda_group * self._group_loss_for_total(loss_lga, loss_vga)
            )
        if self.hparams.enable_entity_pred:
            total_loss = (
                total_loss
                + self.hparams.lambda_esp * loss_entity
            )
        if self.hparams.enable_pseudo_triplet_pred:
            total_loss = (
                total_loss
                + self.hparams.lambda_ptp * loss_pseudo
            )
        if self.hparams.enable_medklip:
            total_loss = (
                total_loss
                + self.hparams.lambda_medklip_cls * loss_medklip_cls
                + self.hparams.lambda_medklip_loc * loss_medklip_loc
            )

        log = self._build_train_log_dict(
            total_loss,
            loss_ga,
            loss_pfa,
            loss_ica,
            loss_pca,
            loss_lga,
            loss_vga,
            loss_group,
            loss_entity,
            loss_pseudo,
            loss_medklip_cls,
            loss_medklip_loc,
            acc1,
            acc5,
        )
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
        )
        return total_loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.hparams.learnable_temperature:
            torch.nn.utils.clip_grad_norm_(
                [
                    self.ga_temp,
                    self.pfa_temp,
                    self.ica_temp,
                    self.pca_temp,
                    self.gro_temp,
                ],
                max_norm=1.0,
            )
        
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            if not self.hparams.instance_prototype is None:
                for param in self.instance_prototype_layer.parameters():
                    param.grad = None
            if not self.hparams.token_prototype is None:
                for param in self.token_prototype_layer.parameters():
                    param.grad = None

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        # LightningModule hook signature (not Callback). Clamp log-temps after the step
        # so learnable τ stays in [min_temp, max_temp].
        with torch.no_grad():
            if self.hparams.learnable_temperature:
                self.ga_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))
                self.pfa_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))
                self.ica_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))
                self.pca_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))
                self.gro_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))
            if self.medklip_temp is not None:
                self.medklip_temp.clamp_(
                    math.log(self.min_temp), math.log(self.max_temp)
                )
        if self.training and 0 < self.hparams.threshold_momentum < 1.0:
            if self.hparams.language_group:
                self.language_group_threshold = self.hparams.threshold_momentum * self.language_group_threshold + \
                    (1 - self.hparams.threshold_momentum) * self.patch_cur_threshold
                self.language_group_threshold = max(0.001, min(0.99, self.language_group_threshold))
            if self.hparams.visual_group:
                self.token_group_threshold = self.hparams.threshold_momentum * self.token_group_threshold + \
                    (1 - self.hparams.threshold_momentum) * self.token_cur_threshold
                self.token_group_threshold = max(0.001, min(0.99, self.token_group_threshold))
            
    def validation_step(self, batch, batch_idx):
        (
            loss_ga,
            loss_pfa,
            loss_ica,
            loss_pca,
            loss_lga,
            loss_vga,
            loss_group,
            loss_entity,
            loss_pseudo,
            loss_medklip_cls,
            loss_medklip_loc,
            acc1,
            acc5,
        ) = self(batch, batch_idx, "valid")

        if self._use_entity_align() and self.hparams.hyperparam_independ:
            ga_term = loss_ga
        else:
            ga_term = self.hparams.lambda_ga * loss_ga
        hp = self.hparams
        loss = ga_term + hp.lambda_pca * loss_pca
        if hp.enable_loss_pfa:
            loss = loss + hp.lambda_pfa * loss_pfa
        if hp.enable_loss_ica:
            loss = loss + hp.lambda_ica * loss_ica
        if hp.language_group or hp.visual_group:
            loss = (
                loss
                + hp.lambda_group * self._group_loss_for_total(loss_lga, loss_vga)
            )
        if self.hparams.enable_entity_pred:
            loss = loss + self.hparams.lambda_esp * loss_entity
        if self.hparams.enable_pseudo_triplet_pred:
            loss = loss + self.hparams.lambda_ptp * loss_pseudo
        if self.hparams.enable_medklip:
            loss = (
                loss
                + self.hparams.lambda_medklip_cls * loss_medklip_cls
                + self.hparams.lambda_medklip_loc * loss_medklip_loc
            )

        log = self._build_val_log_dict(
            loss,
            loss_ga,
            loss_pfa,
            loss_ica,
            loss_pca,
            loss_lga,
            loss_vga,
            loss_group,
            loss_entity,
            loss_pseudo,
            loss_medklip_cls,
            loss_medklip_loc,
            acc1,
            acc5,
        )
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        temp_ids = set()
        if self.hparams.learnable_temperature:
            for _t in (
                self.ga_temp,
                self.pfa_temp,
                self.ica_temp,
                self.pca_temp,
                self.gro_temp,
            ):
                if isinstance(_t, nn.Parameter):
                    temp_ids.add(id(_t))
        decay_params, no_decay_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in temp_ids:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        if self.hparams.learnable_temperature and no_decay_params:
            param_groups = [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        else:
            param_groups = [
                {
                    "params": [p for p in self.parameters() if p.requires_grad],
                    "weight_decay": self.hparams.weight_decay,
                }
            ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.1)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs
