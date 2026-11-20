import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from medeha.constants import BERT_CONFIG, BERT_DIR
from medeha.models.backbones import cnn_backbones
from medeha.models.backbones.med import BertModel
from medeha.models.backbones.vits import create_vit
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

logging.set_verbosity_error()


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "vit" in model_name:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict, strict=False)

            self.global_embed = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )

            self.local_embed = LocalEmbedding(
                vision_width, hidden_dim, output_dim
            )

        else:
            model_function = getattr(
                cnn_backbones, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )

            # Average pooling
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

            self.local_embed = LocalEmbedding(
                self.interm_feature_dim, hidden_dim, output_dim
            )

    def resnet_forward(self, x, get_local=True):
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()

    def vit_forward(self, x, propompt=None):
        return self.model(x, propompt, register_blk=11)

    def forward(self, x, propompt=None, get_local=False):
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x, propompt)
            if propompt is None:
                return img_feat[:, 0].contiguous(), \
                    img_feat[:, 1:].contiguous(), \
                    img_feat[:, 1:].contiguous()
            else:
                return img_feat[:, 0].contiguous(), \
                    img_feat[:, 1:-propompt.size(0)].contiguous(), \
                    img_feat[:, -propompt.size(0):].contiguous()


class BertEncoder(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True):
        super(BertEncoder, self).__init__()
        # self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        self.bert_type = str(BERT_DIR)
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        # self.max_sent_num = 10

        self.config = BertConfig.from_json_file(str(BERT_CONFIG))
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,
        )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

        self.global_embed = GlobalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt

    def forward(self, ids, attn_mask, token_type, get_local=False):
        outputs = self.model(ids, attn_mask, token_type,
                             return_dict=True, mode="text")

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, ids, last_layer_attn)
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in ids]

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        return report_feat, word_feat, last_atten_pt, sents

class TransformerEncoderLayerWithCrossAttn(nn.Module):
    """Transformer encoder layer with optional cross-attention to external features"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.gelu, layer_norm_eps=1e-5, batch_first=True, cross_attention=False):
        super().__init__()
        self.batch_first = batch_first
        self.cross_attention = cross_attention
                
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention layer (optional)
        if cross_attention:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = activation
    
    def forward(self, src, memory=None, src_mask=None, memory_mask=None,
                src_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: input sequence [batch_size, seq_len, d_model]
            memory: external memory for cross-attention [batch_size, mem_len, d_model] (optional)
            src_mask: self-attention mask
            memory_mask: cross-attention mask
            src_key_padding_mask: padding mask for src
            memory_key_padding_mask: padding mask for memory
        Returns:
            output: transformed sequence [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        src2 = self.self_attn(src, src, src, 
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask,
                             need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Cross-attention with residual connection (if memory provided)
        if memory is not None and self.cross_attention:
            src2 = self.cross_attn(src, memory, memory,
                                  attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask,
                                  need_weights=False)[0]
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Feedforward network with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        
        return src


class TransformerFusionLayer(nn.Module):
    """Transformer-based fusion layer with optional cross-attention to report features"""
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len, 
                 fusion_type="global", cross_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.cross_attention = cross_attention
        
        # Positional embeddings (FP32 for numerical stability)
        self.positional_embedding = nn.Parameter(
            torch.randn(max_seq_len + 1, embed_dim, dtype=torch.float32) / embed_dim ** 0.5
        )
        
        # CLS token for global feature extraction
        self.cls_token = nn.Parameter(torch.randn(embed_dim, dtype=torch.float32))
        
        # Transformer encoder layers with optional cross-attention
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttn(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
                cross_attention = cross_attention
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, report_feat=None, attention_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            x: input features [batch_size, seq_len, embed_dim]
            report_feat: optional report features for cross-attention [batch_size, report_seq_len, embed_dim]
            attention_mask: optional mask [batch_size, seq_len]
            memory_key_padding_mask: optional [batch_size, report_seq_len], True = pad position (ignore in cross-attn)
        Returns:
            global_feature: [batch_size, embed_dim] if fusion_type=="global"
            or all_features: [batch_size, seq_len, embed_dim] if fusion_type=="local"
            or split_features: [batch_size, embed_dim], [batch_size, seq_len, embed_dim] if fusion_type=="split"
            attention_weights: approximate attention weights from last layer
        """
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype
        device = x.device
        
        # Convert positional embeddings to input dtype
        pos_embed = self.positional_embedding[:seq_len+1].unsqueeze(0).expand(batch_size, -1, -1)
        pos_embed = pos_embed.to(device).to(dtype)
        
        # Prepare input with CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        cls_tokens = cls_tokens.to(device).to(dtype)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, embed_dim]
        x_with_cls = x_with_cls + pos_embed
        
        # Create attention mask for transformer
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
            full_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            key_padding_mask = ~full_attention_mask
        else:
            key_padding_mask = None
        
        # Prepare report features for cross-attention (if enabled)
        memory = None
        memory_key_padding_mask = None
        if self.cross_attention and report_feat is not None:
            # Project report features to same dimension if needed
            if report_feat.size(-1) != self.embed_dim:
                # Use linear projection if dimensions don't match
                proj = nn.Linear(report_feat.size(-1), self.embed_dim).to(device)
                memory = proj(report_feat)
            else:
                memory = report_feat

            if memory_key_padding_mask is not None:
                memory_key_padding_mask = memory_key_padding_mask.to(device)
            else:
                # Legacy: treat all memory positions as valid
                memory_key_padding_mask = torch.zeros(
                    batch_size, memory.size(1),
                    device=device, dtype=torch.bool
                )
        
        # Transformer encoding with optional cross-attention
        output = x_with_cls
        for layer in self.layers:
            output = layer(
                output, 
                memory=memory,
                src_key_padding_mask=key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # Apply final layer normalization
        output = self.norm(output)
        
        # Extract attention weights (approximate - uniform distribution as placeholder)
        attention_weights = torch.ones(batch_size, seq_len, device=device) / seq_len
        
        # Return features based on fusion type
        if self.fusion_type == "global":
            # Extract global feature (CLS token)
            return output[:, 0], attention_weights
        
        if self.fusion_type == "split":
            # Return both global feature and sequence features
            return output[:, 0], output[:, 1:], attention_weights
        
        # Default: return sequence features (excluding CLS token)
        return output[:, 1:], attention_weights

# class TransformerFusionLayer(nn.Module):
#     """Transformer-based fusion layer with learnable CLS token"""
#     def __init__(self, embed_dim, num_heads, num_layers, max_seq_len, fusion_type="global"):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.fusion_type = fusion_type
        
#         # Positional embeddings
#         self.positional_embedding = nn.Parameter(
#             torch.randn(max_seq_len + 1, embed_dim, dtype=torch.float32) / embed_dim ** 0.5
#         )
        
#         # CLS token for global feature extraction
#         self.cls_token = nn.Parameter(torch.randn(embed_dim, dtype=torch.float32))
        
#         # Transformer encoder layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, 
#             num_layers=2 if num_layers is None else num_layers,
#             norm=nn.LayerNorm(embed_dim)
#         )
        
#         # Layer normalization
#         self.norm = nn.LayerNorm(embed_dim)
    
#     def forward(self, x, attention_mask=None):
#         """
#         Args:
#             x: input features [batch_size, seq_len, embed_dim]
#             attention_mask: optional mask [batch_size, seq_len]
#         Returns:
#             global_feature: [batch_size, embed_dim] if fusion_type=="global"
#             or all_features: [batch_size, seq_len, embed_dim] if fusion_type=="local"
#             or split_features: [batch_size, embed_dim], [batch_size, seq_len, embed_dim] if fusion_type=="split"
#             attention_weights: attention weights from last layer
#         """
#         batch_size, seq_len, _ = x.shape
#         dtype = x.dtype
        
#         # Add positional embeddings
#         pos_embed = self.positional_embedding[:seq_len+1].unsqueeze(0).expand(batch_size, -1, -1)
#         pos_embed = pos_embed.to(x.device).to(dtype)
        
#         # Prepare input with CLS token
#         cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
#         cls_tokens = cls_tokens.to(x.device).to(dtype)
        
#         x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, embed_dim]
#         x_with_cls = x_with_cls + pos_embed
        
#         # Create attention mask if needed
#         if attention_mask is not None:
#             # Add CLS token to attention mask
#             cls_mask = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
#             full_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
#             # PyTorch Transformer uses True for positions to ignore
#             key_padding_mask = ~full_attention_mask
#         else:
#             key_padding_mask = None
        
#         # Transformer encoding
#         with torch.cuda.amp.autocast(dtype=dtype):
#             output = self.transformer_encoder(
#                 x_with_cls,
#                 src_key_padding_mask=key_padding_mask if key_padding_mask is not None else None
#             )
        
#         # Get attention weights (approximate from last layer)
#         # This is a simplified version - full attention weights would require hooking into the transformer
#         attention_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len
        
#         if self.fusion_type == "global":
#             # Extract global feature (CLS token)
#             return output[:, 0], attention_weights
#         if self.fusion_type == "split":
#             return output[:, 0], output[:, 1:], attention_weights
#         else:
#             # Return all features except CLS token
#             return output[:, 1:], attention_weights

class AttentionPool2d(nn.Module):
    def __init__(self, context_length: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(context_length + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, weights = nn.functional.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x.squeeze(0), weights[:,0,1:]