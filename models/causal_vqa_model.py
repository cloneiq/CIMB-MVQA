import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_encoders.clip_model import build_model, adapt_position_encoding
from models.vision_encoders import swin_transformer as swin
from models.language_encoders.bert_model import BertConfig, BertCrossLayer
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
import numpy as np
from utils.lossfc_tools import contrastive_decoupling_loss
from models.cross_modal_graph import CausalGraphNetwork
from . import simple_cnn
from . import auto_encoder
from models.utils import init_weights


def get_extended_suppression_attention_mask(v_mask, suppress_value=-5.0):
    attn_bias = torch.zeros_like(v_mask).to(v_mask.device)
    attn_bias[v_mask == 0.0] = -1e9  # Non-structural region (directly masked)
    attn_bias[v_mask == -1.0] = suppress_value  # Co-occurrence structural region (soft suppression)
    attn_bias[v_mask == 1.0] = 0.0  # Pathological region (keep)
    return attn_bias[:, None, None, :]


def get_extended_suppression_key_words_attention_mask(q_mask, suppress_value=-3.0):
    attn_bias = torch.zeros_like(q_mask).to(q_mask.device)
    attn_bias[q_mask == 0.0] = -1e9  # Non-structural region (directly masked)
    attn_bias[q_mask == 1.0] = suppress_value
    return attn_bias[:, None, None, :]


def avg_pool(x, output_size):
    # x: [B, T, C]
    x = x.permute(0, 2, 1)  # [B, C, T]
    x = F.adaptive_avg_pool1d(x, output_size)  # [B, C, output_size]
    return x.permute(0, 2, 1)  # [B, output_size, C]


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CausalVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        label_num = config.get('num_hid', False)
        self.config = config
        self.vision_encoder = build_model(name=config.get('visual_backbone', 'ViT-B/16'),
                                          resolution_after=config.get('image_size', 384))
        # Load pretrained weights
        roberta_config = RobertaConfig.from_pretrained('roberta-base')

        # Main language encoder - load pretrained parameters and freeze
        self.language_encoder = RobertaModel.from_pretrained('roberta-base', config=roberta_config)

        # do_language encoder - load pretrained parameters but allow updating
        # self.do_language_encoder = RobertaModel.from_pretrained('roberta-base', config=roberta_config)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_token>']})

        self.embedding_dim = self.language_encoder.embeddings.word_embeddings.weight.shape[1]

        self.multi_modal_language_proj = nn.Linear(config.get('input_text_embed_size', 768),
                                                   config.get('hidden_size', 768))
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config.get('input_image_embed_size', 768),
                                                 config.get('hidden_size', 768))
        self.multi_modal_vision_proj.apply(init_weights)
        self.multi_modal_do_language_proj = nn.Linear(config.get('input_text_embed_size', 768),
                                                      config.get('hidden_size', 768))
        self.multi_modal_do_language_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(3, config.get("hidden_size", 768))
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_language_layers.apply(init_weights)
        self.multi_modal_do_language_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_do_language_layers.apply(init_weights)

        self.multi_modal_vision_post_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 2))])
        self.multi_modal_vision_post_layers.apply(init_weights)


        self.multi_modal_vision_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_language_pooler.apply(init_weights)
        self.multi_modal_do_language_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_do_language_pooler.apply(init_weights)
        self.multi_modal_graph_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_graph_pooler.apply(init_weights)

        ae_weight_path = config.get('ae_weight_path', 'pretrained_weights/pretrained_ae.pth')
        self.auto_encoder = auto_encoder.Auto_Encoder_Model()
        self.auto_encoder.load_state_dict(torch.load(ae_weight_path, map_location='cpu'))
        maml_weight_path = config.get('maml_weight_path', 'pretrained_weights/pretrained_maml.weights')
        self.maml_model = simple_cnn.SimpleCNN(maml_weight_path)
        self.convert = nn.Linear(16384, 64)
        self.convert.apply(init_weights)

        # Create a separate pure vision feature MLP instead of overwriting the original
        self.pure_vision_MLP = nn.Sequential(
            nn.Linear(128, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        self.pure_vision_MLP.apply(init_weights)

        self.vision_embedding_MLP = nn.Sequential(
            nn.Linear(768, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        self.vision_embedding_MLP.apply(init_weights)

        self.do_position_embeddings = nn.Embedding(1026, config.get("hidden_size", 768))
        self.do_position_embeddings.apply(init_weights)

        self.do_token_type_embeddings = nn.Embedding(2, config.get("hidden_size", 768))
        self.do_token_type_embeddings.apply(init_weights)

        self.visual_modality_embedding = nn.Parameter(torch.randn(1, 1, config.get("hidden_size", 768)))

        self.text_modal_graph = CausalGraphNetwork(hidden_dim=config.get("hidden_size", 768),
                                                   num_layers=config.get("num_layers", 2),
                                                   heads=config.get("graph_heads", 8),
                                                   dropout=config.get("graph_dropout", 0.1))
        self.text_modal_graph.apply(init_weights)

        # self.PFBAN = ProgressiveFineBAN(hidden=config.get('hidden_size', 768), glimpses=config.get("glimpses", 2))

        self.vqa_head = nn.Sequential(
            nn.Linear(config.get("hidden_size", 768) * 4, config.get("hidden_size", 768) * 8),
            nn.LayerNorm(config.get("hidden_size", 768) * 8),
            nn.GELU(),
            nn.Linear(config.get("hidden_size", 768) * 8, label_num),
        )
        self.vqa_head.apply(init_weights)

        ckpt = torch.load(config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        state_dict = adapt_position_encoding(state_dict,
                                             after=config.get('image_size', 384),
                                             patch_size=config.get('patch_size', 16))

        # Load pretrained weights into the model
        self.load_state_dict(state_dict, strict=False)
        self.language_encoder.resize_token_embeddings(len(tokenizer))

    def forward(self, images, questions_ids, attention_mask, do_questions_ids, do_attention_mask,
                image_token_type_idx=1, special_visual_token_type_idx=2, ae_images=None, maml_images=None,
                v_mask_pre=None, v_mask=None, q_mask_pre=None, pattern_embedding=None, entity_embedding=None, epoch=0,
                causal_start_epoch=0, training=True):
        #  == Begin: Ori Text Encoding ==
        # Use the frozen language_encoder to process the original text
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=questions_ids)
        text_input_shape = attention_mask.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(attention_mask, text_input_shape,
                                                                                questions_ids.device)
        do_extended_text_masks = self.language_encoder.get_extended_attention_mask(do_attention_mask,
                                                                                   do_attention_mask.size(),
                                                                                   do_questions_ids.device)

        for layer in self.language_encoder.encoder.layer:
            q_feats = uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)

        # == Begin: do Text Encoding ==
        decoder = None
        text_ori_embedding = self.language_encoder.embeddings.word_embeddings(do_questions_ids)
        do_uni_modal_text_feats, decoder = self._do_language_encoder(do_questions_ids,
                                                                     text_ori_embedding, ae_images,
                                                                     maml_images,
                                                                     attention_mask=do_extended_text_masks)
        do_uni_modal_text_feats = self.multi_modal_do_language_proj(do_uni_modal_text_feats)

        # == Begin: Ori Image Encoding ==
        if training and v_mask is None:
            images_feats = self.vision_encoder(images)
            ori_image_feats, pos_image_feats, neg_image_feats = torch.chunk(images_feats, 3, dim=0)
            v_feat = uni_modal_image_feats = ori_image_feats
        elif training and v_mask is not None and epoch % 2 != 0:
            images_feats = self.vision_encoder(images)
            ori_image_feats, pos_image_feats, neg_image_feats = torch.chunk(images_feats, 3, dim=0)
            uni_modal_image_feats = torch.cat([ori_image_feats, neg_image_feats], dim=0)
        elif training and v_mask is not None and epoch % 2 == 0:
            images_feats = self.vision_encoder(images)
            ori_image_feats, pos_image_feats, neg_image_feats = torch.chunk(images_feats, 3, dim=0)
            uni_modal_image_feats = torch.cat([pos_image_feats, neg_image_feats], dim=0)
        else:
            uni_modal_image_feats = self.vision_encoder(images)
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=images.device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 images.device)
        # == End: Ori Image Encoding ==
        uni_modal_text_feats, uni_modal_image_feats, uni_modal_do_text_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(attention_mask)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
            do_uni_modal_text_feats + self.modality_type_embeddings(
                torch.full_like(do_attention_mask, special_visual_token_type_idx))
        )

        if v_mask_pre is not None:
            extended_image_masks_pre = self.language_encoder.get_extended_attention_mask(v_mask_pre, v_mask_pre.size(),
                                                                                         images.device)
        else:
            extended_image_masks_pre = extended_image_masks
        if v_mask is not None:
            extended_image_suppression_masks = get_extended_suppression_attention_mask(v_mask)
        else:
            extended_image_suppression_masks = extended_image_masks
        if q_mask_pre is not None:
            extended_text_masks_pre = self.language_encoder.get_extended_attention_mask(q_mask_pre, q_mask_pre.size(),
                                                                                        do_questions_ids.device)
        else:
            extended_text_masks_pre = extended_text_masks
        x, y, z = uni_modal_text_feats, uni_modal_image_feats, uni_modal_do_text_feats
        for layer_idx, (text_layer, image_layer, do_text_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                                 self.multi_modal_vision_layers,
                                                                                 self.multi_modal_do_language_layers)):
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            z1 = do_text_layer(z, y, do_extended_text_masks, extended_image_masks, output_attentions=True)
            x, y, z = x1[0], y1[0], z1[0]

        for layer_idx, post_image_layer in enumerate(self.multi_modal_vision_post_layers):
            if layer_idx < 1:
                y1 = post_image_layer(y, x, extended_image_masks_pre, extended_text_masks_pre)
                y = y1[0]
            else:
                y1 = post_image_layer(y, x, extended_image_suppression_masks, extended_text_masks_pre)
                y = y1[0]
        # # == End: do Co-Attention ==
        #
        multi_modal_do_text_cls_feats = self.multi_modal_do_language_pooler(z)

        mask_expanded = attention_mask.unsqueeze(-1).expand(uni_modal_text_feats.size())
        uni_modal_text_cls_feats = x[:, 0, :] + torch.sum(x[:, 1:, :] * mask_expanded[:, 1:, :], dim=1) / (
            torch.clamp(mask_expanded.sum(1), min=1e-9))
        uni_modal_text_cls_do_feats = self.text_modal_graph(uni_modal_text_cls_feats, pattern_embedding,
                                                            entity_embedding, multi_modal_do_text_cls_feats)
        uni_modal_text_cls_do_feats = self.multi_modal_graph_pooler(uni_modal_text_cls_do_feats)
        x = torch.cat([uni_modal_text_cls_do_feats.unsqueeze(1) + x[:, 0, :], x[:, 1:, :]], dim=1)
        # == Begin: do Pooling ==
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        multi_modal_cls_feats = torch.cat(
            [multi_modal_text_cls_feats, multi_modal_image_cls_feats, multi_modal_do_text_cls_feats, uni_modal_text_cls_do_feats], dim=-1)
        logits = self.vqa_head(multi_modal_cls_feats)
        if training and v_mask is None:
            cf_loss = None
            if epoch >= causal_start_epoch:
                pool_ori_image_feats = ori_image_feats[:, 0, :] + ori_image_feats[:, 1:, :].mean(dim=1)
                pool_pos_image_feats = pos_image_feats[:, 0, :] + pos_image_feats[:, 1:, :].mean(dim=1)
                pool_neg_image_feats = neg_image_feats[:, 0, :] + neg_image_feats[:, 1:, :].mean(dim=1)
                cf_loss = contrastive_decoupling_loss(pool_ori_image_feats, pool_pos_image_feats, pool_neg_image_feats)
            return logits, cf_loss, v_feat, q_feats, decoder
        elif training and v_mask is not None:
            return logits
        else:
            return logits

    def _do_language_encoder(self, input_ids, embedding_input, ae_data, maml_data,
                             attention_mask=None, visual_token_id=50265, position_strategy='first',
                             token_type_id_for_visual=1):
        B, L = input_ids.size()
        device = input_ids.device
        decoder = None
        encoder = self.auto_encoder.forward_pass(ae_data)
        decoder = self.auto_encoder.reconstruct_pass(encoder)
        ae_v_emb = encoder.view(encoder.shape[0], -1)
        ae_v_emb = self.convert(ae_v_emb)
        maml_v_emb = self.maml_model(maml_data)
        combined_features = torch.cat([maml_v_emb, ae_v_emb], dim=1)
        visual_features = self.pure_vision_MLP(combined_features)
        visual_embedding = self.vision_embedding_MLP(visual_features)

        visual_mask = (input_ids == visual_token_id)
        visual_indices = visual_mask.nonzero(as_tuple=False)
        embedding_input[visual_indices[:, 0], visual_indices[:, 1]] = visual_embedding[visual_indices[:, 0]]

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        if position_strategy == 'first':
            position_ids = torch.arange(1, L + 1, device=device).unsqueeze(0).expand(B, -1)
            for i in range(B):
                visual_pos = (input_ids[i] == visual_token_id).nonzero(as_tuple=True)[0]
                if len(visual_pos) > 0:
                    position_ids[i, visual_pos[0]] = 0
        elif position_strategy == 'last':
            for i in range(B):
                visual_pos = (input_ids[i] == visual_token_id).nonzero(as_tuple=True)[0]
                if len(visual_pos) > 0:
                    position_ids[i, visual_pos[0]] = L + 1
        elif isinstance(position_strategy, int):
            for i in range(B):
                visual_pos = (input_ids[i] == visual_token_id).nonzero(as_tuple=True)[0]
                if len(visual_pos) > 0:
                    position_ids[i, visual_pos[0]] = position_strategy

        position_embed = self.do_position_embeddings(position_ids)

        token_type_ids = torch.zeros_like(input_ids)
        for i in range(B):
            visual_pos = (input_ids[i] == visual_token_id).nonzero(as_tuple=True)[0]
            if len(visual_pos) > 0:
                token_type_ids[i, visual_pos[0]] = token_type_id_for_visual

        token_type_embed = self.do_token_type_embeddings(token_type_ids)

        # === Insert learnable modality embedding ===
        modality_embed = torch.zeros_like(embedding_input)
        for i in range(B):
            visual_pos = (input_ids[i] == visual_token_id).nonzero(as_tuple=True)[0]
            if len(visual_pos) > 0:
                modality_embed[i, visual_pos[0]] = self.visual_modality_embedding.squeeze(0)

        embeddings = embedding_input + position_embed + token_type_embed + modality_embed
        embeddings = self.language_encoder.embeddings.LayerNorm(embeddings)
        embeddings = self.language_encoder.embeddings.dropout(embeddings)

        # Use the updatable do_language_encoder for processing
        encoder_outputs = self.language_encoder.encoder(
            embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = encoder_outputs.last_hidden_state
        return sequence_output, decoder

    def get_trainable_roberta_params(self):
        return [p for p in self.language_encoder.parameters() if p.requires_grad]

    def freeze_roberta_except_last_n_layers(self, n=2):
        for param in self.language_encoder.parameters():
            param.requires_grad = False

        encoder_layers = self.language_encoder.encoder.layer
        total_layers = len(encoder_layers)

        for i in range(total_layers - n, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        return self.get_trainable_roberta_params()
