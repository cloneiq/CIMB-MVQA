import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_encoders.clip_model import build_model, adapt_position_encoding
from models.language_encoders.bert_model import BertConfig, BertCrossLayer
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
import numpy as np


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


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
    
class M3AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        label_num = config.get('num_hid', False)
        self.config = config
        self.vision_encoder = build_model(name=config.get('visual_backbone', 'ViT-B/16'),
                                          resolution_after=config.get('image_size', 384))
        roberta_config = RobertaConfig.from_pretrained('roberta-base')
        self.language_encoder = RobertaModel.from_pretrained('roberta-base', config=roberta_config)

        self.multi_modal_language_proj = nn.Linear(config.get('input_text_embed_size', 768),
                                                   config.get('hidden_size', 768))
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config.get('input_image_embed_size', 768),
                                                 config.get('hidden_size', 768))
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(4, config.get("hidden_size", 768))
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(roberta_config) for _ in range(config.get('num_top_layer', 6))])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(config.get("hidden_size", 768))
        self.multi_modal_language_pooler.apply(init_weights)

        self.vqa_head = nn.Sequential(
            nn.Linear(config.get("hidden_size", 768) * 2, config.get("hidden_size", 768) * 4),
            nn.LayerNorm(config.get("hidden_size", 768) * 4),
            nn.GELU(),
            nn.Linear(config.get("hidden_size", 768) * 4, label_num),
        )
        self.vqa_head.apply(init_weights)

        ckpt = torch.load(config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        state_dict = adapt_position_encoding(state_dict,
                                             after=config.get('image_size', 384),
                                             patch_size=config.get('patch_size', 16))
        self.load_state_dict(state_dict, strict=False)

    def forward(self, images, questions_ids, attention_mask, image_token_type_idx=1):
         #  == Begin: Ori Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=questions_ids)
        text_input_shape = attention_mask.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(attention_mask, text_input_shape,
                                                                                questions_ids.device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == Begin: do Text Encoding ==

        
        # == Begin: Ori Image Encoding ==
        
        uni_modal_image_feats = self.vision_encoder(images)
        
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=images.device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 images.device)
        # == End: Ori Image Encoding ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(attention_mask)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                                 self.multi_modal_vision_layers)):
                x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
                y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
                x, y = x1[0], y1[0]
            

        # == End: do Co-Attention ==
           # == Begin: do Pooling ==
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        
        multi_modal_cls_feats = torch.cat(
            [multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)

        logits = self.vqa_head(multi_modal_cls_feats)
            
        return logits

    

    

