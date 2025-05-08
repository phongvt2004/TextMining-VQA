from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import os
# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_sentencepiece_model_for_beit3(sentencepiece_model):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model)
class Beit3Processing():
    def __init__(self, sentencepiece_model="beit3.spm", num_max_bpe_tokens=64, input_size=480):
        self.tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    def _get_image(self, image):
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, image, question: str, data: dict):
        img = self._get_image(image)
        data["image"] = img.reshape(1, img.size(0), img.size(1), img.size(2))
        tokens = self.tokenizer.tokenize(question)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        data["language_tokens"] = [language_tokens]
        data["padding_mask"] = [padding_mask]

    def __call__(self, image, question: str):
        data = dict()
        self._get_image_text_example(image, question, data)
        batch_tensors = {}
        for tensor_key in data:
            batch_tensors[tensor_key] = torch.tensor(data[tensor_key])
        return batch_tensors

class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self,
            args,
            num_classes,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
            nn.Softmax(dim=1)
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question,
            visual_tokens=image,
            text_padding_position=padding_mask,
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep)

@register_model
def beit3_base_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_768_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=768, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, **kwargs)
    return model
