# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries # 存储最大检测目标数量
        self.transformer = transformer # 存储 Transformer 模块
        self.hidden_dim = hidden_dim = transformer.d_model  # 获取 Transformer 模型的特征维度
        self.num_feature_levels = num_feature_levels # 存储特征层数量
        self.nheads = nheads # 存储多头注意力的头数
        self.max_text_len = 256 # 存储最大文本长度
        self.sub_sentence_present = sub_sentence_present # 存储是否使用子句的标志

        # setting query dim
        self.query_dim = query_dim # 存储查询维度
        assert query_dim == 4# 确保查询维度为 4

        # for dn training
        self.num_patterns = num_patterns # 存储模式数量
        self.dn_number = dn_number  # 存储去噪训练的数量
        self.dn_box_noise_scale = dn_box_noise_scale # 存储边界框噪声缩放比例
        self.dn_label_noise_ratio = dn_label_noise_ratio  # 存储标签噪声比例
        self.dn_labelbook_size = dn_labelbook_size  # 存储标签库大小

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type) # 获取分词器
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)# 获取预训练的 BERT 语言模型
        self.bert.pooler.dense.weight.requires_grad_(False) # 冻结 BERT 池化层的权重和偏置
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert) # 封装 BERT 模型

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True) # 定义一个线性层，将 BERT 输出的特征维度映射到模型的隐藏维度
        nn.init.constant_(self.feat_map.bias.data, 0) # 初始化线性层的偏置为 0
        nn.init.xavier_uniform_(self.feat_map.weight.data)  # 使用 Xavier 均匀初始化线性层的权重
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"]) # 将特殊标记转换为对应的 ID

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels) # 获取骨干网络输出的通道数数量
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_] # 获取骨干网络第 _ 层的输出通道数
                input_proj_list.append( # 定义一个输入投影层，包含 1x1 卷积和组归一化
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append( # 定义额外的输入投影层，包含 3x3 卷积和组归一化
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list) # 将输入投影层列表转换为 ModuleList
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"   # 当特征层数量为 1 时，确保两阶段类型为 "no"
            self.input_proj = nn.ModuleList(# 定义一个输入投影层，包含 1x1 卷积和组归一化
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone  # 存储骨干网络模块
        self.aux_loss = aux_loss  # 存储是否使用辅助损失的标志
        self.box_pred_damping = box_pred_damping = None  # 存储边界框预测阻尼参数

        self.iter_update = iter_update  # 存储是否使用迭代更新的标志
        assert iter_update, "Why not iter_update?" # 确保使用迭代更新

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share # 存储解码器预测边界框嵌入是否共享的标志
        # prepare class & box embed
        _class_embed = ContrastiveEmbed() # 初始化类别嵌入模块

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # 初始化边界框嵌入模块，使用 MLP 网络
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)  # 初始化 MLP 最后一层的权重为 0
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)  # 初始化 MLP 最后一层的偏置为 0

        if dec_pred_bbox_embed_share:  # 如果解码器预测边界框嵌入共享，使用同一个边界框嵌入模块多次
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else: # 如果不共享，复制多个边界框嵌入模块
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]  # 类别嵌入模块使用多次
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)    # 将边界框嵌入模块列表转换为 ModuleList
        self.class_embed = nn.ModuleList(class_embed_layerlist)  # 将类别嵌入模块列表转换为 ModuleList
        self.transformer.decoder.bbox_embed = self.bbox_embed # 将边界框嵌入模块列表赋值给 Transformer 解码器
        self.transformer.decoder.class_embed = self.class_embed  # 将类别嵌入模块列表赋值给 Transformer 解码器

        # two stage
        self.two_stage_type = two_stage_type # 存储两阶段类型
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share: # 如果两阶段边界框嵌入共享，确保解码器预测边界框嵌入也共享
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed# 将同一个边界框嵌入模块赋值给 Transformer 编码器输出
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)  # 如果不共享，复制一个边界框嵌入模块赋值给 Transformer 编码器输出

            if two_stage_class_embed_share:# 如果两阶段类别嵌入共享，确保解码器预测边界框嵌入也共享
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None  # 初始化参考点嵌入为 None

        self._reset_parameters()    # 初始化模型参数

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)  # 使用 Xavier 均匀初始化输入投影层的卷积层权重
            nn.init.constant_(proj[0].bias, 0)  # 初始化输入投影层的卷积层偏置为 0

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  # 将输入转换为 NestedTensor 类型
        self.features, self.poss = self.backbone(samples) # 通过骨干网络提取图像特征和位置编码

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features # 删除存储的图像特征
        if hasattr(self,'poss'):
            del self.poss   # 删除存储的位置编码

    def set_image_features(self, features , poss):  # 直接设置图像特征和位置编码
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):  # 初始化参考点嵌入
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None: # 如果没有提供目标信息，从关键字参数中获取文本描述
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]  # 从目标信息中提取文本描述

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to( # 对文本描述进行分词处理
            samples.device
        )
        # 生成文本自注意力掩码、位置编码和类别到标记的映射掩码
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len: # 如果文本长度超过最大长度，进行截断
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:# 如果使用子句，调整输入到 BERT 模型的参数
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768 # 通过 BERT 模型提取文本嵌入

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model  # 将 BERT 输出的特征维度映射到模型的隐藏维度
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195  # 获取文本标记的掩码
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:# 如果编码后的文本长度超过最大长度，进行截断
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {# 构建文本信息字典
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples) # 将输入转换为 NestedTensor 类型
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples) # 如果没有存储图像特征和位置编码，调用 set_image_tensor 方法进行提取

        srcs = []
        masks = []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose() # 分解特征为图像特征和掩码
            srcs.append(self.input_proj[l](src))# 通过输入投影层处理图像特征
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors) # 对最后一层特征进行输入投影处理
                else:
                    src = self.input_proj[l](srcs[-1]) # 对前一层处理后的特征进行输入投影处理
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0] # 对掩码进行插值处理
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype) # 计算位置编码
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None# 初始化输入查询的边界框、标签、注意力掩码和去噪元数据为 None
        # 将处理后的图像特征、掩码、位置编码和文本信息传入 Transformer 模块进行处理
        # srcs: 经过输入投影层处理后的图像特征列表
        # masks: 图像的掩码列表
        # input_query_bbox: 输入查询的边界框，初始化为 None
        # self.poss: 图像的位置编码列表
        # input_query_label: 输入查询的标签，初始化为 None
        # attn_mask: 注意力掩码，初始化为 None
        # text_dict: 包含编码后的文本、文本标记掩码、位置编码和文本自注意力掩码的字典
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, self.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []# 用于存储每层解码器输出的边界框坐标
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(# 遍历除最后一层外的所有解码器层的参考点、边界框嵌入模块和隐藏状态
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs) # 计算当前层隐藏状态经过边界框嵌入模块后的输出，得到边界框的偏移量
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig) # 将边界框的偏移量与参考点经过逆 sigmoid 函数处理后的值相加，得到未经过 sigmoid 处理的边界框坐标
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()  # 对未经过 sigmoid 处理的边界框坐标应用 sigmoid 函数，将其值映射到 [0, 1] 区间
            outputs_coord_list.append(layer_outputs_unsig)# 将当前层处理后的边界框坐标添加到输出列表中
        outputs_coord_list = torch.stack(outputs_coord_list)# 将存储每层边界框坐标的列表转换为一个张量

        # output
        # 存储每层解码器输出的类别预测结果
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        # 构建最终输出的字典，包含预测的类别 logits 和边界框坐标
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)# 从关键字参数中获取是否删除图像特征和位置编码的标志，默认为 True
        if unset_image_tensor: # 如果需要删除图像特征和位置编码
            self.unset_image_tensor() ## If necessary # 调用 unset_image_tensor 方法删除存储的图像特征和位置编码
        return out # 返回最终的输出字典

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model

