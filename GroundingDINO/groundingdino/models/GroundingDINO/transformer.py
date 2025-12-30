# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from groundingdino.util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict["encoded_text"] = memory_text
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        if self.two_stage_type == "standard":
            #全部候选框（处理0.01-0.09）
            output_memory, output_proposals = gen_encoder_output_proposals( # 生成编码器输出的候选框提议和处理后的特征
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory)) # 对输出特征进行归一化和进一步处理

            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict) # 根据文本信息和输出特征得到未选择的类别预测结果
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory) # 仅根据输出特征得到未选择的类别预测结果

            topk_logits = enc_outputs_class_unselected.max(-1)[0] # 获取每个候选框的最大类别得分
            #预测框+候选框 ？？
            enc_outputs_coord_unselected = ( # 计算未选择的边界框坐标预测结果（加上候选框提议）
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq  # 选择得分最高的 topk 个候选框的索引

            # gather boxes
            #预测框+候选框 的前topk个
            refpoint_embed_undetach = torch.gather(# 收集 topk 个候选框的边界框坐标嵌入（未经过 sigmoid 激活）
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoid
            # 预测框+候选框 的前topk个，分离梯度
            refpoint_embed_ = refpoint_embed_undetach.detach()  # 分离梯度，得到参考点嵌入
            # 候选框 的前topk个
            init_box_proposal = torch.gather( # 收集 topk 个候选框的提议，并经过 sigmoid 激活得到初始框提议
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(  # 收集 topk 个候选框的特征表示
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt: # 如果启用嵌入初始目标，生成初始目标嵌入
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()  # 否则，使用收集到的特征表示作为目标嵌入

            #refpoint_embed--预测框+候选框 的前topk个  tgt--预测框+候选框 的前topk个的特征
            if refpoint_embed is not None: # 如果存在之前的参考点嵌入，将其与新的参考点嵌入拼接
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)  # 将之前的目标嵌入与新的目标嵌入拼接
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_ # 否则，直接使用新的参考点嵌入和目标嵌入

        elif self.two_stage_type == "no":
            tgt_ = ( # 生成初始目标嵌入
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = ( # 生成参考点嵌入
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:  # 如果存在之前的参考点嵌入，将其与新的参考点嵌入拼接
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:# 否则，直接使用新的参考点嵌入和目标嵌入
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0: # 如果存在模式数量，重复目标嵌入和参考点嵌入
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(# 生成模式嵌入并加到目标嵌入上
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()  # 将参考点嵌入经过 sigmoid 激活得到初始框提议

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        # 调用解码器进行解码操作
        # tgt 是目标序列，这里将其维度从 (nq, bs, d_model) 转置为 (bs, nq, d_model)
        # memory 是编码器的输出，同样将其维度从 (hw, bs, d_model) 转置为 (bs, hw, d_model)
        # memory_key_padding_mask 是编码器输出的填充掩码，用于标记哪些位置是填充的
        # pos 是位置编码，将其维度从 (hw, bs, d_model) 转置为 (bs, hw, d_model)
        # refpoints_unsigmoid 是未经过 sigmoid 激活的参考点，将其维度从 (nq, bs, query_dim) 转置为 (bs, nq, query_dim)
        # level_start_index 表示每个特征层的起始索引
        # spatial_shapes 表示每个特征层的空间形状
        # valid_ratios 表示每个特征层的有效区域比例
        # tgt_mask 是目标序列的注意力掩码
        # memory_text 是文本特征
        # text_attention_mask 是文本的注意力掩码，这里取反是因为在代码中 False 表示使用该标记，True 表示填充该标记
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),#预测框+候选框 的前topk个的特征
            memory=memory.transpose(0, 1),#图像特征
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),#refpoint_embed--预测框+候选框 的前topk个
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        # hs 是解码器各层的输出，形状为 (n_dec, bs, nq, d_model)，其中 n_dec 是解码器层数，bs 是批次大小，nq 是查询数量，d_model 是特征维度
        # references 是各层解码器的参考点，形状为 (n_dec + 1, bs, nq, query_dim)，这里 n_dec + 1 是因为包含了初始参考点
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            # 将未分离梯度的目标序列添加一个维度，作为编码器的输出
            # 形状变为 (1, bs, nq, d_model)
            hs_enc = tgt_undetach.unsqueeze(0)
            # 将未分离梯度的参考点经过 sigmoid 激活后添加一个维度，作为编码器的参考点
            # 形状变为 (1, bs, nq, query_dim)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None # 如果两阶段类型不是 "standard"，则编码器的输出和参考点都设为 None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        """
            初始化 Transformer 解码器。

            参数:
                decoder_layer (nn.Module): 单个解码器层的实例，用于构建多层解码器。
                num_layers (int): 解码器的层数。
                norm (nn.Module, 可选): 归一化层，用于对解码器的输出进行归一化处理。
                return_intermediate (bool, 可选): 是否返回每一层的中间输出，这里强制要求为 True。
                d_model (int, 可选): 模型的特征维度，默认为 256。
                query_dim (int, 可选): 查询向量的维度，只能为 2 或 4。
                num_feature_levels (int, 可选): 特征层的数量，默认为 1。
        """
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers) # 使用 _get_clones 函数复制 decoder_layer 多次，构建多层解码器
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)# 初始化一个多层感知机（MLP），用于处理参考点的特征
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        """
           前向传播函数，定义解码器的计算流程。

           参数:
               tgt (Tensor): 目标序列，形状为 (nq, bs, d_model)，其中 nq 是查询的数量，bs 是批次大小，d_model 是特征维度。
               memory (Tensor): 编码器的输出，形状为 (hw, bs, d_model)。
               tgt_mask (Tensor, 可选): 目标序列的注意力掩码。
               memory_mask (Tensor, 可选): 编码器输出的注意力掩码。
               tgt_key_padding_mask (Tensor, 可选): 目标序列的填充掩码。
               memory_key_padding_mask (Tensor, 可选): 编码器输出的填充掩码。
               pos (Tensor, 可选): 位置编码，形状为 (hw, bs, d_model)。
               refpoints_unsigmoid (Tensor, 可选): 未经过 sigmoid 激活的参考点，形状为 (nq, bs, 2/4)。
               level_start_index (Tensor, 可选): 每个特征层的起始索引。
               spatial_shapes (Tensor, 可选): 每个特征层的空间形状。
               valid_ratios (Tensor, 可选): 有效区域的比例。
               memory_text (Tensor, 可选): 文本特征。
               text_attention_mask (Tensor, 可选): 文本的注意力掩码。

           返回:
               list: 包含每一层的中间输出和参考点更新结果的列表。
        """
        output = tgt  # 将 tgt 赋值给 output，作为初始输出

        intermediate = [] # 创建列表用于存储每一层的中间输出
        reference_points = refpoints_unsigmoid.sigmoid()  # 对未经过 sigmoid 激活的参考点应用 sigmoid 函数，得到参考点
        ref_points = [reference_points] # 创建列表用于存储参考点的更新结果

        for layer_id, layer in enumerate(self.layers):  # 遍历每一层解码器

            if reference_points.shape[-1] == 4: # 根据参考点的维度进行不同的处理
                reference_points_input = ( # 若参考点维度为 4，将参考点与有效区域比例组合
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2 # 确保参考点维度为 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :] # 若参考点维度为 2，将参考点与有效区域比例组合
            query_sine_embed = gen_sineembed_for_position( # 生成查询的正弦嵌入
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            # 条件查询生成
            # 通过 ref_point_head 对查询的正弦嵌入进行处理，得到原始的查询位置
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1 # 如果 query_scale 不为 None，则对 output 应用 query_scale 得到位置缩放因子
            query_pos = pos_scale * raw_query_pos# 将位置缩放因子与原始查询位置相乘，得到最终的查询位置
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(# 主要处理流程
                tgt=output,#topk 特征
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,#经过处理的topk的预测+候选框
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            # 检查输出中是否存在 NaN 或 Inf 值
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            # 迭代更新参考点
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)  # 对参考点应用 inverse_sigmoid 函数，得到 sigmoid 之前的参考点
                ##预测框的修正量
                delta_unsig = self.bbox_embed[layer_id](output)     # 通过 bbox_embed 对输出进行处理，得到偏移量
                outputs_unsig = delta_unsig + reference_before_sigmoid  # 将偏移量与 sigmoid 之前的参考点相加，得到更新后的未经过 sigmoid 激活的参考点
                new_reference_points = outputs_unsig.sigmoid() # 对更新后的未经过 sigmoid 激活的参考点应用 sigmoid 函数，得到新的参考点

                reference_points = new_reference_points.detach()  # 将新的参考点从计算图中分离出来，避免梯度传播
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points) # 将新的参考点添加到参考点更新结果列表中

            intermediate.append(self.norm(output))  # 对输出应用归一化层，并将结果添加到中间输出列表中

        return [  # 对中间输出和参考点更新结果进行维度转置，并返回
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,  # 模型的特征维度
        d_ffn=1024,  # 前馈神经网络的隐藏层维度
        dropout=0.1,  # 丢弃率，用于防止过拟合
        activation="relu",  # 激活函数类型
        n_levels=4,  # 特征层的数量
        n_heads=8,  # 多头注意力机制中的头数
        n_points=4,  # 可变形注意力机制中每个特征层采样点的数量
        use_text_feat_guide=False,  # 是否使用文本特征引导
        use_text_cross_attention=False,  # 是否使用文本交叉注意力
    ):
        super().__init__()

        # cross attention
        # 交叉注意力模块
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,  # 嵌入维度
            num_levels=n_levels,  # 特征层数量
            num_heads=n_heads,  # 头数
            num_points=n_points,  # 采样点数量
            batch_first=True,  # 批次维度在前
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 丢弃层，若丢弃率为 0 则使用恒等映射
        self.norm1 = nn.LayerNorm(d_model) # 层归一化层

        # cross attention text
        if use_text_cross_attention: # 文本交叉注意力模块
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        # 前馈神经网络（FFN）
        # 第一个线性层，将输入维度从 d_model 映射到 d_ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide  # 目前不支持使用文本特征引导
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):#移除自注意力模块。
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        """
                将位置编码添加到输入张量中。

                参数:
                    tensor (Tensor): 输入张量。
                    pos (Tensor, 可选): 位置编码。

                返回:
                    Tensor: 添加位置编码后的张量。
                """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt)))) # 经过第一个线性层、激活函数、丢弃层和第二个线性层
        tgt = tgt + self.dropout4(tgt2)   # 残差连接
        tgt = self.norm3(tgt)   # 层归一化
        return tgt

    def forward(
        self,
        # for tgt
        # 目标相关输入
        tgt: Optional[Tensor],  # 目标序列，形状为 (nq, bs, d_model)
        tgt_query_pos: Optional[Tensor] = None,  # 查询的位置编码，经过 MLP(Sine(pos)) 处理
        tgt_query_sine_embed: Optional[Tensor] = None,  # 查询的正弦位置编码，即 Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None, # 目标序列的填充掩码
        tgt_reference_points: Optional[Tensor] = None,   # 目标的参考点，形状为 (nq, bs, 4)
        memory_text: Optional[Tensor] = None, # 文本特征，形状为 (bs, num_token, d_model)
        text_attention_mask: Optional[Tensor] = None, # 文本的注意力掩码，形状为 (bs, num_token)
        # for memory  # 编码器输出相关输入
        memory: Optional[Tensor] = None, # 编码器的输出，形状为 (hw, bs, d_model)
        memory_key_padding_mask: Optional[Tensor] = None, # 编码器输出的填充掩码
        memory_level_start_index: Optional[Tensor] = None,  # num_levels # 每个特征层的起始索引
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2  # 每个特征层的空间形状，形状为 (bs, num_levels, 2)
        memory_pos: Optional[Tensor] = None,  # pos for memory  # 编码器输出的位置编码
        # sa    # 自注意力和交叉注意力掩码
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention  # 自注意力掩码
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention # 交叉注意力掩码
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        # 自注意力模块
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)  # 将位置编码添加到目标序列中
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0] # 进行自注意力计算
            tgt = tgt + self.dropout2(tgt2) # 残差连接
            tgt = self.norm2(tgt)  # 层归一化

        # 文本交叉注意力模块
        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        # 可变形交叉注意力模块
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),# 转置查询的维度
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(), # 转置参考点的维度
            value=memory.transpose(0, 1),# 转置编码器输出的维度
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        # 前馈神经网络
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )
