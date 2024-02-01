from typing import Dict, List, TypedDict
import numpy as np
import torch
import math

from ..Misc import Logger as log

from .BaseProc import CrossAttnProcessorBase
from .BaseProc import BundleType
from ..Misc.BBox import BoundingBox


class InjecterProcessor(CrossAttnProcessorBase):
    def __init__(
        self,
        bundle: BundleType,
        bbox_per_frame: List[BoundingBox],
        name: str,
        strengthen_scale: float = 0.0,
        weaken_scale: float = 1.0,
        is_text2vidzero: bool = False,
    ):
        super().__init__(bundle, is_text2vidzero=is_text2vidzero)
        self.strengthen_scale = strengthen_scale
        self.weaken_scale = weaken_scale
        self.bundle = bundle
        self.num_frames = len(bbox_per_frame)
        self.bbox_per_frame = bbox_per_frame
        self.use_weaken = True
        self.name = name

    def dd_core(self, attention_probs: torch.Tensor):
        """ """

        frame_size = attention_probs.shape[0] // self.num_frames
        num_affected_frames = self.num_frames
        attention_probs_copied = attention_probs.detach().clone()

        token_inds = self.bundle.get("token_inds")
        trailing_length = self.bundle.get("trailing_length")
        trailing_inds = list(
            range(self.len_prompt + 1, self.len_prompt + trailing_length + 1)
        )
        # NOTE: Spatial cross attention editing
        if len(attention_probs.size()) == 4:
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            strengthen_map = self.localized_weight_map(
                attention_probs_copied,
                token_inds=all_tokens_inds,
                bbox_per_frame=self.bbox_per_frame,
            )

            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale

            # weakening
            attention_probs_copied[..., all_tokens_inds] *= weaken_map[
                ..., all_tokens_inds
            ]
            # strengthen
            attention_probs_copied[..., all_tokens_inds] += (
                self.strengthen_scale * strengthen_map[..., all_tokens_inds]
            )
        # NOTE: Temporal cross attention editing
        elif len(attention_probs.size()) == 5:
            strengthen_map = self.localized_temporal_weight_map(
                attention_probs_copied,
                bbox_per_frame=self.bbox_per_frame,
            )
            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale
            # weakening
            attention_probs_copied *= weaken_map
            # strengthen
            attention_probs_copied += self.strengthen_scale * strengthen_map

        return attention_probs_copied
