import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from romatch.utils import get_tuple_transform_ops


@dataclass
class SyntheticWarpConfig:
    max_rotation_deg: float = 12.0
    max_translation: float = 0.08  # normalized
    min_scale: float = 0.9
    max_scale: float = 1.1
    max_shear_deg: float = 8.0
    elastic_alpha: float = 8.0
    elastic_sigma: float = 6.0


class SyntheticWarpPairs(Dataset):
    """
    Build synthetic supervision from already co-registered cross-modal pairs.

    pair_file: text file with one pair per line:
      path/to/im_A path/to/im_B
    paths are relative to data_root unless absolute.
    """

    def __init__(
        self,
        data_root: str,
        pair_file: str,
        ht: int = 384,
        wt: int = 512,
        normalize: bool = True,
        synth_cfg: Optional[SyntheticWarpConfig] = None,
    ) -> None:
        self.data_root = data_root
        self.ht = ht
        self.wt = wt
        self.synth_cfg = synth_cfg or SyntheticWarpConfig()
        self.im_transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=normalize)

        with open(pair_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        self.pairs = [tuple(ln.split()[:2]) for ln in lines]
        if not self.pairs:
            raise ValueError(f"No valid image pairs found in {pair_file}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.data_root, p)

    def _rand_affine(self, device: torch.device) -> torch.Tensor:
        cfg = self.synth_cfg
        angle = (torch.rand(1, device=device) * 2 - 1) * np.deg2rad(cfg.max_rotation_deg)
        scale = cfg.min_scale + torch.rand(1, device=device) * (cfg.max_scale - cfg.min_scale)
        shear = (torch.rand(2, device=device) * 2 - 1) * np.deg2rad(cfg.max_shear_deg)
        tx = (torch.rand(1, device=device) * 2 - 1) * cfg.max_translation
        ty = (torch.rand(1, device=device) * 2 - 1) * cfg.max_translation

        ca, sa = torch.cos(angle), torch.sin(angle)
        shx, shy = torch.tan(shear[0]), torch.tan(shear[1])

        A = torch.tensor(
            [[ca.item(), -sa.item(), 0.0], [sa.item(), ca.item(), 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        Sh = torch.tensor(
            [[1.0, shx.item(), 0.0], [shy.item(), 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        Sc = torch.tensor(
            [[scale.item(), 0.0, 0.0], [0.0, scale.item(), 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        T = torch.tensor(
            [[1.0, 0.0, tx.item()], [0.0, 1.0, ty.item()], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        return T @ Sh @ Sc @ A

    def _elastic_field(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        cfg = self.synth_cfg
        if cfg.elastic_alpha <= 0:
            return torch.zeros(1, h, w, 2, device=device)
        noise = torch.randn(1, 2, h, w, device=device)
        k = max(3, int(cfg.elastic_sigma * 4) | 1)
        smoothed = F.avg_pool2d(noise, kernel_size=k, stride=1, padding=k // 2)
        flow = smoothed * (cfg.elastic_alpha / max(h, w))
        return flow.permute(0, 2, 3, 1)

    def _make_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            indexing="ij",
        )
        base = torch.stack([xs, ys], dim=-1)[None]
        return base

    def __getitem__(self, idx: int):
        im_a_rel, im_b_rel = self.pairs[idx]
        im_a_path = self._resolve(im_a_rel)
        im_b_path = self._resolve(im_b_rel)

        im_a = Image.open(im_a_path).convert("RGB")
        im_b = Image.open(im_b_path).convert("RGB")
        im_a, im_b = self.im_transform_ops((im_a, im_b))

        im_a = im_a[None]
        im_b = im_b[None]
        device = im_a.device
        h, w = im_a.shape[-2:]

        base = self._make_grid(h, w, device)
        A = self._rand_affine(device)
        pts = torch.cat([base, torch.ones(1, h, w, 1, device=device)], dim=-1)
        aff = torch.einsum("ij,bhwj->bhwi", A, pts)[..., :2]
        elastic = self._elastic_field(h, w, device)
        warp = aff + elastic

        valid = ((warp[..., 0] >= -1) & (warp[..., 0] <= 1) & (warp[..., 1] >= -1) & (warp[..., 1] <= 1)).float()
        im_a_syn = F.grid_sample(im_a, warp, mode="bilinear", padding_mode="zeros", align_corners=False)

        return {
            "im_A": im_a_syn[0],
            "im_B": im_b[0],
            "gt_warp": warp[0],
            "gt_prob": valid[0],
            "im_A_path": im_a_path,
            "im_B_path": im_b_path,
        }
