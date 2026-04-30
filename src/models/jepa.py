import torch, torch.nn as nn, timm, copy
import numpy as np


class ZMWM_JEPA(nn.Module):
    """I-JEPA-style joint-embedding predictor.

    Two masking modes (selectable for A/B):
      * mask_mode="block": union of n_blocks rectangular targets (the actual
        I-JEPA recipe). Pairs with mask_ratio ~ 0.4. Forces the predictor to
        extrapolate across contiguous unknown regions, which requires shape
        priors -- this is the regime that produces semantic features.
      * mask_mode="random": per-patch random mask (the original SSLP setup).
        Pairs with mask_ratio ~ 0.75. Solvable by local interpolation, which
        is why the v1 backbone learned texture-only features.

    The context encoder runs only on visible patches; the target encoder
    runs on the full image. This blocks the leakage path where target
    information bleeds into context tokens through encoder self-attention
    (the v1 code ran the context encoder on the full image, then overwrote
    masked positions afterward -- the encoder had already mixed targets in).
    """

    def __init__(self, mask_ratio=0.4, ema=0.996, pred_depth=4,
                 mask_mode="block", n_blocks=4, ar_range=(0.75, 1.5)):
        super().__init__()
        self.context_enc = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="")
        self.target_enc = copy.deepcopy(self.context_enc)
        for p in self.target_enc.parameters(): p.requires_grad = False
        D = self.context_enc.embed_dim
        self.N = self.context_enc.patch_embed.num_patches
        self.side = int(self.N ** 0.5)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pred_pos_embed = nn.Parameter(torch.zeros(1, self.N, D))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pred_pos_embed, std=0.02)
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=D*2,
                                        batch_first=True, norm_first=True),
            num_layers=pred_depth)
        self.pred_norm = nn.LayerNorm(D)
        self.mask_ratio = mask_ratio
        self.ema = ema
        self.mask_mode = mask_mode
        self.n_blocks = n_blocks
        self.ar_range = ar_range

    @torch.no_grad()
    def update_target(self):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(self.ema).add_(pc.data, alpha=1-self.ema)

    def _random_mask(self, B, n_mask, device):
        idx = torch.rand(B, self.N, device=device).argsort(dim=1)
        return idx[:, :n_mask]

    def _block_mask(self, B, n_mask, device):
        """Sample union of n_blocks rectangles, then constrain to exactly n_mask patches."""
        side = self.side
        per_lo = self.mask_ratio / self.n_blocks * 0.8
        per_hi = self.mask_ratio / self.n_blocks * 1.8
        out = np.empty((B, n_mask), dtype=np.int64)
        for b in range(B):
            m = np.zeros(self.N, dtype=bool)
            for _ in range(self.n_blocks):
                ar = np.random.uniform(*self.ar_range)
                area = self.N * np.random.uniform(per_lo, per_hi)
                bh = max(2, min(side, int(round((area * ar) ** 0.5))))
                bw = max(2, min(side, int(round((area / ar) ** 0.5))))
                t = np.random.randint(0, side - bh + 1)
                l = np.random.randint(0, side - bw + 1)
                rows = np.arange(t, t + bh)[:, None]
                cols = np.arange(l, l + bw)[None, :]
                m[(rows * side + cols).ravel()] = True
            idx = np.flatnonzero(m)
            if idx.size >= n_mask:
                idx = np.sort(np.random.choice(idx, n_mask, replace=False))
            else:
                extras = np.random.choice(np.flatnonzero(~m), n_mask - idx.size, replace=False)
                idx = np.sort(np.concatenate([idx, extras]))
            out[b] = idx
        return torch.from_numpy(out).to(device, non_blocking=True)

    def _build_mask(self, B, device):
        n_mask = int(self.N * self.mask_ratio)
        if self.mask_mode == "block":
            mask_idx = self._block_mask(B, n_mask, device)
        else:
            mask_idx = self._random_mask(B, n_mask, device)
        is_masked = torch.zeros(B, self.N, dtype=torch.bool, device=device)
        is_masked.scatter_(1, mask_idx, True)
        visible_idx = is_masked.long().argsort(dim=1, stable=True)[:, :self.N - n_mask]
        return mask_idx, visible_idx

    def _encode_visible(self, enc, x, visible_idx):
        """Context-encoder forward on visible patches only. [B, M, D] out."""
        tokens = enc.patch_embed(x) + enc.pos_embed[:, 1:, :]
        B, _, D = tokens.shape
        vis = tokens.gather(1, visible_idx.unsqueeze(-1).expand(-1, -1, D))
        cls = enc.cls_token.expand(B, -1, -1) + enc.pos_embed[:, :1, :]
        seq = torch.cat([cls, vis], dim=1)
        seq = enc.blocks(enc.pos_drop(seq))
        return enc.norm(seq)[:, 1:]

    def _encode_full(self, enc, x):
        """Target-encoder forward on full image. [B, N, D] out."""
        tokens = enc.patch_embed(x)
        cls = enc.cls_token.expand(tokens.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1) + enc.pos_embed
        seq = enc.blocks(enc.pos_drop(seq))
        return enc.norm(seq)[:, 1:]

    def forward(self, ctx_img, tgt_img):
        B = ctx_img.size(0)
        mask_idx, visible_idx = self._build_mask(B, ctx_img.device)

        ctx_vis = self._encode_visible(self.context_enc, ctx_img, visible_idx)
        with torch.no_grad():
            tgt_full = self._encode_full(self.target_enc, tgt_img)

        D = ctx_vis.size(-1)
        pred_input = self.mask_token.expand(B, self.N, D).contiguous()
        pred_input.scatter_(1, visible_idx.unsqueeze(-1).expand(-1, -1, D), ctx_vis)
        pred_input = pred_input + self.pred_pos_embed
        pred = self.pred_norm(self.predictor(pred_input))

        g = lambda t: t.gather(1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
        return g(pred), g(tgt_full)
