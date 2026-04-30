import torch, torch.nn as nn, timm, copy

class ZMWM_JEPA(nn.Module):
    def __init__(self, mask_ratio=0.75, ema=0.996, pred_depth=4):
        super().__init__()
        self.context_enc = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="")
        self.target_enc = copy.deepcopy(self.context_enc)
        for p in self.target_enc.parameters(): p.requires_grad = False
        D = self.context_enc.embed_dim
        self.N = self.context_enc.patch_embed.num_patches
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

    @torch.no_grad()
    def update_target(self):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(self.ema).add_(pc.data, alpha=1-self.ema)

    def _tokens(self, enc, x):
        x = enc.patch_embed(x)
        cls = enc.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + enc.pos_embed
        x = enc.blocks(enc.pos_drop(x))
        return enc.norm(x)[:, 1:]

    def forward(self, ctx_img, tgt_img):
        B = ctx_img.size(0)
        n_mask = int(self.N * self.mask_ratio)
        idx = torch.rand(B, self.N, device=ctx_img.device).argsort(dim=1)
        mask_idx = idx[:, :n_mask]

        ctx_tok = self._tokens(self.context_enc, ctx_img)
        with torch.no_grad():
            tgt_tok = self._tokens(self.target_enc, tgt_img)

        D = ctx_tok.size(-1)
        pred_input = ctx_tok.clone()
        pred_input.scatter_(1,
            mask_idx.unsqueeze(-1).expand(-1, -1, D),
            self.mask_token.expand(B, n_mask, -1))
        pred_input = pred_input + self.pred_pos_embed
        pred = self.pred_norm(self.predictor(pred_input))

        g = lambda t: t.gather(1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
        return g(pred), g(tgt_tok)
