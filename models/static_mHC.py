# models/static_mHC.py
""" static manifold-constrained hyper-connections """

import torch
import torch.nn as nn
import math


def sinkhorn_logspace(logits, num_iters=20, eps=1e-6, tol=1e-4):
    """Compute a doubly-stochastic matrix via Sinkhorn in log-space."""
    # check if input is positive
    #assert (logits >= 0).all(), f"Sinkhorn input has negative values: min={logits.min().item()}"
    # logits: (B,T,n,n)
    z = logits.float()
    # numerical stabilization, prevent overflow in exp
    z = z - z.amax(dim=(-2, -1), keepdim=True)

    # normalize by iteration in log space
    for _ in range(num_iters):
        # set prev z for early termination
        z_prev = z 

        z = z - torch.logsumexp(z, dim=-1, keepdim=True)  # row normalize in log-space
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)  # col normalize in log-space

        # early termination by check max absolute change across the whole batch
        if (z - z_prev).abs().amax() < tol:
            break

    # exponaniate back to probability space
    p = z.exp()

    # final clean normalize 
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / p.sum(dim=-2, keepdim=True).clamp_min(eps)
    return p.to(logits.dtype)

class SHC(nn.Module):
    """
    Static Hyper-Connections wrapper.

    Minimal rewrite of the MHC class:
    - removes input-dependent (dynamic) projections
    - uses direct learnable static parameters for H_pre, H_post, H_res
    - keeps the same wrapper logic and output behavior
    """

    def __init__(
        self,
        branch: nn.Module,
        hidden_size: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        train_branch: bool = False,
        ablate_mapping = None,
        dropout_res: float = 0.1,
        noise_std: float = 1e-2
    ):
        """Initialize static hyperconnection parameters and wrapped branch."""
        super().__init__()
        self.branch = branch
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.ablate_mapping = ablate_mapping or []
        self.dropout_res = dropout_res
        self.noise_std = noise_std

        try:
            branch_device = next(self.branch.parameters()).device
        except StopIteration:
            branch_device = torch.device("cpu")

        # Static learnable parameters
        # pre/post are parameterized in logit space
        # res is parameterized as logits before Sinkhorn
        self.pre_logits = nn.Parameter(torch.empty(num_streams, device=branch_device))
        self.post_logits = nn.Parameter(torch.empty(num_streams, device=branch_device))
        self.res_logits = nn.Parameter(torch.empty(num_streams, num_streams, device=branch_device))

        #self.stream_dropout = nn.Dropout(stream_dropout) if stream_dropout > 0 else nn.Identity()
        
        if not train_branch:
            for p in self.branch.parameters():
                p.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize static gate/residual logits and apply ablations."""
        # pre_logits: target h_pre[s] = 1/n + small noise, renormalize after
        pre_target = 1.0 / self.num_streams
        pre_bias = math.log(pre_target / (1.0 - pre_target))
        nn.init.constant_(self.pre_logits, pre_bias)
        with torch.no_grad():
            self.pre_logits.add_(torch.randn_like(self.pre_logits) * self.noise_std)
            # renormalize to convex combination: sum(sigmoid(pre_logits)) = 1
            h_pre = torch.sigmoid(self.pre_logits)
            h_pre = h_pre / h_pre.sum()
            self.pre_logits.copy_(torch.log(h_pre / (1.0 - h_pre)))

        # post_logits: zero init + small noise
        nn.init.zeros_(self.post_logits)
        with torch.no_grad():
            self.post_logits.add_(torch.randn_like(self.post_logits) * self.noise_std)

        # res_logits: zero init + noise (Sinkhorn will normalize to doubly stochastic)
        nn.init.zeros_(self.res_logits)
        with torch.no_grad():
            self.res_logits.add_(torch.randn_like(self.res_logits) * self.noise_std)
        
        # remove gradients for ablated components
        if "pre" in self.ablate_mapping:
            self.pre_logits.requires_grad = False
        if "post" in self.ablate_mapping:
            self.post_logits.requires_grad = False
        if "res" in self.ablate_mapping:
            self.res_logits.requires_grad = False

    def _init_streams(self, x):
        """Initialize stream tensor by copying inputs across streams."""
        # initialize streams with copies of the input
        return x.unsqueeze(2).repeat(1, 1, self.num_streams, 1)

    @staticmethod
    def _extract_hidden(out):
        """Extract hidden states from a model output tuple."""
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x, *args, readout=False, **kwargs):
        """Run the wrapped branch and update static hyperconnection streams."""
        # accept either a single hidden state tensor or an existing stream tensor
        if x.dim() == 3:
            X = self._init_streams(x)  # [b, t, d] -> [b, t, n, d]
        elif x.dim() == 4:
            X = x  # already [b, t, n, d]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.shape}")

        B, T, n, D = X.shape
        x_device, x_dtype = X.device, X.dtype

        if "pre" in self.ablate_mapping:
            branch_in = X.mean(dim=2)  # simple read-in when pre is ablated
        else:
            # pre gate chooses how to mix streams into the branch input
            h_pre = torch.sigmoid(self.pre_logits).to(device=x_device, dtype=x_dtype).reshape(1, 1, n)
            branch_in = torch.einsum("btn,btnd->btd", h_pre.expand(B, T, n), X)

        # run the wrapped sub-layer on the mixed input
        branch_out_raw = self.branch(branch_in, *args, **kwargs)
        branch_out = self._extract_hidden(branch_out_raw)  # unwrap (hidden, aux...) if needed

        if "res" in self.ablate_mapping:
            X_res = X  # identity routing when res is ablated
        else:
            # res routing is shared across batch/time, then expanded for sinkhorn
            res_logits_eff = self.res_logits.to(device=x_device)
            
            if self.training and self.dropout_res > 0:
                # generate dropout mask
                mask = torch.bernoulli(
                    torch.full_like(res_logits_eff, self.dropout_res)
                ).bool()
                # safety: avoid fully masked row/col
                row_full = mask.all(dim=-1, keepdim=True)
                col_full = mask.all(dim=-2, keepdim=True)
                mask = mask & ~row_full & ~col_full
                # apply -inf mask to logits
                res_logits_eff = res_logits_eff.masked_fill(mask, float('-inf'))
            
            # compute a doubly stochastic routing matrix via sinkhorn
            h_res = sinkhorn_logspace(
                res_logits_eff.view(1, 1, n, n).expand(B, T, n, n),
                num_iters=self.sinkhorn_iters,
                eps=self.eps,
            ).to(x_dtype)
            
            # route residual streams through the learned mapping
            X_res = torch.einsum("btnm,btmd->btnd", h_res, X)

        if "post" in self.ablate_mapping:
            X_post = branch_out.unsqueeze(2).expand(B, T, n, D)  # broadcast branch out to all streams
        else:
            # post gate scales branch output per stream (factor in (0, 2))
            h_post = (2.0 * torch.sigmoid(self.post_logits)).to(device=x_device, dtype=x_dtype).reshape(1, 1, n)
            X_post = torch.einsum("btn,btd->btnd", h_post.expand(B, T, n), branch_out)

        # combine routed residuals and injected branch outputs
        X_new = X_res + X_post
        out = X_new.mean(dim=2) if readout else X_new  # optionally read out to [b, t, d]

        if isinstance(branch_out_raw, tuple):
            return (out,) + branch_out_raw[1:]
        return out