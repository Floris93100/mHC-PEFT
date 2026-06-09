# models/dynamic_mHC.py
""" dynamic manifold-constrained hyper-connections """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """ root-mean-square normalization layer """
    def __init__(self, dim, eps=1e-6):
        """ initialize RMSNorm parameters """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """ apply RMSNorm to input tensor """
        # rms = root mean square = sqrt( mean(x^2)+ eps )
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # normalize
        return (x / rms) * self.weight


def sinkhorn_logspace(logits, num_iters=20, eps=1e-6):
    """ compute a doubly-stochastic matrix via Sinkhorn in log-space """
    # logits: (B,T,n,n)
    z = logits.float()
    # numerical stabilization, prevent overflow in exp
    z = z - z.amax(dim=(-2, -1), keepdim=True)

    # normalize by iteration in log space
    for _ in range(num_iters):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)  # row normalize in log-space
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)  # col normalize in log-space

    # exponaniate back to probability space
    p = z.exp()

    # final clean normalize 
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / p.sum(dim=-2, keepdim=True).clamp_min(eps)
    return p.to(logits.dtype)


class MHC(nn.Module):
    """
    mHC wrapper based on the DeepSeek manifold Hyperconnections paper

    Maintains n parallel streams of shape (B, T, n, D). On each forward pass:

    1. Flatten & RMSNorm the stream tensor → (B, T, n*D)
    2. Project into three learned mappings (all scaled by a learned temperature α):
          H_pre  = σ(α_pre  · x̃ φ_pre  + b)   ∈ (0,1)   — input gate per stream
          H_post = 2σ(α_post · x̃ φ_post + b)  ∈ (0,2)   — output scale per stream
          H_res  = Sinkhorn(α_res · x̃ φ_res + b)         — doubly-stochastic (n×n) stream routing matrix
    3. Gate & collapse streams into sub-layer:  branch_in  = Σ_i H_pre[i] · X[i]
    4. Run wrapped sub-layer:                   branch_out = branch(branch_in)
    5. Update streams:
          X_new = H_res @ X          # route residuals across streams
                + H_post · branch_out # inject scaled sub-layer output

    6. Return stream 0 (or mean/sum? of streams) as y, shape (B, T, D).

    The Sinkhorn operator iteratively normalises rows then columns (t_max=20 iters).
    Output type signature matches the wrapped sub-layer for drop-in compatibility.
    
    """
    def __init__(
        self,
        branch: nn.Module,
        hidden_size: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        init_std: float = 1e-3,
        train_branch: bool = False,
    ):
        """ initialize the MHC wrapper and projection parameters """
        super().__init__()
        self.branch = branch 
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        # small init std to break symmetry
        self.init_std = init_std

        flat_dim = num_streams * hidden_size

        self.norm = RMSNorm(flat_dim, eps = eps)

        # projection matrix for pre, post and res
        self.pre_proj = nn.Linear(flat_dim, num_streams)
        self.post_proj = nn.Linear(flat_dim, num_streams)
        self.res_proj = nn.Linear(flat_dim, num_streams * num_streams)

        # alpha for pre, post and res
        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        if not train_branch:
            for p in self.branch.parameters():
                p.requires_grad = False
        
                                        # for diagnostics() getter function
        self.diagnostics_enabled = False
        self._last_h_pre = None         
        self._last_h_res = None         
        self._last_h_post = None        

        self.reset_parameters()

    def reset_parameters(self):
        """ initialize projection weights and gating parameters """
        # tiny noise to break symmetry on weights
        nn.init.normal_(self.pre_proj.weight, mean = 0.0, std = self.init_std)
        nn.init.normal_(self.post_proj.weight, mean = 0.0, std = self.init_std)
        nn.init.normal_(self.res_proj.weight, mean = 0.0, std = self.init_std)

        # initialise H_pre so that H_pre[i] is 1/n
        # p = 1/n
        pre_target = 1.0 / self.num_streams
        # bias = log(p/(1-p)) , so sigmoid(bias) = 1/n
        pre_bias = math.log(pre_target / (1.0 - pre_target))
        nn.init.constant_(self.pre_proj.bias, pre_bias)

        # H_post[i] = 2 * sigmoid(0) = 1 so that branch output is initially unscaled
        nn.init.zeros_(self.post_proj.bias)

        # zero bias for res_proj so doubly stochastic matrix starts approximately uniform
        nn.init.zeros_(self.res_proj.bias)

        # alpha starts at 1 so that initial H_pre, H_post, H_res are near uniform
        nn.init.ones_(self.alpha_pre)
        nn.init.ones_(self.alpha_post)
        nn.init.ones_(self.alpha_res)

        # RMSNorm scale starts neutral
        nn.init.ones_(self.norm.weight)

    # initialize streams with copies of the input
    def _init_streams(self, x):
        """ initialize stream tensor by copying inputs across streams """
        X = x.unsqueeze(2).repeat(1, 1, self.num_streams, 1)
        return X

    @staticmethod
    def _extract_hidden(out):
        """ extract hidden states from a model output tuple """
        return out[0] if isinstance(out, tuple) else out

    def enable_diagnostics(self, enabled = True):
        """ enable or disable caching of routing objects for diagnostics """
        self.diagnostics_enabled = enabled
        if not enabled:                 
            self._last_h_pre = None
            self._last_h_res = None
            self._last_h_post = None


    def diagnostics(self):
        """ return latest dynamic routing objects for generic diagnostics """
        if (
            self._last_h_pre is None
            or self._last_h_res is None
            or self._last_h_post is None
        ):
            raise RuntimeError(
                "no cached routing state found; run a forward pass before diagnostics()"
            )

        return {
            "num_streams": self.num_streams,
            "h_pre": self._last_h_pre,
            "h_res": self._last_h_res,
            "h_post": self._last_h_post,
        }

    def forward(self, x, *args, **kwargs):
        """ run the wrapped branch and update hyperconnection streams """
        # x: (B,T,D)
        X = self._init_streams(x)                          # (B,T,n,D)

        # flatten streams into feature vector, normalize
        B, T, n, D = X.shape
        flat = X.reshape(B, T, n * D)
        h = self.norm(flat)

        # add bias after multiplication with alpha to follow the paper
        h_pre = torch.sigmoid(
            self.alpha_pre * F.linear(h, self.pre_proj.weight, bias=None) + self.pre_proj.bias
        )
        h_post = 2.0 * torch.sigmoid(
            self.alpha_post * F.linear(h, self.post_proj.weight, bias=None) + self.post_proj.bias
        )

        # calculate res and make doubly stochastic matrix via Sinkhorn
        res_logits = self.alpha_res * F.linear(h, self.res_proj.weight, bias=None) + self.res_proj.bias
        res_logits = res_logits.view(B, T, n, n)
        h_res = sinkhorn_logspace(res_logits, num_iters=self.sinkhorn_iters, eps=self.eps)

        h_pre = h_pre.to(dtype = X.dtype)
        h_post = h_post.to(dtype = X.dtype)
        h_res = h_res.to(dtype = X.dtype)

                                        # for diagnostics() getter function
        if self.diagnostics_enabled:
            self._last_h_pre = h_pre.detach()
            self._last_h_res = h_res.detach()
            self._last_h_post = h_post.detach()

        # wrap the neural sub layer
        branch_in = torch.sum(h_pre.unsqueeze(-1) * X, dim=2)              # (B,T,D)
        branch_in = branch_in.to(dtype = X.dtype)
        branch_out_raw = self.branch(branch_in, *args, **kwargs)
        branch_out = self._extract_hidden(branch_out_raw)

        # branch output into streams
        X_res = torch.einsum("btij,btjd->btid", h_res, X)                   # (B,T,n,D)
        X_post = h_post.unsqueeze(-1) * branch_out.unsqueeze(2)             # (B,T,n,D)

        # hyperconnection update
        X_new = X_res + X_post
        X_new = X_new.to(dtype = X.dtype)

        y = X_new.mean(dim=2)
        
        # safety for huggingface compatibility
        if isinstance(branch_out_raw, tuple):
            return (y,) + branch_out_raw[1:]
        return y