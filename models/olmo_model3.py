# models/olmo_model.py
# replaces the normal OLMo3 decoder layers with SHC-wrapped layers

"""OLMo3 SHC wrappers and decoder-layer replacements."""

import torch 
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Model,
    Olmo3DecoderLayer,
    Olmo3PreTrainedModel,
)

from models.shc import SHC


class _OlmoAttentionBranch(nn.Module):
    """
    Pre-norm attention branch used inside SHC.
    """

    def __init__(self, layer: Olmo3DecoderLayer):
        """Wrap an OLMo3 attention branch for SHC."""
        super().__init__()
        self.self_attn = layer.self_attn
        self.post_norm = layer.post_attention_layernorm

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: torch.Tensor | None = None,
    #     position_ids: torch.LongTensor | None = None,
    #     past_key_values=None,
    #     use_cache: bool | None = False,
    #     position_embeddings=None,
    #     **kwargs,
    # ):
    #     hidden_states = self.norm(hidden_states)
    #     return self.self_attn(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         position_embeddings=position_embeddings,
    #         **kwargs,
    #     )
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        """Run attention branch and apply post-attention norm."""
        attn_out, *rest = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return (self.post_norm(attn_out), *rest)


class _OlmoMLPBranch(nn.Module):
    """
    Pre-norm MLP branch used inside SHC.
    """

    def __init__(self, layer: Olmo3DecoderLayer):
        """Wrap an OLMo3 MLP branch for SHC."""
        super().__init__()
        self.mlp = layer.mlp
        self.post_norm = layer.post_feedforward_layernorm

    # def forward(self, hidden_states: torch.Tensor, **kwargs):
    #     hidden_states = self.norm(hidden_states)
    #     return self.mlp(hidden_states)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Run MLP branch and apply post-FFN norm."""
        return self.post_norm(self.mlp(hidden_states))


class SHCOlmoDecoderLayer(nn.Module):
    """
    Full OLMo decoder-layer replacement. It replaces the whole decoder layer
    so there is no double residual connection.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        base_layer: Olmo3DecoderLayer,
        hidden_size: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        train_branch: bool = False,
        dropout_res: float = 0.1,
        noise_std: float = 1e-2,
        ablate_mapping = None,
    ):
        """Create an SHC-wrapped OLMo3 decoder layer replacement."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams

        self.attn_hc = SHC(
            branch=_OlmoAttentionBranch(base_layer),
            hidden_size=hidden_size,
            num_streams=num_streams,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            train_branch=train_branch,
            dropout_res=dropout_res,
            noise_std=noise_std,
            ablate_mapping = ablate_mapping
        )
        self.mlp_hc = SHC(
            branch=_OlmoMLPBranch(base_layer),
            hidden_size=hidden_size,
            num_streams=num_streams,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            train_branch=train_branch,
            dropout_res=dropout_res,
            noise_std=noise_std,
            ablate_mapping=ablate_mapping
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention and MLP SHC branches for a decoder layer."""
        hidden_states = self.attn_hc(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(hidden_states, tuple):
            # shc can return (streams, aux); we only propagate streams through depth
            hidden_states = hidden_states[0]

        hidden_states = self.mlp_hc(hidden_states)
        if isinstance(hidden_states, tuple):
            # keep only the updated streams
            hidden_states = hidden_states[0]

        return hidden_states


class SHCOlmoModel(Olmo3Model):
    """
    OLMo model wrapper that carries SHC streams across depth and only reads out
    once at the end, before the final norm and LM head.
    """
    def __init__(
        self, base_model, 
        hidden_size, 
        num_streams=4, 
        sinkhorn_iters=20, 
        eps=1e-6, 
        train_branch=False, 
        ablate_mapping=None,
        dropout_res=0.1,
        noise_std=1e-2,
        softmax_readout=False,
    ):
        """Wrap an OLMo3 model to propagate SHC streams across depth."""
        super().__init__(base_model.config)

        self.num_streams = num_streams
        self.softmax_readout = softmax_readout
        if self.softmax_readout:        # newly added softmax readout
            self.readout_logits = nn.Parameter(torch.zeros(num_streams))
            nn.init.normal_(self.readout_logits, mean = 0.0, std = 1e-3)

        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb

        self.layers = nn.ModuleList(
            [
                SHCOlmoDecoderLayer(
                    config=base_model.config,
                    layer_idx=i,
                    base_layer=layer,
                    hidden_size=hidden_size,
                    num_streams=num_streams,
                    sinkhorn_iters=sinkhorn_iters,
                    eps=eps,
                    train_branch=train_branch,
                    dropout_res=dropout_res,
                    noise_std=noise_std,
                    ablate_mapping=ablate_mapping
                )
                for i, layer in enumerate(base_model.layers)
            ]
        )

    def _init_streams(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize stream tensor from token embeddings."""
        # [b, t, h] -> [b, t, s, h]
        return x.unsqueeze(2).repeat(1, 1, self.num_streams, 1)

    # @staticmethod
    # def _readout(streams: torch.Tensor) -> torch.Tensor:
    #     # simple readout: average over streams dimension
    #     return streams.mean(dim=2)
    def _readout(self, streams: torch.Tensor) -> torch.Tensor:
        """ reads out the stream dimension """
        if not self.softmax_readout:
            return streams.mean(dim = 2)

        weights = torch.softmax(self.readout_logits, dim = 0)
        return torch.einsum("btsh,s->bth", streams, weights)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Run the wrapped OLMo3 model forward pass with SHC streams."""
        if (input_ids is None) == (inputs_embeds is None):
            # hf models expect exactly one input representation
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(self.layers[0].attn_hc.branch.self_attn.q_proj.weight.dtype)

        if use_cache and past_key_values is None:
            # hf uses a cache object to store kv states across decoding steps
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # offset positions when continuing generation with cached kv
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = self._init_streams(inputs_embeds)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            
        hidden_states = self._readout(hidden_states)
        hidden_states = self.norm(hidden_states)
            
        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        # position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        # # replaced below
        # # hidden_states = self._init_streams(inputs_embeds)
        # hidden_states = inputs_embeds

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        #     hidden_states = decoder_layer(
        #         hidden_states,
        #         # pass the full causal mask (includes optional padding mask)
        #         attention_mask=causal_mask,
        #         position_embeddings=position_embeddings,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         use_cache=use_cache,
        #         **kwargs,
        #     )

        # deleted below
        # hidden_states = self._readout(hidden_states)
        #hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def olmo_shc(
    olmo: nn.Module,
    num_streams: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    train_branch: bool = False,
    dropout_res: float = 0.1,
    noise_std: float = 1e-2,
    ablate_mapping = None,
    softmax_readout: bool = False,
):
    """
    Replaces olmo.model with an SHC-based model that does not double-apply
    the original OLMo residual connections.
    """
    hidden_size = olmo.config.hidden_size
    olmo.model = SHCOlmoModel(
        base_model=olmo.model,
        num_streams=num_streams,
        sinkhorn_iters=sinkhorn_iters,
        hidden_size=hidden_size,
        eps=eps,
        train_branch=train_branch,
        dropout_res=dropout_res,
        noise_std=noise_std,
        ablate_mapping=ablate_mapping,
        softmax_readout=softmax_readout,
    )
    return olmo