# callbacks.py
""" custom Trainer callbacks for logging in training"""

import torch
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


class WeightGradStatsCallback(TrainerCallback):
    """
    Logs per-layer weight and gradient statistics to TensorBoard at every logging step.

    Tracked per trainable parameter:
        weights/   {name}/mean|std|min|max   
        gradients/ {name}/mean|std|min|max  
    """

    def __init__(self, model: torch.nn.Module):
        self._model      = model
        self._grad_stats = {}           
        self._writer     = None

        # hook fires on every backward pass and stores the gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda g, n=name: self._grad_stats.update(
                        {n: g.detach().float()}
                    )
                )

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._writer = SummaryWriter(log_dir=args.logging_dir)

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer is not None:
            self._writer.close()

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._writer is None:
            return

        step = state.global_step

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            w = param.detach().float()
            self._writer.add_scalar(f"weights/{name}/mean", w.mean(), step)
            self._writer.add_scalar(f"weights/{name}/std",  w.std(),  step)
            self._writer.add_scalar(f"weights/{name}/min",  w.min(),  step)
            self._writer.add_scalar(f"weights/{name}/max",  w.max(),  step)

        for name, grad in self._grad_stats.items():
            self._writer.add_scalar(f"gradients/{name}/mean", grad.mean(), step)
            self._writer.add_scalar(f"gradients/{name}/std",  grad.std(),  step)
            self._writer.add_scalar(f"gradients/{name}/min",  grad.min(),  step)
            self._writer.add_scalar(f"gradients/{name}/max",  grad.max(),  step)