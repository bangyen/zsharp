# import torch.nn.functional as F
# import torch.nn as nn
import numpy as np
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                p.state["e"] = e

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e" in p.state:
                    p.sub_(p.state["e"])
        self.base_optimizer.step()

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires two-step calls: first_step and second_step"
        )


class ZSharp(SAM):
    def __init__(
        self, params, base_optimizer, rho=0.05, percentile=70, **kwargs
    ):
        super().__init__(params, base_optimizer, rho=rho, **kwargs)
        self.percentile = percentile

    @torch.no_grad()
    def first_step(self):
        grads = [
            p.grad.detach().flatten()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        all_grads = torch.cat(grads)
        mean, std = torch.mean(all_grads), torch.std(all_grads) + 1e-12
        zscores = (all_grads - mean) / std
        threshold = np.percentile(zscores.cpu().numpy(), self.percentile)

        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                mask = ((p.grad - mean) / std) > threshold
                e = p.grad * mask * scale
                p.add_(e)
                p.state["e"] = e
