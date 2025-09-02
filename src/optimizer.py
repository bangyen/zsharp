import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
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
                if not hasattr(p, "state"):
                    p.state = {}
                p.state["e"] = e

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "state") and "e" in p.state:
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
        # Collect gradients by layer/parameter group for layer-wise Z-score
        # computation
        layer_grads = []
        layer_grad_info = (
            []
        )  # Store (param, grad, start_idx, end_idx) for each layer

        start_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_flat = p.grad.detach().flatten()

                # Handle mixed precision
                if grad_flat.dtype == torch.float16:
                    grad_flat = grad_flat.float()

                layer_grads.append(grad_flat)
                end_idx = start_idx + grad_flat.numel()
                layer_grad_info.append((p, p.grad, start_idx, end_idx))
                start_idx = end_idx

        if not layer_grads:
            return

        # Compute Z-scores layer-wise as described in the paper
        zscores_list = []
        for grad_flat in layer_grads:
            # Layer-wise Z-score normalization with numerical stability
            layer_mean, layer_std = (
                torch.mean(grad_flat),
                torch.std(grad_flat) + 1e-8,
            )
            layer_zscores = (grad_flat - layer_mean) / layer_std
            zscores_list.append(layer_zscores)

        # Concatenate Z-scores for percentile computation
        all_zscores = torch.cat(zscores_list)

        # Handle edge case where all Z-scores are the same
        if torch.allclose(all_zscores, all_zscores[0]):
            # If all Z-scores are the same, use a simple threshold
            threshold = 0.0
        else:
            # Use absolute Z-scores for percentile computation as per paper
            threshold = torch.quantile(
                all_zscores.abs(), self.percentile / 100.0
            ).item()

        # Apply filtering to each layer
        for i, (p, original_grad, start_idx, end_idx) in enumerate(
            layer_grad_info
        ):
            if p.grad is None:
                continue

            # Get Z-scores for this layer
            layer_zscores = zscores_list[i]

            # Create mask based on absolute Z-score threshold
            mask = layer_zscores.abs() >= threshold

            # Ensure at least some gradients are kept (numerical stability)
            if not mask.any():
                # Keep top 20% if no gradients pass threshold
                top_k = max(1, int(0.2 * mask.numel()))
                _, indices = torch.topk(layer_zscores.abs(), top_k)
                mask = torch.zeros_like(mask)
                mask[indices] = True

            # Reshape mask to match original gradient shape
            mask = mask.view_as(original_grad)

            # Apply masking to gradients
            p.grad = p.grad * mask

        # Apply SAM perturbation with filtered gradients
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

        # Add numerical stability to gradient scaling
        if grad_norm > 0:
            scale = self.rho / grad_norm
        else:
            scale = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                if not hasattr(p, "state"):
                    p.state = {}
                p.state["e"] = e
