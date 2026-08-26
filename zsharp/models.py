# Copyright (c) 2025 Bangyen Pham
"""Model loading utilities for various architectures.

This module provides functions to load and configure different
PyTorch models including ResNet, VGG, and Vision Transformer variants.

Alongside the torchvision ImageNet models, this module implements the
architectures used in the ZSharp paper (arXiv:2505.02369): CIFAR-style
ResNets (He et al., Sec. 4.2) and the paper's small Vision Transformers.
"""

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torchvision import models
from torchvision.models import vit_b_16

if TYPE_CHECKING:
    from collections.abc import Callable

from zsharp.constants import (
    CIFAR_RESNET_BASE_WIDTH,
    CIFAR_RESNET_STAGES,
    RESNET18_NAME,
    VIT_PAPER_HEADS,
    VIT_PAPER_HIDDEN,
    VIT_PAPER_LAYERS,
    VIT_PAPER_PATCHES_PER_SIDE,
)


class _CifarBasicBlock(nn.Module):
    """Two-convolution residual block for CIFAR-style ResNets."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Initialize the block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride of the first convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Option A shortcut from He et al. Sec. 4.2: downsample by striding
        # and zero-pad the extra channels, keeping the block parameter-free.
        self.pad_channels = 0
        self.stride = stride
        if stride != 1 or in_planes != planes:
            self.pad_channels = (planes - in_planes) // 2

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the parameter-free option-A shortcut."""
        if self.pad_channels == 0 and self.stride == 1:
            return x
        subsampled = x[:, :, :: self.stride, :: self.stride]
        return nn.functional.pad(
            subsampled,
            (0, 0, 0, 0, self.pad_channels, self.pad_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self._shortcut(x)
        return nn.functional.relu(out)


class CifarResNet(nn.Module):
    """CIFAR-style ResNet with 6n+2 layers (He et al., Sec. 4.2).

    Three stages of ``n`` basic blocks operate at 16, 32 and 64 channels on
    32x32 inputs. This differs from the torchvision ImageNet ResNets, which
    use a 7x7 stem and max-pooling; depths such as 56 and 110 exist only in
    this CIFAR family.
    """

    def __init__(self, blocks_per_stage: int, num_classes: int = 10) -> None:
        """Initialize the network.

        Args:
            blocks_per_stage: Number of blocks ``n`` in each of the three
                stages, giving a depth of ``6n + 2``.
            num_classes: Number of output classes.
        """
        super().__init__()
        width = CIFAR_RESNET_BASE_WIDTH
        self.conv1 = nn.Conv2d(
            3,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)

        stages, final_width = self._build_stages(blocks_per_stage, width)
        self.layers = nn.Sequential(*stages)
        self.fc = nn.Linear(final_width, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

    @staticmethod
    def _build_stages(
        blocks_per_stage: int, width: int
    ) -> tuple[list[nn.Module], int]:
        """Build the three residual stages.

        Args:
            blocks_per_stage: Number of blocks in each stage.
            width: Channel count of the first stage.

        Returns:
            tuple: The blocks and the final channel count.
        """
        blocks: list[nn.Module] = []
        in_planes = width
        for stage in range(CIFAR_RESNET_STAGES):
            planes = width * (2**stage)
            for block in range(blocks_per_stage):
                # Each stage after the first halves the spatial resolution.
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(_CifarBasicBlock(in_planes, planes, stride))
                in_planes = planes
        return blocks, in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Input images of shape ``(batch, 3, height, width)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch, num_classes)``.
        """
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        return cast("torch.Tensor", self.fc(out))


class CifarVGG16BN(nn.Module):
    """VGG-16 with batch normalization, adapted for small images.

    Matches the author's reference implementation: the standard VGG-16
    convolutional stack followed by global average pooling and a single
    512-unit linear classifier. The torchvision ImageNet VGG instead ends
    in three 4096-wide layers, which on a 32x32 input would upsample a 1x1
    feature map to 7x7 and carry roughly nine times the parameters.
    """

    # Standard VGG-16 ("D") convolutional configuration.
    # fmt: off
    CONFIG = (
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512,
    )
    # fmt: on

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the network.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for entry in self.CONFIG:
            if entry == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            channels = int(entry)
            layers.extend(
                (
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Input images of shape ``(batch, 3, height, width)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch, num_classes)``.
        """
        out = self.avgpool(self.features(x)).flatten(1)
        return cast("torch.Tensor", self.classifier(out))


class PaperViT(nn.Module):
    """The compact Vision Transformer used in the ZSharp paper.

    Named ``ViT-<layers>/<patches>/<heads>-<mlp>`` in the paper, e.g.
    ``ViT-7/8/8-384`` and ``ViT-7/8/12-768``.

    The paper's prose describes the fields as layers / heads / patch size /
    MLP dimension, but that reading is not self-consistent: it would make
    both variants 8-headed with patch size 8, leaving the differing third
    field unexplained, and 12 patches per side does not divide a 32x32
    input. The author's reference implementation fixes patches at 8 per
    side and varies heads (8 and 12) at a constant embedding width of 384,
    which is the reading used here.

    Note that the second field is the number of patches *per side*, not the
    patch size in pixels: on a 32x32 input, 8 patches per side means each
    patch is 4x4 pixels. Patches are embedded by flattening pixels and
    applying a single linear projection.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 10,
        image_size: int = 32,
        *,
        patches_per_side: int = VIT_PAPER_PATCHES_PER_SIDE,
        num_layers: int = VIT_PAPER_LAYERS,
        num_heads: int = VIT_PAPER_HEADS,
        hidden: int = VIT_PAPER_HIDDEN,
        mlp_hidden: int = VIT_PAPER_HIDDEN,
    ) -> None:
        """Initialize the transformer.

        Args:
            num_classes: Number of output classes.
            image_size: Height and width of the input images.
            patches_per_side: Number of patches along each spatial axis.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of self-attention heads.
            hidden: Embedding dimension.
            mlp_hidden: Feed-forward dimension, the paper's trailing field.
        """
        super().__init__()
        self.patches_per_side = patches_per_side
        self.patch_size = image_size // patches_per_side
        num_patches = patches_per_side**2
        patch_dim = (self.patch_size**2) * 3

        self.embed = nn.Linear(patch_dim, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=mlp_hidden,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only
        # emits a warning; disable it explicitly.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, num_classes)
        )

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split images into flattened patch vectors."""
        size = self.patch_size
        out = x.unfold(2, size, size).unfold(3, size, size)
        out = out.permute(0, 2, 3, 4, 5, 1)
        return out.reshape(x.size(0), self.patches_per_side**2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Input images of shape ``(batch, 3, height, width)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch, num_classes)``.
        """
        out = self.embed(self._to_patches(x))
        cls = self.cls_token.repeat(out.size(0), 1, 1)
        out = torch.cat([cls, out], dim=1) + self.pos_embed
        out = self.encoder(out)
        return cast("torch.Tensor", self.head(out[:, 0]))


def get_model(
    model_name: str = RESNET18_NAME,
    num_classes: int = 10,
    image_size: int = 32,
) -> nn.Module:
    """Get a PyTorch model by name.

    Args:
        model_name: Name of the model to load
        num_classes: Number of output classes
        image_size: Input resolution, used by the paper's ViT variants to
            derive the patch size.

    Returns:
        torch.nn.Module: PyTorch model

    Raises:
        ValueError: If model name is not supported

    """
    model_map: dict[str, Callable[[], nn.Module]] = {
        "resnet18": lambda: models.resnet18(num_classes=num_classes),
        "vgg11": lambda: models.vgg11(num_classes=num_classes),
        "vit_b_16": lambda: vit_b_16(num_classes=num_classes),
        # Architectures from the ZSharp paper.
        "resnet56": lambda: CifarResNet(9, num_classes),
        "resnet110": lambda: CifarResNet(18, num_classes),
        "vgg16_bn": lambda: CifarVGG16BN(num_classes=num_classes),
        "vit_7_8_8_384": lambda: PaperViT(
            num_classes=num_classes,
            image_size=image_size,
            mlp_hidden=384,
        ),
        "vit_7_8_12_768": lambda: PaperViT(
            num_classes=num_classes,
            image_size=image_size,
            num_heads=12,
            mlp_hidden=768,
        ),
    }

    if model_name not in model_map:
        error_msg = f"Unknown model {model_name}"
        raise ValueError(error_msg)

    return model_map[model_name]()
