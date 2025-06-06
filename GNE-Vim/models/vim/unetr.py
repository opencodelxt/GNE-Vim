from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Decoder, ConvBlock2d, Upsampler2d
from .vit import get_vision_transformer

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None


#
# UNETR IMPLEMENTATION [Vision Transformer (ViT from MAE / ViT from SAM) + UNet Decoder from `torch_em`]
#


class UNETR(nn.Module):

    def _load_encoder_from_checkpoint(self, backbone, encoder, checkpoint):

        if isinstance(checkpoint, str):
            if backbone == "sam" and isinstance(encoder, str):
                # If we have a SAM encoder, then we first try to load the full SAM Model
                # (using micro_sam) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    _, model = get_sam_model(
                        model_type=encoder,
                        checkpoint_path=checkpoint,
                        return_sam=True
                    )
                    encoder_state = model.image_encoder.state_dict()
                except Exception:
                    # Try loading the encoder state directly from a checkpoint.
                    encoder_state = torch.load(checkpoint)

            elif backbone == "mae":
                # vit initialization hints from:
                #     - https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L233-L242
                encoder_state = torch.load(checkpoint)["model"]
                encoder_state = OrderedDict({
                    k: v for k, v in encoder_state.items()
                    if (k != "mask_token" and not k.startswith("decoder"))
                })

                # let's remove the `head` from our current encoder (as the MAE pretrained don't expect it)
                current_encoder_state = self.encoder.state_dict()
                if ("head.weight" in current_encoder_state) and ("head.bias" in current_encoder_state):
                    del self.encoder.head

        else:
            encoder_state = checkpoint

        self.encoder.load_state_dict(encoder_state)

    def __init__(
            self,
            img_size: int = 1024,
            backbone: str = "sam",
            encoder: Optional[Union[nn.Module, str]] = "vit_b",
            decoder: Optional[nn.Module] = None,
            out_channels: int = 1,
            use_sam_stats: bool = False,
            use_mae_stats: bool = False,
            resize_input: bool = True,
            encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
            final_activation: Optional[Union[str, nn.Module]] = None,
            use_skip_connection: bool = True,
            embed_dim: Optional[int] = None,
            use_conv_transpose=True,
    ) -> None:
        super().__init__()

        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats
        self.use_skip_connection = use_skip_connection
        self.resize_input = resize_input

        if isinstance(encoder, str):  # "vit_b" / "vit_l" / "vit_h"
            print(f"Using {encoder} from {backbone.upper()}")
            self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=encoder)
            if encoder_checkpoint is not None:
                self._load_encoder_from_checkpoint(backbone, encoder, encoder_checkpoint)

            in_chans = self.encoder.in_chans
            if embed_dim is None:
                embed_dim = self.encoder.embed_dim

        else:  # `nn.Module` ViT backbone
            self.encoder = encoder

            have_neck = False
            for name, _ in self.encoder.named_parameters():
                if name.startswith("neck"):
                    have_neck = True

            if embed_dim is None:
                if have_neck:
                    embed_dim = self.encoder.neck[2].out_channels  # the value is 256
                else:
                    embed_dim = self.encoder.patch_embed.proj.out_channels

            try:
                in_chans = self.encoder.patch_embed.proj.in_channels
            except AttributeError:  # for getting the input channels while using vit_t from MobileSam
                in_chans = self.encoder.patch_embed.seq[0].c.in_channels

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        # choice of upsampler - to use (bilinear interpolation + conv) or conv transpose
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d

        if decoder is None:
            self.decoder = Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=ConvBlock2d,
                sampler_impl=_upsampler
            )
        else:
            self.decoder = decoder

        if use_skip_connection:
            self.deconv1 = Deconv2DBlock(embed_dim, features_decoder[0])
            self.deconv2 = nn.Sequential(
                Deconv2DBlock(embed_dim, features_decoder[0]),
                Deconv2DBlock(features_decoder[0], features_decoder[1])
            )
            self.deconv3 = nn.Sequential(
                Deconv2DBlock(embed_dim, features_decoder[0]),
                Deconv2DBlock(features_decoder[0], features_decoder[1]),
                Deconv2DBlock(features_decoder[1], features_decoder[2])
            )
            self.deconv4 = ConvBlock2d(in_chans, features_decoder[-1])
        else:
            self.deconv1 = Deconv2DBlock(embed_dim, features_decoder[0])
            self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
            self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])
            self.deconv4 = Deconv2DBlock(features_decoder[2], features_decoder[3])

        self.base = ConvBlock2d(embed_dim, features_decoder[0])

        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)

        self.deconv_out = _upsampler(
            scale_factor=2, in_channels=features_decoder[-1], out_channels=features_decoder[-1]
        )

        self.decoder_head = ConvBlock2d(2 * features_decoder[-1], features_decoder[-1])

        self.final_activation = self._get_activation(final_activation)

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)
        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")
        return return_activation()

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def resize_longest_side(self, image: torch.Tensor) -> torch.Tensor:
        """Resizes the image so that the longest side has the correct length.

        Expects batched images with shape BxCxHxW and float format.
        """
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.encoder.img_size)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        if self.use_sam_stats:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1).to(device)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1).to(device)
        elif self.use_mae_stats:
            # TODO: add mean std from mae experiments (or open up arguments for this)
            raise NotImplementedError
        else:
            pixel_mean = torch.Tensor([0.0, 0.0, 0.0]).view(1, -1, 1, 1).to(device)
            pixel_std = torch.Tensor([1.0, 1.0, 1.0]).view(1, -1, 1, 1).to(device)

        # 保存原始输入形状
        input_shape = x.shape[-2:]

        # 对输入进行标准化
        x = (x - pixel_mean) / pixel_std

        # 对于卷积网络，我们可能不需要调整大小和填充，因为它们通常可以处理任意大小的输入
        # 但对于ViT/ViM，我们需要确保输入大小正确
        if hasattr(self.encoder, 'img_size'):
            if self.resize_input:
                x = self.resize_longest_side(x)
                input_shape = x.shape[-2:]

            # 只有当编码器需要固定大小输入时才进行填充
            # 例如ViT/ViM编码器
            h, w = x.shape[-2:]
            if h != self.encoder.img_size or w != self.encoder.img_size:
                padh = max(0, self.encoder.img_size - h)
                padw = max(0, self.encoder.img_size - w)
                x = F.pad(x, (0, padw, 0, padh))

        return x, input_shape

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        if isinstance(x, tuple):
            input_shape = torch.Size([224, 224])
            original_shape = torch.Size([224, 224])
            x = self.encoder(x)
            encoder_outputs = x
        else:
            original_shape = x.shape[-2:]
            x, input_shape = self.preprocess(x)
            encoder_outputs = self.encoder(x)
        use_skip_connection = getattr(self, "use_skip_connection", True)

        # 处理不同类型编码器的输出格式
        # 对于卷积编码器，输出是一个列表，包含最终特征和中间特征列表
        # 对于ViT/ViM编码器，输出可能是单个张量或包含全局注意力层的列表
        if isinstance(encoder_outputs, list) and len(encoder_outputs) == 2 and isinstance(encoder_outputs[1], list):
            # 这是卷积编码器或CrossConv的输出格式 [final_features, [skip_features]]
            z12, from_encoder = encoder_outputs
        elif isinstance(encoder_outputs, list) and isinstance(encoder_outputs[-1], list):
            # 这是ViT/ViM编码器的输出格式，包含全局注意力层
            z12, from_encoder = encoder_outputs
        else:
            # 单个张量输出
            z12 = encoder_outputs
            from_encoder = None

        if use_skip_connection and from_encoder is not None:
            # 确保跳跃连接特征按照从深到浅的顺序
            if not isinstance(from_encoder[0], list):  # 如果不是嵌套列表
                from_encoder = from_encoder[::-1]

            # 确保我们有足够的跳跃连接
            if len(from_encoder) >= 3:
                z9 = self.deconv1(from_encoder[0])
                z6 = self.deconv2(from_encoder[1])
                # z3 = self.deconv3(from_encoder[2])
            else:  # 如果跳跃连接不足，使用最后一个特征图
                last_feature = from_encoder[-1]
                z9 = self.deconv1(last_feature)
                z6 = self.deconv2(last_feature)
                z3 = self.deconv3(last_feature)

            z0 = self.deconv4(x)
        else:
            z9 = self.deconv1(z12)
            z6 = self.deconv2(z9)
            z3 = self.deconv3(z6)
            z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        # 确保z12是正确的形状用于base层
        # 对于卷积网络，z12可能是4D张量 [B, C, H, W]
        # 对于ViT/ViM，z12可能是3D张量 [B, N, C]
        if len(z12.shape) == 3:  # ViT/ViM输出
            x = self.base(z12)
        else:  # 卷积网络输出
            x = self.base(z12)

        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return z12, x


#
#  ADDITIONAL FUNCTIONALITIES
#


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_conv_transpose=True):
        super().__init__()
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d
        self.block = nn.Sequential(
            _upsampler(scale_factor=2, in_channels=in_channels, out_channels=out_channels),
            SingleConv2DBlock(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
