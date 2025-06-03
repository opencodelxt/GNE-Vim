# installation from https://github.com/hustvl/Vim
# encoder from https://github.com/hustvl/Vim
# decoder from https://github.com/constantinpape/torch-em
import os.path
from functools import partial

import timm
import torch
from timm.models.vision_transformer import _cfg
from torch import nn
import torchvision.models as models
# import sys
#
# curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append(curPath)

from models_simple.vim.conv_transformer import CrossAttention
from models_simple.vim.models_mamba import VisionMamba, rms_norm_fn, RMSNorm, layer_norm_fn
from models_simple.vim.unetr import UNETR


# pretrained model weights: vim_t - https://huggingface.co/hustvl/Vim-tiny/blob/main/vim_tiny_73p1.pth


class ViT(nn.Module):
    def __init__(
            self,
            if_cls_token=True,
            checkpoint=None,
            **kwargs
    ):
        super().__init__()
        model = timm.create_model("deit_tiny_patch16_224",
                                  pretrained=False, **kwargs)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])

        # copy the attributes from the model
        for k, v in model.__dict__.items():
            self.__dict__[k] = v
        self.if_cls_token = if_cls_token

    def convert_to_expected_dim(self, inputs_):
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        x = self.forward_features(x, inference_params)
        if self.if_cls_token:  # remove the class token
            x = x[:, 1:, :]
        # let's get the patches back from the 1d tokens
        x = self.convert_to_expected_dim(x)
        return x  # from here, the tokens can be upsampled easily (N x H x W x C)


class ViM(VisionMamba):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def convert_to_expected_dim(self, inputs_):
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # mamba implementation
        residual = None
        hidden_states = x
        for layer in self.layers:
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm = False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        x = self.forward_features(x, inference_params)
        if self.if_cls_token:  # remove the class token
            x = x[:, 1:, :]
        # let's get the patches back from the 1d tokens
        x = self.convert_to_expected_dim(x)
        return x  # from here, the tokens can be upsampled easily (N x H x W x C)


class CrossViM(nn.Module):
    def __init__(self, in_channels, img_size, patch_embed, num_heads=3):
        super().__init__()
        self.cross_attn = CrossAttention(dim=in_channels, num_heads=num_heads, bias=False)
        self.img_size = img_size
        self.patch_embed = patch_embed

    def forward(self, x):
        x1, x2 = x
        return self.cross_attn(x1, x2, x2)
        # return x1 + x2


def get_vim_encoder(model_type="vim_t", with_cls_token=True, checkpoint=None):
    """
    * vim_t_midclstok_76p1acc.pth
      * if_rope=False
      * stride=16
    * vim_tiny_73p1.pth
      * if_rope=True
      * stride=16
    * vim_t_midclstok_ft_78p3acc.pth
      * if_rope=False
      * stride=8
    """
    if model_type == "vgg16" or model_type == "resnet50":
        return get_conv_encoder(model_type=model_type, pretrained=True, checkpoint=None)
    if checkpoint is None:
        if_rope = True
        stride = 16
    elif "vim_t_midclstok_76p1acc.pth" in checkpoint:
        if_rope = False
        stride = 16
    elif "vim_t_midclstok_ft_78p3acc.pth" in checkpoint:
        if_rope = False
        stride = 8
    else:
        # checkpoint == "vim_tiny_73p1.pth":
        if_rope = True
        stride = 16
    if model_type == "vim_t":
        # `vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token`
        # *has an imagenet pretrained model
        encoder = ViM(patch_size=16, embed_dim=192, stride=stride, depth=24, rms_norm=True, residual_in_fp32=True,
                      fused_add_norm=True,
                      final_pool_type='all', if_abs_pos_embed=True, if_rope=if_rope, if_rope_residual=False,
                      bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)
    elif model_type == "vim_s":
        # `vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual`
        # AA: added a class token to the default models
        encoder = ViM(patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                      final_pool_type='all', if_abs_pos_embed=True, if_rope=True, if_rope_residual=False,
                      bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)
    elif model_type == "vit_t":
        # `vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual`
        # AA: added a class token to the default models
        encoder = ViT(mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), checkpoint=checkpoint)

    elif model_type == "vim_b":
        # `vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual`
        # AA: added a class token to the default models
        encoder = ViM(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=24,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='all',
            if_abs_pos_embed=True,
            if_rope=True,
            if_rope_residual=True,
            bimamba_type="v2",
            if_cls_token=with_cls_token,
        )
    else:
        raise ValueError("Choose from 'vim_t' / 'vim_s' / 'vim_b'")

    encoder.default_cfg = _cfg()

    if checkpoint is not None and "vim" in checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        encoder_state = state["model"]
        encoder.load_state_dict(encoder_state)
    return encoder


def get_vimunet_model(
        out_channels, model_type="vim_t", with_cls_token=True, checkpoint=None,
):
    if model_type == "vgg16" or model_type == "resnet50":
        return get_conv_model(out_channels, model_type=model_type, pretrained=True, checkpoint=None)
    encoder = get_vim_encoder(model_type, with_cls_token, checkpoint=checkpoint)

    model_state = None
    if checkpoint is not None and not checkpoint.endswith(".pth"):  # from Vim
        state = torch.load(checkpoint, map_location="cpu")
        model_state = state["model_state"]

    encoder.img_size = encoder.patch_embed.img_size[0]

    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid",
    )

    if model_state is not None:
        model.load_state_dict(model_state)

    return model


def get_cross_vimunet_model(in_channels, out_channels, img_size, patch_embed, num_heads=3):
    encoder = CrossViM(in_channels, img_size, patch_embed, num_heads=num_heads)
    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid",
    )
    return model


def get_conv_encoder(model_type="vgg16", pretrained=True, checkpoint=None):
    """
    获取基于卷积网络的编码器
    
    Args:
        model_type: 卷积网络类型，支持 "vgg16" 或 "resnet50"
        pretrained: 是否使用预训练权重
        checkpoint: 自定义权重文件路径
    
    Returns:
        编码器模型
    """
    if model_type == "vgg16":
        encoder = models.vgg16(pretrained=pretrained).features
        # 设置图像大小属性，以便与UNETR兼容
        encoder.img_size = 224
        # 设置输出特征维度
        encoder.embed_dim = 512
    elif model_type == "resnet50":
        # 移除最后的全连接层
        encoder = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
        # 设置图像大小属性，以便与UNETR兼容
        encoder.img_size = 224
        # 设置输出特征维度
        encoder.embed_dim = 2048
    else:
        raise ValueError("Choose from 'vgg16' or 'resnet50'")

    # 添加forward方法以返回特征图和中间特征列表
    class ConvEncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.img_size = encoder.img_size
            self.embed_dim = encoder.embed_dim

            # 添加patch_embed属性以兼容UNETR类的接口
            # 创建一个简单的占位符对象，具有img_size属性
            class PatchEmbedPlaceholder:
                def __init__(self, img_size):
                    self.img_size = [img_size, img_size]

                    # 添加proj属性以兼容UNETR类的接口
                    class ProjPlaceholder:
                        def __init__(self):
                            pass

                    self.proj = ProjPlaceholder()
                    # 设置out_channels属性为编码器的embed_dim
                    self.proj.out_channels = encoder.embed_dim
                    self.proj.in_channels = 3  # 假设输入是RGB图像

            self.patch_embed = PatchEmbedPlaceholder(self.img_size)

        def forward(self, x):
            # 对于VGG16，我们可以在不同深度获取特征
            if isinstance(self.encoder, models.vgg16().features.__class__):
                features = []
                for i, layer in enumerate(self.encoder):
                    x = layer(x)
                    # 在每个池化层后保存特征
                    if isinstance(layer, nn.MaxPool2d):
                        features.append(x)
                # 返回最终特征和中间特征列表
                return x
            # 对于ResNet，我们需要在每个阶段后获取特征
            else:
                # 假设这是ResNet
                features = []
                # 对于ResNet，我们需要手动提取中间特征
                x = self.encoder[0](x)  # conv1
                x = self.encoder[1](x)  # bn1
                x = self.encoder[2](x)  # relu
                x = self.encoder[3](x)  # maxpool

                # layer1
                x = self.encoder[4](x)
                features.append(x)

                # layer2
                x = self.encoder[5](x)
                features.append(x)

                # layer3
                x = self.encoder[6](x)
                features.append(x)

                # layer4
                x = self.encoder[7](x)

                # 返回最终特征和中间特征列表
                return x

    wrapped_encoder = ConvEncoderWrapper(encoder)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        wrapped_encoder.load_state_dict(state)

    return wrapped_encoder


def get_conv_model(out_channels, model_type="vgg16", pretrained=True, checkpoint=None):
    """
    获取基于卷积网络的UNETR模型
    
    Args:
        out_channels: 输出通道数
        model_type: 卷积网络类型，支持 "vgg16" 或 "resnet50"
        pretrained: 是否使用预训练权重
        checkpoint: 自定义权重文件路径
    
    Returns:
        UNETR模型
    """
    encoder = get_conv_encoder(model_type=model_type, pretrained=pretrained, checkpoint=checkpoint)

    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,  # 对于卷积网络，使用跳跃连接
        final_activation="Sigmoid",
    )

    return model


if __name__ == '__main__':
    model = get_vimunet_model(out_channels=3, model_type="vim_t",
                              checkpoint="../../weights/vim_t_midclstok_ft_78p3acc.pth")
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        out = model(x)
        print(out.shape)

    # 测试卷积模型
    conv_model = get_conv_model(out_channels=3, model_type="vgg16")
    conv_model.eval()
    conv_model.to(device)
    with torch.no_grad():
        out = conv_model(x)
        print(f"Conv model output shape: {out.shape}")
