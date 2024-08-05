# installation from https://github.com/hustvl/Vim
# encoder from https://github.com/hustvl/Vim
# decoder from https://github.com/constantinpape/torch-em
from functools import partial

import torch
import timm
from torch import nn

from models.vim.conv_transformer import CrossAttention, CrossConvTransformerBlock
from models.vim.models_mamba import VisionMamba, rms_norm_fn, RMSNorm, layer_norm_fn
from models.vim.unetr import UNETR
from timm.models.vision_transformer import _cfg


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
    def __init__(self, model_type="vim_t", with_cls_token=True, checkpoint=None):
        super().__init__()
        self.encoder1 = get_vim_encoder(model_type=model_type, with_cls_token=with_cls_token, checkpoint=checkpoint)
        self.encoder2 = get_vim_encoder(model_type=model_type, with_cls_token=with_cls_token, checkpoint=checkpoint)
        self.cross_attn = CrossAttention(dim=192, num_heads=3, bias=False)
        # self.cross_attn = CrossConvTransformerBlock(dim=192, num_heads=3, ffn_expansion_factor=2, bias=False,
        #                                              LayerNorm_type='WithBias')
        self.img_size = self.encoder1.patch_embed.img_size[0]
        self.default_cfg = self.encoder1.default_cfg
        self.patch_embed = self.encoder1.patch_embed

    def forward(self, x):
        x1, x2 = x
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.cross_attn(x1, x2, x2)
        return x


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
        out_channels, model_type="vim_t", with_cls_token=True, device=None, checkpoint=None
):
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


def get_cross_vimunet_model(out_channels, model_type="vim_t", with_cls_token=True, checkpoint=None):
    encoder = CrossViM(model_type=model_type, with_cls_token=with_cls_token, checkpoint=checkpoint)
    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid",
    )
    return model


if __name__ == '__main__':
    # test get_cross_vimunet_model
    model = get_vimunet_model(out_channels=3, model_type="vit_t", with_cls_token=True,
                              checkpoint='../../weights/deit_tiny_patch16_224-a1311bcf.pth')
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
