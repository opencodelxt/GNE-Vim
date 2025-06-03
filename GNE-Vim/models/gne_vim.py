
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from models.vim.vim import get_vimunet_model, get_cross_vimunet_model, get_vim_encoder


class ScoreModel(nn.Module):
    def __init__(self, model_type="vim_t", checkpoint=None):
        super(ScoreModel, self).__init__()
        if model_type == "vgg16":
            self.backbone = torchvision.models.vgg16(pretrained=True).features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
            in_features = 512
        elif model_type == "resnet50":
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            in_features = 2048
        else:
            self.backbone = get_vim_encoder(model_type=model_type, with_cls_token=True,
                                            checkpoint=checkpoint)
            self.backbone = nn.Sequential(self.backbone, nn.AdaptiveAvgPool2d((1, 1)))
            in_features = 192
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, model_type="vim_t"):
        super().__init__()
        if model_type == "vgg16":
            in_channels = 512
        elif model_type == "resnet50":
            in_channels = 2048
        else:
            in_channels = 192
        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )

    def forward(self, x):
        return self.discriminator(x)


# 'vim_t_midclstok_ft_78p3acc' 'vim_tiny_73p1' 'deit_tiny_patch16_224-a1311bcf'
class VimIQAModel(nn.Module):
    def __init__(self, in_channels=3, model_type="vim_t", checkpoint="weights/vim_tiny_73p1.pth"):
        super(VimIQAModel, self).__init__()
        self.noise_generator = get_vimunet_model(in_channels, model_type=model_type, with_cls_token=True,
                                                 checkpoint=checkpoint)
        img_size = self.noise_generator.encoder.patch_embed.img_size[0]
        patch_embed = self.noise_generator.encoder.patch_embed
        if model_type == "vgg16":
            in_features = 512
            num_heads = 8
        elif model_type == "resnet50":
            in_features = 2048
            num_heads = 8
        else:
            in_features = 192
            num_heads = 3
        self.fusion = get_cross_vimunet_model(in_features, in_channels, img_size, patch_embed, num_heads)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):
        # x: distorted image
        # y: reference image
        if y is not None:
            noise_features, noise_img = self.noise_generator(x)
        else:
            noise_features = self.noise_generator.encoder(x)

        noise_features_x = F.adaptive_avg_pool2d(noise_features, (1, 1))
        noise_features_x = noise_features_x.view(noise_features_x.size(0), -1)
        score = self.regressor(noise_features_x)
        if y is not None:
            ref_features = self.noise_generator.encoder(y)
            _, pseudo_distort_img = self.fusion((noise_features, ref_features))
            return pseudo_distort_img, score, noise_img
        else:
            return score


if __name__ == '__main__':
    from thop import profile, clever_format

    model = VimIQAModel(in_channels=3, model_type="vit_t", checkpoint=None)
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    model.cuda()
    x, y = x.cuda(), y.cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
