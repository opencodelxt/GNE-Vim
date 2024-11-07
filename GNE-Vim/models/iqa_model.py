
import torch
from torch import nn

from models.unet.unet_model import UNet
from models.vae.vqvae import VQVAE
from models.vim.vim import get_vimunet_model, get_vim_encoder, get_cross_vimunet_model
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile
from thop import clever_format

class ScoreModel(nn.Module):
    def __init__(self, model_type="vim_t", checkpoint=None):
        super(ScoreModel, self).__init__()
        # self.backbone = torchvision.models.vgg16(pretrained=True).features
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone = get_vim_encoder(model_type=model_type, with_cls_token=True,
                                        checkpoint=checkpoint)
        self.regressor = nn.Sequential(
            nn.Linear(192, 512),
            # training
            nn.BatchNorm1d(512),
            # testing
            # nn.LayerNorm(512),

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
        x = self.backbone.forward_features(x)
        x = x.mean(dim=1)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x



class Generator(nn.Module):
    def __init__(self, in_channels=3, model_type="vim_t", checkpoint="weights/vim_tiny_73p1.pth"):
        super(Generator, self).__init__()
        self.noise_generator = get_vimunet_model(in_channels, model_type=model_type, with_cls_token=True,
                                                 checkpoint=checkpoint)

        self.score_net = ScoreModel(model_type=model_type, checkpoint=checkpoint)

    def forward(self, x):
        # x: distorted image
        # y: reference image
        noise_img = self.noise_generator(x)
        noise_img = (noise_img - 0.5) / 0.5
        score = self.score_net(noise_img)

        return score


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 1)
        )

    def forward(self, x):
        return self.discriminator(x)



class VimIQAModel(nn.Module):
    def __init__(self, in_channels=3, model_type="vim_t", checkpoint="weights/vim_tiny_73p1.pth"):
        super(VimIQAModel, self).__init__()
        self.noise_generator = get_vimunet_model(in_channels, model_type=model_type, with_cls_token=True,
                                                 checkpoint=checkpoint)
        self.fusion = get_cross_vimunet_model(in_channels, model_type=model_type, with_cls_token=True,
                                              checkpoint=checkpoint)

        self.score_net = ScoreModel(model_type=model_type, checkpoint=checkpoint)

    def forward(self, x, y):
        # x: distorted image
        # y: reference image
        noise_img = self.noise_generator(x)
        noise_img = (noise_img - 0.5) / 0.5
        score = self.score_net(noise_img)

        # training
        pseudo_distort_img = self.fusion((noise_img, y))
        pseudo_distort_img = (pseudo_distort_img - 0.5) / 0.5
        # return pseudo_distort_img, noise_img, score
        return pseudo_distort_img, score



if __name__ == '__main__':
    in_channels = 3
    img_size = 224
    model = Generator(in_channels, model_type="vim_t", checkpoint="../weights/vim_tiny_73p1.pth")
    device = torch.device("cuda:0")
    model = model.to(device)
    model.train()
    input = torch.randn(1, in_channels, img_size, img_size, device=device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
