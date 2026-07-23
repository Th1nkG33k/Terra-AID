import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg, key, default=None):

    if isinstance(cfg, dict):
        return cfg.get(key, default)
    
    return getattr(cfg, key, default)

# ---------------------------------------------------------------------
# ResNet-50 encoder-decoder reconstruction model.
# Sentinel-2/multimodal inputs can have more than 3 channels. A 1x1
# projection maps the full input stack to 3 channels so ImageNet pretrained
# ResNet weights can still be used, while the decoder reconstructs the full
# original channel stack.
# ---------------------------------------------------------------------
class ResNetAutoencoder(nn.Module):

    def __init__(self, in_channels=15, backbone="resnet50", pretrained=True, decoder_channels=(512, 256, 128, 64)):
        
        super().__init__()
        
        self.in_channels = int(in_channels)
        self.backbone_name = str(backbone or "resnet50").lower()
        self.pretrained = bool(pretrained)
        self.input_projection = nn.Conv2d(self.in_channels, 3, kernel_size=1) if self.in_channels != 3 else nn.Identity()

        try:

            from torchvision import models
        
        except Exception as exc:
        
            raise ImportError("ResNet autoencoder requires torchvision. " \
                              "Install torchvision or choose mae/cnn_autoencoder."
                  ) from exc

        if self.backbone_name != "resnet50":
            raise ValueError("Only resnet50 is currently supported by Terra-AId's ResNet autoencoder.")

        weights = None

        if self.pretrained:
            
            try:

                weights = models.ResNet50_Weights.DEFAULT

            except AttributeError:
                weights = "DEFAULT"

        resnet = models.resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = 2048

        blocks = []
        ch = self.out_channels

        for out_ch in decoder_channels:

            blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                        nn.Conv2d(ch, int(out_ch), kernel_size=3, padding=1),
                                        nn.BatchNorm2d(int(out_ch)),
                                        nn.ReLU(inplace=True),
            ))

            ch = int(out_ch)

        self.decoder = nn.Sequential(*blocks)
        self.output_layer = nn.Conv2d(ch, self.in_channels, kernel_size=1)

    def encode(self, x):

        y = self.input_projection(x)
        y = self.stem(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        return y

    def forward(self, x):

        input_size = x.shape[-2:]
        z = self.encode(x)
        y = self.decoder(z)
        recon = self.output_layer(y)
        
        if recon.shape[-2:] != input_size:
            recon = F.interpolate(recon, size=input_size, mode="bilinear", align_corners=False)
        
        return recon, {}

    def predict(self, x):
        was_training = self.training
        self.eval()
        recon, meta = self.forward(x)
        if was_training:
            self.train()
        return {
            "reconstruction": recon,
            "metadata": meta,
            "model_type": self.get_model_type(),
        }

    def get_model_type(self) -> str:
        return "resnet_autoencoder"


def build_resnet_autoencoder(config):

    pretrained = _get(config, "pretrained", True)
    
    if isinstance(pretrained, str):
        pretrained = pretrained.lower() in {"true", "1", "yes", "y"}
    
    return ResNetAutoencoder(in_channels=_get(config, "num_channels", 15),
                             backbone=_get(config, "backbone", "resnet50"),
                             pretrained=pretrained,
    )
