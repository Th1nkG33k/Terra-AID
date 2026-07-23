import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg, key, default=None):

    if isinstance(cfg, dict):
        return cfg.get(key, default)
    
    return getattr(cfg, key, default)


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
    
        super().__init__()
    
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ===========================================================================
# Classic CNN AutoEncoder
#
#    Simple convolutional autoencoder baseline for reconstruction-based
#    anomaly detection. It is deliberately plain so it represents a classic CNN
#    baseline against MAE/ViT-style reconstruction.
# ===========================================================================
class ClassicCNNAutoencoder(nn.Module):
 
    def __init__(self, in_channels=15, base_channels=32, depth=4, latent_channels=None):

        super().__init__()
        self.in_channels = int(in_channels)
        self.depth = int(depth)

        enc = []
        ch = self.in_channels
        out_ch = int(base_channels)

        for _ in range(self.depth):
            enc.append(ConvBlock(ch, out_ch))
            ch = out_ch
            out_ch *= 2
        
        self.encoder_blocks = nn.ModuleList(enc)
        self.out_channels = ch

        latent_channels = int(latent_channels or ch)
        self.bottleneck = ConvBlock(ch, latent_channels)

        dec = []
        ch = latent_channels

        for _ in range(self.depth):

            next_ch = max(int(base_channels), ch // 2)
            dec.append(ConvBlock(ch, next_ch))
            ch = next_ch
        
        self.decoder_blocks = nn.ModuleList(dec)
        self.output_layer = nn.Conv2d(ch, self.in_channels, kernel_size=1)

    def encode(self, x):

        z = x
        
        for block in self.encoder_blocks:
            z = block(z)
            z = F.max_pool2d(z, kernel_size=2, stride=2, ceil_mode=True)
        
        return self.bottleneck(z)

    def forward(self, x):
        
        input_size = x.shape[-2:]
        z = self.encode(x)
        y = z

        for block in self.decoder_blocks:
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = block(y)

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
        return "cnn_autoencoder"


def build_cnn_autoencoder(config):

    return ClassicCNNAutoencoder(in_channels=_get(config, "num_channels", 15),
                                 base_channels=_get(config, "base_channels", 32),
                                 depth=_get(config, "encoder_depth", _get(config, "depth", 4)),
                                 latent_channels=_get(config, "latent_channels", None),
    )
