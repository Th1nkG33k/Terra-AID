
import torch
import torch.nn as nn

# ---------------------------------------------------------
# Basic Model building blocks
# ---------------------------------------------------------

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
    
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, in_ch, base_ch, depth):

        super().__init__()
        layers = []
        ch = in_ch

        for i in range(depth):

            layers.append(ConvBlock(ch, base_ch))
            ch = base_ch
            base_ch *= 2  # progressively increase channels

        self.encoder = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.encoder(x)

    def encode(self, x):
        return self.forward(x)


class Decoder(nn.Module):

    def __init__(self, in_ch, out_ch, depth):

        super().__init__()
        layers = []
        ch = in_ch

        for i in range(depth):

            layers.append(ConvBlock(ch, ch // 2))
            ch = ch // 2

        layers.append(nn.Conv2d(ch, out_ch, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ---------------------------------------------------------
# Masked Autoencoder
# ---------------------------------------------------------

class MAE(nn.Module):

    def __init__(self, encoder, decoder, mask_ratio):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, apply_mask=True):
        # -------------------------------------------------
        # Masking
        # -------------------------------------------------
        # Training keeps the MAE behaviour: reconstruct from a randomly masked
        # input. Prediction can call forward(..., apply_mask=False) through
        # predict() so anomaly maps are deterministic and are not dominated by
        # whichever pixels happened to be masked at inference time.
        B, C, H, W = x.shape
        num_pixels = H * W

        if apply_mask and float(self.mask_ratio or 0.0) > 0.0:
            num_mask = int(num_pixels * self.mask_ratio)

            # random mask per sample
            mask = torch.rand(B, num_pixels, device=x.device)
            _, idx = torch.topk(mask, num_mask, dim=1)

            mask_full = torch.ones(B, num_pixels, device=x.device)
            mask_full.scatter_(1, idx, 0)
            mask_full = mask_full.view(B, 1, H, W)
            x_in = x * mask_full
        else:
            mask_full = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
            x_in = x

        # -------------------------------------------------
        # Encode + Decode
        # -------------------------------------------------
        latent = self.encoder(x_in)
        recon = self.decoder(latent)

        return recon, mask_full

    def predict(self, x):
        was_training = self.training
        self.eval()
        recon, meta = self.forward(x, apply_mask=False)
        if was_training:
            self.train()
        result = {
            "reconstruction": recon,
            "model_type": self.get_model_type(),
        }
        if torch.is_tensor(meta):
            result["mask"] = meta
        elif isinstance(meta, dict):
            result.update(meta)
        return result

    def get_model_type(self) -> str:
        return "mae"


def build_mae(config):
    """
    Build a MAE model dynamically from a config dictionary.
    """

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    in_channels = _get(config, "num_channels", 15)
    base_channels = _get(config, "base_channels", 32)
    encoder_depth = _get(config, "encoder_depth", 4)
    decoder_depth = _get(config, "decoder_depth", 3)
    mask_ratio = _get(config, "mask_ratio", 0.75)

    # Encoder
    encoder = Encoder(in_ch=in_channels,
                      base_ch=base_channels,
                      depth=encoder_depth
                    )

    # Decoder
    decoder = Decoder(in_ch=encoder.out_channels,
                      out_ch=in_channels,
                      depth=decoder_depth
                    )

    # Full MAE
    model = MAE(encoder=encoder,
                decoder=decoder,
                mask_ratio=mask_ratio
            )

    return model