"""PyTorch code shared by the scripts: backbone (frame -> features) and the temporal heads.

realtime.py needs only common.py + video_io.py + activity_tracker.py (not config.py), so the
deployable app is: realtime.py + activity_tracker.py + common.py + video_io.py + model.pt + requirements.txt
"""
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FEATURE_DIMS = {"mobilenet_v3_small": 576, "mobilenet_v3_large": 960, "resnet18": 512}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ----------------------------------------------------------------------------------
# Frames -> features
# ----------------------------------------------------------------------------------
def build_backbone(name, pretrained=True):
    """Pretrained CNN that turns ONE frame into ONE feature vector. It is never trained."""
    from torchvision import models

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        net = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        net = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    else:
        raise ValueError(f"Unknown backbone '{name}'. Use one of {list(FEATURE_DIMS)}")
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def frames_to_tensor(frames_rgb):
    arr = np.stack(frames_rgb).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def compute_features(backbone, frames_rgb, batch_size=32):
    """list of prepared frames -> numpy array (num_frames, feature_dim)."""
    dev = next(backbone.parameters()).device     # CPU, unless the caller moved the backbone (extract_features does on a GPU)
    out = []
    for i in range(0, len(frames_rgb), batch_size):
        out.append(backbone(frames_to_tensor(frames_rgb[i:i + batch_size]).to(dev)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------------
# Temporal heads: input (batch, time, feature_dim) -> output (batch, num_classes)
# ----------------------------------------------------------------------------------
def add_motion(x, dstd):
    """(batch, time, d) standardised features -> (batch, time, 2d): the features plus how much each one
    CHANGED since the previous frame (zero for the first frame), scaled by dstd. Typing vs sitting still,
    or writing vs resting the head, differ mostly in this change, which single frames hardly show."""
    d = torch.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    return torch.cat([x, d / dstd], dim=2)


class GRUHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=128, layers=1, dropout=0.4, noise=0.1, motion=False):
        super().__init__()
        # Feature mean/std are filled in by train.py and saved with the model.
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.register_buffer("dstd", torch.ones(in_dim))      # std of the frame-to-frame change (motion=True)
        self.noise, self.motion = noise, bool(motion)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim * (2 if motion else 1), hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.motion:
            x = add_motion(x, self.dstd)
        if self.training and self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        x = self.proj(x)
        out, _ = self.gru(x)                 # (batch, time, hidden)
        return self.fc(self.drop(out.mean(dim=1)))


class TCNHead(nn.Module):
    """1-D convolutions over time (kernel 3, dilation 1, 2, 4, ...)."""

    def __init__(self, in_dim, num_classes, hidden=128, layers=2, dropout=0.4, noise=0.1, motion=False):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.register_buffer("dstd", torch.ones(in_dim))
        self.noise, self.motion = noise, bool(motion)
        blocks, channels = [], in_dim * (2 if motion else 1)
        for i in range(layers):
            d = 2 ** i
            blocks += [nn.Conv1d(channels, hidden, kernel_size=3, padding=d, dilation=d),
                       nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            channels = hidden
        self.net = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.motion:
            x = add_motion(x, self.dstd)
        if self.training and self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        x = self.net(x.transpose(1, 2))      # (batch, channels, time)
        return self.fc(x.mean(dim=2))


def build_head(kind, in_dim, num_classes, hidden, layers, dropout, motion=False):
    if kind == "gru":
        return GRUHead(in_dim, num_classes, hidden, layers, dropout, motion=motion)
    if kind == "tcn":
        return TCNHead(in_dim, num_classes, hidden, layers, dropout, motion=motion)
    raise ValueError(f"Unknown head '{kind}'. Use 'gru' or 'tcn'.")
