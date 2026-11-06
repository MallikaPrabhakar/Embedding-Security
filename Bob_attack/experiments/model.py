# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch

# facenet_pytorch is a local directory at the project root -> needs root on sys.path.
# Prefer PROJECT_ROOT from a .env (if python-dotenv is available), else derive it from
# this file's location (Bob_attack/experiments/model.py -> repo root is 2 levels up).
try:
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv())
except ModuleNotFoundError:
    pass
PROJECT_ROOT = os.getenv('PROJECT_ROOT') or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(PROJECT_ROOT)

from facenet_pytorch import InceptionResnetV1


class InceptionResnetV1PipeFE(InceptionResnetV1):
    """Facenet backbone (unnormalised last_bn, 512-d) followed by one of the
    PIPE_attack/datagen fuzzy-extractor procedures. This is the single wrapper for
    ALL schemes -- unprotected, facialfe_e8, facialfe_leech, facialfe_leech_randomised,
    ourfe (our defense) -- selected by `scheme`.

    By default the FE step is the DIFFERENTIABLE straight-through version (pipe_fe_torch):
    the forward value is the bit-exact numpy FE (so saved data / the trained attacker see
    the exact protected helper), while the backward pass uses an analytic surrogate so the
    ID loss / latent optimisation can backprop through the FE to the backbone. ourfe
    (no clean surrogate) falls back to the non-differentiable numpy FE under no_grad
    (self.fe_differentiable is False). The seeded schemes (facialfe_leech_randomised,
    ourfe) need the image name, passed through as img_id.
    """
    def __init__(self, scheme, same_r='none', pretrained=None, classify=False,
                 num_classes=None, dropout_prob=0.6, device=None, dataset='celeba', centered=False):
        super().__init__(pretrained, classify, num_classes, dropout_prob, device)
        self.scheme = scheme
        self.same_r = same_r
        self.dataset = dataset
        self.centered = centered
        _here = os.path.dirname(os.path.abspath(__file__))  # Bob_attack/experiments (holds pipe_fe_*.py)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        # Prefer the DIFFERENTIABLE straight-through FE (pipe_fe_torch): forward value is the
        # bit-exact numpy FE, backward uses an analytic surrogate so gradients reach the backbone
        # (needed for ID-loss / latent-optimisation attacks). ourfe has no clean surrogate ->
        # fall back to the non-differentiable numpy FE under no_grad.
        from pipe_fe_transforms import get_transform
        try:
            from pipe_fe_torch import get_torch_transform
            self.fe = get_torch_transform(scheme, dataset=dataset, centered=centered)
            self.fe_differentiable = True
        except (ImportError, NotImplementedError):
            self.fe = get_transform(scheme, dataset=dataset, centered=centered)
            self.fe_differentiable = False
        if device is not None:
            self.to(self.device)

    def forward(self, x, img_id=None):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        # x: (bs, 512) unnormalised last_bn == PIPE unprotected embedding
        if self.fe_differentiable:
            # torch straight-through FE: forward == exact numpy FE (bit-identical), backward
            # via the analytic surrogate -> gradients flow back through the backbone.
            out = self.fe(x, img_id).to(torch.float32)
        else:
            with torch.no_grad():
                emb = x.detach().cpu().numpy().astype(np.float64)
                names = img_id if img_id is not None else [None] * emb.shape[0]
                out = np.stack([np.asarray(self.fe(emb[i], names[i]), dtype=np.float64)
                                for i in range(emb.shape[0])])
                out = torch.from_numpy(out).to(torch.float32).to(self.device)
        return out
