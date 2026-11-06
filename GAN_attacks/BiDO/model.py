# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import torch.nn as nn
import sys
sys.path.append('../..')
from facenet_pytorch import InceptionResnetV1

####################################### Unprotected model #############################

class InceptionResnetV1Unprotected(InceptionResnetV1):
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__(pretrained, classify, num_classes, dropout_prob, device)
        # Classification layer remains 512-dim
        if device is not None:
            self.to(self.device)

    def forward(self, x, iden=None):
        """Wrapper to include optional iden argument. For compatibility with other models only 
        """
        return super().forward(x)

####################################### PIPE FE (unified wrapper) #############################

class InceptionResnetV1PipeFE(InceptionResnetV1):
    """Facenet backbone (unnormalised last_bn, 512-d) followed by one of the
    PIPE_attack/datagen fuzzy-extractor procedures, selected by `scheme`:
    unprotected, facialfe_e8, facialfe_leech, facialfe_leech_randomised, ourfe.

    Mirrors Bob_attack/experiments/model.py: the FE step is the numpy / ctypes(Leech C lib)
    port in Bob_attack/experiments/pipe_fe_transforms.py, which imports leech_helper and the
    PIPE datagen scale constants directly (no FE logic is duplicated here). The FE step is
    non-differentiable, so it runs on the CPU under no_grad -- exactly like the original
    FacialFE transform. The seeded schemes (facialfe_leech_randomised, ourfe) need the image
    name, passed through as img_id.
    """
    def __init__(self, scheme, sameR=True, pretrained=None, classify=False,
                 num_classes=None, dropout_prob=0.6, device=None, dataset='celeba', centered=True,
                 differentiable=False):
        super().__init__(pretrained, classify, num_classes, dropout_prob, device)
        self.scheme = scheme
        self.dataset = dataset
        self.centered = centered
        self.differentiable = differentiable
        # pipe_fe_transforms lives in Bob_attack/experiments (same dir as Bob's model.py). It
        # self-locates leech_helper + the PIPE datasets from its OWN path, so we only need it on
        # sys.path. From GAN_attacks/BiDO/model.py that is two levels up + Bob_attack/experiments.
        _helper = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..',
            'Bob_attack', 'experiments'))
        if _helper not in sys.path:
            sys.path.insert(0, _helper)
        from pipe_fe_transforms import get_transform
        # The seeded schemes derive their codeword/keys from the image filename (per-image
        # randomness): per_image_seed=True is the only setting used here.
        self.fe = get_transform(scheme, dataset=dataset, centered=centered, per_image_seed=True)
        # Differentiable (straight-through) FE for attacks: forward stays bit-exact (the trained
        # head sees identical features), backward gives a usable gradient to z. See pipe_fe_torch.
        # ourfe is decode-first with a one-way keyed-hash helper and has NO clean surrogate ->
        # get_torch_transform raises NotImplementedError. Rather than crash, fall back to the
        # exact non-differentiable numpy FE: the attack still runs, but the identity loss carries
        # no gradient to z (z is driven by the GAN prior only), which honestly reflects that the
        # defense leaves no differentiable path to invert.
        self.fe_torch = None
        if differentiable:
            from pipe_fe_torch import get_torch_transform
            try:
                self.fe_torch = get_torch_transform(scheme, dataset=dataset, centered=centered, per_image_seed=True)
            except NotImplementedError as e:
                print(f"[PipeFE] no differentiable surrogate for '{scheme}' -> "
                      f"non-differentiable FE (identity loss will not backprop to z). {e}")
                self.differentiable = False
        # The FE output dim is scheme-dependent (e.g. 504 for the leech schemes, 512 unprotected,
        # 42 ourfe), so the inherited Linear(512 -> num_classes) head cannot consume it.
        # Probe the transform once to size a fresh classifier head to the FE output. The FE step
        # is non-differentiable (numpy under no_grad), so only this head trains.
        if classify and num_classes is not None:
            _probe = np.asarray(self.fe(np.zeros(512, dtype=np.float64), None), dtype=np.float64).flatten()
            self.fe_out_dim = int(_probe.shape[0])
            self.logits = nn.Linear(self.fe_out_dim, num_classes)
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
        if self.differentiable:
            # STE torch FE: forward bit-exact, backward carries a gradient to x (and thus to z).
            out = self.fe_torch(x, img_id)
        else:
            with torch.no_grad():
                emb = x.detach().cpu().numpy().astype(np.float64)
                names = img_id if img_id is not None else [None] * emb.shape[0]
                out = np.stack([np.asarray(self.fe(emb[i], names[i]), dtype=np.float64)
                                for i in range(emb.shape[0])])
                out = torch.from_numpy(out).to(torch.float32).to(self.device)
        # out: (bs, fe_out_dim) protected helper. Apply the classifier head if requested.
        # (out is detached from the backbone via the numpy round-trip, so only logits trains.)
        if self.classify:
            out = self.logits(out)
        return out