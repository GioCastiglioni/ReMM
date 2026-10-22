"""Multimodal self-distillation: MM-BYOL and MM-DINO on one skeleton.

Read this as WoMM with two changes and nothing else:

  1. the joint (prototype) side of every pair comes from an EMA teacher instead of
     the online network, so the target is stale and detached by construction;
  2. there is no explicit regulariser -- no negatives, no variance term. Collapse
     is meant to be prevented by the teacher/student asymmetry alone.

The pairs themselves are untouched: mask `m` of view 1 against the joint of view 2,
mask `m` of view 2 against the joint of view 1, and joint against joint. That keeps
the probes, the mask structure and the biased/unbiased arms directly comparable to
the contrastive and variance families. With `bidirectional` the teacher also encodes
the unimodal masks and each unimodal-joint pair is used in both roles, as CoMM's
symmetric InfoNCE does.
"""
import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from losses.selfdistill_loss import MMSelfDistillLoss
from models.mmfusion import MMFusion
from models.selfdistill import BYOLPredictor, DINOHead
from pl_modules.base import BaseModel
from utils import LinearWarmupCosineAnnealingLR, set_weight_decay_per_param


class MMSD(BaseModel):
    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict,
                 ema_base: float = 0.996,
                 ema_final: float = 1.0,
                 predictor_kwargs: Optional[Dict] = None,
                 cross_view: Optional[bool] = None,
                 lr_warmup_frac: float = 0.1,
                 wd_mult_end: Optional[float] = 10.0,
                 freeze_last_layer_epochs: int = 1,
                 bidirectional: bool = False,
                 unimodal_to_joint: bool = True):
        """
        Args:
            encoder: multimodal fusion encoder (the online / student network).
            projection: MLP projector to the latent space.
            optim_kwargs: optimisation hyper-parameters.
            loss_kwargs: given to `MMSelfDistillLoss`; `objective` selects the method.
            ema_base / ema_final: teacher momentum, cosine-ramped between them.
            predictor_kwargs: for the BYOL predictor.
            cross_view: whether the student of view v is matched against the teacher
                on view 1-v. Defaults to True, which is what BYOL and DINO do: both
                are two-view methods.
            lr_warmup_frac: fraction of `max_epochs` spent linearly warming the
                learning rate up to the value the train config sets, after which it
                is cosine-annealed. The peak is unchanged.
            wd_mult_end: weight decay is cosine-ramped from `optim_kwargs`'s value to
                that value times this factor. DINO's 0.04 -> 0.4 is a 10x rise; the
                factor transfers across architectures where the absolutes do not.
                None keeps weight decay flat. Read by DINO only -- BYOL's paper
                specifies a constant decay, so it keeps `optim_kwargs`'s value.
            freeze_last_layer_epochs: epochs during which the DINO prototype layer
                receives no gradient. Ignored by the other objectives.
            bidirectional: the teacher encodes every mask, not only the joint one, and
                the joint student also predicts each unimodal teacher output.
            unimodal_to_joint: keep the unimodal student's predictions of the joint
                teacher. False needs `bidirectional`, which then leaves the joint
                student's predictions (of each unimodal and of the joint teacher) only.
        """
        super(MMSD, self).__init__(optim_kwargs)

        self.encoder = encoder
        self.head = projection

        loss_kwargs = dict(loss_kwargs)
        self.objective = loss_kwargs.get("objective", "byol")
        self.bidirectional = bool(bidirectional)
        self.unimodal_to_joint = bool(unimodal_to_joint)
        if not (self.bidirectional or self.unimodal_to_joint):
            raise ValueError("unimodal_to_joint=False needs bidirectional=True: no unimodal pair would remain")
        n_centers = encoder.num_modalities + 1 if self.bidirectional else 1
        self.loss = MMSelfDistillLoss(**loss_kwargs, n_centers=n_centers)

        dim = self._head_out_dim(projection)
        pk = dict(predictor_kwargs or {})
        if self.objective == "byol":
            self.predictor = BYOLPredictor(dim, **pk)
        elif self.objective == "dino":
            # DINO has no predictor: the asymmetry is centering + sharpening. The
            # prototype layer is shared, so it must live on both sides and is
            # therefore part of the EMA-copied stack.
            self.predictor = nn.Identity()
            self.head = nn.Sequential(OrderedDict([
                ("proj", projection),
                ("proto", DINOHead(dim, out_dim=loss_kwargs.get("out_dim", 2048), **pk)),
            ]))
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        self.cross_view = True if cross_view is None else bool(cross_view)

        # the teacher: a frozen copy updated only by EMA
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_head = copy.deepcopy(self.head)
        for p in list(self.teacher_encoder.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False

        self.ema_base = ema_base
        self.ema_final = ema_final
        self._ema_now = ema_base

        self.lr_warmup_frac = lr_warmup_frac
        self.wd_mult_end = wd_mult_end
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

    def configure_optimizers(self):
        """Warmup + cosine on top of the train config's learning rate, which stays the peak."""
        optimizer = torch.optim.AdamW(
            set_weight_decay_per_param(self, weight_decay=self.optim_kwargs["weight_decay"]),
            lr=self.optim_kwargs["lr"])
        max_epochs = self.trainer.max_epochs
        warmup = max(1, int(round(max_epochs * self.lr_warmup_frac)))
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup, max_epochs=max_epochs,
            warmup_start_lr=self.optim_kwargs["lr"] * 1e-3, eta_min=0.0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def _set_weight_decay(self):
        """DINO's cosine ramp of weight decay, on the decayed parameter group only.

        `set_weight_decay_per_param` puts biases and norms in a second group at 0,
        which must stay at 0.

        DINO-only on purpose: the 0.04 -> 0.4 ramp is part of DINO's recipe, while
        BYOL specifies a small constant decay. Each objective runs as its own paper
        states, so the comparison is between the asymmetry mechanisms and not
        between one method and another method's schedule.
        """
        if self.objective != "dino" or self.wd_mult_end is None:
            return
        wd_start = self.optim_kwargs["weight_decay"]
        wd_end = wd_start * self.wd_mult_end
        denom = max(1, self.trainer.max_epochs - 1)
        t = min(1.0, self.current_epoch / denom)
        wd = wd_end + 0.5 * (wd_start - wd_end) * (1 + math.cos(math.pi * t))
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                if group.get("weight_decay", 0) > 0:
                    group["weight_decay"] = wd
        self.log("optim/weight_decay", wd, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._set_weight_decay()

    def on_after_backward(self):
        """DINO's freeze_last_layer: the prototype layer gets no gradient early on."""
        if self.objective != "dino" or self.current_epoch >= self.freeze_last_layer_epochs:
            return
        for param in self.head.proto.last_layer.parameters():
            if param.grad is not None:
                param.grad.zero_()

    @staticmethod
    def _head_out_dim(projection: nn.Module) -> int:
        for m in reversed(list(projection.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
        raise ValueError("Cannot infer the projector output width")

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def gen_all_possible_masks(self, n_mod: int) -> List[List[bool]]:
        """`n_mod` single-modality masks, then the joint one last."""
        masks = [[s == L for s in range(n_mod)] for L in range(n_mod)]
        masks.append([True] * n_mod)
        return masks

    # ---- teacher ---------------------------------------------------------
    def _ema_momentum(self) -> float:
        # `self.trainer` raises when unattached, so go through the private handle
        trainer = getattr(self, "_trainer", None)
        total = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
        if not total or not math.isfinite(total) or total <= 0:
            return self.ema_base
        t = min(1.0, float(self.global_step) / float(total))
        return self.ema_final - (self.ema_final - self.ema_base) * (math.cos(math.pi * t) + 1) / 2

    @torch.no_grad()
    def update_teacher(self):
        m = self._ema_momentum()
        self._ema_now = m
        for online, target in ((self.encoder, self.teacher_encoder),
                               (self.head, self.teacher_head)):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.copy_(bo)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher()
        self.log("ema_momentum", self._ema_now, on_step=True, sync_dist=False)

    # ---- forward ---------------------------------------------------------
    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]) -> Dict:
        return self._forward_pooled(x1, x2)

    def _forward_pooled(self, x1, x2) -> Dict:
        n_mod = len(x1)
        masks = self.gen_all_possible_masks(n_mod)
        joint = [[True] * n_mod]

        student = []
        for x in (x1, x2):
            z = self.encoder(x, mask_modalities=masks)
            student.append([self.predictor(self.head(zi)) for zi in z])

        teacher = []
        with torch.no_grad():
            for x in (x1, x2):
                if self.bidirectional:
                    z = self.teacher_encoder(x, mask_modalities=masks)
                    teacher.append([self.teacher_head(zi).detach() for zi in z])
                else:
                    z = self.teacher_encoder(x, mask_modalities=joint)
                    teacher.append(self.teacher_head(z[0]).detach())

        return {"student": student, "teacher": teacher, "cross_view": self.cross_view,
                "bidirectional": self.bidirectional, "unimodal_to_joint": self.unimodal_to_joint,
                "prototype": -1}

    # ---- probing ---------------------------------------------------------
    # Which network each original method evaluates, and why they differ: the probed
    # network is the one that sees the complete input. Both of BYOL's views are full
    # images, so it keeps the online encoder. DINO's student also gets local crops,
    # so it evaluates the teacher. In this skeleton the modality masks play the same
    # role: the student runs on all M+1 masks, the teacher only on the joint one.
    PROBE_NETWORK = {"byol": "student", "dino": "teacher"}

    @property
    def probe_encoder(self) -> nn.Module:
        """The encoder the probes read, following each method's own protocol."""
        if self.PROBE_NETWORK[self.objective] == "student":
            return self.encoder
        return self.teacher_encoder

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """Pooled output of `probe_encoder`, never the projector or the predictor."""
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                output = self.probe_encoder(X_, **kwargs)
                if isinstance(output, list):
                    output = output[0]
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)
