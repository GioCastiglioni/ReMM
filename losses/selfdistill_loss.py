"""The three self-distillation objectives behind one interface.

Every objective sees the same pairs as CoMM: mask `m` of one view against the
*joint* embedding of the other view. The only change from `WoMMLoss` is that the
joint side comes from an EMA teacher instead of the online network, and that there
is no explicit regularisation term -- collapse is meant to be prevented by the
teacher/student asymmetry alone. That is what makes this family a different point
in the design space rather than another regulariser.

By default each unimodal-joint pair runs in one direction only, the unimodal student
predicting the joint teacher. With `bidirectional` outputs the teacher also encodes
the unimodal masks, and the joint student predicts them too, so every pair is used in
both roles as CoMM's symmetric InfoNCE uses it in both directions. Dropping
`unimodal_to_joint` on top keeps only the joint student's predictions: a unimodal
branch cannot know what the joint teacher holds about the other modality, and with
no negatives to compensate that target only pulls the representation towards a
constant.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class MMSelfDistillLoss(nn.Module):
    OBJECTIVES = ("byol", "dino")

    def __init__(self,
                 objective: str = "byol",
                 weights: Optional[List[float]] = None,
                 # dino
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.07,
                 warmup_teacher_temp: float = 0.04,
                 warmup_teacher_temp_frac: float = 0.3,
                 center_momentum: float = 0.9,
                 out_dim: int = 2048,
                 n_centers: int = 1):
        super().__init__()
        if objective not in self.OBJECTIVES:
            raise ValueError(f"Unknown objective {objective!r}, expected one of {self.OBJECTIVES}")
        self.objective = objective
        self.weights = weights
        self.student_temp = student_temp
        self.teacher_temp_final = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_frac = warmup_teacher_temp_frac
        self.center_momentum = center_momentum
        # One DINO center per teacher mask: unimodal and joint teacher outputs have
        # different means, and a shared center would leave each of them off-centre.
        self.register_buffer("center", torch.zeros(n_centers, out_dim))
        self.register_buffer("teacher_temp", torch.tensor(float(warmup_teacher_temp)))

    # ---- schedules -------------------------------------------------------
    def step(self, epoch: int, max_epochs: int):
        """Linear warmup of the teacher temperature, as in DINO.

        Starting sharp makes the teacher confident before it is any good, which is
        the reported route to collapse in DINO.
        """
        if self.objective != "dino":
            return
        n = max(1, int(round(max_epochs * self.warmup_teacher_temp_frac)))
        t = min(1.0, float(epoch) / n)
        self.teacher_temp.fill_(self.warmup_teacher_temp
                                + t * (self.teacher_temp_final - self.warmup_teacher_temp))

    # ---- per-objective discrepancies -------------------------------------
    def _byol(self, student: torch.Tensor, teacher: torch.Tensor, k: int = 0) -> torch.Tensor:
        s = F.normalize(student, dim=-1, p=2)
        t = F.normalize(teacher, dim=-1, p=2)
        return (2.0 - 2.0 * (s * t).sum(dim=-1)).mean()

    def _dino(self, student: torch.Tensor, teacher: torch.Tensor, k: int = 0) -> torch.Tensor:
        t = F.softmax((teacher - self.center[k:k + 1]) / self.teacher_temp, dim=-1)
        s = F.log_softmax(student / self.student_temp, dim=-1)
        return -(t * s).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher: torch.Tensor, k: int = 0):
        if self.objective != "dino":
            return
        batch_center = teacher.mean(dim=0, keepdim=True)
        self.center[k:k + 1].mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum)

    # ---- diagnostics -----------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _effective_rank(z: torch.Tensor) -> torch.Tensor:
        """exp(entropy of the normalised singular values).

        The collapse metric for this family: there is no regularisation term whose
        value would reveal a shrinking representation, so it has to be measured.
        """
        z = z.float()
        z = z - z.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(z)
        p = s / s.sum().clamp_min(1e-12)
        p = p.clamp_min(1e-12)
        return torch.exp(-(p * p.log()).sum())

    # ---- forward ---------------------------------------------------------
    def forward(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        return self._forward_pooled(outputs)

    def _weights(self, n: int, device) -> Optional[torch.Tensor]:
        if self.weights is None:
            return None
        if len(self.weights) != n:
            raise ValueError(f"`weights` has {len(self.weights)} entries for {n} masks")
        return torch.tensor(self.weights, device=device, dtype=torch.float32)

    def _forward_pooled(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        student, teacher = outputs["student"], outputs["teacher"]
        n_mask = len(student[0])
        d = self._byol if self.objective == "byol" else self._dino
        # DINO's rule that a student crop is never matched to the teacher's view of
        # the same crop is what `cross_view` implements here
        cv = bool(outputs.get("cross_view", True))
        bidir = bool(outputs.get("bidirectional", False))
        to_joint = bool(outputs.get("unimodal_to_joint", True))
        joint = n_mask - 1
        other = (lambda v: 1 - v) if cv else (lambda v: v)
        # Teacher output of view v under mask i, and the DINO center it is read against:
        # a single joint output and center unless the teacher encodes every mask.
        tch = (lambda v, i: teacher[v][i]) if bidir else (lambda v, i: teacher[v])
        ctr = (lambda i: i) if bidir else (lambda i: 0)
        per_mask, forward, backward = [], {}, {}
        for i in range(n_mask):
            fwd = 0.5 * (d(student[0][i], tch(other(0), joint), ctr(joint))
                         + d(student[1][i], tch(other(1), joint), ctr(joint)))
            if i == joint:
                per_mask.append(fwd)
                continue
            terms = [fwd] if to_joint else []
            if bidir:
                bwd = 0.5 * (d(student[0][joint], tch(other(0), i), ctr(i))
                             + d(student[1][joint], tch(other(1), i), ctr(i)))
                forward[i], backward[i] = fwd.detach(), bwd
                terms.append(bwd)
            per_mask.append(torch.stack(terms).mean())
        per_mask = torch.stack(per_mask)
        w = self._weights(n_mask, per_mask.device)
        loss = torch.mean(per_mask * w) if w is not None else per_mask.mean()

        out = {"loss": loss}
        out.update({f"sd_loss_{i}": v for i, v in enumerate(per_mask)})
        out.update({f"sd_loss_{i}_to_joint": v for i, v in forward.items()})
        out.update({f"sd_loss_joint_to_{i}": v for i, v in backward.items()})
        with torch.no_grad():
            out["teacher_rank"] = self._effective_rank(tch(0, joint))
            out["student_rank"] = self._effective_rank(student[0][-1])
            if self.objective == "dino":
                p = F.softmax((tch(0, joint) - self.center[ctr(joint):ctr(joint) + 1])
                              / self.teacher_temp, dim=-1)
                out["teacher_entropy"] = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
                out["teacher_pmax"] = p.max(dim=-1).values.mean()
                out["teacher_temp_now"] = self.teacher_temp.clone()
        for i in (range(n_mask) if bidir else [joint]):
            self.update_center(torch.cat([tch(0, i), tch(1, i)], dim=0), ctr(i))
        return out
