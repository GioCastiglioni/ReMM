import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributed as dist
from utils import all_gather_batch_with_grad

def all_reduce(tensor, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        reduce_op = dist.ReduceOp.MAX if op == "MAX" else dist.ReduceOp.AVG
        dist.all_reduce(tensor, op=reduce_op)
    return tensor

class WoMMLoss(nn.Module):
    NEG_SAMPLE_REGS = ('neg-samples', 'sem-aware', 'neg-samples-multi', 'neg-samples-dcl',
                       'neg-samples-dcl-cross', 'neg-samples-cross')
    PER_EMBED_NEG_REGS = ('neg-samples', 'sem-aware', 'neg-samples-dcl',
                          'neg-samples-dcl-cross', 'neg-samples-cross')
    PROTO_ONLY_REGS = ('sigreg-multi', 'visreg-multi')
    PER_MASK_REGS = ('sigreg-permask',)
    # name -> (mode, gaussian regularizer, contrastive regularizer), combined as
    # 0.5 * ((1 - mix_rho) * gaussian + mix_rho * contrastive).
    #   mix  : both act on the whole embedding.
    #   split: the contrastive one sees the first half of the coordinates and the
    #          gaussian one the second half. With the MSE alignment averaging over
    #          coordinates, each half is then its own single-regularizer objective.
    COMPOSITE_REGS = {
        'mix-sigreg-neg-samples': ('mix', 'sigreg', 'neg-samples'),
        'split-sigreg-neg-samples': ('split', 'sigreg', 'neg-samples'),
        'mix-sigreg-neg-samples-cross': ('mix', 'sigreg', 'neg-samples-cross'),
        'split-sigreg-neg-samples-cross': ('split', 'sigreg', 'neg-samples-cross'),
    }
    SEM_AWARE_MARGIN = 0.5

    def __init__(
        self, 
        weights=None, 
        reconstruction='mse', 
        regularization='sigreg', 
        temperature=0.1,
        sigma_max=2.0, 
        sigma_min=0.5, 
        reg_weight=0.05,
        mix_rho=0.95,
        K=4096,
        stop_grad=False,
        vicreg_inv_coeff=25.0,
        vicreg_std_coeff=25.0,
        vicreg_cov_coeff=1.0,
        use_geco=False,
        geco_warmup_frac=0.25,
        geco_kappa_calib_frac=0.05,
        geco_kappa_ramp_frac=0.10,
        geco_kappa_mode='reference',
        geco_kappa_slack=0.05,
        geco_kappa_gap_frac=0.5,
        geco_kappa_reference_trials=16,
        geco_tolerance_margin=0.95,
        geco_response_per_epoch=1.10,
        geco_max_lambda_change_per_epoch=1.25,
        geco_ema_halflife_epochs=3.0,
        geco_deadband_frac=0.25,
        geco_lambda_range=50.0,
        geco_updates_per_epoch=10,
        sim_short_halflife_epochs=1.0,
        sim_long_halflife_epochs=6.0,
        geco_degradation_sigmas=2.0,
        steps_per_epoch=None,
    ):
        super().__init__()
        self.weights = weights
        self.reconstruction = reconstruction
        self.regularization = regularization
        self.temperature = temperature
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma = sigma_max
        self.reg_weight = reg_weight
        self.mix_rho = mix_rho
        if regularization in self.COMPOSITE_REGS:
            if reconstruction == 'cosine':
                raise ValueError(f"{regularization} needs an unnormalized alignment: a gaussian "
                                 "regularizer cannot be satisfied by unit vectors")
            if not 0.0 <= mix_rho <= 1.0:
                raise ValueError(f"mix_rho must lie in [0, 1], got {mix_rho}")
        self.stop_grad = stop_grad
        self.K = K
        self.INF = 1e8
        
        self.vicreg_std_coeff = vicreg_std_coeff
        self.vicreg_cov_coeff = vicreg_cov_coeff
        self.vicreg_inv_coeff = vicreg_inv_coeff
        
        self.use_geco = use_geco
        self.geco_warmup_frac = geco_warmup_frac
        self.geco_kappa_calib_frac = geco_kappa_calib_frac
        self.geco_kappa_ramp_frac = geco_kappa_ramp_frac
        self.geco_kappa_mode = geco_kappa_mode
        self.geco_kappa_slack = geco_kappa_slack
        self.geco_kappa_gap_frac = geco_kappa_gap_frac
        self.geco_kappa_reference_trials = geco_kappa_reference_trials
        self.geco_tolerance_margin = geco_tolerance_margin
        self.geco_response_per_epoch = geco_response_per_epoch
        self.geco_max_lambda_change_per_epoch = geco_max_lambda_change_per_epoch
        self.geco_ema_halflife_epochs = geco_ema_halflife_epochs
        self.geco_deadband_frac = geco_deadband_frac
        self.geco_lambda_range = geco_lambda_range
        self.geco_updates_per_epoch = geco_updates_per_epoch
        self.sim_short_halflife_epochs = sim_short_halflife_epochs
        self.sim_long_halflife_epochs = sim_long_halflife_epochs
        self.geco_degradation_sigmas = geco_degradation_sigmas

        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('lagrange_lambda', torch.tensor(reg_weight, dtype=torch.float32))
        self.register_buffer('constraint_ma', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('constraint_ma_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('kappa_start', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa_target', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa_reference', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa_anchor', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa_ref_init', torch.tensor(False, dtype=torch.bool))   # attempted
        self.register_buffer('kappa_ref_valid', torch.tensor(False, dtype=torch.bool))  # usable
        self.register_buffer('geco_initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('geco_epoch0', torch.tensor(0, dtype=torch.long))
        self.register_buffer('reg_calib_sum', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('reg_calib_count', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('sim_ema_short', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('sim_ema_long', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('sim_sq_dev_ema', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('sim_ema_count', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('sim_ema_init', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('sim_trend_z', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('brake_level', torch.tensor(0, dtype=torch.long))
        self.register_buffer('geco_factor', torch.tensor(1.0, dtype=torch.float32))

        self._schedule_ready = False
        self.steps_per_epoch = None
        self.total_epochs = None
        if steps_per_epoch:
            self.configure_schedule(steps_per_epoch)

        self._cached_B = -1
        self._cached_target = None

        gaussian = self.COMPOSITE_REGS[regularization][1] if regularization in self.COMPOSITE_REGS \
            else regularization
        if gaussian in ('sigreg', 'sigreg-multi', 'sigreg-permask'):
            self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=self.K)

    def configure_schedule(self, steps_per_epoch: int, total_epochs: int = 100):
        """Derive every step-level GECO constant from the run's own shape.

        Nothing here is a tuned hyper-parameter: the phase lengths come from
        total_epochs and the time constants from steps_per_epoch, so one config
        transfers across datasets with very different budgets and epoch sizes
        (Sen1Floods11: 3 steps/epoch, CREMA-D: 89). Idempotent.
        """
        S = max(1, int(steps_per_epoch))
        E = max(1, int(total_epochs))
        if self._schedule_ready and self.steps_per_epoch == S and self.total_epochs == E:
            return

        self.steps_per_epoch = S
        self.total_epochs = E

        # phase lengths as fractions of the budget, not absolute epoch counts
        self.warmup_epochs = max(1, int(round(self.geco_warmup_frac * E)))
        self.kappa_calib_epochs = max(1, int(round(self.geco_kappa_calib_frac * E)))
        self.kappa_ramp_epochs = max(1, int(round(self.geco_kappa_ramp_frac * E)))
        U = max(1, int(self.geco_updates_per_epoch))
        self.lbd_step = max(1, int(round(S / U)))
        self.u_eff = S / self.lbd_step

        self.alpha_c = 0.5 ** (1.0 / max(1e-6, self.geco_ema_halflife_epochs * S))
        self.alpha_short = 0.5 ** (1.0 / max(1e-6, self.sim_short_halflife_epochs * self.u_eff))
        self.alpha_long = 0.5 ** (1.0 / max(1e-6, self.sim_long_halflife_epochs * self.u_eff))

        self.margin_gap = max(1e-6, 1.0 - self.geco_tolerance_margin)
        self.log_f_max = math.log(self.geco_max_lambda_change_per_epoch) / self.u_eff
        self.gain = math.log(self.geco_response_per_epoch) / (self.u_eff * self.margin_gap)
        self.deadband = self.geco_deadband_frac * self.margin_gap

        base = max(abs(float(self.reg_weight)), 1e-8)
        self.lambda_min = base / self.geco_lambda_range
        self.lambda_max = base * self.geco_lambda_range

        # the trend brake needs enough samples before its running std is meaningful
        self.sim_brake_min_updates = max(4.0, self.sim_long_halflife_epochs * self.u_eff)

        self._schedule_ready = True
        if self.use_geco:
            print(
                f"[GECO] steps_per_epoch={S} epochs={E} warmup={self.warmup_epochs} calib={self.kappa_calib_epochs} ramp={self.kappa_ramp_epochs} lbd_step={self.lbd_step} updates/epoch={self.u_eff:.2f} "
                f"alpha_c={self.alpha_c:.5f} f_max={math.exp(self.log_f_max):.5f} gain={self.gain:.4f} "
                f"deadband={self.deadband:.5f} lambda=[{self.lambda_min:.5g}, {self.lambda_max:.5g}]",
                flush=True,
            )

    def step(self, current_epoch, total_epochs):
        self.current_epoch.fill_(current_epoch)
        if self.reconstruction == 'rbf':
            self.sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * current_epoch / total_epochs))

    @torch.no_grad()
    def _reference_samples(self, shape, device, dtype, normalize):
        z = torch.randn(*shape, device=device, dtype=dtype)
        return func.normalize(z, p=2, dim=-1) if normalize else z

    @torch.no_grad()
    def calibrate_kappa_reference(self, n_emb, B, D, device, dtype):
        """Score the regularizer on samples drawn from the distribution it rewards.

        Each regularizer here is a statistic with a known optimum: sigreg and visreg
        measure departure from an isotropic Gaussian, vicreg from a decorrelated
        unit-variance one, and the InfoNCE-style terms from uniformity on the sphere.
        Evaluating them on actual samples of that distribution, at the shapes seen in
        training, yields the achievable floor -- finite-batch bias included -- without
        any reference to reg_weight. Returns None when the combination has no
        well-defined target (the caller then falls back to warmup calibration).

        The InfoNCE terms are always scored positive-free, because that is the
        quantity the controller constrains: on two independent draws the positive is
        just another negative, so the coupled value would instead measure the floor
        at zero alignment -- log(2N-2) + 1/(2 T^2 D) plus the e^0 of an unaligned
        pair -- and asking training to reach it means asking it to unlearn.
        """
        reg = self.regularization
        trials = max(1, int(self.geco_kappa_reference_trials))
        if reg in self.NEG_SAMPLE_REGS and self.reconstruction != 'cosine':
            return None
        if reg not in ('sigreg', 'visreg', 'vicreg') + self.NEG_SAMPLE_REGS \
                + self.PROTO_ONLY_REGS + self.PER_MASK_REGS:
            return None

        # the slicing seed must not advance: it would desynchronise the training draws
        saved_step = self.sigreg.global_step.clone() \
            if reg in ('sigreg', 'sigreg-multi', 'sigreg-permask') else None
        vals = []
        for _ in range(trials):
            if reg == 'sigreg':
                vals.append(self.sigreg(self._reference_samples((2 * n_emb * B, D), device, dtype, False)))
            elif reg in ('sigreg-multi', 'sigreg-permask'):
                vals.append(self.sigreg(self._reference_samples((2 * B, D), device, dtype, False)))
            elif reg == 'visreg-multi':
                vals.append(0.5 * (self.forward_visreg(self._reference_samples((1, B, D), device, dtype, False))
                                   + self.forward_visreg(self._reference_samples((1, B, D), device, dtype, False))))
            elif reg == 'visreg':
                vals.append(0.5 * (self.forward_visreg(self._reference_samples((n_emb, B, D), device, dtype, False))
                                   + self.forward_visreg(self._reference_samples((n_emb, B, D), device, dtype, False))))
            elif reg == 'vicreg':
                vals.append(0.5 * (self.forward_vicreg(self._reference_samples((n_emb * B, D), device, dtype, False))
                                   + self.forward_vicreg(self._reference_samples((n_emb * B, D), device, dtype, False))))
            else:
                margin = self.SEM_AWARE_MARGIN if reg == 'sem-aware' else None
                vals.append(self._neg_samples_reduce(
                    self._neg_samples_matrix(
                        self._reference_samples((B, D), device, dtype, True),
                        self._reference_samples((B, D), device, dtype, True),
                        margin, reg in ('neg-samples-dcl-cross', 'neg-samples-cross')),
                    True))
        if saved_step is not None:
            self.sigreg.global_step.copy_(saved_step)
        return torch.stack(vals).float().mean()

    def _refresh_kappa(self):
        """Linearly ramp kappa from where warmup left the regularizer to the target."""
        ramp = max(1, int(self.kappa_ramp_epochs))
        t = float(self.current_epoch - self.geco_epoch0) / ramp
        t = min(1.0, max(0.0, t))
        self.kappa.copy_(self.kappa_start + t * (self.kappa_target - self.kappa_start))

    @torch.no_grad()
    def _geco_update(self, sim_val, reg_val):
        """Advance the GECO controller by one training step."""
        reg_val = all_reduce(reg_val.detach().float().clone())
        sim_val = all_reduce(sim_val.detach().float().clone())

        if self.current_epoch < self.warmup_epochs:
            # accumulate the kappa calibration window over the tail of the warmup
            if self.current_epoch >= self.warmup_epochs - self.kappa_calib_epochs:
                self.reg_calib_sum.add_(reg_val)
                self.reg_calib_count.add_(1.0)
            return

        if not self.geco_initialized:
            if self.reg_calib_count > 0:
                reg_calib = self.reg_calib_sum / self.reg_calib_count
            else:
                reg_calib = reg_val
            self.kappa_start.copy_(reg_calib)

            use_ref = self.geco_kappa_mode == 'reference' and bool(self.kappa_ref_valid)
            if self.geco_kappa_mode == 'reference' and not use_ref:
                print("[GECO] kappa_mode='reference' but no reference was calibrated for "
                      f"({self.reconstruction}, {self.regularization}); falling back to 'warmup'.",
                      flush=True)
            if use_ref:
                anchor = self.kappa_reference + self.geco_kappa_slack * self.kappa_reference.abs()
                frac = min(1.0, max(0.0, float(self.geco_kappa_gap_frac)))
                target = self.kappa_start + frac * (anchor - self.kappa_start)
            else:
                anchor = reg_calib - (1.0 - self.geco_tolerance_margin) * reg_calib.abs()
                target = anchor
            self.kappa_anchor.copy_(anchor)
            self.kappa_target.copy_(target)

            self.geco_epoch0.copy_(self.current_epoch)
            self.geco_initialized.fill_(True)
            print(f"[GECO] kappa calibrated: start={self.kappa_start.item():.6g} "
                  f"-> target={self.kappa_target.item():.6g} "
                  f"(mode={'reference' if use_ref else 'warmup'}, "
                  f"anchor={self.kappa_anchor.item():.6g}, "
                  f"reference={self.kappa_reference.item():.6g}, "
                  f"gap_frac={self.geco_kappa_gap_frac}) "
                  f"ramp over {int(self.kappa_ramp_epochs)} epochs", flush=True)
        self._refresh_kappa()

        c_rel = (reg_val - self.kappa) / (self.kappa.abs() + 1e-8)

        if not self.constraint_ma_init:
            self.constraint_ma.copy_(c_rel)
            self.constraint_ma_init.fill_(True)
        elif self.brake_level == 0:
            # frozen while braking: anti-windup, the integrator keeps its state
            self.constraint_ma.mul_(self.alpha_c).add_(c_rel, alpha=1.0 - self.alpha_c)

        do_update = (self.global_step % self.lbd_step) == 0
        self.global_step.add_(1)
        if not do_update:
            return

        if not self.sim_ema_init:
            self.sim_ema_short.copy_(sim_val)
            self.sim_ema_long.copy_(sim_val)
            self.sim_sq_dev_ema.zero_()
            self.sim_ema_init.fill_(True)
        else:
            innov = sim_val - self.sim_ema_short
            self.sim_ema_short.mul_(self.alpha_short).add_(sim_val, alpha=1.0 - self.alpha_short)
            self.sim_ema_long.mul_(self.alpha_long).add_(sim_val, alpha=1.0 - self.alpha_long)
            self.sim_sq_dev_ema.mul_(self.alpha_long).add_(innov * innov, alpha=1.0 - self.alpha_long)
        self.sim_ema_count.add_(1.0)

        sigma_sim = self.sim_sq_dev_ema.clamp_min(0.0).sqrt()
        z = (self.sim_ema_short - self.sim_ema_long) / (sigma_sim + 1e-8)
        self.sim_trend_z.copy_(z)

        if self.sim_ema_count < self.sim_brake_min_updates:
            level = 0
        elif z > 2.0 * self.geco_degradation_sigmas:
            level = 2
        elif z > self.geco_degradation_sigmas:
            level = 1
        else:
            level = 0
        self.brake_level.fill_(level)

        if level == 0 and self.constraint_ma.abs() < self.deadband:
            self.geco_factor.fill_(1.0)
            return

        log_factor = (self.gain * self.constraint_ma).clamp(-self.log_f_max, self.log_f_max)
        if level == 1:
            log_factor = log_factor.clamp_max(0.0)
        elif level == 2:
            log_factor = torch.full_like(log_factor, -self.log_f_max)

        factor = torch.exp(log_factor)
        self.geco_factor.copy_(factor)
        self.lagrange_lambda.mul_(factor).clamp_(self.lambda_min, self.lambda_max)

    def geco_metrics(self):
        """Controller internals, for logging on training steps only."""
        if not self.use_geco:
            return {}
        return {
            "geco/lambda": self.lagrange_lambda.item(),
            "geco/kappa": self.kappa.item(),
            "geco/kappa_start": self.kappa_start.item(),
            "geco/kappa_target": self.kappa_target.item(),
            "geco/kappa_anchor": self.kappa_anchor.item(),
            "geco/kappa_reference": self.kappa_reference.item(),
            "geco/c_rel_ma": self.constraint_ma.item(),
            "geco/factor": self.geco_factor.item(),
            "geco/sim_trend_z": self.sim_trend_z.item(),
            "geco/brake_level": float(self.brake_level.item()),
            "geco/active": float(bool(self.geco_initialized)),
        }

    @torch.no_grad()
    def _alignment_mse(self, z1_all, z2_all, prototype, w_tensor):
        """Raw MSE of each view against the prototype: a sigma-free alignment probe.

        With 'rbf' the loss is a correntropy whose Taylor expansion weights every even
        moment of the error by 1/sigma^(2k), so as sigma anneals the reported value
        climbs even while the underlying error shrinks. Anything that reads alignment
        as a TREND -- the brake, and the logs -- must therefore read the raw error
        instead. For 'cosine' (unit vectors, mse = 2 - 2cos) and for 'mse' this is a
        positive affine transform of loss_sim, so the brake's z-score is unchanged.
        Never enters the loss.
        """
        vals = []
        for i in range(len(z1_all)):
            m1 = func.mse_loss(z1_all[i], z2_all[prototype], reduction='none').mean()
            m2 = func.mse_loss(z2_all[i], z1_all[prototype], reduction='none').mean()
            vals.append((m1 + m2) / 2.0)
        stacked = torch.stack(vals)
        return torch.mean(stacked * w_tensor) if w_tensor is not None else stacked.mean()

    def calc_similarity(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        if self.reconstruction == 'mse':
            return func.mse_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'l1':
            return func.l1_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'huber':
            return func.huber_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'rbf':
            mse = func.mse_loss(x, y, reduction='none').mean(dim=-1)
            correntropy = torch.exp(-mse / (2 * self.sigma ** 2))
            return 1.0 - correntropy
        elif self.reconstruction == 'cosine':
            return - (x * y).sum(dim=-1) / self.temperature
            
        raise ValueError(f"Unknown reconstruction metric: {self.reconstruction}")

    def _neg_samples_matrix(self, x, y, semantic_margin=None, cross_only=False):
        """The [2N, 2N] score matrix whose row-wise logsumexp is the InfoNCE denominator.

        Rows 0..N-1 come from x, rows N..2N-1 from y. Every row holds one positive
        and 2N-2 negatives, and the positive sits exactly on the main diagonal:
        sim_Z[i, i] = sim_xy[i, i] and sim_Z[N+i, N+i] = sim_xy.T[i, i]. The two
        self-similarity diagonals land off-diagonal, at [i, N+i] and [N+i, i], so
        masking eye(2N) removes the positives and nothing else.
        """
        # dim: [N, D]
        N = len(x)

        sim_xx = -self.calc_similarity(x.unsqueeze(1), x.unsqueeze(0))
        sim_yy = -self.calc_similarity(y.unsqueeze(1), y.unsqueeze(0))
        sim_xy = -self.calc_similarity(x.unsqueeze(1), y.unsqueeze(0))

        if semantic_margin is not None:
            with torch.no_grad():
                if self.reconstruction == 'cosine':
                    prior_xx = torch.clamp(sim_xx, min=0.0)
                    prior_yy = torch.clamp(sim_yy, min=0.0)
                    prior_xy = torch.clamp(sim_xy, min=0.0)
                else:
                    prior_xx = sim_xx
                    prior_yy = sim_yy
                    prior_xy = sim_xy

            sim_xx = sim_xx - semantic_margin * prior_xx
            sim_yy = sim_yy - semantic_margin * prior_yy
            sim_xy = sim_xy - semantic_margin * prior_xy

        # `cross_only` drops the whole same-set blocks rather than just their
        # diagonals: rows of x are then scored against y alone. What remains is the
        # cross term, which is where a cross-modal conjunction can live at all --
        # repelling z_x,i from z_x,j is instance discrimination within one set and
        # carries no information about the pairing.
        eye = torch.eye(N, device=x.device)
        sim_xx = sim_xx - self.INF * (torch.ones_like(eye) if cross_only else eye)
        sim_yy = sim_yy - self.INF * (torch.ones_like(eye) if cross_only else eye)

        # Matrix shape: [2N, 2N]
        sim_Z1 = torch.cat([sim_xy, sim_xx], dim=1)
        sim_Z2 = torch.cat([sim_yy, sim_xy.T], dim=1)
        return torch.cat([sim_Z1, sim_Z2], dim=0)

    def _neg_samples_reduce(self, sim_Z, drop_pos):
        if drop_pos:
            sim_Z = sim_Z - self.INF * torch.eye(sim_Z.shape[0], device=sim_Z.device)
        return torch.logsumexp(sim_Z, dim=1).mean()

    def _neg_samples_pair(self, x, y, semantic_margin=None, drop_pos=False,
                          cross_only=False):
        sim_Z = self._neg_samples_matrix(x, y, semantic_margin, cross_only)
        if drop_pos:
            reg = self._neg_samples_reduce(sim_Z, True)
            return reg, reg.detach()
        return (self._neg_samples_reduce(sim_Z, False),
                self._neg_samples_reduce(sim_Z.detach(), True))

    def calc_neg_samples_reg(self, x, y, drop_pos=False):
        return self._neg_samples_reduce(self._neg_samples_matrix(x, y), drop_pos)

    def calc_semantic_neg_samples_reg(self, x, y, semantic_margin=SEM_AWARE_MARGIN, drop_pos=False):
        return self._neg_samples_reduce(
            self._neg_samples_matrix(x, y, semantic_margin), drop_pos)

    def _composite_reg(self, z1_all, z2_all, prototype, w_tensor):
        """A regularizer of COMPOSITE_REGS, and the constraint GECO would read from it."""
        mode, gaussian, contrastive = self.COMPOSITE_REGS[self.regularization]
        D = z1_all[0].shape[-1]
        if mode == 'split':
            c_dims, g_dims = slice(0, D // 2), slice(D // 2, D)
        else:
            c_dims = g_dims = slice(None)

        def reduce(vals):
            vals = torch.stack(vals)
            return torch.mean(vals * w_tensor) if w_tensor is not None else vals.mean()

        drop_pos = contrastive in ('neg-samples-dcl', 'neg-samples-dcl-cross')
        cross = contrastive in ('neg-samples-cross', 'neg-samples-dcl-cross')
        regs, cons = [], []
        for i in range(len(z1_all)):
            tgt1 = z2_all[prototype].detach() if self.stop_grad else z2_all[prototype]
            tgt2 = z1_all[prototype].detach() if self.stop_grad else z1_all[prototype]
            reg1, cons1 = self._neg_samples_pair(z1_all[i][..., c_dims], tgt1[..., c_dims],
                                                 None, drop_pos, cross)
            reg2, cons2 = self._neg_samples_pair(z2_all[i][..., c_dims], tgt2[..., c_dims],
                                                 None, drop_pos, cross)
            regs.append((reg1 + reg2) / 2.0)
            cons.append((cons1 + cons2) / 2.0)
        reg_c, cons_c = reduce(regs), reduce(cons)

        if gaussian == 'sigreg':
            reg_g = self.sigreg(torch.cat(list(z1_all) + list(z2_all), dim=0)[..., g_dims])
        elif gaussian == 'sigreg-permask':
            reg_g = reduce([self.sigreg(torch.cat([z1_all[i], z2_all[i]], dim=0)[..., g_dims])
                            for i in range(len(z1_all))])
        else:
            raise ValueError(f"Unsupported gaussian part {gaussian!r} in {self.regularization}")

        rho = self.mix_rho
        return (0.5 * ((1 - rho) * reg_g + rho * reg_c),
                0.5 * ((1 - rho) * reg_g.detach() + rho * cons_c))

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward_vicreg(self, z: torch.Tensor) -> torch.Tensor:
        # dim: [N, D]
        N, D = z.shape
        z = z - z.mean(dim=0)
        
        std_z = torch.sqrt(z.var(dim=0) + 1e-04)
        std_loss = torch.mean(func.relu(1.0 - std_z))
        
        cov_z = (z.T @ z) / (N - 1)
        cov_loss = self.off_diagonal(cov_z).pow_(2).sum().div(D)
        
        return (self.vicreg_std_coeff * std_loss) + (self.vicreg_cov_coeff * cov_loss)

    def forward_visreg(self, z: torch.Tensor) -> torch.Tensor:
        # dim: [M, B, D]
        _, B, D = z.shape
        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = func.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return 0.9 * scale_loss + 1.2 * shape_loss + 0.9 * center_loss

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, outputs):
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)

        if self.reconstruction == 'cosine':
            z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
            z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        
        Z = all_gather_batch_with_grad(z1 + z2)
        z1_all, z2_all = Z[:n_emb], Z[n_emb:]

        loss_sim = []
        loss_reg_local = []
        reg_cons_local = []

        for i in range(n_emb):
            tgt1 = z2_all[prototype].detach() if self.stop_grad else z2_all[prototype]
            tgt2 = z1_all[prototype].detach() if self.stop_grad else z1_all[prototype]

            sim1 = self.calc_similarity(z1_all[i], tgt1).mean()
            sim2 = self.calc_similarity(z2_all[i], tgt2).mean()
            loss_sim.append((sim1 + sim2) / 2.0)

            if self.regularization in self.PER_EMBED_NEG_REGS:
                margin = self.SEM_AWARE_MARGIN if self.regularization == 'sem-aware' else None
                drop_pos = self.regularization in ('neg-samples-dcl', 'neg-samples-dcl-cross')
                cross = self.regularization in ('neg-samples-dcl-cross',
                                                'neg-samples-cross')
                reg1, cons1 = self._neg_samples_pair(z1_all[i], tgt1, margin, drop_pos, cross)
                reg2, cons2 = self._neg_samples_pair(z2_all[i], tgt2, margin, drop_pos, cross)
                loss_reg_local.append((reg1 + reg2) / 2.0)
                reg_cons_local.append((cons1 + cons2) / 2.0)

        losses_dict = {"sim_loss_%i"%i: l for i, l in enumerate(loss_sim)}
        w_tensor = torch.tensor(self.weights, device=z1_all[0].device) if self.weights is not None else None
        
        if w_tensor is not None:
            total_sim_loss = torch.mean(torch.stack(loss_sim) * w_tensor)
        else:
            total_sim_loss = torch.mean(torch.stack(loss_sim))

        total_sim_loss = (self.vicreg_inv_coeff if self.regularization == 'vicreg' else 1.0) * total_sim_loss

        # monitoring + brake signal only; detached, never part of total_loss
        align_mse = self._alignment_mse(z1_all, z2_all, prototype, w_tensor)
        
        # what GECO constrains: for the InfoNCE family the alignment-free denominator,
        # for everything else the regularizer itself (already alignment-free)
        reg_constraint = None

        if self.regularization == 'visreg':
            loss_reg = 0.5 * (self.forward_visreg(torch.stack(z1_all, dim=0)) + self.forward_visreg(torch.stack(z2_all, dim=0)))
        elif self.regularization == 'vicreg':
            z1_global = torch.cat(z1_all, dim=0)
            z2_global = torch.cat(z2_all, dim=0)
            loss_reg = 0.5 * (self.forward_vicreg(z1_global) + self.forward_vicreg(z2_global))
        elif self.regularization == 'sigreg':
            z_all_global = torch.cat(Z, dim=0)
            loss_reg = self.sigreg(z_all_global)
        elif self.regularization == 'sigreg-permask':
            # Each mask must look isotropic on its own; the union looking right is
            # not enough. Views are pooled within a mask so the test keeps 2N samples.
            per_mask = torch.stack([
                self.sigreg(torch.cat([z1_all[i], z2_all[i]], dim=0)) for i in range(n_emb)])
            loss_reg = (torch.mean(per_mask * w_tensor) if w_tensor is not None
                        else per_mask.mean())
        elif self.regularization == 'sigreg-multi':
            # Only the joint embedding is asked to be Gaussian; the unimodal masks
            # keep their alignment term and receive no distributional constraint.
            loss_reg = self.sigreg(torch.cat([z1_all[prototype], z2_all[prototype]], dim=0))
        elif self.regularization == 'visreg-multi':
            loss_reg = 0.5 * (self.forward_visreg(z1_all[prototype].unsqueeze(0))
                              + self.forward_visreg(z2_all[prototype].unsqueeze(0)))
        elif self.regularization in self.COMPOSITE_REGS:
            loss_reg, reg_constraint = self._composite_reg(z1_all, z2_all, prototype, w_tensor)
        elif self.regularization in self.PER_EMBED_NEG_REGS:
            if w_tensor is not None:
                loss_reg = torch.mean(torch.stack(loss_reg_local) * w_tensor)
                reg_constraint = torch.mean(torch.stack(reg_cons_local) * w_tensor)
            else:
                loss_reg = torch.mean(torch.stack(loss_reg_local))
                reg_constraint = torch.mean(torch.stack(reg_cons_local))
        elif self.regularization == 'neg-samples-multi':
            tgt1 = z2_all[prototype].detach() if self.stop_grad else z2_all[prototype]
            tgt2 = z1_all[prototype].detach() if self.stop_grad else z1_all[prototype]
            reg1, cons1 = self._neg_samples_pair(z1_all[prototype], tgt1)
            reg2, cons2 = self._neg_samples_pair(z2_all[prototype], tgt2)
            loss_reg = (reg1 + reg2) / 2.0
            reg_constraint = (cons1 + cons2) / 2.0
        else:
            loss_reg = torch.tensor(0.0, device=z1_all[0].device)

        if reg_constraint is None:
            reg_constraint = loss_reg.detach()

        if self.use_geco:
            if self.training:
                if not self._schedule_ready:
                    raise RuntimeError(
                        "WoMMLoss: GECO is enabled but configure_schedule(steps_per_epoch) was never "
                        "called. BaseModel.on_train_epoch_start does this automatically."
                    )
                if self.geco_kappa_mode == 'reference' and not self.kappa_ref_init:
                    ref = self.calibrate_kappa_reference(
                        n_emb, z1_all[0].shape[0], z1_all[0].shape[-1],
                        z1_all[0].device, z1_all[0].dtype)
                    if ref is not None:
                        self.kappa_reference.copy_(ref)
                        print(f"[GECO] kappa reference for ({self.reconstruction}, "
                              f"{self.regularization}) = {ref.item():.6g}", flush=True)
                    self.kappa_ref_init.fill_(True)
                    self.kappa_ref_valid.fill_(ref is not None)
                self._geco_update(align_mse, reg_constraint)

            if self.geco_initialized:
                lambda_value = self.lagrange_lambda.detach()
            else:
                lambda_value = torch.as_tensor(
                    self.reg_weight, dtype=total_sim_loss.dtype, device=total_sim_loss.device)
            total_loss = total_sim_loss + lambda_value * loss_reg
            lambda_out = float(lambda_value)
        else:
            total_loss = total_sim_loss + self.reg_weight * loss_reg
            lambda_out = self.reg_weight

        return {"loss": total_loss, "loss_sim": total_sim_loss, "loss_reg": loss_reg,
                "loss_reg_cons": reg_constraint, "align_mse": align_mse,
                "lambda": lambda_out, **losses_dict}

class SlicingUnivariateTest(torch.nn.Module):
    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
            
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats

class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        return all_reduce(x, op="AVG")

    @property
    def world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

class EppsPulley(UnivariateTest):
    def __init__(
        self, t_max: float = 5, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt 
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # dim: (*, N, K, n_points)
        x_t = x.unsqueeze(-1) * self.t 
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # dim: (*, n_points)
        cos_mean = cos_vals.mean(-3) 
        sin_mean = sin_vals.mean(-3) 

        cos_mean = all_reduce(cos_mean, op="AVG")
        sin_mean = all_reduce(sin_mean, op="AVG")

        err = (cos_mean - self.phi).square() + sin_mean.square()

        return (err @ self.weights) * N * self.world_size