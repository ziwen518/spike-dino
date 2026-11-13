"""Spike-DINO v7: the DINO objective on a constrained sphere.

Why this file exists
--------------------
v6 ended with a measured negative: the decoupled-GMM distillation half adds
nothing at matched schedule (instboth e150 60.40 against instonly 60.43), and
the best arm in the project is the one with the distillation term switched
off (``--dino-weight 0``, 62.97 dev / 61.10 official test).  That is not a
DINO result -- with ``dino_weight = 0`` the model is MoCo-style InfoNCE with an
EMA teacher, and the GMM only describes the Teacher's features.

The v7 line therefore goes back to the DINO objective itself and treats the
v6 collapse as a *geometry* problem rather than a loss-term problem.

What DINO actually constrains
-----------------------------
The projection output is L2-normalised, so every feature lives on the unit
sphere, and the logits are cosines against a table of prototypes.  Nothing in
the loss asks those prototypes -- or the features -- to be spread out; the
cross-entropy only asks each feature to be closer to *some* prototype than to
the others.  The cheap joint minimum is the observed one: the prototypes bunch
together, the features bunch with them, and the target becomes uniform or
one-hot while the loss still falls.  That is Partial Prototype Collapse
(Govindarajan et al., BMVC 2024; Arteaga et al., ICLR 2026), and v6 reproduced
it with a decoupled table too (distill-only 4/256 unique prototypes at
eps=0.5, nearest-neighbour cosine 0.991), so decoupling alone does not protect
against it.

Two SNN-specific facts make the sphere worse than it is in the ANN case
----------------------------------------------------------------------
1. **The features do not start on the sphere; they start in a cap of it.**
   The projection bottleneck is a spike *rate*: ``r in [0, 1]^d``.  Every
   sample therefore has a large positive shared component, measured on this
   backbone at ``||mean|| / mean||x|| = 0.752``.  Cosine on the raw rate is
   dominated by that DC term -- which is why the v6 NT-Xent control had to
   subtract the Teacher batch mean before its negatives separated anything
   (``sdino_v6.py``), and why v5's ``teacher_logit_std`` sat at the
   random-alignment value 1/sqrt(d) for 1400 steps.
2. **The variance of a rate is not the variance of a feature.**  A rate is a
   binomial proportion with ``n`` = the number of spike times; its variance
   scales with the mean, so low-rate units are systematically shrunk by L2
   normalisation and high-rate units dominate.  ``arcsin(sqrt(r))`` is the
   variance-stabilising transform for exactly this family and is applied
   before centring when ``arcsin_sqrt`` is on.

What v7 does differently
------------------------
* **The DC is removed before normalisation, by an EMA buffer.**  The origin is
  a v7-owned buffer updated from the Student's global views only, applied to
  both Student and Teacher, so the cosine geometry is a sphere around the
  feature distribution's own centre rather than around the spike-rate origin.
  A batch-mean origin (the v6 NT-Xent choice) makes training and eval
  disagree; an EMA buffer does not.
* **The prototype table is trainable again** (v6 froze it and fitted a GMM by
  EM; that is the design Arteaga et al. already published, and it lost 9.1
  points there).  What changes is that the table now has a *geometry* term.
* **The DINO loss is stock**: cosine logits, Teacher centring, Teacher
  temperature warmup, Student temperature, two global + two temporal views
  with the v5 reliability weight.

The three constraints (all off by default, so the v7 baseline is a clean
control with only the DC removal and the trainable table changed):

``proto_frame_weight``
    Frame potential ``mean_{i != j} (mu_i . mu_j)^2`` on the unit prototype
    table.  For unit prototypes this equals half of ``||mu mu^T - I||^2``, so
    minimising it pushes the table towards an equiangular tight frame -- the
    exact "spread the prototypes on the sphere" statement.  It is a soft
    penalty: it competes with the cross-entropy.

``proto_orthonormal``
    A hard retraction of the table onto the Stiefel manifold after every
    optimiser step (polar decomposition, ``U V^T`` of the SVD).  With
    ``K == d`` the prototypes then *are* an orthonormal basis, so the logits
    are the coordinates of the feature in that basis and the only way the
    table can move is by rotating.  Collapse-by-bunching is removed from the
    feasible set by construction; the question becomes whether a rotation is
    enough for the objective.

``koleo_weight`` / ``proto_balance_weight``
    Feature-side and usage-side spread terms (DINOv2's KoLeo; marginal
    assignment entropy).  They constrain the features or the assignment rather
    than the table, so they are a different hypothesis from the two above.

A note on what this file deliberately does *not* have: no InfoNCE term, no
queue, no GMM, no learnable logit scale.  The v6 evidence is that the
contrastive term is what supplies the escape from collapse, and the point of
v7 is to buy that escape with geometry instead of with a second objective.  A
learnable logit scale is also omitted: a scale shared by Student and Teacher
can shrink both sides of the softmax together, which lowers the loss without
moving any feature.
"""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .sdino_v4 import (SpikeDinoV4Encoder, _running_stats_frozen,
                       bottleneck_regularizers, independent_forward, koleo_loss)


def spherical_feature(rate, center=None, arcsin_sqrt=False, eps=1e-6):
    """Spike rate -> unit sphere.

    ``arcsin(sqrt(r))`` first (variance stabilisation for binomial rates),
    then the EMA origin is removed, then L2 normalisation.  With
    ``center=None`` the result is the raw-rate direction, i.e. the geometry
    v5 used, which is the ablation for the DC removal.
    """
    x = rate.float()
    if arcsin_sqrt:
        x = torch.asin(x.clamp(0., 1.).sqrt())
    if center is not None:
        x = x - center
    return F.normalize(x, dim=-1, eps=eps)


def prototype_frame_penalty(table, eps=1e-6):
    """Mean squared off-diagonal cosine between prototypes (frame potential).

    Zero iff the prototypes are mutually orthogonal.  For unit vectors
    ``sum_{i != j} (mu_i . mu_j)^2 = ||mu mu^T - I||_F^2``, so this is the
    Gram-matrix deviation penalty written in the form that is numerically
    stable when the table has not been normalised yet.
    """
    weight = F.normalize(table.float(), dim=-1, eps=eps)
    gram = weight @ weight.t()
    count = gram.shape[0]
    if count < 2:
        return weight.new_zeros(())
    off = gram - torch.diag_embed(torch.diagonal(gram))
    return off.pow(2).sum() / (count * (count - 1))


@torch.no_grad()
def project_to_stiefel(table):
    """Polar retraction: the nearest semi-orthonormal matrix, ``U V^T``.

    For ``K <= d`` the rows become orthonormal; for ``K > d`` the columns do
    (a Parseval tight frame either way).  This is the hard form of the frame
    constraint: it removes the component of the gradient that bunches
    prototypes together and leaves only the rotation.
    """
    weight = table.float()
    u, _, vh = torch.linalg.svd(weight, full_matrices=False)
    table.copy_((u @ vh).to(table.dtype))
    return table


@torch.no_grad()
def unique_prototypes(table, eps):
    """Greedy representative count at cosine tolerance ``eps``.

    Definition 2.1 of Govindarajan et al. (BMVC 2024): prototypes are unit
    vectors and a partial collapse of degree M at distance eps means M
    representatives such that every prototype is within ``1 - cos < eps`` of
    its representative.  Kept index-ordered and CPU-side so the numbers match
    ``tmp/_proto_diversity.py`` exactly; K is 256, so the transfer is 256 KB.
    """
    unit = F.normalize(table.float(), dim=-1).cpu()
    similarity = unit @ unit.t()
    taken = torch.zeros(len(unit), dtype=torch.bool)
    groups = 0
    largest = 0
    for index in range(len(unit)):
        if taken[index]:
            continue
        members = ((1. - similarity[index]) < eps) & ~taken
        taken |= members
        groups += 1
        largest = max(largest, int(members.sum()))
    return groups, largest


def koleo_hinge(features, distance_floor=0., eps=1e-8):
    """KoLeo with the nearest-neighbour distance shifted by ``distance_floor``.

    With ``distance_floor <= 0`` this is ``sdino_v4.koleo_loss`` bit for bit --
    it *calls* it, so the equivalence is by construction rather than by a
    copied expression that could drift.

    With a positive floor the log is taken on ``d + floor`` instead of ``d``:

        L = -log(d + floor)      instead of      L = -log(d)

    which leaves the term almost unchanged wherever ``d >> floor`` but bounds
    both the value (``-log(floor)`` at ``d = 0``) and the gradient magnitude
    (``1/floor``) at the near-duplicate tail.  Note what it does NOT do: unlike
    a ``clamp_min`` it does not switch the push off on the closest pairs -- it
    keeps a finite repulsion there rather than an infinite one.  A clamp would
    be the wrong shape, because the pairs a diversity term exists to separate
    are exactly the ones it would silence.

    Why the floor is needed at all.  Unclamped KoLeo is the one regulariser
    this project has already launched and lost to a gradient explosion: at
    weight 0.1 (``runs/sdino_v7_koleo01_c10_e300_20261110``) the arm read
    pre-clip ``grad_norm`` 167.58 against the DINO term's 2.27-3.23 and was
    stopped by budget at e4.71, so its effect on the representation was never
    measured.  Its own ``proto_koleo`` 1.1417 there means a mean
    nearest-neighbour distance of ``exp(-1.1417) = 0.319``: the *typical* pair
    is harmless, and the explosion comes entirely from the ``d -> 0`` tail
    through the ``1/d`` gradient.  The measured ``student_feature_cos_max``
    across every arm is 0.95-1.00, i.e. ``d = sqrt(2 - 2 cos)`` from 0.31 down
    to 0, so the tail is real and populated; a floor near 0.1 sits just inside
    it (cos 0.995) and caps it by ~10x while leaving the typical row alone.
    """
    if distance_floor <= 0:
        return koleo_loss(features, eps=eps)
    unit = F.normalize(features, dim=-1, eps=eps)
    with torch.no_grad():
        similarity = unit @ unit.t()
        similarity.fill_diagonal_(-2.)
        neighbour = similarity.argmax(1)
    distance = F.pairwise_distance(unit, unit[neighbour], p=2, eps=eps)
    return -torch.log(distance + distance_floor + eps).mean()


def cross_view_spectral_stats(feature_a, feature_b, eps=1e-8):
    """CSSP statistics of two paired global views, entirely in sample space.

    ``feature_a`` and ``feature_b`` are (N, d) unit-norm spherical features of
    the SAME N records seen under two augmentations.  With

        X = A - mean(A),   Y = B - mean(B),   C_AB = X^T Y / (N - 1)

    the three CSSP readouts are

        V_A      = tr(C_AA) = ||X||_F^2 / (N - 1)
        A_pair   = tr(C_AB) / sqrt(V_A V_B) = <X, Y>_F / (||X||_F ||Y||_F)
        R_shared = participation ratio of the squared singular values of C_AB

    and none of them needs a d x d object.  ``A_pair`` is exactly the cosine
    between the two flattened, centred feature matrices.  For ``R_shared``,
    sigma(C_AB) = sigma(X^T Y), so the nonzero eigenvalues of C_AB C_AB^T are
    those of P Q with P = X X^T and Q = Y Y^T, and therefore

        tr(PQ)   = sum_i sigma_i^2 = <P, Q>_F
        tr(PQ^2) = sum_i sigma_i^4 = sum_ij (PQ)_ij (PQ)_ji

    Both are Frobenius-level quantities of two N x N matrices (N = 64), i.e.
    cheaper than one 256 x 256 eigen-decomposition and exact up to float error
    (checked against a brute-force SVD in ``tmp/_cssp_design_check.py``).

    ``R_shared`` is **scale-free**, which is both its purpose and its hazard:
    X and Y are normalised before P and Q are formed because the ratio is
    invariant to their magnitude and the raw form underflows exactly when the
    features shrink -- the regime the regulariser exists to police.  Because it
    is scale-free it cannot detect a collapse to a point: on unit-norm features
    a point mass plus high-dimensional jitter scores at the *top* of its range
    (~35 of a ~43 noise ceiling), so ``V_A`` must carry that job.

    ``shared_energy_corrected`` is a fourth, **log-only** quantity added for
    the first CSSP arm: ``tr(P_h Q_h) - 1/(N-1)``, the total cross-view shared
    energy with the exact random-permutation null subtracted.  It is zero for
    independent views and positive only for genuinely paired structure, so it
    prices the one cheat ``R_shared`` admits (raise the ratio with
    high-dimensional jitter that shares nothing).  It is never added to a loss.

    All three are differentiable in the features; the caller detaches whatever
    it only wants to log.
    """
    a = feature_a.float()
    b = feature_b.float()
    x = a - a.mean(0, keepdim=True)
    y = b - b.mean(0, keepdim=True)
    count = max(1, x.shape[0] - 1)
    norm_a = x.pow(2).sum().clamp_min(eps)
    norm_b = y.pow(2).sum().clamp_min(eps)
    shared_strength = (x * y).sum() / count
    # Unit-Frobenius copies: R_shared is invariant to this and the raw form
    # underflows once the features start to shrink.
    xh = x / norm_a.sqrt()
    yh = y / norm_b.sqrt()
    product = (xh @ xh.t()) @ (yh @ yh.t())
    trace = torch.diagonal(product).sum()
    squared = (product * product.t()).sum().clamp_min(eps)
    # ---- permutation-corrected shared energy (log only, never a loss) -------
    # With the unit-Frobenius copies used above,
    #     trace = tr(P_h Q_h) = ||C_AB||_F^2 / (V_A V_B),
    # the *whole* Frobenius energy of the cross-covariance.  ``a_pair`` is a
    # different object: it reports only tr(C_AB), i.e. the projection onto the
    # single mean direction, so the two coincide only when C_AB is rank one.
    # Neither bounds the other in general, which is exactly why both are
    # logged.  For an independent random permutation of the rows of Y -- which
    # keeps both marginals and destroys only the pairing -- the centring makes
    #     E_perm ||C_AB||_F^2 = V_A V_B / (N - 1)   =>   E_perm trace = 1/(N-1),
    # so subtracting that null gives a statistic that is ZERO in expectation
    # for independent views at any N and moves only with genuine per-sample
    # shared structure.  Measured on CPU (tmp/_cssp_eshared_check.py): the
    # permutation mean is 0.015871 against the predicted 1/63 = 0.015873, and
    # the naive 1/N = 0.015625 is outside the sampling sd.
    # A NEGATIVE value is meaningful, not an error: it says the pairing
    # carries LESS cross-energy than a random pairing of the same marginals.
    #
    # Why it is needed at all: R_shared is a *ratio* of energies, so it can be
    # raised by adding high-dimensional jitter to both views without sharing
    # anything (measured in tmp/_cssp_grad_check.py: independent noise scores
    # R_shared 42 against aligned 35).  ``shared_energy_corrected`` is the
    # readout that separates the two, and it is deliberately NOT in the loss --
    # the first CSSP arm logs it so the cheat signature (R_shared up while
    # E_shared does not follow) is visible without changing the objective
    # under test.
    permutation_baseline = 1. / count
    return {'v_a': norm_a / count, 'v_b': norm_b / count,
            'a_pair': shared_strength * count / (norm_a * norm_b).sqrt(),
            'r_shared': trace.pow(2) / squared,
            'shared_strength': shared_strength,
            'shared_energy': trace,
            'shared_energy_corrected': trace - permutation_baseline,
            'permutation_baseline': trace.new_tensor(permutation_baseline)}


def cssp_hinges(stats, amp_floor, align_floor, rank_floor, energy_floor=0.):
    """The four one-sided hinges of CSSP.

    Each is identically zero while its floor holds, so the whole term is inert
    on a healthy spectrum and cannot drag a working model anywhere.  The
    switch-on is the risk, not the steady state: it is why every sub-term's
    pre-clip gradient norm is logged separately.

    ``shared`` is the only SCALE-FREE term, and that is a defect in an objective
    rather than a feature.  A participation ratio is invariant to any isotropic
    rescaling, so the cheapest way to raise it is to *shrink the dominant shared
    direction* instead of growing new ones -- and that is what the first CSSP
    arm measured: on step_00140.pt, ``shared_rank`` 2.145 against the base arm's
    1.201 (+79 %) while ``shared_energy_corrected`` fell to 0.0151 against the
    base arm's 0.0453 (-67 %), and the same inversion held at e002/e005.  A
    floor on the *corrected* shared energy prices that shortcut, because rank
    can then no longer be bought by discarding signal.  It is one-sided, so it
    is inert once the spectrum carries at least ``energy_floor`` of null-
    corrected cross-view energy; and ``energy_floor <= 0`` skips it entirely,
    which is what keeps every arm written before this option bit-identical.
    """
    amp = ((amp_floor - stats['v_a']).clamp_min(0.).pow(2)
           + (amp_floor - stats['v_b']).clamp_min(0.).pow(2))
    align = (align_floor - stats['a_pair']).clamp_min(0.).pow(2)
    shared = (math.log(rank_floor)
              - stats['r_shared'].clamp_min(1e-12).log()).clamp_min(0.).pow(2)
    if energy_floor > 0:
        signal = (energy_floor - stats['shared_energy_corrected']).clamp_min(0.).pow(2)
    else:
        signal = stats['r_shared'].new_zeros(())
    return {'amp': amp, 'align': align, 'shared': shared, 'signal': signal}


class SpikeDinoV7(nn.Module):
    """Spike-DINO with a trainable cosine prototype table on a centred sphere."""

    def __init__(self, *, student_temp=.1, center_momentum=.9,
                 teacher_bn_mode='train', temporal_weight=1.,
                 reliability_floor=.25,
                 proto_count=256, center_rate=True, rate_center_momentum=.99,
                 arcsin_sqrt=False,
                 proto_frame_weight=0., koleo_weight=0.,
                 koleo_distance_floor=0.,
                 proto_balance_weight=0., balance_temp=.1,
                 proto_orthonormal=False,
                 bottleneck_var_weight=0., bottleneck_decorr_weight=0.,
                 bottleneck_var_gamma=.15,
                 cssp_amp_weight=0., cssp_align_weight=0.,
                 cssp_shared_weight=0., cssp_amp_floor=0.,
                 cssp_align_floor=0., cssp_rank_floor=1.,
                 cssp_signal_weight=0., cssp_energy_floor=0.,
                 cssp_grad_log=False,
                 bottleneck=256, **encoder_kwargs):
        super().__init__()
        if teacher_bn_mode not in ('train', 'running_eval'):
            raise ValueError('Unknown Teacher BN mode')
        if not 0. <= reliability_floor <= 1.:
            raise ValueError('Reliability floor must be in [0, 1]')
        if student_temp <= 0:
            raise ValueError('Student temperature must be positive')
        if temporal_weight < 0:
            raise ValueError('Temporal weight must be nonnegative')
        if min(proto_frame_weight, koleo_weight, proto_balance_weight,
               bottleneck_var_weight, bottleneck_decorr_weight,
               cssp_amp_weight, cssp_align_weight, cssp_shared_weight,
               cssp_signal_weight) < 0:
            raise ValueError('Regulariser weights must be nonnegative')
        if min(cssp_amp_floor, cssp_align_floor, cssp_rank_floor,
               cssp_energy_floor) < 0 or cssp_rank_floor <= 0:
            raise ValueError('CSSP floors must be nonnegative and rank_floor positive')
        if koleo_distance_floor < 0:
            raise ValueError('KoLeo distance floor must be nonnegative')
        if cssp_signal_weight and cssp_energy_floor <= 0:
            # Without a floor the hinge is identically zero (see cssp_hinges),
            # so a positive weight would silently train nothing.  Refuse: a
            # weight and a floor are one option, not two.
            raise ValueError('cssp_signal_weight requires a positive cssp_energy_floor')
        if proto_balance_weight and balance_temp <= 0:
            raise ValueError('Balance temperature must be positive')
        if not 0. <= rate_center_momentum <= 1.:
            raise ValueError('Rate centre momentum must be in [0, 1]')
        if proto_orthonormal and int(proto_count) > int(bottleneck):
            # K mutually orthogonal vectors need K <= d.  With K > d the polar
            # retraction returns a tight frame instead of a basis, the rows are
            # no longer unit-norm, and the "logits are coordinates in an
            # orthonormal basis" reading -- the whole point of the hard
            # constraint -- does not hold.  Refuse rather than silently run a
            # different constraint than the one requested.
            raise ValueError('proto_orthonormal requires proto_count <= bottleneck')
        self.teacher_bn_mode = teacher_bn_mode
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.temporal_weight = float(temporal_weight)
        self.reliability_floor = float(reliability_floor)
        self.center_rate = bool(center_rate)
        self.rate_center_momentum = float(rate_center_momentum)
        self.arcsin_sqrt = bool(arcsin_sqrt)
        self.proto_frame_weight = float(proto_frame_weight)
        self.koleo_weight = float(koleo_weight)
        # 0 means the stock DINOv2 term, bit-identical to ``sdino_v4.koleo_loss``
        # and to every arm before this option existed.  A positive value shifts
        # the log onto ``d + floor`` and bounds the near-duplicate tail; see
        # ``koleo_hinge`` for the measurement that motivates it.
        self.koleo_distance_floor = float(koleo_distance_floor)
        self.proto_balance_weight = float(proto_balance_weight)
        self.balance_temp = float(balance_temp)
        self.proto_orthonormal = bool(proto_orthonormal)
        self.bottleneck_var_weight = float(bottleneck_var_weight)
        self.bottleneck_decorr_weight = float(bottleneck_decorr_weight)
        self.bottleneck_var_gamma = float(bottleneck_var_gamma)
        self.cssp_amp_weight = float(cssp_amp_weight)
        self.cssp_align_weight = float(cssp_align_weight)
        self.cssp_shared_weight = float(cssp_shared_weight)
        self.cssp_amp_floor = float(cssp_amp_floor)
        self.cssp_align_floor = float(cssp_align_floor)
        self.cssp_rank_floor = float(cssp_rank_floor)
        # The energy floor and its weight.  A separate weight from the rank
        # floor on purpose: the two statistics have different units and
        # different gradient scales, so one calibrated multiplier cannot serve
        # both.  Both default to 0, so the term is not even built.
        self.cssp_signal_weight = float(cssp_signal_weight)
        self.cssp_energy_floor = float(cssp_energy_floor)
        # Split-gradient logging.  Off by default, so every arm written before
        # this option is bit-identical: when it is off, forward performs no
        # extra autograd call and stores nothing.
        self.cssp_grad_log = bool(cssp_grad_log)
        # Set to (loss_dino, loss_cssp) on every forward that has a live CSSP
        # graph, so the trainer can separate the two gradients on the Encoder
        # without changing forward's return signature.
        self.loss_parts = None
        # The prototype table is the encoder's own ``head.prototypes``, so the
        # EMA Teacher copies it parameter-for-parameter in ``update_teacher``
        # exactly as DINO does.  v6 replaced it with EM buffers; that design is
        # published (Arteaga et al., ICLR 2026) and lost 9.1 points there, so
        # v7 keeps the standard learnable table and constrains it instead.
        encoder_kwargs.pop('prototypes', None)
        encoder_kwargs['prototypes'] = int(proto_count)
        self.student = SpikeDinoV4Encoder(bottleneck=bottleneck, **encoder_kwargs)
        self.teacher = deepcopy(self.student).requires_grad_(False)
        self.proto_count = int(self.student.head.prototypes.shape[0])
        self.bottleneck = int(bottleneck)
        # Teacher centring (DINO): an EMA of the Teacher's mean logit profile.
        self.register_buffer('center', torch.zeros(1, self.proto_count))
        # The sphere's origin.  Owned by v7 rather than by the encoder so that
        # Student and Teacher subtract *the same* vector -- the Teacher is an
        # EMA of the Student, so its own EMA centre would lag by one more EMA
        # and the two logits would be computed against different origins.
        if self.center_rate:
            self.register_buffer('feature_center', torch.zeros(1, bottleneck))
            # Counts updates so the first batch can define the origin exactly
            # instead of contributing 1 % of it: at momentum .99 a plain lerp
            # from zero leaves the DC dominant for ~100 steps, and a DC-
            # dominated feature is the same direction for every sample, i.e. a
            # uniform target and no instance information at all.
            self.register_buffer('center_updates', torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        if self.teacher_bn_mode == 'running_eval':
            self.teacher.eval()
        return self

    def _transform(self, rate):
        x = rate.float()
        if self.arcsin_sqrt:
            x = torch.asin(x.clamp(0., 1.).sqrt())
        return x

    def _sphere(self, rate):
        origin = self.feature_center if self.center_rate else None
        return spherical_feature(rate, origin, self.arcsin_sqrt)

    @torch.no_grad()
    def update_feature_center(self, rate):
        """EMA origin from the Student's global views only (label-free)."""
        if not self.center_rate or not self.training:
            return
        mean = self._transform(rate.detach()).mean(0, keepdim=True)
        if int(self.center_updates.item()) == 0:
            self.feature_center.copy_(mean)
        else:
            self.feature_center.lerp_(mean, 1 - self.rate_center_momentum)
        self.center_updates.add_(1)

    @torch.no_grad()
    def project_prototypes(self):
        """Hard frame constraint; call once per optimiser step if enabled.

        With K = d the retraction makes the table exactly orthonormal, so the
        rows are a basis of the feature space and prototype bunching is not in
        the feasible set.  ``__init__`` refuses K > d for this reason.
        """
        if self.proto_orthonormal:
            project_to_stiefel(self.student.head.prototypes)

    def forward(self, views, teacher_temp, autocast_dtype=None):
        """views = [global0, global1, temporal0, temporal1]."""
        self.loss_parts = None
        if len(views) != 4:
            raise ValueError('Spike-DINO v7 needs 2 global + 2 temporal views')
        if any(view.shape != views[0].shape for view in views):
            raise ValueError('v7 views must share a shape')
        autocast = (torch.autocast('cuda', dtype=autocast_dtype)
                    if autocast_dtype is not None else torch.autocast('cuda', enabled=False))
        batch = views[0].shape[1]
        with autocast:
            with torch.no_grad():
                teacher_out = independent_forward(self.teacher, torch.cat(views[:2], 1))
                teacher_rate = teacher_out['bottleneck_rate'].float()
            groups = [views[:2], views[2:]]
            student = []
            for group_index, group in enumerate(groups):
                freeze_running = (self.teacher_bn_mode == 'running_eval'
                                  and group_index > 0)
                with _running_stats_frozen(self.student, freeze_running):
                    output = independent_forward(self.student, torch.cat(group, 1))
                if group_index == 0:
                    student_global_rate = output['bottleneck_rate'].float()
                for index in range(len(group)):
                    part = slice(index * batch, (index + 1) * batch)
                    student.append({
                        'rate': output['bottleneck_rate'][part].float(),
                        'cls_spikes': output['cls_spikes'][:, part],
                        'patch_spikes': output['patch_spikes'][:, part],
                        'pooled_spikes': output['pooled_spikes'][:, part]})
        # Geometry is computed outside the autocast block on purpose: the
        # cosine logits span [-1, 1] and are divided by a Teacher temperature
        # of .04, so a bf16 matmul would quantise the target to ~0.1 of a
        # logit step.  The matmuls are [2B, d] x [d, K] -- negligible next to
        # the backbone, so fp32 costs nothing here.
        teacher_feature = self._sphere(teacher_rate)
        student_feature = [self._sphere(out['rate']) for out in student]
        teacher_logits = (teacher_feature @ F.normalize(
            self.teacher.head.prototypes.float(), dim=-1, eps=1e-6).t()).float()
        weight = F.normalize(self.student.head.prototypes.float(), dim=-1, eps=1e-6)
        student_logits = [(feature @ weight.t()).float() for feature in student_feature]

        targets_all = F.softmax((teacher_logits - self.center) / teacher_temp, -1)
        q0, q1 = targets_all.chunk(2)
        e0, e1 = teacher_feature.chunk(2)
        # Reliability: how much the Teacher agrees with itself across the two
        # global views, measured on the same centred features the target is
        # built from (v5/v6 measured it on the normalised raw rate).
        agreement = (e0 * e1).sum(-1).clamp(0., 1.)
        weight_sample = self.reliability_floor + (1. - self.reliability_floor) * agreement

        def term_vector(target, logits):
            return -(target * F.log_softmax(logits / self.student_temp, -1)).sum(-1)

        global_terms = torch.stack([term_vector(q0, student_logits[1]),
                                    term_vector(q1, student_logits[0])])
        dino_global = global_terms.mean()
        temporal_terms = torch.stack([term_vector(target, student_logits[index])
                                      for target in (q0, q1) for index in (2, 3)])
        dino_temporal = (temporal_terms * weight_sample.unsqueeze(0)).mean() / weight_sample.mean()
        loss = dino_global + self.temporal_weight * dino_temporal
        # Kept separate from ``loss`` so the trainer can take the DINO and CSSP
        # gradients apart on the Encoder.  This is the quantity the weight
        # calibration has to compare against: ``clip_grad_norm_`` later measures
        # the sum, and a sum cannot say which half moved the model.
        dino_objective = loss

        frame = loss.new_zeros(())
        if self.proto_frame_weight:
            frame = prototype_frame_penalty(self.student.head.prototypes)
            loss = loss + self.proto_frame_weight * frame
        koleo = loss.new_zeros(())
        if self.koleo_weight:
            koleo = .5 * (koleo_hinge(student_feature[0],
                                      self.koleo_distance_floor)
                          + koleo_hinge(student_feature[1],
                                        self.koleo_distance_floor))
            loss = loss + self.koleo_weight * koleo
        # CSSP: three one-sided floors on the cross-view shared spectrum of the
        # two *global* views.  All three weights default to 0, so every arm
        # written before this option is bit-identical.
        cssp_amp = cssp_align = cssp_shared = cssp_signal = loss.new_zeros(())
        cssp_objective = loss.new_zeros(())
        if (self.cssp_amp_weight or self.cssp_align_weight
                or self.cssp_shared_weight or self.cssp_signal_weight):
            hinges = cssp_hinges(
                cross_view_spectral_stats(student_feature[0], student_feature[1]),
                self.cssp_amp_floor, self.cssp_align_floor, self.cssp_rank_floor,
                self.cssp_energy_floor)
            cssp_amp, cssp_align, cssp_shared, cssp_signal = (
                hinges['amp'], hinges['align'], hinges['shared'], hinges['signal'])
            cssp_objective = (self.cssp_amp_weight * cssp_amp
                              + self.cssp_align_weight * cssp_align
                              + self.cssp_shared_weight * cssp_shared
                              + self.cssp_signal_weight * cssp_signal)
            loss = loss + cssp_objective
        balance = loss.new_zeros(())
        if self.proto_balance_weight:
            # Negative marginal entropy of the Student's own assignment: a
            # prototype that no sample ever selects still occupies the sphere,
            # so the table is only "spread" if it is also *used*.
            marginal_student = F.softmax(
                torch.cat(student_logits[:2], 0) / self.balance_temp, -1).mean(0)
            balance = (marginal_student * marginal_student.clamp_min(1e-12).log()).sum()
            loss = loss + self.proto_balance_weight * balance
        if self.bottleneck_var_weight or self.bottleneck_decorr_weight:
            bottleneck_var, bottleneck_decorr = bottleneck_regularizers(
                student_global_rate, self.bottleneck_var_gamma)
            loss = (loss + self.bottleneck_var_weight * bottleneck_var
                    + self.bottleneck_decorr_weight * bottleneck_decorr)
            bottleneck_var = bottleneck_var.detach()
            bottleneck_decorr = bottleneck_decorr.detach()
        else:
            bottleneck_var = bottleneck_decorr = loss.new_zeros(())
        self.update_feature_center(student_global_rate)

        # ---- split gradients, measured at the features -----------------------
        # Both objectives act on the same two tensors (the spherical features
        # of the two global views), so this separates them exactly and costs
        # one extra backward through the CSSP branch plus one through the
        # prototype matmul -- both trivial next to the backbone.  It is the
        # cheap per-step quantity; the Encoder-space calibration is the
        # trainer's job and only runs for a few steps.
        # ``retain_graph`` is essential: forward returns ``loss`` and the
        # trainer calls ``loss.backward()`` afterwards.
        cssp_grad_feature = dino_grad_feature = cssp_grad_cos = None
        if (self.cssp_grad_log and torch.is_grad_enabled()
                and cssp_objective.requires_grad):
            pair = (student_feature[0], student_feature[1])
            grad_c = torch.autograd.grad(cssp_objective, pair, retain_graph=True,
                                         allow_unused=True)
            grad_d = torch.autograd.grad(dino_objective, pair, retain_graph=True,
                                         allow_unused=True)
            zero = loss.new_zeros(student_feature[0].shape)
            vector_c = torch.cat([g if g is not None else zero for g in grad_c])
            vector_d = torch.cat([g if g is not None else zero for g in grad_d])
            cssp_grad_feature = vector_c.norm()
            dino_grad_feature = vector_d.norm()
            cssp_grad_cos = ((vector_c * vector_d).sum()
                             / (cssp_grad_feature * dino_grad_feature).clamp_min(1e-12))
        if ((cssp_objective.requires_grad or koleo.requires_grad)
                and torch.is_grad_enabled()):
            # ``cssp_shared``, ``cssp_signal`` and ``koleo`` are the UNWEIGHTED
            # hinges and are exposed for a specific reason: the trainer
            # calibrates each weight against ``||grad L_h||``, and if it
            # measured ``||grad cssp_objective||`` instead, that gradient
            # *already contains* the current weight.  The resulting update
            # lambda_new = target*||g_D|| / (lambda_old*||g_h||)
            # is a multiplicative feedback that oscillates between 1 and
            # lambda* instead of converging to lambda*; with the unweighted
            # hinge the ratio is weight-free and one measurement is enough.
            # ``koleo`` is exposed for the same reason and is the *mean* of the
            # two views' terms, i.e. exactly the quantity the weight multiplies.
            # The guard admits either family: a KoLeo-only arm has a CSSP part
            # with no graph (it is a plain zero), and vice versa.
            self.loss_parts = (dino_objective, cssp_objective, cssp_shared,
                               cssp_signal, koleo)

        with torch.no_grad():
            self.center.lerp_(teacher_logits.mean(0, keepdim=True),
                              1 - self.center_momentum)
            marginal = targets_all.mean(0)
            entropy = -(targets_all * targets_all.clamp_min(1e-12).log()).sum(-1).mean()
            marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
            student_probs = F.softmax(student_logits[0].float() / self.student_temp, -1)
            # ---- sphere health, Teacher side (the target's geometry) --------
            torch_gram = teacher_feature @ teacher_feature.t()
            count = torch_gram.shape[0]
            off_teacher = torch_gram - torch.eye(count, device=torch_gram.device)
            student_pair = torch.cat(student_feature[:2], 0)
            student_gram = student_pair @ student_pair.t()
            student_count = student_gram.shape[0]
            off_student = student_gram - torch.eye(student_count, device=student_gram.device)
            transformed = self._transform(teacher_rate)
            dc_share = (transformed.mean(0).norm()
                        / transformed.norm(dim=-1).mean().clamp_min(1e-8))
            # ---- sphere health, prototype side -------------------------------
            table = F.normalize(self.student.head.prototypes.float(), dim=-1, eps=1e-6)
            proto_gram = table @ table.t()
            proto_off = proto_gram - torch.diag_embed(torch.diagonal(proto_gram))
            proto_nn = proto_gram.fill_diagonal_(-2.).max(-1).values
            # ---- feature spread ----------------------------------------------
            centered_teacher = teacher_feature - teacher_feature.mean(0, keepdim=True)
            spectrum = torch.linalg.svdvals(centered_teacher.float())
            energy = spectrum.pow(2)
            effective_rank = (energy.sum().pow(2)
                              / energy.pow(2).sum().clamp_min(1e-12))
            per_dim_std = teacher_feature.std(0, unbiased=False)
            # ---- Student side and the cross-view shared spectrum ------------
            # Logging only.  No Student effective rank existed before this, so
            # "does the Student's spectrum fall before the Teacher's" could not
            # be asked of any earlier log.
            student_stats = cross_view_spectral_stats(student_feature[0],
                                                      student_feature[1])
            student_ranks = []
            for feature in student_feature[:2]:
                student_centred = feature - feature.mean(0, keepdim=True)
                student_spectrum = torch.linalg.svdvals(student_centred.float())
                student_energy = student_spectrum.pow(2)
                student_ranks.append(
                    student_energy.sum().pow(2)
                    / student_energy.pow(2).sum().clamp_min(1e-12))
            cls_rate = student[0]['cls_spikes'].float().mean(0)
            metrics = {
                'loss': loss.detach(), 'dino': dino_global.detach(),
                'dino_global': dino_global.detach(),
                'dino_temporal': dino_temporal.detach(),
                'proto_frame': frame.detach(), 'proto_koleo': koleo.detach(),
                'proto_balance': balance.detach(),
                'bottleneck_var_hinge': bottleneck_var,
                'bottleneck_decorr': bottleneck_decorr,
                'temporal_weight': loss.new_tensor(self.temporal_weight),
                'reliability_mean': agreement.mean(),
                'reliability_min': agreement.min(),
                'reliability_weight_mean': weight_sample.mean(),
                'teacher_entropy': entropy,
                'teacher_marginal_entropy': marginal_entropy,
                'teacher_information': marginal_entropy - entropy,
                'teacher_top1_mass': targets_all.max(-1).values.mean(),
                'student_entropy': -(student_probs * student_probs.clamp_min(
                    1e-12).log()).sum(-1).mean(),
                'teacher_logit_std': (teacher_logits - self.center).std(-1).mean(),
                'teacher_logit_range': ((teacher_logits - self.center).max(-1).values
                                        - (teacher_logits - self.center).min(-1).values).mean(),
                'teacher_proto_usage': targets_all.new_tensor(
                    targets_all.argmax(-1).unique().numel() / targets_all.shape[-1]),
                'student_proto_usage': student_logits[0].new_tensor(
                    student_logits[0].argmax(-1).unique().numel()
                    / student_logits[0].shape[-1]),
                # Teacher feature geometry: mean off-diagonal cosine -> 1 means
                # every sample collapsed onto one direction; max is the worst
                # pair in the batch.  Effective rank is the participation ratio
                # of the covariance spectrum, capped at the 2B = 128 samples.
                'teacher_feature_cos_mean': off_teacher.mean(),
                'teacher_feature_cos_std': off_teacher.std(),
                'teacher_feature_cos_max': off_teacher.max(),
                'teacher_feature_eff_rank': effective_rank,
                'teacher_feature_std_min': per_dim_std.min(),
                'teacher_feature_dead_units': (per_dim_std < 1e-3).float().mean(),
                'student_feature_cos_mean': off_student.mean(),
                'student_feature_cos_max': off_student.max(),
                'feature_dc_share': dc_share,
                # Prototype geometry: the readouts the collapse papers use.
                'proto_gram_rms': proto_off.pow(2).mean().sqrt(),
                'proto_gram_max_off': proto_off.max(),
                'proto_nn_cos_mean': proto_nn.mean(),
                'proto_nn_cos_max': proto_nn.max(),
                'proto_norm_mean': self.student.head.prototypes.float().norm(dim=-1).mean(),
                'cls_spike_rate': cls_rate.mean(),
                'patch_spike_rate': student[0]['patch_spikes'].float().mean(),
                'pooled_spike_rate': student[0]['pooled_spikes'].float().mean(),
                'cls_feature_std': cls_rate.std(0, unbiased=False).mean(),
                'teacher_bottleneck_rate': teacher_rate.mean(),
                'teacher_bottleneck_std': teacher_rate.std(0, unbiased=False).mean(),
                'student_bottleneck_rate': student_global_rate.mean(),
                'student_bottleneck_std': student_global_rate.std(0, unbiased=False).mean(),
                'feature_center_norm': (self.feature_center.norm()
                                        if self.center_rate else loss.new_tensor(0.)),
                # Cosine between the Student's global view and the Teacher's
                # *other* global view, on the sphere features the loss uses.
                'positive_cosine': (student_feature[0] * e1).sum(-1).mean(),
                # ---- Student-side and cross-view geometry (CSSP logging) -----
                # The Teacher-only effective rank above measures the branch that
                # EMITS the target; these measure the branch that is TRAINED.
                # `student_feature_trace` is tr(C) = N(1-||z_bar||^2)/(N-1) on
                # unit-norm features, i.e. exactly "how much survives the
                # sphere's own mean" -- the quantity the point-mass collapse
                # destroys and a PR hinge cannot see.
                'student_feature_eff_rank_0': student_ranks[0],
                'student_feature_eff_rank_1': student_ranks[1],
                'student_feature_trace_0': student_stats['v_a'],
                'student_feature_trace_1': student_stats['v_b'],
                'shared_rank': student_stats['r_shared'],
                'pair_alignment': student_stats['a_pair'],
                'shared_strength': student_stats['shared_strength'],
                # Log only -- NEVER in the loss.  ``shared_energy_corrected``
                # is the total shared energy with its exact random-permutation
                # null (1/(N-1)) removed, so it is 0 for independent views and
                # positive only for genuinely paired structure.  R_shared is a
                # ratio and can be raised by adding high-dimensional jitter to
                # both views without sharing anything (measured on CPU: a
                # genuine pair degraded by independent noise reads R_shared
                # 5.92 -> 41.51 while E_shared falls 0.1151 -> 0.0001).  The
                # cheat signature to watch is therefore "shared_rank up while
                # shared_energy_corrected does not follow".
                'shared_energy': student_stats['shared_energy'],
                'shared_energy_corrected': student_stats['shared_energy_corrected'],
                'cssp_amp': cssp_amp.detach(),
                'cssp_align': cssp_align.detach(),
                'cssp_shared': cssp_shared.detach(),
                'cssp_signal': cssp_signal.detach(),
                'cssp_shared_weight': loss.new_tensor(self.cssp_shared_weight),
                'cssp_signal_weight': loss.new_tensor(self.cssp_signal_weight),
                # Feature-space split gradient (log only, NaN when the
                # ``--cssp-grad-log`` probe is off).  The absolute CSSP norm is
                # the direct check on the rank-1 stationary point: at exactly
                # rank-1 the shared hinge is positive with zero gradient, so
                # "loss > 0" does not imply "a gradient exists".
                'cssp_grad_feature': (cssp_grad_feature if cssp_grad_feature is not None
                                      else loss.new_tensor(float('nan'))),
                'dino_grad_feature': (dino_grad_feature if dino_grad_feature is not None
                                      else loss.new_tensor(float('nan'))),
                'cssp_grad_cosine': (cssp_grad_cos if cssp_grad_cos is not None
                                     else loss.new_tensor(float('nan'))),
            }
        return loss, metrics

    @torch.no_grad()
    def prototype_health(self, epsilons=(.5, .025)):
        """Unique-prototype counts, for the trainer's log rows only."""
        report = {}
        for eps in epsilons:
            unique, largest = unique_prototypes(self.student.head.prototypes, eps)
            report[f'proto_unique_{eps}'] = unique
            report[f'proto_redundancy_{eps}'] = largest
        teacher_unique, teacher_largest = unique_prototypes(
            self.teacher.head.prototypes, epsilons[0])
        report[f'teacher_proto_unique_{epsilons[0]}'] = teacher_unique
        report[f'teacher_proto_redundancy_{epsilons[0]}'] = teacher_largest
        return report

    @torch.no_grad()
    def update_teacher(self, momentum):
        for target, source in zip(self.teacher.parameters(),
                                  self.student.parameters()):
            target.lerp_(source.detach(), 1 - momentum)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)
