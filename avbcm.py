import torch
import torch.nn.functional as F


class AVBCM:
    """
    AVBCM
    Prior-Gated Adaptive Vulnerability Boundary Calibration

    Full model:
        Boundary Gap
            鈫?
        EMA Boundary State
            鈫?
        Adaptive Boundary Strength
            鈫?
        Prior Gate
            鈫?
        Class-Prior Bias
            鈫?
        Calibrated Logits

    ------------------------------------------------------------
    Ablation switches
    ------------------------------------------------------------

    use_ema:
        Whether to smooth the boundary gap using EMA.

    use_boundary:
        Whether to use boundary-aware adaptive calibration
        strength.

    use_prior_gate:
        Whether to use the imbalance-aware prior gate.

    use_prior_bias:
        Whether to use log(class_prior) as the calibration
        direction.

    fixed_lambda:
        Fixed-位 ablation. When set to a float, 位_base is
        frozen at that constant for every batch (gap and
        gap_ema are still computed and logged). Default
        None keeps the adaptive 位_base = f(g_t).

    ------------------------------------------------------------
    Recommended configurations
    ------------------------------------------------------------

    Full AVBCM:
        use_boundary=True
        use_ema=True
        use_prior_gate=True
        use_prior_bias=True

    w/o EMA:
        use_boundary=True
        use_ema=False
        use_prior_gate=True
        use_prior_bias=True

    w/o Prior Gate:
        use_boundary=True
        use_ema=True
        use_prior_gate=False
        use_prior_bias=True

    w/o Prior Bias:
        use_boundary=True
        use_ema=True
        use_prior_gate=True
        use_prior_bias=False

    Prior Bias only:
        use_boundary=False
        use_ema=False
        use_prior_gate=False
        use_prior_bias=True

    ------------------------------------------------------------
    Input
    ------------------------------------------------------------

    logits:
        [N, C]

    labels:
        [N]

    class_prior:
        [C]

    ------------------------------------------------------------
    Output
    ------------------------------------------------------------

    calibrated_logits
    lambda_t
    gap_ema
    """

    def __init__(
        self,
        beta=0.9,
        use_boundary=True,
        use_ema=True,
        use_prior_gate=True,
        use_prior_bias=True,
        gate_strength=0.5,
        fixed_lambda=None
    ):

        # ========================================================
        # Configuration
        # ========================================================

        self.beta = beta

        self.use_boundary = use_boundary
        self.use_ema = use_ema
        self.use_prior_gate = use_prior_gate
        self.use_prior_bias = use_prior_bias
        self.gate_strength = gate_strength
        self.fixed_lambda = fixed_lambda

        # ========================================================
        # Running boundary state
        # ========================================================

        self.boundary_gap_ema = None
        self.last_raw_gap = None

    # ============================================================
    # Boundary gap
    # ============================================================

    @torch.no_grad()
    def compute_boundary_gap(
        self,
        logits,
        labels
    ):
        """
        Estimate the vulnerability boundary gap.

            gap =
                E[z_pos | y=1]
                -
                E[z_pos | y=0]

        For binary classification:
            z_pos = logits[:, 1]

        If one class is absent from the current batch,
        return zero.
        """

        pos_mask = labels == 1
        neg_mask = labels == 0

        # --------------------------------------------------------
        # Current batch contains only one class
        # --------------------------------------------------------

        if (
            pos_mask.sum() == 0
            or
            neg_mask.sum() == 0
        ):
            return torch.tensor(
                0.0,
                device=logits.device,
                dtype=logits.dtype
            )

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        # --------------------------------------------------------
        # Binary classification
        # --------------------------------------------------------

        if logits.size(1) == 2:

            pos_score = pos_logits[:, 1]
            neg_score = neg_logits[:, 1]

        # --------------------------------------------------------
        # Generic multi-class fallback
        # --------------------------------------------------------

        else:

            pos_score = pos_logits.mean(dim=1)
            neg_score = neg_logits.mean(dim=1)

        gap = (
            pos_score.mean()
            -
            neg_score.mean()
        )

        return gap

    # ============================================================
    # EMA boundary update
    # ============================================================

    @torch.no_grad()
    def update_boundary(
        self,
        gap
    ):
        """
        Update the running boundary state.

        If EMA is disabled, return the current batch gap directly.
        """

        # --------------------------------------------------------
        # No EMA ablation
        # --------------------------------------------------------

        if not self.use_ema:

            return gap.detach()

        # --------------------------------------------------------
        # EMA enabled
        # --------------------------------------------------------

        if self.boundary_gap_ema is None:

            self.boundary_gap_ema = gap.detach().clone()

        else:

            self.boundary_gap_ema = (
                self.beta * self.boundary_gap_ema
                +
                (1.0 - self.beta) * gap.detach()
            )

        return self.boundary_gap_ema

    # ============================================================
    # Adaptive boundary strength
    # ============================================================

    @torch.no_grad()
    def compute_lambda(
        self,
        gap,
        logits
    ):
        """
        Compute adaptive calibration strength.

        Full boundary-aware formulation:

            boundary_factor =
                sigmoid(-gap)

            margin_factor =
                1 / (1 + |gap|)

            lambda =
                boundary_factor
                *
                (1 + margin_factor)

        If boundary calibration is disabled:

            lambda = 1

        This makes the ablation equivalent to removing
        boundary-aware adaptive weighting while preserving
        the calibration direction.
        """

        # --------------------------------------------------------
        # Boundary component disabled
        # --------------------------------------------------------

        if not self.use_boundary:

            return torch.tensor(
                1.0,
                device=logits.device,
                dtype=logits.dtype
            )

        # --------------------------------------------------------
        # Boundary-aware adaptive strength
        # --------------------------------------------------------

        if gap is None:

            return torch.tensor(
                0.0,
                device=logits.device,
                dtype=logits.dtype
            )

        boundary_factor = torch.sigmoid(-gap)

        margin_factor = (
            1.0
            /
            (1.0 + torch.abs(gap))
        )

        lambda_t = (
            boundary_factor
            *
            (1.0 + margin_factor)
        )

        return lambda_t

    # ============================================================
    # Prior gate
    # ============================================================

    @torch.no_grad()
    def compute_prior_gate(
        self,
        class_prior
    ):
        """
        Automatically determine the strength of prior-aware
        calibration according to class imbalance.

        minority_prior:
            min(P(y=0), P(y=1))

        imbalance_strength:
            1 - minority_prior / 0.5

        gate:
            1 + 0.5 * imbalance_strength

        The fixed 0.5 is part of the original v3.1 design.
        """

        prior = class_prior.float()

        # Defensive normalization

        prior = (
            prior
            /
            prior.sum().clamp_min(1e-12)
        )

        minority_prior = torch.min(prior)

        imbalance_strength = (
            1.0
            -
            minority_prior / 0.5
        )

        imbalance_strength = imbalance_strength.clamp(
            min=0.0,
            max=1.0
        )

        gate = (
            1.0
            +
            self.gate_strength * imbalance_strength
        )

        return gate

    # ============================================================
    # Main calibration
    # ============================================================

    def calibrate(
        self,
        logits,
        labels,
        class_prior
    ):
        """
        Perform AVBCM calibration.

        The ablation switches affect only the corresponding
        component. All other operations remain unchanged.
        """

        # ========================================================
        # 1. Boundary estimation
        # ========================================================

        gap = self.compute_boundary_gap(
            logits,
            labels
        )

        self.last_raw_gap = gap.detach()

        # ========================================================
        # 2. Temporal smoothing
        # ========================================================

        gap_ema = self.update_boundary(
            gap
        )

        # ========================================================
        # 3. Boundary-aware adaptive strength
        # ========================================================

        if self.fixed_lambda is not None:

            # Fixed-位 ablation: freeze 位_base at a constant.
            # gap / gap_ema are still computed and returned so
            # boundary logging stays comparable.

            lambda_base = torch.tensor(
                float(self.fixed_lambda),
                device=logits.device,
                dtype=logits.dtype
            )

        else:

            lambda_base = self.compute_lambda(
                gap_ema,
                logits
            )

            lambda_base = lambda_base.to(
                device=logits.device,
                dtype=logits.dtype
            )

        # ========================================================
        # 4. Prior gate
        # ========================================================

        if self.use_prior_gate:

            prior_gate = self.compute_prior_gate(
                class_prior
            )

            prior_gate = prior_gate.to(
                device=logits.device,
                dtype=logits.dtype
            )

        else:

            # Remove prior-gating mechanism.
            prior_gate = torch.tensor(
                1.0,
                device=logits.device,
                dtype=logits.dtype
            )

        # ========================================================
        # 5. Final calibration strength
        # ========================================================

        lambda_t = (
            lambda_base
            *
            prior_gate
        )

        # ========================================================
        # 6. Class prior
        # ========================================================

        prior = class_prior.to(
            device=logits.device,
            dtype=logits.dtype
        )

        prior = (
            prior
            /
            prior.sum().clamp_min(1e-12)
        )

        # ========================================================
        # 7. Calibration direction
        # ========================================================

        if self.use_prior_bias:

            bias = torch.log(
                prior.clamp_min(1e-12)
            )

        else:

            # Remove class-prior calibration direction.
            bias = torch.zeros_like(
                prior,
                device=logits.device,
                dtype=logits.dtype
            )

        # ========================================================
        # 8. Apply calibration
        # ========================================================

        calibrated_logits = (
            logits
            +
            lambda_t * bias
        )

        # ========================================================
        # 9. Return statistics
        # ========================================================

        return (
            calibrated_logits,
            lambda_t.detach(),
            gap_ema.detach()
        )