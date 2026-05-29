"""
Boltz-2 inference wrapper with pair-representation bias for ROCKET.

Implements the ROCKET-Boltz2 (naive pair-embedding-bias) baseline described in
BOLTZ2_INTEGRATION.md.  The learnable bias is applied to the trunk's final pair
representation z AFTER the last PairFormer recycling step and BEFORE z is consumed
by the diffusion conditioning module.

This is deliberately NOT guided diffusion – no gradient is injected into the
denoising update itself.  The learnable parameters are only w_pair and b_pair
(multiplicative + additive bias on z).
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch import Tensor

# Boltz-2 imports – these live in the boltz source tree which must be on PYTHONPATH
try:
    from boltz.model.models.boltz2 import Boltz2
    from boltz.main import (
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
    )
except ImportError as exc:
    raise ImportError(
        "boltz package not found.  Add /path/to/boltz/src to PYTHONPATH or install it."
    ) from exc

# Default checkpoint path; can be overridden via env or constructor arg
_DEFAULT_CACHE = Path(os.environ.get("BOLTZ_CACHE", Path.home() / ".boltz"))
_DEFAULT_CKPT = _DEFAULT_CACHE / "boltz2_conf.ckpt"

# Known token_z for the released Boltz-2 conf checkpoint (inferred from architecture)
BOLTZ2_TOKEN_Z = 128
BOLTZ2_TOKEN_S = 384


class Boltz2PairBias(nn.Module):
    """
    Boltz-2 wrapped with a learnable pair-representation bias.

    The only trainable parameters are::

        z_biased = z @ w_pair + b_pair

    applied to the pre-PairFormer pair representation z [B, N, N, 128].

    Three sampling modes control how gradients flow back to w_pair/b_pair:

    ``"truncated_bptt"`` (original)
        Full 200-step stochastic reverse diffusion; gradients retained only
        through the last ``truncated_backprop_steps`` (K) steps.  In practice
        the stochastic noise washes out the signal from the pair bias.

    ``"single_step"``
        Single deterministic denoising step at σ_max:
        x̂₀ = D_θ(σ_max·ε, σ_max, z_biased).
        One network call; clean gradient through PairFormer + one diffusion step.
        Inspired by ConForNets (arxiv 2604.18559).  7× more structurally
        sensitive than TBPTT; 2.3× lower seed variance.

    ``"ddim"``
        N deterministic DDIM steps with fixed initial noise; gradient flows
        through all N steps AND the PairFormer.  Better structural precision
        than single-step (N × denoising iterations) while keeping the gradient
        fully deterministic.  N controlled by ``ddim_steps`` (default 20).

    Parameters
    ----------
    checkpoint_path:
        Path to the boltz2_conf.ckpt checkpoint.
    truncated_backprop_steps:
        K for ``"truncated_bptt"`` mode.  Ignored for other modes.
    sampling_mode:
        ``"truncated_bptt"`` | ``"single_step"`` | ``"ddim"``
    ddim_steps:
        Number of deterministic steps for ``"ddim"`` mode.
    diffusion_seed:
        Fixed RNG seed for the initial noise draw.  Same seed is used at every
        optimisation step — the only source of randomness between calls is the
        MSA subsampling (when applicable).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        truncated_backprop_steps: int = 5,
        diffusion_seed: Optional[int] = None,
        num_sampling_steps: int = 200,
        recycling_steps: int = 3,
        sampling_mode: str = "truncated_bptt",
        ddim_steps: int = 20,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        if checkpoint_path is None:
            checkpoint_path = _DEFAULT_CKPT
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Boltz-2 checkpoint not found at {checkpoint_path}.  "
                "Download it with: boltz predict --model boltz2 <any_input>"
            )

        self.K = truncated_backprop_steps
        self.diffusion_seed = diffusion_seed
        self.num_sampling_steps = num_sampling_steps
        self.recycling_steps = recycling_steps
        self.sampling_mode = sampling_mode
        self.ddim_steps = ddim_steps
        self._device = device

        logger.info(f"Loading Boltz-2 from {checkpoint_path} …")
        diffusion_params = Boltz2DiffusionParams()
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(use_paired_feature=True)

        self._boltz: Boltz2 = Boltz2.load_from_checkpoint(
            str(checkpoint_path),
            strict=True,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            predict_args={
                "recycling_steps": recycling_steps,
                "sampling_steps": num_sampling_steps,
                "diffusion_samples": 1,
                "max_parallel_samples": 1,
            },
        )
        self._boltz.eval()
        for p in self._boltz.parameters():
            p.requires_grad_(False)

        self._token_z = BOLTZ2_TOKEN_Z
        self._token_s = BOLTZ2_TOKEN_S

        # Bias parameters – initialised lazily once we know n_tokens from the first
        # batch.  Stored as plain tensors in a dict so they are NOT part of the
        # nn.Module parameter tree (they live only in the optimizer passed from outside).
        self._bias: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    # def init_bias(self, n_tokens: int, device: str | torch.device) -> dict[str, Tensor]:
    #     """
    #     Create (or reset) the multiplicative/additive pair-bias tensors.

    #     Returns a dict with keys ``'w_pair'`` and ``'b_pair'`` as leaf tensors
    #     with requires_grad=True.  Pass them to your Adam optimizer.
    #     """
    #     shape = (1, n_tokens, n_tokens, self._token_z)
    #     w = torch.ones(shape, device=device, requires_grad=True)
    #     b = torch.zeros(shape, device=device, requires_grad=True)
    #     self._bias = {"w_pair": w, "b_pair": b}
    #     return self._bias

    def init_bias(self, device: str | torch.device) -> dict[str, Tensor]:
        """
        Create channel-wise affine transform matrices.
        Independent of protein length (N), matching ConforNets.
        """
        # self._token_z is 128. Shape: [128, 128]
        w = torch.eye(self._token_z, device=device, requires_grad=True)
        # Shape: [128]
        b = torch.zeros(self._token_z, device=device, requires_grad=True)
        
        self._bias = {"w_pair": w, "b_pair": b}
        return self._bias

    @property
    def bias_params(self) -> list[Tensor]:
        return list(self._bias.values())

    def freeze(self):
        for p in self._boltz.parameters():
            p.requires_grad_(False)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self._boltz = self._boltz.to(*args, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Core forward: trunk + biased z + truncated-backprop diffusion
    # ------------------------------------------------------------------

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """
        Run Boltz-2 trunk with pair-bias applied, then sample coordinates via
        truncated-backprop reverse diffusion.

        Returns a dict containing at least:
          - ``sample_atom_coords``  [1, n_atoms, 3]  (Å)
          - ``plddt``               [1, n_tokens]    (if confidence_prediction)
          - ``s``                   [1, n_tokens, token_s]
          - ``z``                   [1, n_tokens, n_tokens, token_z]  (biased)
          - ``pbfactor``            [1, n_tokens, num_bins]   (if predict_bfactor)
        """
        if recycling_steps is None:
            recycling_steps = self.recycling_steps
        if num_sampling_steps is None:
            num_sampling_steps = self.num_sampling_steps

        model = self._boltz
        w_pair = self._bias.get("w_pair")
        b_pair = self._bias.get("b_pair")
        if w_pair is None:
            raise RuntimeError("Call init_bias() before forward().")

        # --- 1. Trunk (no grad through Boltz-2 weights, but grad through bias) ---
        s, z = self._run_trunk(feats, recycling_steps, w_pair, b_pair)

        # --- 2. Distogram (no grad needed) ---
        with torch.no_grad():
            pdistogram = model.distogram_module(z.detach())

        dict_out: dict[str, Tensor] = {"pdistogram": pdistogram, "s": s, "z": z}

        # --- 3. B-factor prediction (detach s; B is used in SFC but not differentiated) ---
        if model.predict_bfactor:
            with torch.no_grad():
                pbfactor = model.bfactor_module(s.detach())
            dict_out["pbfactor"] = pbfactor

        # --- 4. Diffusion conditioning (grad flows through z) ---
        from boltz.model.modules.diffusion_conditioning import DiffusionConditioning  # noqa: PLC0415

        relative_position_encoding = model.rel_pos(feats)
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            model.diffusion_conditioning(
                s_trunk=s,
                z_trunk=z,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
            )
        )
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        # s_inputs for the diffusion module
        s_inputs = model.input_embedder(feats)

        # --- 5. Diffusion sampling (mode-dependent) ---
        _s = s.float()
        _si = s_inputs.float()
        with torch.autocast("cuda", enabled=False):
            if self.sampling_mode == "single_step":
                sample_out = self._sample_one_step(
                    s_trunk=_s, s_inputs=_si, feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                    num_sampling_steps=num_sampling_steps,
                )
            elif self.sampling_mode == "ddim":
                sample_out = self._sample_ddim(
                    s_trunk=_s, s_inputs=_si, feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                    num_sampling_steps=num_sampling_steps,
                )
            else:  # "truncated_bptt" (default)
                sample_out = self._sample_truncated(
                    s_trunk=_s, s_inputs=_si, feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                    num_sampling_steps=num_sampling_steps,
                )
        dict_out.update(sample_out)

        # --- 6. Confidence (stop-grad on s and z as in original Boltz-2) ---
        if model.confidence_prediction:
            with torch.no_grad():
                conf_out = model.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=pdistogram[:, :, :, 0].detach(),
                    multiplicity=1,
                    run_sequentially=True,
                    use_kernels=False,
                )
            dict_out.update(conf_out)

        return dict_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # def _run_trunk(
    #     self,
    #     feats: dict[str, Tensor],
    #     recycling_steps: int,
    #     w_pair: Tensor,
    #     b_pair: Tensor,
    # ) -> tuple[Tensor, Tensor]:
    #     """Run the Boltz-2 trunk with pair bias on the final recycling output."""
    #     model = self._boltz

    #     s_inputs = model.input_embedder(feats)
    #     s_init = model.s_init(s_inputs)
    #     z_init = (
    #         model.z_init_1(s_inputs)[:, :, None]
    #         + model.z_init_2(s_inputs)[:, None, :]
    #     )
    #     relative_position_encoding = model.rel_pos(feats)
    #     z_init = z_init + relative_position_encoding
    #     z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    #     z_init = z_init + model.contact_conditioning(feats)

    #     mask = feats["token_pad_mask"].float()
    #     pair_mask = mask[:, :, None] * mask[:, None, :]

    #     s = torch.zeros_like(s_init)
    #     z = torch.zeros_like(z_init)

    #     for i in range(recycling_steps + 1):
    #         is_last = i == recycling_steps
    #         # Only keep gradient graph on the final recycling iteration
    #         ctx = torch.enable_grad() if is_last else torch.no_grad()

    #         with ctx:
    #             s = s_init + model.s_recycle(model.s_norm(s.detach() if not is_last else s))
    #             z = z_init + model.z_recycle(model.z_norm(z.detach() if not is_last else z))

    #             if model.use_templates:
    #                 z = z + model.template_module(z, feats, pair_mask, use_kernels=False)

    #             z = z + model.msa_module(z, s_inputs, feats, use_kernels=False)
    #             s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)

    #     # Apply pair bias AFTER trunk, before diffusion conditioning
    #     # z shape: [B, N, N, token_z]

    #     z_biased = w_pair * z + b_pair
    #     return s, z_biased

    def _run_trunk(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int,
        w_pair: Tensor,
        b_pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the Boltz-2 trunk with ConforNets channel-wise pre-Pairformer bias."""

        model = self._boltz

        # Standard Boltz-2 Latent Initialization
        s_inputs = model.input_embedder(feats)
        s_init = model.s_init(s_inputs)
        z_init = (
            model.z_init_1(s_inputs)[:, :, None]
            + model.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = model.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + model.token_bonds(feats["token_bonds"].float())
        z_init = z_init + model.contact_conditioning(feats)

        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # Recycling Loop
        for i in range(recycling_steps + 1):
            is_last = i == recycling_steps
            # Enable gradients ONLY on the final recycling pass
            ctx = torch.enable_grad() if is_last else torch.no_grad()

            with ctx:
                s_in = s.detach() if not is_last else s
                z_in = z.detach() if not is_last else z

                s = s_init + model.s_recycle(model.s_norm(s_in))
                z = z_init + model.z_recycle(model.z_norm(z_in))

                if model.use_templates:
                    z = z + model.template_module(z, feats, pair_mask, use_kernels=False)

                z = z + model.msa_module(z, s_inputs, feats, use_kernels=False)

                # --- CONFORNETS CRITICAL INJECTION POINT ---
                if is_last:
                    # 1. Apply the channel-wise transform right before the Pairformer blocks
                    # z shape: [B, N, N, 128] @ w_pair shape: [128, 128] -> [B, N, N, 128]
                    z = torch.matmul(z, w_pair) + b_pair

                    # 2. Replicate chunk size logic from PairformerModule code
                    chunk_size_tri_attn = 128 if z.shape[1] > 512 else 512

                    # 3. Step-by-block fine-grained activation checkpointing
                    # This completely bypasses the 'if self.training' block check in Boltz-2 source code
                    for layer in model.pairformer_module.layers:
                        # Default-argument captures `layer` by value, avoiding Python's
                        # late-binding closure semantics.  Without this, during backward
                        # recompute all run_block closures would reference the last layer,
                        # producing completely wrong gradients.
                        def run_block(s_tensor, z_tensor, _layer=layer):
                            return _layer(
                                s_tensor,
                                z_tensor,
                                mask,
                                pair_mask,
                                chunk_size_tri_attn,
                                False,
                            )
                        s, z = checkpoint.checkpoint(run_block, s, z, use_reentrant=False)
                else:
                    # Clean un-differentiable execution for early recycling passes
                    s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)

        return s, z

    def _sample_truncated(
        self,
        s_trunk: Tensor,
        s_inputs: Tensor,
        feats: dict[str, Tensor],
        diffusion_conditioning: dict[str, Tensor],
        num_sampling_steps: int,
    ) -> dict[str, Tensor]:
        """
        Reverse diffusion with truncated backpropagation through the last K steps.

        Steps 0 … (T-K-1) are run under torch.no_grad().
        Steps (T-K) … (T-1) are run with the autograd graph retained.
        """
        from boltz.model.modules.diffusionv2 import AtomDiffusion, compute_random_augmentation  # noqa: PLC0415
        from math import sqrt as _sqrt  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        diff_module: AtomDiffusion = self._boltz.structure_module
        atom_mask = feats["atom_pad_mask"].float()

        # Noise schedule
        sigmas = diff_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > diff_module.gamma_min, diff_module.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # Fix noise seed within this gradient step for reproducibility
        if self.diffusion_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.diffusion_seed)

        shape = (*atom_mask.shape, 3)
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=atom_mask.device)

        if self.diffusion_seed is not None:
            torch.set_rng_state(rng_state)

        detach_boundary = num_sampling_steps - self.K  # keep grad only for last K steps

        # Build network_condition_kwargs once
        network_condition_kwargs = dict(
            s_trunk=s_trunk,
            s_inputs=s_inputs,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            multiplicity=1,
        )

        # Disable coordinate augmentation during ROCKET (would scramble gradients)
        orig_aug = diff_module.coordinate_augmentation_inference
        diff_module.coordinate_augmentation_inference = False

        step_scale = diff_module.step_scale

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            sigma_tm_f = sigma_tm.item()
            sigma_t_f = sigma_t.item()
            gamma_f = gamma.item()

            t_hat = sigma_tm_f * (1 + gamma_f)
            noise_var = diff_module.noise_scale ** 2 * (t_hat ** 2 - sigma_tm_f ** 2)
            eps = _sqrt(noise_var) * torch.randn_like(atom_coords)
            atom_coords_noisy = atom_coords.detach() + eps

            use_grad = step_idx >= detach_boundary
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                atom_coords_denoised = diff_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    network_condition_kwargs=network_condition_kwargs,
                )

            if diff_module.alignment_reverse_diff:
                from boltz.model.modules.diffusionv2 import weighted_rigid_align  # noqa: PLC0415
                with torch.no_grad():
                    atom_coords_noisy_aligned = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.detach().float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    ).to(atom_coords_denoised)
            else:
                atom_coords_noisy_aligned = atom_coords_noisy

            denoised_over_sigma = (atom_coords_noisy_aligned - atom_coords_denoised) / t_hat
            atom_coords_next = atom_coords_noisy_aligned + step_scale * (sigma_t_f - t_hat) * denoised_over_sigma

            if use_grad:
                atom_coords = atom_coords_next
            else:
                atom_coords = atom_coords_next.detach()

        diff_module.coordinate_augmentation_inference = orig_aug

        return {"sample_atom_coords": atom_coords}

    # ------------------------------------------------------------------
    # Single-step denoising (ConForNets approach)
    # ------------------------------------------------------------------

    def _sample_one_step(
        self,
        s_trunk: Tensor,
        s_inputs: Tensor,
        feats: dict,
        diffusion_conditioning: dict,
        num_sampling_steps: int,
    ) -> dict[str, Tensor]:
        """
        Single deterministic denoising step at σ_max.

        x̂₀ = D_θ(σ_max · ε, σ_max, z_biased)

        One call to the preconditioned denoising network — fully differentiable
        with respect to z_biased (and hence w_pair/b_pair) through PairFormer
        and the single diffusion network call.  No stochastic noise in the
        gradient path.

        Inspired by ConForNets (arxiv 2604.18559): "for objectives defined on
        coordinates, we use a single deterministic denoising step."
        """
        diff_module = self._boltz.structure_module
        atom_mask   = feats["atom_pad_mask"].float()

        sigmas    = diff_module.sample_schedule(num_sampling_steps)
        sigma_max = sigmas[0].item()

        # Fixed initial noise — identical every optimisation step
        if self.diffusion_seed is not None:
            rng = torch.get_rng_state()
            torch.manual_seed(self.diffusion_seed)
        shape       = (*atom_mask.shape, 3)
        atom_noisy  = sigma_max * torch.randn(shape, device=atom_mask.device)
        if self.diffusion_seed is not None:
            torch.set_rng_state(rng)

        orig_aug = diff_module.coordinate_augmentation_inference
        diff_module.coordinate_augmentation_inference = False

        network_condition_kwargs = dict(
            s_trunk=s_trunk, s_inputs=s_inputs, feats=feats,
            diffusion_conditioning=diffusion_conditioning, multiplicity=1,
        )
        x_hat = diff_module.preconditioned_network_forward(
            atom_noisy, sigma_max,
            network_condition_kwargs=network_condition_kwargs,
        )

        diff_module.coordinate_augmentation_inference = orig_aug
        return {"sample_atom_coords": x_hat}

    # ------------------------------------------------------------------
    # DDIM multi-step deterministic denoising
    # ------------------------------------------------------------------

    def _sample_ddim(
        self,
        s_trunk: Tensor,
        s_inputs: Tensor,
        feats: dict,
        diffusion_conditioning: dict,
        num_sampling_steps: int,
    ) -> dict[str, Tensor]:
        """
        N deterministic DDIM steps with fixed initial noise.

        Uses ``self.ddim_steps`` steps subsampled evenly from the full
        ``num_sampling_steps`` schedule.  Each step is a deterministic Euler
        update (gamma=0, no stochastic noise injection):

            x_{t-1} = x_t + step_scale · (σ_{t-1} − σ_t) · (x_t − x̂₀) / σ_t

        Gradient flows through ALL steps AND through PairFormer (via the
        diffusion conditioning that depends on z_biased at each step).

        Because the trajectory is fully deterministic (fixed initial noise +
        no stochastic Euler noise), the gradient is clean and reproducible.
        This gives better structural precision than single-step (more denoising
        iterations → sharper prediction) while avoiding the noise that kills
        the stochastic TBPTT gradient.
        """
        diff_module = self._boltz.structure_module
        atom_mask   = feats["atom_pad_mask"].float()

        # Subsample schedule to ddim_steps intervals
        sigmas_full = diff_module.sample_schedule(num_sampling_steps)
        n = min(self.ddim_steps, num_sampling_steps)
        indices = torch.linspace(0, num_sampling_steps, n + 1).long()
        sigmas  = sigmas_full[indices]   # shape [n+1]

        # Fixed initial noise (same every optimisation step)
        if self.diffusion_seed is not None:
            rng = torch.get_rng_state()
            torch.manual_seed(self.diffusion_seed)
        shape       = (*atom_mask.shape, 3)
        atom_coords = sigmas[0].item() * torch.randn(shape, device=atom_mask.device)
        if self.diffusion_seed is not None:
            torch.set_rng_state(rng)

        orig_aug = diff_module.coordinate_augmentation_inference
        diff_module.coordinate_augmentation_inference = False

        network_condition_kwargs = dict(
            s_trunk=s_trunk, s_inputs=s_inputs, feats=feats,
            diffusion_conditioning=diffusion_conditioning, multiplicity=1,
        )
        step_scale = diff_module.step_scale

        for i in range(n):
            sigma_tm = sigmas[i].item()       # current noise level
            sigma_t  = sigmas[i + 1].item()   # next (lower) noise level

            # gamma=0: no stochastic inflation → t_hat = sigma_tm exactly
            t_hat = sigma_tm

            atom_coords_denoised = diff_module.preconditioned_network_forward(
                atom_coords, t_hat,
                network_condition_kwargs=network_condition_kwargs,
            )

            # Deterministic Euler step — no noise injection, gradient through all steps
            denoised_over_sigma = (atom_coords - atom_coords_denoised) / t_hat
            atom_coords = atom_coords + step_scale * (sigma_t - t_hat) * denoised_over_sigma

        diff_module.coordinate_augmentation_inference = orig_aug
        return {"sample_atom_coords": atom_coords}

    # ------------------------------------------------------------------
    # B-factor extraction
    # ------------------------------------------------------------------

    def get_bfactors_from_pbfactor(
        self, pbfactor_logits: Tensor, bin_min: float = 0.0, bin_max: float = 100.0
    ) -> Tensor:
        """
        Convert BFactorModule bin logits → per-token B-factor scalars (Å²).

        Boltz-2 uses a histogram over [bin_min, bin_max] with num_bins bins.
        We return the expected value under the predicted distribution.
        """
        num_bins = pbfactor_logits.shape[-1]
        bin_centers = torch.linspace(
            bin_min, bin_max, num_bins, device=pbfactor_logits.device
        )
        probs = torch.softmax(pbfactor_logits.float(), dim=-1)
        return (probs * bin_centers).sum(dim=-1)  # [B, N]

    @staticmethod
    def token_b_to_atom_b(
        token_b: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        """
        Broadcast per-token B-factors to per-atom B-factors using
        the atom_to_token index tensor from feats.

        Parameters
        ----------
        token_b: [B, N_tokens]
        atom_to_token: [B, N_atoms]  (int index)
        """
        return token_b[:, atom_to_token.long()]  # [B, N_atoms]
