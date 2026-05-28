"""
Atom-coordinate extraction utilities for Boltz-2 ROCKET.

Boltz-2 represents atoms differently from AlphaFold2:
  - AF2: fixed 37-atom-type layout per residue (N_res × 37, sparse)
  - Boltz-2: flat per-atom layout with atom_to_token one-hot mapping

The SFC (SFcalculator) uses cra_name format "chain-resid-resname-atomname"
where resid is a 0-based sequential index matching the full sequence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:
    from boltz.data import const as boltz_const
except ImportError:
    boltz_const = None  # type: ignore[assignment]

from rocket import utils as rk_utils
from rocket.coordinates import iterative_kabsch_alignment


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _boltz_available() -> None:
    if boltz_const is None:
        raise ImportError(
            "boltz package not found.  Add the boltz src/ directory to PYTHONPATH."
        )


def decode_atom_names(ref_atom_name_chars: Tensor) -> list[str]:
    """
    Decode per-atom names from Boltz-2's character one-hot encoding.

    Parameters
    ----------
    ref_atom_name_chars : Tensor
        Shape [N_atoms, 4, 64].  Each atom name is encoded as 4 characters,
        each character as a one-hot over 64 classes where class ``k``
        corresponds to ASCII character ``k + 32``.  Class 0 is the null/pad.

    Returns
    -------
    list[str]
        Length N_atoms.  Each entry is the stripped atom name (e.g. "CA").
    """
    # [N_atoms, 4]  -- argmax over the 64 character classes
    char_indices = ref_atom_name_chars.argmax(-1).cpu().numpy()
    names: list[str] = []
    for atom_chars in char_indices:
        name = "".join(chr(int(c) + 32) for c in atom_chars if c > 0).strip()
        names.append(name)
    return names


def _token_res_type_to_name(res_type_onehot: Tensor) -> list[str]:
    """
    Convert one-hot token res_type [N_tokens, num_tokens] to 3-letter names.

    Uses boltz_const.tokens which is indexed by the one-hot class index.
    """
    _boltz_available()
    indices = res_type_onehot.argmax(-1).cpu().numpy()  # [N_tokens]
    return [boltz_const.tokens[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_allatoms_boltz2(
    sample_atom_coords: Tensor,
    feats: dict[str, Tensor],
    cra_name_sfc: list[str],
    plddt_tokens: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """
    Extract atom coordinates from Boltz-2 output and reorder to match SFC.

    The SFC's cra_name uses the format ``"chain-resid-resname-atomname"``
    where:
      - chain  : single letter (A, B, …) from 0-based asym_id
      - resid  : 0-based sequential residue index within the chain
                 (Boltz-2 stores this as ``residue_index`` in feats)
      - resname: 3-letter code from ``boltz_const.tokens``
      - atomname: stripped atom name (e.g. "CA")

    Parameters
    ----------
    sample_atom_coords : Tensor [1, N_atoms, 3]
        Output from Boltz-2 diffusion.
    feats : dict
        Batch of featurizer-v2 features (leading batch dim = 1).
        Required keys: ``atom_pad_mask``, ``atom_to_token``,
        ``residue_index``, ``asym_id``, ``res_type``, ``ref_atom_name_chars``.
    cra_name_sfc : list[str]
        Atom name list from the SFC (used for ordering and topology check).
    plddt_tokens : Tensor [1, N_tokens] or None
        Per-token plddt in [0, 1] from the Boltz-2 confidence module.
        If provided, returns per-atom plddt in SFC ordering.

    Returns
    -------
    positions : Tensor [n_sfc_atoms, 3]
    plddt_atom : Tensor [n_sfc_atoms] or None
        Per-atom plddt in [0, 1], or None if plddt_tokens was not provided.
    """
    _boltz_available()

    b = 0  # batch index (always 0 for inference)

    # --- atom pad mask: True for real atoms (not padding) ---
    atom_mask_np = rk_utils.assert_numpy(feats["atom_pad_mask"][b] > 0.5)  # [N_atoms]

    # --- raw atom coordinates [N_atoms, 3], discard batch and pad dims ---
    coords_all = sample_atom_coords[b]  # [N_atoms, 3]

    # --- atom → token index [N_atoms] ---
    # feats["atom_to_token"] is one-hot [B, N_atoms, N_tokens]
    atom_to_token_idx = feats["atom_to_token"][b].argmax(-1)  # [N_atoms]

    # --- per-token features → per-atom via index ---
    residue_index = feats["residue_index"][b]   # [N_tokens], 0-based per-chain
    asym_id       = feats["asym_id"][b]         # [N_tokens], 0-based chain int
    res_type_oh   = feats["res_type"][b]        # [N_tokens, num_tokens] one-hot

    atom_res_idx  = residue_index[atom_to_token_idx]  # [N_atoms]
    atom_asym_id  = asym_id[atom_to_token_idx]        # [N_atoms]
    atom_res_type = res_type_oh[atom_to_token_idx]    # [N_atoms, num_tokens]

    # --- decode atom names ---
    # feats["ref_atom_name_chars"] is [B, N_atoms, 4, 64]
    all_atom_names = decode_atom_names(feats["ref_atom_name_chars"][b])

    # --- per-token residue names ---
    tok_res_names = _token_res_type_to_name(res_type_oh)  # [N_tokens] list
    # map to per-atom
    tok_idx_np = rk_utils.assert_numpy(atom_to_token_idx, arr_type=int)
    atom_res_names = [tok_res_names[i] for i in tok_idx_np]

    # --- build cra_name for every real (non-pad) atom ---
    atom_res_idx_np  = rk_utils.assert_numpy(atom_res_idx, arr_type=int)
    atom_asym_id_np  = rk_utils.assert_numpy(atom_asym_id, arr_type=int)

    cra_name_boltz2: list[str] = []
    real_indices: list[int] = []

    for i in range(len(all_atom_names)):
        if not atom_mask_np[i]:
            continue
        chain_letter = chr(ord("A") + atom_asym_id_np[i])
        res_num      = atom_res_idx_np[i]    # 0-based, matches SFC
        res_name     = atom_res_names[i]
        atom_name    = all_atom_names[i]
        if not atom_name:
            continue  # skip atoms with empty name (should not occur)
        cra_name_boltz2.append(f"{chain_letter}-{res_num}-{res_name}-{atom_name}")
        real_indices.append(i)

    # --- filter coords to real atoms ---
    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long,
                                   device=coords_all.device)
    valid_coords = coords_all[real_idx_tensor]  # [n_real, 3]

    # --- reorder to match SFC topology ---
    try:
        reorder_index = np.array(
            [cra_name_boltz2.index(name) for name in cra_name_sfc], dtype=np.int64
        )
    except ValueError as exc:
        missing = [n for n in cra_name_sfc if n not in cra_name_boltz2]
        raise ValueError(
            f"Topology mismatch: {len(missing)} SFC atoms not found in Boltz-2 output.\n"
            f"First missing: {missing[:5]}"
        ) from exc

    assert np.all(
        np.array(cra_name_boltz2)[reorder_index] == np.array(cra_name_sfc)
    ), "Topology mismatch between Boltz-2 and SFC after reordering!"

    positions = valid_coords[
        torch.tensor(reorder_index, dtype=torch.long, device=valid_coords.device)
    ]

    # --- optional per-atom plddt ---
    if plddt_tokens is not None:
        # plddt_tokens: [1, N_tokens], Boltz-2 range [0, 1]
        plddt_tok = plddt_tokens[b]  # [N_tokens]
        # per-atom (all atoms, including pad)
        atom_plddt_all = plddt_tok[atom_to_token_idx]  # [N_atoms]
        # filter to real atoms
        valid_plddt = atom_plddt_all[real_idx_tensor]  # [n_real]
        # reorder
        plddt_atom = valid_plddt[
            torch.tensor(reorder_index, dtype=torch.long, device=valid_plddt.device)
        ]
    else:
        plddt_atom = None

    return positions, plddt_atom


# ---------------------------------------------------------------------------
# Position-alignment helper (mirrors refinement_utils.position_alignment)
# ---------------------------------------------------------------------------

def position_alignment_boltz2(
    model_output: dict[str, Tensor],
    feats: dict[str, Tensor],
    cra_name_sfc: list[str],
    best_pos: Tensor,
    exclude_res: Optional[list[int]] = None,
    domain_segs: Optional[list[int]] = None,
    reference_bfactor: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Extract coordinates from Boltz-2 output, convert to pseudo-B weights,
    and run iterative Kabsch alignment against a reference position.

    Parameters
    ----------
    model_output : dict
        Output of ``Boltz2PairBias.forward()``.  Must contain
        ``sample_atom_coords`` [1, N_atoms, 3] and optionally ``plddt``
        [1, N_tokens] in [0, 1].
    feats : dict
        Featurizer-v2 features (batch dim = 1).
    cra_name_sfc : list[str]
        SFC atom name list.
    best_pos : Tensor [n_atoms, 3]
        Reference coordinates (typically previous-best or crystal model).
    exclude_res : list[int] or None
        0-based residue indices to exclude from alignment.
    domain_segs : list[int] or None
        Residue-index boundaries between rigid domains.
    reference_bfactor : Tensor [n_atoms] or None
        If provided, use these B-factors for alignment weights instead of
        the model's predicted pseudo-B.

    Returns
    -------
    aligned_xyz : Tensor [n_atoms, 3]   (with grad if input has grad)
    plddt_per_token : Tensor [N_tokens]  (per-token, [0,1])
    pseudo_Bs : Tensor [n_atoms]         (detached pseudo-B factors, Å²)
    """
    # plddt per-token: [1, N_tokens] or None
    plddt_tokens: Optional[Tensor] = model_output.get("plddt")

    xyz_orth_sfc, plddt_atom = extract_allatoms_boltz2(
        model_output["sample_atom_coords"],
        feats,
        cra_name_sfc,
        plddt_tokens=plddt_tokens,
    )

    # --- pseudo-B factors from plddt ---
    if plddt_atom is not None:
        # plddt_atom is in [0, 1]; plddt2pseudoB_pt expects [0, 100]
        pseudo_Bs = rk_utils.plddt2pseudoB_pt(plddt_atom * 100.0)
    else:
        # Fall back to a uniform medium B if confidence was not run
        pseudo_Bs = torch.full(
            (xyz_orth_sfc.shape[0],), 30.0,
            dtype=xyz_orth_sfc.dtype, device=xyz_orth_sfc.device
        )

    # --- alignment weights ---
    if reference_bfactor is None:
        b_np     = rk_utils.assert_numpy(pseudo_Bs)
        cutoff1  = np.quantile(b_np, 0.3)
        cutoff2  = cutoff1 * 1.5
        weights  = rk_utils.weighting(b_np, cutoff1, cutoff2)
    else:
        assert reference_bfactor.shape == pseudo_Bs.shape, (
            "reference_bfactor shape must match n_sfc_atoms"
        )
        ref_b_np = rk_utils.assert_numpy(reference_bfactor)
        cutoff1  = np.quantile(ref_b_np, 0.3)
        cutoff2  = cutoff1 * 1.5
        weights  = rk_utils.weighting(ref_b_np, cutoff1, cutoff2)

    # --- iterative Kabsch alignment ---
    aligned_xyz = iterative_kabsch_alignment(
        xyz_orth_sfc,
        best_pos,
        cra_name_sfc,
        weights=weights,
        exclude_res=exclude_res,
        domain_segs=domain_segs,
    )

    # per-token plddt for logging / downstream use (detached)
    if plddt_tokens is not None:
        plddt_per_token = plddt_tokens[0].detach()
    else:
        n_tokens = feats["token_pad_mask"].shape[1]
        plddt_per_token = torch.ones(n_tokens, device=xyz_orth_sfc.device) * 0.7

    return aligned_xyz, plddt_per_token, pseudo_Bs.detach()
