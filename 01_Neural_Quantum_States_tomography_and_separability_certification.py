#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:30:37 2025


@author: Marcin Plodzien
 
Quantum State Tomography with NNS / SNNS, non-k-separability, and entanglement certification 
====================================================================

High-level idea
---------------
We study how well Neural Network States (NNS) and Structured NNS (SNNS) can
learn quantum N-qubit pure states only from measurement statistics, mimicking
a realistic experiment.

Pipeline for one target state:

  1) Construct a target pure state |psi_target> on N qubits as a vector
     in the computational (Z) basis. Examples include GHZ, W, Dicke, and
     states built from Bell pairs.

  2) Choose local measurement bases for each qubit. A "basis string"
     of length N encodes a product basis, such as "ZXZYYZ", where
     each character in {Z, X, Y} is the local Pauli axis.

  3) Generate measurement data:
       - For each chosen basis, rotate |psi_target> from Z basis into
         that product basis using only local 2x2 rotations with tensordot.
       - Obtain probabilities p_target(outcome | basis) via the Born rule.
       - For finite shots_per_basis:
           * sample measurement outcomes according to these probabilities,
             and build empirical frequencies p_data.
         For "infinite" shots_per_basis:
           * use p_target directly as p_data.

  4) Define a parametric model |psi_theta> using:
       - NNS: one global RBM-like network for amplitude and another for phase.
       - SNNS: a product of NNS blocks defined on partitions of the qubits.

  5) Compute model probabilities p_model(outcome | basis) from |psi_theta>
     in each basis, again via local basis rotations and Born rule.

  6) Optimize parameters theta by minimizing the negative log-likelihood (NLL):

         L(theta) = - (1 / B) * sum_b sum_k
                     [ p_data(k | basis b) * log p_model(k | basis b; theta) ]

     where:
       - B is the number of distinct bases used in the experiment
         (either a fixed set, or a pool of random local bases).

  7) For diagnostics (not for training), compute the fidelity F between
     |psi_theta> and |psi_target>.

Measurement bases
-----------------
We support two main regimes.

1) Fixed global bases:
   - random_bases_per_shot = False
   - measurement_bases = ["Z"] or ["X", "Y", "Z"], etc.
   - Each element "X", "Y", or "Z" means: measure all N qubits in that axis.
   - For each of these bases, we compute p_data and include it in the NLL.

2) Random local bases with a finite pool:
   - random_bases_per_shot = True
   - measurement_bases is interpreted as the set of allowed local axes
     per qubit, e.g. ["X", "Y", "Z"].
   - We generate a pool of num_random_bases random basis strings, each of
     length N, such as "XYZZYX", where each character is in the allowed set.
   - We compute p_data for each of these basis strings.
   - For finite shots_per_basis, each basis in the pool gets exactly
     shots_per_basis projective measurement shots.

Filename conventions
--------------------
  - Measurement data:
      ./measurement_data/measdata_<case>_<suffix>.txt

  - Figures:
      ./figures/fig_nns_snns_loss_<suffix>.png
      ./figures/fig_nns_snns_fid_<suffix>.png
      ./figures/fig_nns_snns_topology_<suffix>.png

The suffix has the form:

  N<N>_bases-<bases_tag>_shots-<shots_tag>_nbases-<B>_shpb-<shpb>

where:

  - N<N>             : number of qubits, e.g. N6
  - bases_tag        :
        for fixed bases:       "Z", "XYZ", etc.
        for random local bases: "rand-XYZ" if allowed axes are {X, Y, Z}
  - shots_tag        : shots_per_basis number (shots per basis), or "inf"
                       if shots_per_basis <= 0
  - nbases           : number of effective bases B used in the NLL
  - shpb             : shots_per_basis, repeated for clarity

"""

import os
import copy
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import random, tree_util, value_and_grad, lax
from itertools import combinations
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

jax.config.update("jax_enable_x64", True)

# Make directory for figures
if not os.path.exists("./figures"):
    os.makedirs("./figures")

# Directory for raw simulated measurement data (ASCII text)
MEAS_DIR = "./measurement_data"
if not os.path.exists(MEAS_DIR):
    os.makedirs(MEAS_DIR)


CONFIG = {
    # --------------------------------------------------------------
    # Global RNG and system size
    # --------------------------------------------------------------
    "seed": 42,
    # Number of qubits. The Hilbert-space dimension is 2**N.
    "N": 12,

    # --------------------------------------------------------------
    # Training mode
    # --------------------------------------------------------------
    "loss_mode": "measurements",

    # --------------------------------------------------------------
    # Measurement bases configuration
    # --------------------------------------------------------------
    # Two regimes:
    #
    # 1) random_bases_per_shot == False
    #    measurement_bases is a list of global bases, e.g. ["Z"], ["X", "Y", "Z"].
    #    - "Z" means: all qubits measured in the Z eigenbasis.
    #    - "X" means: all qubits measured in the X eigenbasis.
    #    - "Y" means: all qubits measured in the Y eigenbasis.
    #
    #    In this regime the number of effective bases B is simply
    #    len(measurement_bases), and we compute one probability vector
    #    p_data(outcome | basis) per entry in measurement_bases.
    #
    # 2) random_bases_per_shot == True
    #    measurement_bases is a list of allowed local axes, e.g. ["X", "Y", "Z"].
    #
    #    - We construct a pool of num_random_bases random basis strings,
    #      each of length N, such as "XYZZYX".
    #    - Each such string assigns a local axis in {X, Y, Z} to each qubit.
    #    - We compute one probability vector per basis string in the pool.
    "measurement_bases": ["Z"],

    # --------------------------------------------------------------
    # Shot budget (statistics)
    # --------------------------------------------------------------
    # shots and shots_per_basis MUST be the same number.
    #
    # shots_per_basis:
    #   - If shots_per_basis <= 0:
    #       we do not sample; instead we use the exact Born probabilities
    #       as p_data (infinite statistics).
    #   - If shots_per_basis > 0:
    #       for each basis (fixed or random-pool element) we sample
    #       exactly shots_per_basis projective outcomes and build
    #       empirical frequencies.
    #
    # shots:
    #   - Kept for backward compatibility; must equal shots_per_basis.
    "shots": 0,
    "shots_per_basis": 0,

    # --------------------------------------------------------------
    # Random local basis pool (optional)
    # --------------------------------------------------------------
    # If True:
    #   - Act in random-basis regime (see above).
    #   - measurement_bases is the set of allowed local axes.
    #   - num_random_bases sets the size of the pool.
    "random_bases_per_shot": False,

    # Number of distinct random basis strings in the pool when
    # random_bases_per_shot == True.
    #
    # If None or <= 0, we choose:
    #   B_desired = min(30, number_of_possible_bases),
    # where number_of_possible_bases = len(measurement_bases) ** N.
    "num_random_bases": 30,

    # --------------------------------------------------------------
    # NNS hyperparameters (unconstrained ansatz)
    # --------------------------------------------------------------
    # The amplitude network is an RBM-like model:
    #   - N visible units (spins in {+1, -1})
    #   - H_amp hidden units
    #
    # The phase network is another RBM-like model with H_phase hidden units.
    "H_amp": 40,
    "H_phase": 40,

    # --------------------------------------------------------------
    # SNNS hyperparameters (per block)
    # --------------------------------------------------------------
    # For the structured ansatz SNNS we partition the qubits into
    # blocks, and each block is given its own NNS with:
    #   - H_amp_per hidden units in the amplitude network
    #   - H_phase_per hidden units in the phase network
    "H_amp_per": 40,
    "H_phase_per": 40,

    # --------------------------------------------------------------
    # Optimization parameters
    # --------------------------------------------------------------
    "epochs": 1500,    # Number of gradient steps with Adam
    "lr": 0.005,       # Learning rate
    "log_every": 1500,   # Record loss/fidelity every log_every epochs

    # --------------------------------------------------------------
    # Target state zoo
    # --------------------------------------------------------------
    # Each entry describes a target state:
    #   - "kind" selects the construction (GHZ, W, Dicke, etc.)
    #   - extra parameters are passed as needed (e.g. "m" for Dicke)
    "target_cases": [
        {"name": "1. GHZ State",            "kind": "GHZ"},
        # {"name": "2. W State",              "kind": "W"},
        # {"name": "3. Dicke k=2",            "kind": "dicke", "m": 2},
        # {"name": "4. Dicke k=3",            "kind": "dicke", "m": 3},
        # {"name": "5. 3 Bell Pairs",         "kind": "3bell"},
        # {"name": "6. 2 Bell + 2 Product",   "kind": "hybrid"},
    ],

    # --------------------------------------------------------------
    # Family of ansatzes to compare
    # --------------------------------------------------------------
    # "Unconstrained" is a single global NNS.
    # The SNNS variants have explicit partitions (blocks) of the qubits.
    "ansatzes": [
        
        {
            "name": "Unconstrained", 
            "type": "NNS"
        },
        

        {
            "name": "(N/6| N/6 | N/6 | N/6| N/6| N/6)",
            "type": "SNNS",
            "partition_mode": "explicit",
            "partitions": [[0,1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        },        

        
        {
            "name": "(N/4 | N/4 | N/4 | N/4)",
            "type": "SNNS",
            "partition_mode": "explicit",
            "partitions": [[0,1,2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        },
        
        {
            "name": "(N/3 | N/3 | N/3)",
            "type": "SNNS",
            "partition_mode": "explicit",
            "partitions": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        },    
        
        {
            "name": "(N/2 | N/2)",
            "type": "SNNS",
            "partition_mode": "explicit",
            "partitions": [[0, 1, 2, 3, 4, 5] , [6, 7, 8, 9, 10, 11]],
        }, 
        
    ],
}

CONFIG_LIST: List[Dict] = []

# Example config: random local XYZ bases with a finite pool
cfg4 = copy.deepcopy(CONFIG)
cfg4["name"] = "Random local XYZ bases, 5000 shots per basis"
cfg4["measurement_bases"] = ["X", "Y", "Z"]   # allowed local axes per qubit
# IMPORTANT: shots and shots_per_basis must be equal
cfg4["shots"] = 5000                          # shots per basis
cfg4["shots_per_basis"] = 5000
cfg4["random_bases_per_shot"] = True
cfg4["num_random_bases"] = 100                # random basis strings in the pool
CONFIG_LIST.append(cfg4)


# =====================================================================
# Basic utilities: spin configurations
# =====================================================================

def bin_to_spin(bin_array: jnp.ndarray) -> jnp.ndarray:
    """
    Map {0,1} bits to Ising spins {+1, -1}.
    """
    return 2 * bin_array - 1


def get_all_configs(N: int) -> jnp.ndarray:
    """
    Generate all 2**N spin configurations (+1 or -1) for N qubits.

    The output s has shape (2**N, N). Each row is a spin configuration.
    """
    ints = jnp.arange(2**N, dtype=jnp.int32)
    bits = (ints[:, None] >> jnp.arange(N)) & 1
    return bin_to_spin(bits)


# =====================================================================
# NNS ansatz: amplitude and phase networks
# =====================================================================

def amp_wavefunction(p: Dict[str, jnp.ndarray], s: jnp.ndarray) -> jnp.ndarray:
    """
    Amplitude network for NNS, RBM-like structure.

    Given spins s in {+1, -1}^N:

        a_term(s)  = exp(sum_i a_i s_i)
        lin_j(s)   = sum_i W_{ij} s_i + b_j
        prod_term  = prod_j 2 cosh(lin_j(s))

        amp(s) = a_term(s) * prod_term

    Args:
        p: dict with keys 'a', 'b', 'W'
        s: array (..., N) of spins

    Returns:
        amplitude values amp(s) as an array (...,).
    """
    a_term = jnp.exp(jnp.einsum("...i,i->...", s, p["a"]))
    lin = jnp.einsum("...i,ij->...j", s, p["W"]) + p["b"]
    prod_term = jnp.prod(2 * jnp.cosh(lin), axis=-1)
    return a_term * prod_term


def phase_function(p: Dict[str, jnp.ndarray], s: jnp.ndarray) -> jnp.ndarray:
    """
    Phase network for NNS.

    We choose a simple nonlinear model for the phase:

        phi(s) = sum_i c_i s_i + sum_j f( sum_i U_{ij} s_i + d_j ),

    where f is a bounded nonlinearity mapping R to approximately [0, 1]:

        f(x) = (2/pi) * arctan(tanh(x)) + 1/2

    Args:
        p: dict with keys 'c', 'd', 'U'
        s: array (..., N) of spins

    Returns:
        real-valued phases phi(s) as array (...,).
    """
    c_term = jnp.einsum("...i,i->...", s, p["c"])
    lin = jnp.einsum("...i,ij->...j", s, p["U"]) + p["d"]
    f = (2.0 / jnp.pi) * jnp.arctan(jnp.tanh(lin)) + 0.5
    sum_f = jnp.sum(f, axis=-1)
    return c_term + sum_f


def nns_wavefunction(params: Dict[str, Dict[str, jnp.ndarray]],
                     s: jnp.ndarray) -> jnp.ndarray:
    """
    Full NNS wavefunction:

        psi(s) = amp(s) * exp( 2 pi i * phi(s) )

    Args:
        params: dict with "amp" and "phase" parameter sub-dicts
        s: array of spins of shape (num_configs, N)

    Returns:
        complex-valued amplitudes psi(s) as array (num_configs,).
    """
    amp = amp_wavefunction(params["amp"], s)
    phase = phase_function(params["phase"], s)
    return amp * jnp.exp(2j * jnp.pi * phase)


def init_params(key: random.PRNGKey,
                N: int,
                H_amp: int,
                H_phase: int,
                scale: float = 0.01) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Initialize amplitude and phase parameters with small Gaussian noise.

    scale sets the standard deviation of the initial weights and biases.
    """
    keys = random.split(key, 6)
    amp_a = scale * random.normal(keys[0], (N,))
    amp_b = scale * random.normal(keys[1], (H_amp,))
    amp_W = scale * random.normal(keys[2], (N, H_amp))
    phase_c = scale * random.normal(keys[3], (N,))
    phase_d = scale * random.normal(keys[4], (H_phase,))
    phase_U = scale * random.normal(keys[5], (N, H_phase))
    return {
        "amp":   {"a": amp_a,   "b": amp_b,   "W": amp_W},
        "phase": {"c": phase_c, "d": phase_d, "U": phase_U},
    }


# =====================================================================
# Fidelity (diagnostic only)
# =====================================================================

def fidelity(params: Dict[str, Dict[str, jnp.ndarray]],
             target_amp: jnp.ndarray,
             all_s: jnp.ndarray) -> float:
    """
    Compute fidelity:

        F = |<psi_model | psi_target>|^2 / ||psi_model||^2

    psi_target is assumed normalized. psi_model is normalized internally.
    This is NOT used as a loss function, only as a diagnostic readout.
    """
    psi_values = nns_wavefunction(params, all_s)
    psi_norm_sq = jnp.sum(jnp.abs(psi_values) ** 2)
    inner = jnp.sum(psi_values * jnp.conj(target_amp))
    return jnp.abs(inner) ** 2 / psi_norm_sq


# =====================================================================
# Adam optimizer
# =====================================================================

def adam_update(params, grads, m, v,
                lr: float, beta1: float, beta2: float, eps: float, t: int):
    """
    One Adam update step for arbitrary JAX PyTrees.

    params, grads, m, v share the same tree structure.
    """
    def update_m(mm, g):
        return beta1 * mm + (1.0 - beta1) * g

    def update_v(vv, g):
        return beta2 * vv + (1.0 - beta2) * (g ** 2)

    m = tree_util.tree_map(update_m, m, grads)
    v = tree_util.tree_map(update_v, v, grads)

    beta1_t = beta1 ** t
    beta2_t = beta2 ** t

    m_hat = tree_util.tree_map(lambda mm: mm / (1.0 - beta1_t), m)
    v_hat = tree_util.tree_map(lambda vv: vv / (1.0 - beta2_t), v)

    def param_update(p, mh, vh):
        return p - lr * mh / (jnp.sqrt(vh) + eps)

    params = tree_util.tree_map(param_update, params, m_hat, v_hat)
    return params, m, v


# =====================================================================
# SNNS: structured NNS
# =====================================================================

def snns_wavefunction(sub_params: List[Dict[str, Dict[str, jnp.ndarray]]],
                      partitions: List[List[int]],
                      s: jnp.ndarray) -> jnp.ndarray:
    """
    SNNS wavefunction as product over blocks:

        psi_SNNS(s) = product over blocks B: psi_B( s_B )

    where s_B is the restriction of the spin configuration to the block.
    """
    psi_total = jnp.ones(s.shape[0], dtype=jnp.complex128)
    for idx, part in enumerate(partitions):
        s_part = s[:, part]
        psi_total *= nns_wavefunction(sub_params[idx], s_part)
    return psi_total


def snns_fidelity(sub_params: List[Dict[str, Dict[str, jnp.ndarray]]],
                  partitions: List[List[int]],
                  target_amp: jnp.ndarray,
                  all_s: jnp.ndarray) -> float:
    """
    Fidelity for SNNS, same definition as for NNS.
    """
    psi_values = snns_wavefunction(sub_params, partitions, all_s)
    psi_norm_sq = jnp.sum(jnp.abs(psi_values) ** 2)
    inner = jnp.sum(psi_values * jnp.conj(target_amp))
    return jnp.abs(inner) ** 2 / psi_norm_sq


def init_snns_params(key: random.PRNGKey,
                     partitions: List[List[int]],
                     H_amp_per: int,
                     H_phase_per: int,
                     scale: float = 0.01) -> List[Dict[str, Dict[str, jnp.ndarray]]]:
    """
    Initialize one NNS parameter set per partition block.
    """
    sub_keys = random.split(key, len(partitions))
    sub_params = []
    for sk, part in zip(sub_keys, partitions):
        sub_N = len(part)
        sub_params.append(init_params(sk, sub_N, H_amp_per, H_phase_per, scale))
    return sub_params


def adam_update_snns(sub_params, grads, m, v,
                     lr, beta1, beta2, eps, t):
    """
    Block-wise Adam update for SNNS parameter sets.
    """
    new_sub_params = []
    new_m = []
    new_v = []
    for sp, g, mm, vv in zip(sub_params, grads, m, v):
        updated_sp, updated_mm, updated_vv = adam_update(sp, g, mm, vv, lr, beta1, beta2, eps, t)
        new_sub_params.append(updated_sp)
        new_m.append(updated_mm)
        new_v.append(updated_vv)
    return new_sub_params, new_m, new_v


# =====================================================================
# Partition helpers
# =====================================================================

def fully_separable_partitions(N: int) -> List[List[int]]:
    """
    Fully separable: each qubit in its own block.
    """
    return [[i] for i in range(N)]


def build_partitions_from_groups(N: int,
                                 groups: List[List[int]]) -> List[List[int]]:
    """
    Build partitions from given groups plus any leftover singletons.

    groups is a list of lists, where each inner list is a group of
    qubit indices. Any qubit not in any group becomes its own block.
    """
    used = set()
    partitions: List[List[int]] = []
    for g in groups:
        g_sorted = sorted(set(g))
        if any(i < 0 or i >= N for i in g_sorted):
            raise ValueError("Invalid qubit index in group {}".format(g_sorted))
        if any(i in used for i in g_sorted):
            raise ValueError("Qubit appears in multiple groups: {}".format(g_sorted))
        used.update(g_sorted)
        partitions.append(g_sorted)
    for i in range(N):
        if i not in used:
            partitions.append([i])
    return partitions


def validate_explicit_partitions(N: int,
                                 partitions: List[List[int]]) -> None:
    """
    Check that explicit partitions cover each qubit index exactly once.
    """
    flat = sorted(i for block in partitions for i in block)
    if flat != list(range(N)):
        raise ValueError(
            "Explicit partitions must cover 0..{} once. Got indices {}."
            .format(N - 1, flat)
        )


def resolve_partitions_from_config(N: int, ansatz_cfg: Dict) -> List[List[int]]:
    """
    Resolve partition structure from ansatz configuration.

    Modes:
      - "full_sep": fully separable (each qubit its own block)
      - "groups":   user specifies multi-qubit groups; others become singletons
      - "explicit": user supplies the full list of blocks
    """
    mode = ansatz_cfg.get("partition_mode", "full_sep")
    if mode == "full_sep":
        return fully_separable_partitions(N)
    elif mode == "groups":
        groups = ansatz_cfg.get("groups", [])
        if not groups:
            raise ValueError("partition_mode='groups' requires 'groups'.")
        return build_partitions_from_groups(N, groups)
    elif mode == "explicit":
        partitions = ansatz_cfg.get("partitions", [])
        if not partitions:
            raise ValueError("partition_mode='explicit' requires 'partitions'.")
        validate_explicit_partitions(N, partitions)
        return partitions
    else:
        raise ValueError("Unknown partition_mode: {}".format(mode))


# =====================================================================
# Local Basis rotations  
# =====================================================================

# Encode axes as integers: 0 -> Z, 1 -> X, 2 -> Y.
AXIS_CHARS = ["Z", "X", "Y"]
AXIS_TO_INDEX = {c: i for i, c in enumerate(AXIS_CHARS)}

# LOCAL_BASIS_MATS[axis_index] is a 2x2 unitary matrix that takes Z-basis
# kets to eigenstates of sigma_axis.
LOCAL_BASIS_MATS = jnp.stack([
    # Z basis: identity
    jnp.array([[1.0, 0.0],
               [0.0, 1.0]], dtype=jnp.complex128),
    # X basis: Hadamard-like
    (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0,  1.0],
                                       [1.0, -1.0]], dtype=jnp.complex128),
    # Y basis: eigenstates of sigma_y
    (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, -1j],
                                       [1.0,  1j]], dtype=jnp.complex128),
], axis=0)  # shape (3, 2, 2)


def axis_string_to_indices(basis_spec: str, N: int) -> List[int]:
    """
    Convert a basis string like 'ZXZYYZ' to a list of integer axis indices.

    If the string length is 1, e.g. 'Z', it is broadcast to length N.
    """
    if len(basis_spec) == 1:
        basis_spec = basis_spec * N
    if len(basis_spec) != N:
        raise ValueError("basis_spec must have length 1 or N={}.".format(N))
    return [AXIS_TO_INDEX[c.upper()] for c in basis_spec]


def apply_local_basis_to_state_indices(psi_Z: jnp.ndarray,
                                       N: int,
                                       basis_axes: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate an N-qubit state from Z basis to the product basis defined by
    basis_axes (integer-encoded axes for each qubit).

    Implementation:
      - Reshape psi_Z to an N-index tensor of shape (2, 2, ..., 2).
      - For each qubit q:
          * pick U_q = LOCAL_BASIS_MATS[axis_q]
          * contract along axis q via tensordot
          * permute axes to keep qubit ordering consistent
      - Finally, flatten back to a vector of length 2**N.

    This avoids constructing any 2^N x 2^N matrices explicitly.
    """
    psi = psi_Z.reshape((2,) * N)
    for q in range(N):
        axis_idx = basis_axes[q]
        U = LOCAL_BASIS_MATS[axis_idx]  # shape (2, 2)
        axis_index = q
        # Contract psi along axis_index with U's column index (index 1)
        psi = jnp.tensordot(psi, U, axes=[[axis_index], [1]])
        # After tensordot, the new axis for the output qubit is at the end.
        # We move it back to position axis_index to keep a fixed ordering.
        num_axes = psi.ndim
        new_axis_pos = num_axes - 1
        axes_order = list(range(num_axes))
        axes_order.insert(axis_index, axes_order.pop(new_axis_pos))
        psi = jnp.transpose(psi, axes_order)
    return psi.reshape(-1)


def apply_local_basis_to_state_string(psi_Z: jnp.ndarray,
                                      N: int,
                                      basis_spec: str) -> jnp.ndarray:
    """
    Same as apply_local_basis_to_state_indices, but starting from
    a human-readable basis string like "XZYYZX".
    """
    axes = axis_string_to_indices(basis_spec, N)
    basis_axes = jnp.array(axes, dtype=jnp.int32)
    return apply_local_basis_to_state_indices(psi_Z, N, basis_axes)


# =====================================================================
# Data generation from target state
# =====================================================================

def make_data_probs_from_target(target_amp: jnp.ndarray,
                                N: int,
                                basis_labels: List[str],
                                shots: int = None,
                                seed: int = None,
                                random_bases_per_shot: bool = False,
                                num_random_bases: int = None) -> Tuple[jnp.ndarray,
                                                                        List[str],
                                                                        List[Dict]]:
    """
    From target wavefunction, compute measurement probabilities and simulate
    (optionally finite) measurement data.

    IMPORTANT: here "shots" is interpreted as shots_per_basis:
      - For fixed bases: each basis gets exactly "shots" samples (if shots > 0).
      - For random-basis pool: each basis in the pool gets exactly "shots"
        samples (if shots > 0).

    Modes:

    1) random_bases_per_shot == False (fixed global bases)
       ---------------------------------------------------
       basis_labels is a list of strings, each either:
         - "Z", "X", "Y" (broadcast to all qubits), or
         - a length-N string of X/Y/Z specifying local axes.
       For each label:
         - we compute psi in that basis and its Born probabilities p_k,
         - if shots > 0, we sample "shots" outcomes and form empirical
           frequencies,
         - if shots <= 0, we use p_k directly.

       Effective number of bases B is len(basis_labels).

    2) random_bases_per_shot == True (random local basis pool)
       --------------------------------------------------------
       basis_labels is interpreted as a list of allowed local axes,
       e.g. ["X", "Y", "Z"].
       Steps:
         - Let A = number of allowed axes, so there are A**N possible
           basis strings.
         - We choose B (<= num_random_bases, and B <= A**N).
         - We sample B distinct random basis strings of length N from
           the allowed axes.
         - For each basis in the pool we compute psi and Born probabilities p_k.
         - If shots <= 0: we return these exact probabilities
           (infinite statistics).
         - If shots > 0: for each basis in the pool we sample exactly
           "shots" outcomes and form empirical frequencies.

    Returns:
        data_probs_array: array of shape (B, 2**N)
            Empirical or exact probabilities for each basis.
        basis_specs: list of length B of full basis strings (length N).
        measurements: list of raw shot dicts with fields:
            "outcome_int": integer in [0, 2**N),
            "basis_spec":  basis string used for that shot.
    """
    # Normalize target state just in case
    target_amp = target_amp / jnp.sqrt(jnp.sum(jnp.abs(target_amp) ** 2))
    dim = 2 ** N
    measurements: List[Dict] = []

    rng = np.random.RandomState(seed if seed is not None else None)
    use_sampling = shots is not None and shots > 0

    # ------------------------------------------------------------------
    # Mode 2: random pool of local bases
    # ------------------------------------------------------------------
    if random_bases_per_shot:
        allowed_axes = [str(b).upper() for b in basis_labels]
        if len(allowed_axes) == 0:
            raise ValueError("measurement_bases must be non-empty when random_bases_per_shot=True.")

        # Total number of possible basis strings (A**N).
        num_possible = len(allowed_axes) ** N

        # Choose desired number of random bases in the pool.
        if num_random_bases is not None and num_random_bases > 0:
            B_desired = min(num_random_bases, num_possible)
        else:
            # Default pool size: at most 30, limited by number of possible bases.
            B_desired = min(30, num_possible)

        B = B_desired

        # Sample B distinct random basis strings.
        basis_pool: List[str] = []
        basis_set = set()
        while len(basis_pool) < B:
            basis_chars = [rng.choice(allowed_axes) for _ in range(N)]
            basis_full = "".join(basis_chars)
            if basis_full not in basis_set:
                basis_set.add(basis_full)
                basis_pool.append(basis_full)

        # Precompute target probabilities for each basis in the pool.
        probs_pool: List[jnp.ndarray] = []
        for basis_full in basis_pool:
            psi_basis = apply_local_basis_to_state_string(target_amp, N, basis_full)
            p = jnp.abs(psi_basis) ** 2
            p = p / jnp.sum(p)
            probs_pool.append(p)

        # Infinite-statistics mode: just return exact probabilities.
        if not use_sampling:
            data_probs_array = jnp.stack(probs_pool, axis=0)
            basis_specs = basis_pool
            return data_probs_array, basis_specs, measurements

        # Finite shots: each basis gets exactly "shots" outcomes.
        hist = np.zeros((B, dim), dtype=np.float64)
        for b_idx in range(B):
            p_np = np.asarray(probs_pool[b_idx], dtype=float)
            samples = rng.choice(dim, size=shots, p=p_np)
            counts = np.bincount(samples, minlength=dim).astype(np.float64)
            hist[b_idx] = counts
            for outcome in samples:
                measurements.append(
                    {
                        "outcome_int": int(outcome),
                        "basis_spec": basis_pool[b_idx],
                    }
                )

        probs_list = []
        for b_idx in range(B):
            counts = hist[b_idx]
            total_b = counts.sum()
            if total_b > 0:
                freqs = counts / float(total_b)
            else:
                # In a degenerate case where this basis gets 0 shots
                # (should not happen here), fall back to exact p.
                freqs = np.asarray(probs_pool[b_idx], dtype=float)
            probs_list.append(jnp.array(freqs))

        data_probs_array = jnp.stack(probs_list, axis=0)
        basis_specs = basis_pool
        return data_probs_array, basis_specs, measurements

    # ------------------------------------------------------------------
    # Mode 1: fixed global bases
    # ------------------------------------------------------------------
    probs_list = []
    basis_specs: List[str] = []
    for label in basis_labels:
        basis_full = label * N if len(label) == 1 else label
        basis_specs.append(basis_full)

        psi_basis = apply_local_basis_to_state_string(target_amp, N, basis_full)
        p = jnp.abs(psi_basis) ** 2
        p = p / jnp.sum(p)

        if use_sampling:
            # Sample "shots" outcomes from this basis.
            p_np = np.asarray(p, dtype=float)
            samples = rng.choice(dim, size=shots, p=p_np)
            counts = np.bincount(samples, minlength=dim).astype(np.float64)
            freqs = counts / float(shots)
            probs_list.append(jnp.array(freqs))
            for idx in samples:
                measurements.append(
                    {
                        "outcome_int": int(idx),
                        "basis_spec": basis_full,
                    }
                )
        else:
            probs_list.append(p)

    data_probs_array = jnp.stack(probs_list, axis=0)
    return data_probs_array, basis_specs, measurements


# =====================================================================
# NLL with local rotations  
# =====================================================================

def encode_basis_specs_to_axes(basis_specs: List[str], N: int) -> jnp.ndarray:
    """
    Convert a list of basis strings into an integer array of shape (B, N),
    where each entry is in {0, 1, 2} corresponding to {Z, X, Y}.
    """
    axes_rows = []
    for spec in basis_specs:
        axes_rows.append(axis_string_to_indices(spec, N))
    return jnp.array(axes_rows, dtype=jnp.int32)


def nll_loss_nns_local(params,
                       data_probs_array: jnp.ndarray,
                       all_s: jnp.ndarray,
                       basis_axes_array: jnp.ndarray,
                       N: int,
                       eps: float = 1e-12) -> float:
    """
    Negative log-likelihood for NNS using local basis rotations only.

    data_probs_array has shape (B, 2**N), one probability vector per basis.
    basis_axes_array has shape (B, N), each row encoding a basis string.

    NLL is:
        L = (1 / B) * sum_b cross_entropy(p_data_b, p_model_b),
    where:
        cross_entropy(p, q) = - sum_k p_k log(q_k).
    """
    psi_Z = nns_wavefunction(params, all_s)
    psi_Z = psi_Z / jnp.sqrt(jnp.sum(jnp.abs(psi_Z) ** 2))

    B = data_probs_array.shape[0]
    total_ce = 0.0
    for b in range(B):
        basis_axes = basis_axes_array[b]
        psi_basis = apply_local_basis_to_state_indices(psi_Z, N, basis_axes)
        p_model = jnp.abs(psi_basis) ** 2
        p_model = p_model / jnp.sum(p_model)
        ce_b = -jnp.sum(data_probs_array[b] * jnp.log(p_model + eps))
        total_ce = total_ce + ce_b
    return total_ce / float(B)


def nll_loss_snns_local(sub_params,
                        partitions,
                        data_probs_array: jnp.ndarray,
                        all_s: jnp.ndarray,
                        basis_axes_array: jnp.ndarray,
                        N: int,
                        eps: float = 1e-12) -> float:
    """
    Negative log-likelihood for SNNS using local basis rotations only.
    """
    psi_Z = snns_wavefunction(sub_params, partitions, all_s)
    psi_Z = psi_Z / jnp.sqrt(jnp.sum(jnp.abs(psi_Z) ** 2))

    B = data_probs_array.shape[0]
    total_ce = 0.0
    for b in range(B):
        basis_axes = basis_axes_array[b]
        psi_basis = apply_local_basis_to_state_indices(psi_Z, N, basis_axes)
        p_model = jnp.abs(psi_basis) ** 2
        p_model = p_model / jnp.sum(p_model)
        ce_b = -jnp.sum(data_probs_array[b] * jnp.log(p_model + eps))
        total_ce = total_ce + ce_b
    return total_ce / float(B)


# =====================================================================
# Training loops (measurement-only)
# =====================================================================

def train_nns(key: random.PRNGKey,
              N: int,
              H_amp: int,
              H_phase: int,
              target_amp: jnp.ndarray,
              all_s: jnp.ndarray,
              epochs: int,
              lr: float,
              data_probs_array: jnp.ndarray,
              basis_axes_array: jnp.ndarray,
              log_every: int):
    """
    Train the unconstrained NNS using measurement-based NLL only.
    """
    params = init_params(key, N, H_amp, H_phase)
    m = tree_util.tree_map(jnp.zeros_like, params)
    v = tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def one_step(carry, step_idx):
        params, m, v, t = carry

        def loss_fn(p):
            return nll_loss_nns_local(p, data_probs_array, all_s, basis_axes_array, N)

        loss_val, grads = value_and_grad(loss_fn)(params)
        new_t = t + 1
        new_params, new_m, new_v = adam_update(params, grads, m, v,
                                               lr, beta1, beta2, eps, new_t)
        current_fid = fidelity(new_params, target_amp, all_s)
        new_carry = (new_params, new_m, new_v, new_t)
        metrics = (loss_val, current_fid)
        return new_carry, metrics

    one_step_jit = jax.jit(one_step)

    init_carry = (params, m, v, 0)
    step_indices = jnp.arange(epochs, dtype=jnp.int32)
    final_carry, metrics_all = lax.scan(one_step_jit, init_carry, step_indices)

    final_params, _, _, _ = final_carry
    loss_all, fid_all = metrics_all

    idx = np.arange(0, epochs, log_every)
    loss_hist = np.asarray(loss_all)[idx].tolist()
    fid_hist = np.asarray(fid_all)[idx].tolist()
    steps = idx.tolist()
    return final_params, loss_hist, fid_hist, steps


def train_snns(key: random.PRNGKey,
               partitions: List[List[int]],
               H_amp_per: int,
               H_phase_per: int,
               target_amp: jnp.ndarray,
               all_s: jnp.ndarray,
               epochs: int,
               lr: float,
               data_probs_array: jnp.ndarray,
               basis_axes_array: jnp.ndarray,
               N: int,
               log_every: int):
    """
    Train the structured NNS (SNNS) using measurement-based NLL only.
    """
    key, key_init = random.split(key)
    sub_params = init_snns_params(key_init, partitions, H_amp_per, H_phase_per)
    m = [tree_util.tree_map(jnp.zeros_like, sp) for sp in sub_params]
    v = [tree_util.tree_map(jnp.zeros_like, sp) for sp in sub_params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def one_step(carry, step_idx):
        sub_params, m_list, v_list, t = carry

        def loss_fn(sp):
            return nll_loss_snns_local(sp, partitions, data_probs_array,
                                       all_s, basis_axes_array, N)

        loss_val, grads = value_and_grad(loss_fn)(sub_params)
        new_t = t + 1
        new_sub_params, new_m_list, new_v_list = adam_update_snns(sub_params, grads,
                                                                  m_list, v_list,
                                                                  lr, beta1, beta2,
                                                                  eps, new_t)
        current_fid = snns_fidelity(new_sub_params, partitions, target_amp, all_s)
        new_carry = (new_sub_params, new_m_list, new_v_list, new_t)
        metrics = (loss_val, current_fid)
        return new_carry, metrics

    one_step_jit = jax.jit(one_step)

    init_carry = (sub_params, m, v, 0)
    step_indices = jnp.arange(epochs, dtype=jnp.int32)
    final_carry, metrics_all = lax.scan(one_step_jit, init_carry, step_indices)

    final_sub_params, _, _, _ = final_carry
    loss_all, fid_all = metrics_all

    idx = np.arange(0, epochs, log_every)
    loss_hist = np.asarray(loss_all)[idx].tolist()
    fid_hist = np.asarray(fid_all)[idx].tolist()
    steps = idx.tolist()
    return final_sub_params, loss_hist, fid_hist, steps


# =====================================================================
# Target state definitions
# =====================================================================

def get_ghz_state(N: int) -> jnp.ndarray:
    """
    GHZ state: (|0...0> + |1...1>) / sqrt(2).
    """
    amp = jnp.zeros(2**N, dtype=jnp.complex128)
    amp = amp.at[0].set(1 / jnp.sqrt(2))
    amp = amp.at[2**N - 1].set(1 / jnp.sqrt(2))
    return amp


def get_w_state(N: int) -> jnp.ndarray:
    """
    W state: equal superposition of all states with exactly one excitation.
    """
    amp = jnp.zeros(2**N, dtype=jnp.complex128)
    for q in range(N):
        idx = 1 << q
        amp = amp.at[idx].set(1.0)
    return amp / jnp.sqrt(N)


def get_dicke_state(N: int, m: int) -> jnp.ndarray:
    """
    Dicke state |N, m> with exactly m excitations (Hamming weight m).
    """
    if m < 0 or m > N:
        raise ValueError("m must be between 0 and N.")
    amp = jnp.zeros(2**N, dtype=jnp.complex128)
    norm = 1 / jnp.sqrt(comb(N, m))
    for pos in combinations(range(N), m):
        index = sum(1 << p for p in pos)
        amp = amp.at[index].set(norm)
    return amp


def get_3bell_state(N: int) -> jnp.ndarray:
    """
    3 Bell pairs on 6 qubits: (0,1), (2,3), (4,5).
    """
    if N != 6:
        raise ValueError("get_3bell_state is defined for N=6 only.")
    bell = jnp.array([1, 0, 0, 1], dtype=jnp.complex128) / jnp.sqrt(2)
    amp = bell
    amp = jnp.kron(amp, bell)
    amp = jnp.kron(amp, bell)
    return amp


def get_hybrid_state(N: int) -> jnp.ndarray:
    """
    Hybrid state for N=6:

      Bell(0,1) tensor Bell(2,3) tensor |+>_4 tensor |+>_5,

    where |+> = (|0> + |1>) / sqrt(2).
    """
    if N != 6:
        raise ValueError("get_hybrid_state is defined for N=6 only.")
    bell = jnp.array([1, 0, 0, 1], dtype=jnp.complex128) / jnp.sqrt(2)
    plus = jnp.array([1, 1], dtype=jnp.complex128) / jnp.sqrt(2)
    amp = bell
    amp = jnp.kron(amp, bell)
    amp = jnp.kron(amp, plus)
    amp = jnp.kron(amp, plus)
    return amp


def get_time_evolved_state(N: int, t: float) -> jnp.ndarray:
    """
    One-axis twisting state:

        |psi(t)> = exp(-i * t * pi * Sz^2) |+>^N,

    where Sz = (1/2) sum_i sigma_z^i, and t is a dimensionless time.
    """
    k = jnp.arange(2**N, dtype=jnp.int32)
    bits = (k[:, None] >> jnp.arange(N)) & 1
    w = jnp.sum(bits, axis=1)
    sz = N / 2.0 - w
    phase = jnp.exp(-1j * jnp.pi * (sz ** 2) * t)
    amp = phase / jnp.sqrt(2**N)
    return amp


def build_target_state(N: int, case_cfg: Dict) -> jnp.ndarray:
    """
    Construct target state based on case configuration.

    Supported kinds:
      - "GHZ", "W", "dicke", "3bell", "hybrid", "time"
    """
    kind = case_cfg["kind"]
    if kind == "GHZ":
        amp = get_ghz_state(N)
    elif kind == "W":
        amp = get_w_state(N)
    elif kind == "dicke":
        m = case_cfg.get("m", 1)
        amp = get_dicke_state(N, m)
    elif kind == "3bell":
        amp = get_3bell_state(N)
    elif kind == "hybrid":
        amp = get_hybrid_state(N)
    elif kind == "time":
        t = case_cfg["t"]
        amp = get_time_evolved_state(N, t)
    else:
        raise ValueError("Unknown target kind '{}'".format(kind))
    amp = amp / jnp.sqrt(jnp.sum(jnp.abs(amp) ** 2))
    return amp


# =====================================================================
# Wavefunction comparison helper
# =====================================================================

def print_wavefunction_comparison(target_amp: jnp.ndarray,
                                  learned_amp: jnp.ndarray,
                                  N: int,
                                  case_name: str,
                                  ansatz_name: str,
                                  max_rows: int = 64) -> None:
    """
    Print a row-wise comparison of target and learned wavefunctions in
    the computational (Z) basis.

    Both target_amp and learned_amp are internally normalized so that
    sum_k |psi_k|^2 = 1 for each state. This guarantees that the
    printed amplitudes have magnitudes at most 1.

    Rows are indexed by bitstrings |b_{N-1} ... b_0>, where each b_q in {0,1}.

    For each basis configuration we print:

        bitstring | Re[target] | Re[learned] | Im[target] | Im[learned] |
        Abs(Re[target - learned]) | Abs(Im[target - learned])

    The parameter max_rows controls how many basis states are printed.
    For large N you may want to keep this small; for N = 6, max_rows = 64
    prints the full 2**N basis.
    """
    if(N <= 6):
        # Defensive: normalize both states here
        t_norm = jnp.sqrt(jnp.sum(jnp.abs(target_amp) ** 2))
        l_norm = jnp.sqrt(jnp.sum(jnp.abs(learned_amp) ** 2))
    
        # Avoid division by zero in case of numerical disasters
        if float(t_norm) == 0.0:
            raise ValueError("Target amplitude vector has zero norm.")
        if float(l_norm) == 0.0:
            raise ValueError("Learned amplitude vector has zero norm.")
    
        target_amp = target_amp / t_norm
        learned_amp = learned_amp / l_norm
    
        dim = target_amp.shape[0]
        if learned_amp.shape[0] != dim:
            raise ValueError(
                "Target and learned amplitudes must have the same length. "
                "Got {} and {}.".format(dim, learned_amp.shape[0])
            )
    
        # Cast to NumPy for pretty printing
        t_np = np.asarray(target_amp, dtype=np.complex128)
        l_np = np.asarray(learned_amp, dtype=np.complex128)
    
        # Optionally truncate
        num_rows = min(dim, int(max_rows))
    
        print("")
        print("    Wavefunction comparison for case '{}' and ansatz '{}'".format(
            case_name, ansatz_name
        ))
        print("    Showing {} of {} basis states (Z basis).".format(num_rows, dim))
        print("    Columns:")
        print("      bitstring | Re[target] | Re[learned] | Im[target] | Im[learned] | "
              "Abs(Re[target - learned]) | Abs(Im[target - learned])")
        print("")
    
        for k in range(num_rows):
            bitstring = format(k, "0{}b".format(N))
    
            t_val = t_np[k]
            l_val = l_np[k]
    
            re_t = float(np.real(t_val))
            im_t = float(np.imag(t_val))
            re_l = float(np.real(l_val))
            im_l = float(np.imag(l_val))
    
            re_diff = abs(re_t - re_l)
            im_diff = abs(im_t - im_l)
    
            print(
                "    {}  |  {: .3f}  |  {: .3f} |  {: .3f}  |  {: .3f}  |  {: .3f}  |  {: .3f}".format(
                    bitstring,
                    re_t,
                    re_l,
                    im_t,
                    im_l,
                    re_diff,
                    im_diff,
                )
            )
        print("")


# =====================================================================
# Visualization helpers
# =====================================================================

def create_viz_mask(N: int, num_hidden: int, partitions: List[List[int]]) -> np.ndarray:
    """
    Create a visibility mask for topology visualization.

    mask[v, h] = 1 if visible qubit v connects to hidden unit h
    (conceptually, for drawing graphs).
    """
    mask = np.zeros((N, num_hidden), dtype=np.float32)
    if num_hidden <= 0:
        return mask
    total_qubits = float(sum(len(g) for g in partitions))
    if total_qubits == 0:
        return mask
    current_h = 0
    for block_index, group in enumerate(partitions):
        if block_index == len(partitions) - 1:
            h_end = num_hidden
        else:
            fraction = len(group) / total_qubits
            h_count = max(1, int(round(fraction * num_hidden)))
            h_end = min(num_hidden, current_h + h_count)
        for v in group:
            mask[v, current_h:h_end] = 1.0
        current_h = h_end
    return mask


def get_ansatz_partitions_for_viz(N: int, ansatz_cfg: Dict) -> List[List[int]]:
    """
    For visualization we treat the NNS as a single block and the SNNS
    according to its explicit partitions.
    """
    if ansatz_cfg["type"] == "NNS":
        return [[i for i in range(N)]]
    else:
        return resolve_partitions_from_config(N, ansatz_cfg)


def config_suffix(cfg: Dict) -> str:
    """
    Build a suffix string encoding key configuration parameters.

    Includes:
      - N: number of qubits
      - bases_tag: description of bases
          * fixed: "Z", "XYZ", etc.
          * random local: "rand-XYZ" if allowed axes are {X, Y, Z}
      - shots_tag: shots_per_basis or "inf"
      - nbases_effective: number of effective bases B (if provided)
      - shots_per_basis_nominal: here equal to shots_per_basis (if provided)
    """
    bases_list = cfg.get("measurement_bases", [])
    if cfg.get("random_bases_per_shot", False):
        bases_tag = "rand-" + "".join(str(b) for b in bases_list)
    else:
        bases_tag = "".join(str(b) for b in bases_list)

    # Interpret cfg["shots"] as shots_per_basis
    shots_pb = cfg.get("shots", 0)
    shots_tag = "inf" if shots_pb <= 0 else str(shots_pb)

    nbases = cfg.get("nbases_effective", None)
    if nbases is not None:
        nbases_tag = "_nbases-{}".format(nbases)
    else:
        nbases_tag = ""

    shpb = cfg.get("shots_per_basis_nominal", None)
    if shpb is not None:
        shpb_tag = "_shpb-{}".format(shpb)
    else:
        shpb_tag = ""

    return "N{}_bases-{}_shots-{}{}{}".format(
        cfg["N"], bases_tag, shots_tag, nbases_tag, shpb_tag
    )


def target_case_tag(case_cfg: Dict) -> str:
    """
    Short label for target case, used in filenames.
    """
    kind = case_cfg["kind"]
    if kind == "dicke":
        return "dicke_m{}".format(case_cfg.get("m", 1))
    elif kind == "time":
        return "time_t{}".format(case_cfg["t"])
    else:
        return kind


def save_measurement_data(N: int,
                          measurements: List[Dict],
                          case_cfg: Dict,
                          cfg: Dict,
                          out_dir: str = MEAS_DIR) -> None:
    """
    Save simulated measurement data to a text file.

    Each row has the format:
        bitstring,b0,b1,...,b_{N-1}

    where:
      - bitstring is a binary string of length N (e.g. "010011")
      - bq is the axis character ("X", "Y", or "Z") used for qubit q
    """
    if not measurements:
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    case_tag = target_case_tag(case_cfg)
    suffix = config_suffix(cfg)
    filename = "measdata_{}_{}.txt".format(case_tag, suffix)
    path = os.path.join(out_dir, filename)

    with open(path, "w") as f:
        header = "bitstring," + ",".join("b{}".format(i) for i in range(N))
        f.write(header + "\n")
        for rec in measurements:
            outcome_int = rec["outcome_int"]
            basis_spec = rec["basis_spec"]
            basis_full = basis_spec if len(basis_spec) == N else basis_spec * N
            bitstring = format(outcome_int, "0{}b".format(N))
            bases_per_qubit = list(basis_full)
            row = ",".join([bitstring] + bases_per_qubit)
            f.write(row + "\n")

    print("    Saved measurement data to {}".format(os.path.abspath(path)))


def plot_topology_graphs(N: int, ansatzes: List[Dict], cfg: Dict, viz_hidden: int = 12):
    """
    Plot visible-hidden connectivity graphs for each ansatz.

    This is a schematic visualization using networkx, not the actual
    trained weights.
    """
    num_ans = len(ansatzes)
    fig, axes = plt.subplots(1, num_ans, figsize=(5 * num_ans, 5))
    if num_ans == 1:
        axes = [axes]

    for ax, ans_cfg in zip(axes, ansatzes):
        name = ans_cfg["name"]
        partitions = get_ansatz_partitions_for_viz(N, ans_cfg)
        mask = create_viz_mask(N, viz_hidden, partitions)

        G = nx.Graph()
        for v in range(N):
            for h in range(viz_hidden):
                if mask[v, h] > 0.5:
                    G.add_edge("V{}".format(v), "H{}".format(h))

        pos = {}
        for i in range(N):
            pos["V{}".format(i)] = (i * 2.0, 0.0)
        if viz_hidden > 1:
            scale = ((N - 1) * 2.0) / (viz_hidden - 1)
        else:
            scale = 1.0
        for i in range(viz_hidden):
            pos["H{}".format(i)] = (i * scale, 1.0)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=["V{}".format(i) for i in range(N)],
                               node_color="#1f77b4",
                               node_size=150,
                               ax=ax)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=["H{}".format(i) for i in range(viz_hidden)],
                               node_color="#d62728",
                               node_size=50,
                               ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")

    shots_pb = cfg.get("shots", 0)
    shots_tag = "inf" if shots_pb <= 0 else str(shots_pb)
    plt.suptitle(
        "Ansatz Topologies\nN = {}, Bases = {}, Shots_per_basis = {}".format(
            cfg["N"], cfg["measurement_bases"], shots_tag
        ),
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = config_suffix(cfg)
    plt.savefig("./figures/fig_nns_snns_topology_{}.png".format(suffix), dpi=300)
    plt.show()


def plot_results_grid(results_loss, results_fid, ansatzes, cfg):
    """
    Plot grids of loss and fidelity curves across target states.
    """
    shots_pb = cfg.get("shots", 0)
    shots_tag = "inf" if shots_pb <= 0 else str(shots_pb)
    loss_label = "NLL (bitstring statistics)"

    styles = {
        "Unconstrained":            {"c": "black",   "ls": "-",  "lw": 3.0},
        "Bipartite (3|3)":          {"c": "#ff7f0e", "ls": "--", "lw": 2.0},
        "Tripartite (2|2|2)":       {"c": "#2ca02c", "ls": "-.", "lw": 2.5},
        "Quad-partite (2|2|1|1)":   {"c": "#d62728", "ls": ":",  "lw": 3.0},
    }

    num_cases = len(results_loss)
    nrows, ncols = 2, 3
    suffix = config_suffix(cfg)

    # Loss grid
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes1 = axes1.flatten()
    for i, (case_name, case_dict) in enumerate(results_loss):
        if i >= len(axes1):
            break
        ax = axes1[i]
        for ans_name, (steps, curve) in case_dict.items():
            style = styles.get(ans_name, {})
            ax.plot(steps, curve, label=ans_name, **style)
        ax.set_title(case_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(loss_label)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    for j in range(i + 1, len(axes1)):
        axes1[j].axis("off")
    plt.suptitle(
        "Bases = {}, Shots_per_basis = {}".format(cfg["measurement_bases"], shots_tag),
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("./figures/fig_nns_snns_loss_{}.png".format(suffix), dpi=300)
    plt.show()

    # Fidelity grid
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes2 = axes2.flatten()
    for i, (case_name, case_dict) in enumerate(results_fid):
        if i >= len(axes2):
            break
        ax = axes2[i]
        for ans_name, (steps, curve) in case_dict.items():
            style = styles.get(ans_name, {})
            ax.plot(steps, curve, label=ans_name, **style)
        ax.set_title(case_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fidelity")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)
    for j in range(i + 1, len(axes2)):
        axes2[j].axis("off")
    plt.suptitle(
        "Bases = {}, Shots_per_basis = {}".format(cfg["measurement_bases"], shots_tag),
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("./figures/fig_nns_snns_fid_{}.png".format(suffix), dpi=300)
    plt.show()


# =====================================================================
# Main experiment loop
# =====================================================================

def run_experiment(cfg: Dict):
    """
    Run a single experiment configuration:
      - loop over target states
      - generate measurement data
      - train all ansatzes
      - collect loss and fidelity histories
      - save measurement files and plots
      - after each training, print a basis-by-basis comparison of
        psi_target and psi_learned in the Z basis.

    IMPORTANT: we enforce cfg["shots"] == cfg["shots_per_basis"].
    """
    if cfg["loss_mode"] != "measurements":
        raise ValueError("Only measurement-based training is supported.")

    # Enforce shots == shots_per_basis and propagate to both keys
    shots_pb = cfg.get("shots_per_basis", None)
    shots_cfg = cfg.get("shots", None)

    if shots_pb is None and shots_cfg is None:
        shots_pb = 0
        shots_cfg = 0
    elif shots_pb is None:
        shots_pb = shots_cfg
    elif shots_cfg is None:
        shots_cfg = shots_pb
    elif shots_pb != shots_cfg:
        raise ValueError("Config error: 'shots' and 'shots_per_basis' must be equal.")

    cfg["shots"] = shots_cfg
    cfg["shots_per_basis"] = shots_pb
    shots = shots_pb  # use this as shots_per_basis everywhere

    seed = cfg["seed"]
    N = cfg["N"]
    basis_labels = cfg["measurement_bases"]
    random_bases_flag = cfg.get("random_bases_per_shot", False)
    num_random_bases = cfg.get("num_random_bases", None)
    H_amp = cfg["H_amp"]
    H_phase = cfg["H_phase"]
    H_amp_per = cfg["H_amp_per"]
    H_phase_per = cfg["H_phase_per"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    log_every = cfg["log_every"]
    target_cases = cfg["target_cases"]
    ansatzes = cfg["ansatzes"]

    key = random.PRNGKey(seed)
    all_s = get_all_configs(N)

    results_loss = []
    results_fid = []

    # We will set nbases_effective and shots_per_basis_nominal after
    # the first call to make_data_probs_from_target.
    suffix_params_initialized = False

    print("\n====================================")
    print("Experiment:", cfg.get("name", "<unnamed>"))
    print("------------------------------------")
    print("N = {}, bases = {}, shots_per_basis = {}, random_bases_per_shot = {}, num_random_bases = {}".format(
        N, basis_labels,
        "inf" if shots is None or shots <= 0 else shots,
        random_bases_flag,
        num_random_bases,
    ))

    for case_cfg in target_cases:
        case_name = case_cfg["name"]
        print("\nAnalyzing target: {}".format(case_name))
        target_amp = build_target_state(N, case_cfg)

        data_probs_array, basis_specs, measurements = make_data_probs_from_target(
            target_amp, N, basis_labels, shots=shots, seed=seed,
            random_bases_per_shot=random_bases_flag,
            num_random_bases=num_random_bases,
        )

        # Encode bases as integer axes for the NLL.
        basis_axes_array = encode_basis_specs_to_axes(basis_specs, N)

        # Initialize filename suffix parameters once, based on the effective
        # number of bases in use (same for all target cases).
        if not suffix_params_initialized:
            nbases_effective = len(basis_specs)
            cfg["nbases_effective"] = nbases_effective
            if shots is None or shots <= 0:
                cfg["shots_per_basis_nominal"] = "inf"
            else:
                # Here "nominal" is exactly the configured shots_per_basis.
                cfg["shots_per_basis_nominal"] = shots
            suffix_params_initialized = True

        # Save raw measurement shots, if any (also useful if you want to
        # simulate training from actual experimental data later).
        save_measurement_data(N, measurements, case_cfg, cfg)

        case_loss_dict = {}
        case_fid_dict = {}

        for ans_cfg in ansatzes:
            ans_name = ans_cfg["name"]
            ans_type = ans_cfg["type"]
            print("  Training ansatz: {}".format(ans_name))

            key, subkey = random.split(key)

            if ans_type == "NNS":
                params, loss_hist, fid_hist, steps = train_nns(
                    key=subkey,
                    N=N,
                    H_amp=H_amp,
                    H_phase=H_phase,
                    target_amp=target_amp,
                    all_s=all_s,
                    epochs=epochs,
                    lr=lr,
                    data_probs_array=data_probs_array,
                    basis_axes_array=basis_axes_array,
                    log_every=log_every,
                )

                # Learned wavefunction in Z basis (not explicitly normalized here;
                # the comparison helper will normalize internally).
                psi_learned_Z = nns_wavefunction(params, all_s)

            elif ans_type == "SNNS":
                partitions = resolve_partitions_from_config(N, ans_cfg)
                params_snns, loss_hist, fid_hist, steps = train_snns(
                    key=subkey,
                    partitions=partitions,
                    H_amp_per=H_amp_per,
                    H_phase_per=H_phase_per,
                    target_amp=target_amp,
                    all_s=all_s,
                    epochs=epochs,
                    lr=lr,
                    data_probs_array=data_probs_array,
                    basis_axes_array=basis_axes_array,
                    N=N,
                    log_every=log_every,
                )

                psi_learned_Z = snns_wavefunction(params_snns, partitions, all_s)

            else:
                raise ValueError("Unknown ansatz type '{}'".format(ans_type))

            # Print a simple loss monitor: loss vs epoch (only ASCII).
            print("    Loss monitor for ansatz '{}':".format(ans_name))
            for s_step, L in zip(steps, loss_hist):
                print("      epoch {:4d}: loss = {:.6f}".format(int(s_step), float(L)))

            # After training, print basis-by-basis comparison of amplitudes
            # in the computational Z basis.
            print_wavefunction_comparison(
                target_amp=target_amp,
                learned_amp=psi_learned_Z,
                N=N,
                case_name=case_name,
                ansatz_name=ans_name,
                max_rows=2**N  # for N=6 this prints all 64 states
            )

            case_loss_dict[ans_name] = (steps, np.array(loss_hist))
            case_fid_dict[ans_name] = (steps, np.array(fid_hist))

        results_loss.append((case_name, case_loss_dict))
        results_fid.append((case_name, case_fid_dict))

    # Plot global grids of loss and fidelity across cases and ansatzes.
    plot_results_grid(results_loss, results_fid, ansatzes, cfg)
    # Plot schematic ansatz topologies (independent of target case).
    plot_topology_graphs(N, ansatzes, cfg, viz_hidden=12)


def main():
    cfg_list = CONFIG_LIST if CONFIG_LIST else [CONFIG]
    for cfg in cfg_list:
    #     run_experiment(cfg)
        cfg_name = cfg.get("name", "<unnamed>")
    
        start = time.perf_counter()
        run_experiment(cfg)
        end = time.perf_counter()
    
        elapsed = end - start
        print("\n====================================")
        print("Execution time for config '{}': {:.3f} seconds".format(cfg_name, elapsed))
        print("====================================\n")



if __name__ == "__main__":
    main()
