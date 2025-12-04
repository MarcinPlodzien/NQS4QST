#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:11:59 2025

@author: Marcin Plodzien  
 

 
===============================================================================
Neural Quantum State (NQS) Ensemble Tomography and Entanglement Scanner
===============================================================================
 
-------------------------------------------------------------------------------
OVERVIEW
-------------------------------------------------------------------------------
This module implements a Neural Quantum State (NQS) Ensemble Ansatz for 
mixed-state quantum tomography and entanglement certification using JAX.

Each component of the ensemble is a pure-state neural quantum state (NQS)
represented by two real-valued Restricted Boltzmann Machines (RBMs):
one modeling the log-amplitude and one modeling the phase of the wavefunction.

This decomposition,
    Psi_theta(s) = exp[ A_theta(s) + i * Phi_theta(s) ],
improves numerical stability, supports automatic differentiation via JAX,
and provides an explicit amplitude-phase factorization of complex wavefunctions.

The ensemble defines a low-rank mixed quantum state:
    rho_theta = sum_k w_k * |psi_theta_k><psi_theta_k|,
where the ensemble weights w_k are positive (enforced by softmax)
and normalized (sum_k w_k = 1). 
This guarantees Hermiticity, positivity, and trace normalization of rho_theta.

-------------------------------------------------------------------------------
PHYSICS CONTEXT
-------------------------------------------------------------------------------
Quantum State Tomography (QST)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quantum hardware produces finite-shot measurement histograms in local bases
(for example, Z, X, Y). The goal of QST is to reconstruct the underlying 
state rho from such measurement data. Exact tomography scales exponentially 
with the number of qubits (O(2^N)), so data-driven variational models are 
essential for near-term devices.

Entanglement and Non-k-Separability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By constraining RBM connectivity, one can enforce separability across 
specific qubit clusters. A masked RBM implements a k-separable ansatz:
    Psi_theta(s) = Product_m Phi_m(s_Cm),
where each cluster C_m is connected only to its private group of hidden units.
Comparing optimized negative log-likelihood (NLL) values across constrained 
and unconstrained architectures yields an operational measure of entanglement 
depth. A consistent loss hierarchy across these models certifies non-k-separability 
and genuine multipartite entanglement directly from measurement data.

-------------------------------------------------------------------------------
MATHEMATICAL FORMULATION
-------------------------------------------------------------------------------
1. RBM Energy Function
   For visible spins s_i in {-1, +1} and hidden spins h_j in {-1, +1}:
       E_theta(s, h) = -sum_i a_i s_i - sum_j b_j h_j - sum_{i,j} s_i W_ij h_j.

2. Marginalization over Hidden Units
   Exploiting the factorization of the bipartite graph:
       Psi_tilde_theta(s) = exp(sum_i a_i s_i) * Product_j 2 cosh(b_j + sum_i W_ij s_i).

3. Normalized Amplitude
       Psi_theta(s) = Psi_tilde_theta(s) / sqrt(sum_s |Psi_tilde_theta(s)|^2).

4. Amplitude-Phase RBM Decomposition
       Psi_theta(s) = exp[A_theta(s) + i * Phi_theta(s)],
       where
           A_theta(s) = sum_i a_i s_i + sum_j log(2 cosh(b_j + sum_i W_ij s_i)),
           Phi_theta(s) = sum_i c_i s_i + sum_j f(d_j + sum_i U_ij s_i),
           f(x) = (2/pi) * arctan(tanh(x)) + 0.5.

   Both A_theta and Phi_theta are modeled by real-valued RBMs with parameters
   {a, b, W} (amplitude) and {c, d, U} (phase).

5. Mixed-State Ensemble Ansatz
       rho_theta = sum_{k=1..K} w_k |psi_theta_k><psi_theta_k|,
       with
           w_k = softmax(lambda_k) = exp(lambda_k) / sum_j exp(lambda_j).

   Each |psi_theta_k> is a normalized neural quantum state with its own 
   independent RBM parameters. The ensemble rank K controls the maximum 
   mixedness of rho_theta.

6. Born Probabilities and Measurement Likelihood
   For a measurement basis B (for example, tensor product of Pauli operators):
       p_theta(s | B) = <s| U_B rho_theta U_B^dagger |s>.
   The model is trained by minimizing the negative log-likelihood (NLL):
       L_NLL(theta) = -sum_{B,s} f_B(s) * log p_theta(s | B),
   where f_B(s) are empirical frequencies from experimental histograms.
   Gradients dL/dtheta are computed automatically via JAX autodiff.

-------------------------------------------------------------------------------
IMPLEMENTATION DETAILS
-------------------------------------------------------------------------------
Framework:      JAX (with just-in-time compilation and autodiff)
Core Libraries: jax, jax.numpy, optax
Ensemble Rank:  Configurable (controls mixed-state rank)
Measurement Bases: Any combination of {Z, X, Y} or random local Pauli sets

Physical Constraints:
   - Each psi_k normalized individually.
   - Ensemble weights normalized by softmax (sum w_k = 1).
   - rho_theta is Hermitian and positive semidefinite by construction.

Training Objective:
   - Negative Log-Likelihood (Born-likelihood) minimization.

Entanglement Certification:
   - Unconstrained RBMs represent fully entangled ansatzes.
   - Masked RBMs impose separability constraints (bipartite, tripartite, etc.).
   - Comparing NLL gaps between these families provides operational 
     witnesses of entanglement depth.

-------------------------------------------------------------------------------
KEY FUNCTIONS
-------------------------------------------------------------------------------
init_nqs_params():        Initializes amplitude and phase RBM parameters.
nqs_log_psi():            Computes log-amplitude and phase of Psi_theta(s).
get_ensemble_psi_stack(): Builds normalized wavefunctions for all ensemble members.
ensemble_probs_vectorized(): Computes Born probabilities for all measurement bases.
nll_loss():               Implements the negative log-likelihood objective.
train_step():             Performs one gradient update using jax.value_and_grad.
reconstruct_rho():        Reconstructs rho_theta = sum_k w_k |psi_k><psi_k|.

 
===============================================================================
"""


import os
import copy
import jax
import jax.numpy as jnp
from jax import random, tree_util, value_and_grad, lax, vmap, jit
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import optax  # Standard JAX optimizer library

# Enable 64-bit precision.
jax.config.update("jax_enable_x64", True)

# Directory for figures
FIGURES_DIR = "./figures_mixed_states_NQS_ensemble"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Directory for raw simulated measurement data (ASCII text)
MEAS_DIR = "./measurement_data_mixed_states_NQS_ensemble"
if not os.path.exists(MEAS_DIR):
    os.makedirs(MEAS_DIR)


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

CONFIG = {
    # ==============================================================================
    # 1. SYSTEM & REPRODUCIBILITY
    # ==============================================================================
    # "seed": 
    #   Initializes JAX PRNGKey. Crucial for reproducible research.
    #   Simulations with the same seed will produce identical shot noise and network initialization.
    "seed": 42,

    # "N": 
    #   Number of Qubits. 
    #   - Hilbert space dimension = 2^N.
    #   - Density matrix dimension = 2^N x 2^N.
    #   Note: For exact likelihood calculation (this script), computational cost 
    #   scales as O(K * 2^N). For N > 14, one must switch to MCMC sampling methods.
    "N": 3,

    # ==============================================================================
    # 2. OPTIMIZATION OBJECTIVE
    # ==============================================================================
    # "loss_mode":
    #   - "measurements": Minimizes Negative Log-Likelihood (NLL) based on bitstring statistics.
    #     Math: L(theta) = - sum_{basis} sum_{s} P_data(s|basis) * log( P_model(s|basis, theta) )
    #     This is equivalent to minimizing the Kullback-Leibler (KL) divergence 
    #     between the experimental distribution and the model distribution.
    "loss_mode": "measurements",

    # ==============================================================================
    # 3. MEASUREMENT STRATEGY (TOMOGRAPHY)
    # ==============================================================================
    # "measurement_bases":
    #   Defines the Observables.
    #   - ["X", "Y", "Z"] is "Pauli-6" tomography (measuring all 3 axes).
    #   - ["Z"] would result in "Classical Shadow" blindness (phase information is lost).
    #   - For N=3, standard tomography requires measuring 3^N = 27 basis settings 
    #     to be "Informationally Complete" (IC).
    "measurement_bases": ["X", "Y", "Z"],

    # ==============================================================================
    # 4. EXPERIMENTAL SIMULATION (SHOT NOISE)
    # ==============================================================================
    # "shots" / "shots_per_basis":
    #   Simulates the collapse of the wavefunction.
    #   - Value <= 0: "Infinite Statistics" regime. We use exact probabilities P(s) = <s|rho|s>.
    #   - Value > 0: "Finite Sampling" regime (NISQ). We sample bitstrings from a multinomial 
    #     distribution. This introduces "Shot Noise" (statistical uncertainty scaling as 1/sqrt(shots)).
    "shots": 1000,
    "shots_per_basis": 1000,

    # "random_bases_per_shot":
    #   - False: We strictly use the global bases defined in 'measurement_bases' 
    #     (e.g., all qubits measured in X, then all in Y...).
    #   - True: Randomized Tomography. We generate a random pool of Pauli strings 
    #     (e.g., Q0=X, Q1=Z, Q2=Y) to sample the Hilbert space more efficiently.
    "random_bases_per_shot": False,
    "num_random_bases": 27,

    # ==============================================================================
    # 5. OPEN QUANTUM SYSTEM (NOISE) MODEL
    # ==============================================================================
    # "noise_model":
    #   The physical error channel applied to the pure target state.
    #   - "local_dephasing": T2 relaxation. Off-diagonal elements of rho decay. 
    #     Simulates loss of quantum information (phase) without losing energy.
    #   - "depolarizing": White noise. Mixes the state with the Identity matrix (Max Mixed).
    #   - "amplitude_damping": T1 relaxation. Energy relaxation to the ground state |0>.
    "noise_model": "local_dephasing",

    # "p_noise":
    #   Noise strength (probability or rate).
    #   - 0.05 means roughly 5% probability of error per qubit.
    #   - Physics: rho_noisy = (1-p)*rho + p*Error(rho).
    "p_noise": 0.05,

    # ==============================================================================
    # 6. NEURAL NETWORK ARCHITECTURE (RBM ANSATZ)
    # ==============================================================================
    # "H_amp" / "H_phase":
    #   Number of Hidden Units (Neurons) in the Restricted Boltzmann Machine (RBM).
    #   - "amp": Models the magnitude |psi(s)|.
    #   - "phase": Models the phase arg(psi(s)).
    #   - Ratio alpha = H / N determines "Expressibility". 
    #     Higher alpha = better approximation capability but harder to train (vanishing gradients).
    #     Alpha ~ 4 to 8 is usually sufficient for small entangled states.
    "H_amp": 24,
    "H_phase": 24,

    # "H_amp_per" / "H_phase_per":
    #   Hidden units *per block* for Structured Ansatzes.
    #   Since partitioned blocks are smaller (fewer qubits), they need fewer neurons 
    #   to achieve the same density of correlations.
    "H_amp_per": 12,
    "H_phase_per": 12,

    # ==============================================================================
    # 7. OPTIMIZATION HYPERPARAMETERS
    # ==============================================================================
    # "epochs":
    #   Number of gradient descent steps. 2000 is conservative for N=3 (convergence is usually fast).
    "epochs": 2000,

    # "lr":
    #   Learning Rate for the Adam optimizer.
    #   - Too high (>0.1): Unstable, loss diverges.
    #   - Too low (<0.001): Slow convergence, might get stuck in local minima.
    #   - 0.02 is a "Goldilocks" zone for NQS.
    "lr": 0.02,

    # "log_every":
    #   Frequency of printing metrics (Fidelity, Purity, Loss) to console.
    "log_every": 10,

    # ==============================================================================
    # 8. ENSEMBLE RANK (MIXED STATE PARAMETER)
    # ==============================================================================
    # "ensemble_rank_K":
    #   The number of pure states used to reconstruct the density matrix.
    #   Math: rho = sum_{k=1}^K p_k |psi_k><psi_k|.
    #   - K=1: Forces the model to learn a Pure State (Rank-1).
    #   - K>1: Allows learning Mixed States.
    #   - Theoretically, full rank rho requires K = 2^N. However, most physical 
    #     states can be approximated by low-rank ensembles (K << 2^N).
    "ensemble_rank_K": 5,

    # ==============================================================================
    # 9. TARGET QUANTUM STATES
    # ==============================================================================
    # States we attempt to reconstruct.
    # - GHZ: |000> + |111>. "Cat state". Maximally entangled but fragile (1 loss kills all entanglement).
    # - W: |100> + |010> + |001>. Robust entanglement (retains bipartite entanglement if 1 qubit is lost).
    "target_cases": [
        {"name": "GHZ State", "kind": "GHZ"},
        {"name": "W State",   "kind": "W"},
    ],

    # ==============================================================================
    # 10. ANSATZ ZOO (ENTANGLEMENT SCANNER)
    # ==============================================================================
    # We train multiple models with different connectivity constraints to detect entanglement.
    # Logic: If a Separable model fails (High Loss) but Unconstrained succeeds (Low Loss),
    # the state MUST be entangled.
    "ansatzes": [
        {
            # 1. Unconstrained Model
            # Topology: All-to-All.
            # Physics: Capable of capturing multipartite entanglement across all qubits.
            # Expectation: Should reach highest Fidelity ~ 1.0 (minus noise limit).
            "name": "Unconstrained",
            "type": "NQS_Ensemble",
            "partition_mode": "non_sep", # Interpreted as single global block
            "partition_explicit": None
        },
        {
            # 2. Bi-Separable Model (Cut 0 | 1-2)
            # Topology: Q0 is isolated. Q1-Q2 are entangled.
            # Physics: Ansatz factorizes as |psi_0> (x) |psi_12>.
            # Test: If target is GHZ, this should FAIL (Low Fidelity).
            "name": "Bi Separable 0|12",
            "type": "Structured_Ensemble",
            "partition_mode": "explicit",
            "partitions": [[0], [1, 2]] 
        },
        {
            # 3. Fully Separable Model (Product State)
            # Topology: No connections between different qubits.
            # Physics: Ansatz factorizes as |psi_0> (x) |psi_1> (x) |psi_2>.
            # Math: Effectively Mean Field Theory.
            # Test: Should fail for any entangled state (GHZ or W).
            "name": "Fully Separable",
            "type": "Structured_Ensemble",
            "partition_mode": "explicit",
            "partitions": [[i] for i in range(3)]
        },
    ],
}


# =====================================================================
#  CONFIG LIST  
# =====================================================================

CONFIG_LIST: List[Dict] = []

# SCENARIO 1: "Rank Capacity" Test
cfg_rank1 = copy.deepcopy(CONFIG)
cfg_rank1["name"] = "Rank_Collapse_Test_K=1"
cfg_rank1["noise_model"] = "depolarizing"
cfg_rank1["p_noise"] = 0.15
cfg_rank1["ensemble_rank_K"] = 1
CONFIG_LIST.append(cfg_rank1)

cfg_rank5 = copy.deepcopy(CONFIG)
cfg_rank5["name"] = "Rank_Success_Test_K=5"
cfg_rank5["noise_model"] = "depolarizing"
cfg_rank5["p_noise"] = 0.05
cfg_rank5["ensemble_rank_K"] = 5
CONFIG_LIST.append(cfg_rank5)

# SCENARIO 2: "Phase Blindness" vs "Phase Detection"
cfg_blind = copy.deepcopy(CONFIG)
cfg_blind["name"] = "Phase_Blindness_Z_Only"
cfg_blind["noise_model"] = "local_dephasing"
cfg_blind["p_noise"] = 0.02
cfg_blind["measurement_bases"] = ["Z"]
cfg_blind["ensemble_rank_K"] = 3
CONFIG_LIST.append(cfg_blind)

cfg_sight = copy.deepcopy(CONFIG)
cfg_sight["name"] = "Phase_Detection_XYZ"
cfg_sight["noise_model"] = "local_dephasing"
cfg_sight["p_noise"] = 0.05
cfg_sight["measurement_bases"] = ["X", "Y", "Z"]
cfg_sight["ensemble_rank_K"] = 3
CONFIG_LIST.append(cfg_sight)

# SCENARIO 3: "Entanglement Sudden Death"
cfg_death = copy.deepcopy(CONFIG)
cfg_death["name"] = "Entanglement_Death_High_T1"
cfg_death["noise_model"] = "amplitude_damping"
cfg_death["p_noise"] = 0.05
cfg_death["measurement_bases"] = ["X", "Y", "Z"]
CONFIG_LIST.append(cfg_death)

# SCENARIO 4: NISQ-like finite shots
cfg_nisq = copy.deepcopy(CONFIG)
cfg_nisq["name"] = "NISQ_Simulation_1k_Shots"
cfg_nisq["noise_model"] = "depolarizing"
cfg_nisq["p_noise"] = 0.02
cfg_nisq["shots"] = 1000
cfg_nisq["shots_per_basis"] = 1000
cfg_nisq["measurement_bases"] = ["X", "Y", "Z"]
CONFIG_LIST.append(cfg_nisq)

# SCENARIO 5: Random local XYZ bases
cfg_rand_xyz = copy.deepcopy(CONFIG)
cfg_rand_xyz["name"] = "Random_Local_XYZ_Bases"
cfg_rand_xyz["noise_model"] = "depolarizing"
cfg_rand_xyz["p_noise"] = 0.05
cfg_rand_xyz["measurement_bases"] = ["X", "Y", "Z"]
cfg_rand_xyz["shots"] = 1000
cfg_rand_xyz["shots_per_basis"] = 1000
cfg_rand_xyz["random_bases_per_shot"] = True
cfg_rand_xyz["num_random_bases"] = 3 ** cfg_rand_xyz["N"]
cfg_rand_xyz["ensemble_rank_K"] = 3
CONFIG_LIST.append(cfg_rand_xyz)


# =====================================================================
# 2. UTILITIES
# =====================================================================

def bin_to_spin(bin_array: jnp.ndarray) -> jnp.ndarray:
    return 2 * bin_array - 1

def get_all_configs(N: int) -> jnp.ndarray:
    ints = jnp.arange(2**N, dtype=jnp.int32)
    bits = (ints[:, None] >> jnp.arange(N)) & 1
    return bin_to_spin(bits)

# --- Axis encoding and local basis matrices (2x2) ---
AXIS_CHARS = ["Z", "X", "Y"]
AXIS_TO_INDEX = {c: i for i, c in enumerate(AXIS_CHARS)}

# LOCAL_BASIS_MATS[axis_index]
LOCAL_BASIS_MATS = jnp.stack([
    # Z basis: identity
    jnp.eye(2, dtype=jnp.complex128),
    # X basis: Hadamard
    (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0,  1.0], [1.0, -1.0]], dtype=jnp.complex128),
    # Y basis: eigenstates of sigma_y
    (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, -1j], [1.0,  1j]], dtype=jnp.complex128),
], axis=0)

@jit
def apply_single_basis_rotation(psi_flat: jnp.ndarray, basis_axes: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate state |psi> into a specific product basis (vectorized over qubits).
    """
    N = basis_axes.shape[0]
    psi = psi_flat.reshape((2,) * N)
    
    # Apply local unitaries one qubit at a time (unrolled loop)
    for q in range(N):
        axis_idx = basis_axes[q]
        U = LOCAL_BASIS_MATS[axis_idx]
        # Contract axis q
        psi = jnp.tensordot(psi, U, axes=[[q], [1]])
        # Move the new axis back to position q
        perm = list(range(N-1))
        perm.insert(q, N-1)
        psi = jnp.transpose(psi, perm)

    return psi.reshape(-1)

# VMAP: This allows rotating one state against MANY bases in parallel.
# Input: (2^N), (Batch_Bases, N) -> Output: (Batch_Bases, 2^N)
batch_apply_rotations = vmap(apply_single_basis_rotation, in_axes=(None, 0))

def axis_string_to_indices(basis_spec: str, N: int) -> List[int]:
    if len(basis_spec) == 1:
        basis_spec = basis_spec * N
    if len(basis_spec) != N:
        raise ValueError(f"basis_spec must have length 1 or N={N}.")
    return [AXIS_TO_INDEX[c.upper()] for c in basis_spec]

def encode_basis_specs_to_axes(basis_specs: List[str], N: int) -> jnp.ndarray:
    rows = []
    for spec in basis_specs:
        rows.append(axis_string_to_indices(spec, N))
    return jnp.array(rows, dtype=jnp.int32)


# =====================================================================
# 3. NOISE MODELS & TARGET RHO
# =====================================================================

def apply_kraus_map(rho: jnp.ndarray, kraus_ops: List[jnp.ndarray]) -> jnp.ndarray:
    rho_new = jnp.zeros_like(rho)
    for K in kraus_ops:
        rho_new += K @ rho @ K.conj().T
    return rho_new

def get_local_kraus_channel(rho: jnp.ndarray, kraus_single: List[jnp.ndarray]) -> jnp.ndarray:
    d = rho.shape[0]
    N = int(np.round(np.log2(d)))
    current_rho = rho
    for q in range(N):
        full_kraus_ops = []
        for K_loc in kraus_single:
            # Construct K_full = I x ... x K_loc x ... x I
            # Efficient implementation: build full matrix only once per Kraus op
            lst = [jnp.eye(2, dtype=jnp.complex128)] * N
            lst[q] = K_loc
            K_full = lst[0]
            for i in range(1, N):
                K_full = jnp.kron(K_full, lst[i])
            full_kraus_ops.append(K_full)
        current_rho = apply_kraus_map(current_rho, full_kraus_ops)
    return current_rho

def apply_noise_channel(rho: jnp.ndarray, p: float, channel: str) -> jnp.ndarray:
    d = rho.shape[0]

    if channel == "depolarizing":
        return (1 - p) * rho + p * jnp.eye(d) / d

    elif channel == "phase_damping" or channel == "local_dephasing":
        indices = jnp.arange(d, dtype=jnp.int32)
        xor_matrix = jnp.bitwise_xor(indices[:, None], indices[None, :])
        hamming_dist = jax.lax.population_count(xor_matrix)
        decay_mask = jnp.power(1.0 - p, hamming_dist)
        return rho * decay_mask

    elif channel == "amplitude_damping":
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - p)]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=jnp.complex128)
        return get_local_kraus_channel(rho, [K0, K1])

    elif channel == "bit_flip":
        I = jnp.eye(2, dtype=jnp.complex128)
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
        K0 = jnp.sqrt(1 - p) * I
        K1 = jnp.sqrt(p) * X
        return get_local_kraus_channel(rho, [K0, K1])

    elif channel == "thermal":
        pop_ground = 0.8
        sqrt_pop = jnp.sqrt(pop_ground)
        sqrt_exc = jnp.sqrt(1 - pop_ground)
        sqrt_1_p = jnp.sqrt(1 - p)
        sqrt_p = jnp.sqrt(p)
        K0 = sqrt_pop * jnp.array([[1.0, 0.0], [0.0, sqrt_1_p]], dtype=jnp.complex128)
        K1 = sqrt_pop * jnp.array([[0.0, sqrt_p], [0.0, 0.0]], dtype=jnp.complex128)
        K2 = sqrt_exc * jnp.array([[sqrt_1_p, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
        K3 = sqrt_exc * jnp.array([[0.0, 0.0], [sqrt_p, 0.0]], dtype=jnp.complex128)
        return get_local_kraus_channel(rho, [K0, K1, K2, K3])

    else:
        print(f"Warning: Unknown noise channel '{channel}'. Returning original rho.")
        return rho

def get_ghz_state(N: int) -> jnp.ndarray:
    psi = jnp.zeros(2**N, dtype=jnp.complex128)
    psi = psi.at[0].set(1 / jnp.sqrt(2))
    psi = psi.at[-1].set(1 / jnp.sqrt(2))
    return psi

def get_w_state(N: int) -> jnp.ndarray:
    psi = jnp.zeros(2**N, dtype=jnp.complex128)
    for i in range(N):
        psi = psi.at[1 << i].set(1.0)
    return psi / jnp.sqrt(N)

def build_target_rho(N: int, case_cfg: Dict, noise_model: str, p_noise: float) -> jnp.ndarray:
    kind = case_cfg["kind"]
    if kind == "GHZ":
        psi = get_ghz_state(N)
    elif kind == "W":
        psi = get_w_state(N)
    else:
        raise ValueError(f"Unknown target kind {kind}")

    rho_pure = jnp.outer(psi, jnp.conj(psi))
    rho_noisy = apply_noise_channel(rho_pure, p_noise, noise_model)
    return rho_noisy


# =====================================================================
# 3b. Measurement data generation 
# =====================================================================

def make_data_probs_from_rho(rho: jnp.ndarray,
                             N: int,
                             basis_labels: List[str],
                             shots: int,
                             seed: int,
                             random_bases_per_shot: bool = False,
                             num_random_bases: Optional[int] = None
                             ) -> Tuple[jnp.ndarray, List[str], List[Dict]]:
    """
    Computes probabilities for ALL bases in parallel using JAX VMAP.
    """
    dim = rho.shape[0]
    rng = np.random.RandomState(seed if seed is not None else None)

    # 1. Determine the list of bases
    use_sampling = shots is not None and shots > 0
    measurements: List[Dict] = []

    if random_bases_per_shot:
        allowed_axes = [str(b).upper() for b in basis_labels]
        if len(allowed_axes) == 0:
            raise ValueError("measurement_bases must be non-empty.")
        num_possible = len(allowed_axes) ** N
        
        if num_random_bases is not None and num_random_bases > 0:
            B_desired = min(num_random_bases, num_possible)
        else:
            B_desired = min(30, num_possible)

        basis_pool: List[str] = []
        basis_set = set()
        while len(basis_pool) < B_desired:
            chars = [rng.choice(allowed_axes) for _ in range(N)]
            spec = "".join(chars)
            if spec not in basis_set:
                basis_set.add(spec)
                basis_pool.append(spec)
        basis_specs = basis_pool
    else:
        basis_specs = []
        for label in basis_labels:
            full = label * N if len(label) == 1 else label
            basis_specs.append(full)

    # 2. Encode bases for JAX
    basis_axes = encode_basis_specs_to_axes(basis_specs, N)

    # 3. Compute Probabilities Vectorized
    # P(s|B) = sum_j lambda_j |<s|U_B|j>|^2
    evals, evecs = jnp.linalg.eigh(rho)
    evals = jnp.clip(evals, 0.0, None)
    evals = evals / jnp.sum(evals)

    # evecs is (dim, dim) columns. Transpose to (dim, dim) rows
    psi_j_list = evecs.T 

    # Helper: rotate one state 'psi' against ALL bases 'basis_axes'
    def rotate_against_all_bases(psi):
        return batch_apply_rotations(psi, basis_axes)

    # Outer vmap: over eigenvectors j
    # Inner vmap: over bases b (inside rotate_against_all_bases)
    # Result shape: (dim_j, num_bases, dim_hilbert)
    psi_j_rotated = vmap(rotate_against_all_bases)(psi_j_list)

    # Prob(s) for each eigenvector component
    probs_j = jnp.abs(psi_j_rotated) ** 2

    # Weighted sum: sum_j lambda_j * probs_j
    # einsum: j=eigen_idx, b=basis_idx, s=state_idx
    probs_exact_array = jnp.einsum('j, jbs -> bs', evals, probs_j)
    
    # Normalize to be safe
    probs_exact_array = probs_exact_array / jnp.sum(probs_exact_array, axis=1, keepdims=True)

    # 4. (Optional) Sampling
    final_probs_list = []
    probs_np = np.array(probs_exact_array)

    for b_idx, basis_full in enumerate(basis_specs):
        p = probs_np[b_idx]
        if use_sampling:
            samples = rng.choice(dim, size=shots, p=p)
            counts = np.bincount(samples, minlength=dim).astype(np.float64)
            freqs = counts / float(shots)
            final_probs_list.append(jnp.array(freqs))
            for idx in samples:
                measurements.append({
                    "outcome_int": int(idx),
                    "basis_spec": basis_full,
                })
        else:
            final_probs_list.append(jnp.array(p))

    data_probs_array = jnp.stack(final_probs_list, axis=0)
    return data_probs_array, basis_specs, measurements


# =====================================================================
# 4. NQS / SNQS (Log-Domain for Stability)
# =====================================================================

def init_nqs_params(key, N, H_amp, H_phase, scale=0.01):
    keys = random.split(key, 6)
    return {
        "a": scale * random.normal(keys[0], (N,)),
        "b": scale * random.normal(keys[1], (H_amp,)),
        "W": scale * random.normal(keys[2], (N, H_amp)),
        "c": scale * random.normal(keys[3], (N,)),
        "d": scale * random.normal(keys[4], (H_phase,)),
        "U": scale * random.normal(keys[5], (N, H_phase)),
    }

def nqs_log_psi(p, s):
    """
    Computes log(psi) to avoid underflow, then returns psi.
    Uses log-cosh trick.
    """
    # Amplitude: s @ W + b
    lin_amp = jnp.dot(s, p["W"]) + p["b"]
    # log(cosh(x)) ~ |x| - log(2)
    log_cosh = jnp.sum(jnp.log(2.0 * jnp.cosh(lin_amp)), axis=-1)
    log_amp = jnp.dot(s, p["a"]) + log_cosh

    # Phase: nonlinear
    lin_ph = jnp.dot(s, p["U"]) + p["d"]
    f = (2.0 / jnp.pi) * jnp.arctan(jnp.tanh(lin_ph)) + 0.5
    phase = jnp.sum(f, axis=-1) + jnp.dot(s, p["c"])

    return jnp.exp(log_amp + 2j * jnp.pi * phase)

# Vectorized over batch of states 's'
vmap_nqs_psi = vmap(nqs_log_psi, in_axes=(None, 0))

def init_ensemble_params(key, K, N, partitions, cfg):
    k1, k2 = random.split(key)
    weights_logit = 0.01 * random.normal(k1, (K,))
    sub_keys = random.split(k2, K)

    components = []
    H_amp_g, H_phase_g = cfg["H_amp"], cfg["H_phase"]
    H_amp_p, H_phase_p = cfg["H_amp_per"], cfg["H_phase_per"]

    for i in range(K):
        if partitions is None:
            components.append(init_nqs_params(sub_keys[i], N, H_amp_g, H_phase_g))
        else:
            p_list = []
            ks = random.split(sub_keys[i], len(partitions))
            for j, part in enumerate(partitions):
                p_list.append(init_nqs_params(ks[j], len(part), H_amp_p, H_phase_p))
            components.append(p_list)
    return {"weights_logit": weights_logit, "components": components}

def get_ensemble_psi_stack(params, partitions, all_s):
    """
    Returns (K, 2^N) array of normalized pure states.
    """
    components = params["components"]
    psi_list = []
    K = len(components)
    
    # Python loop over K is fine (K is small, e.g., 3 or 5)
    for k in range(K):
        comp_params = components[k]
        if partitions is None:
            psi = vmap_nqs_psi(comp_params, all_s)
        else:
            # Product over blocks
            psi = jnp.ones(all_s.shape[0], dtype=jnp.complex128)
            for idx, part in enumerate(partitions):
                # Extract cols corresponding to this block
                s_part = all_s[:, jnp.array(part)]
                psi = psi * vmap_nqs_psi(comp_params[idx], s_part)
        
        psi = psi / jnp.linalg.norm(psi)
        psi_list.append(psi)
        
    return jnp.stack(psi_list)

@jit
def ensemble_probs_vectorized(params,
                              partitions,
                              all_s,
                              basis_axes_array: jnp.ndarray,
                              N: int) -> jnp.ndarray:
    """
    Compute P_model(s | basis) for ALL bases in basis_axes_array (Batch, N).
    """
    weights = jax.nn.softmax(params["weights_logit"])  # (K,)
    psi_stack_Z = get_ensemble_psi_stack(params, partitions, all_s) # (K, 2^N)
    
    # We map the rotation function over the batch of bases.
    # We want to do this for each component k.
    
    # Helper: rotate a single state 'psi' by ALL bases
    def rotate_all(psi):
        return batch_apply_rotations(psi, basis_axes_array)
    
    # vmap over K components -> result (K, B, 2^N)
    psi_stack_rot = vmap(rotate_all)(psi_stack_Z)
    
    # Component probabilities: |amp|^2
    probs_k = jnp.abs(psi_stack_rot) ** 2
    
    # Weighted sum: sum_k w_k * p_k(s|b)
    # Einsum: k=component, b=basis, s=state
    probs_ensemble = jnp.einsum('k, kbs -> bs', weights, probs_k)
    
    # Normalize
    return probs_ensemble / jnp.sum(probs_ensemble, axis=1, keepdims=True)

def reconstruct_rho(params, partitions, all_s):
    weights = jax.nn.softmax(params["weights_logit"])
    psi_stack = get_ensemble_psi_stack(params, partitions, all_s)
    return jnp.einsum("k,ki,kj->ij", weights, psi_stack, jnp.conj(psi_stack))


# =====================================================================
# 5. METRICS AND TRAINING (With Optax)
# =====================================================================

@jit
def nll_loss(params,
             partitions,
             data_probs: jnp.ndarray,
             all_s: jnp.ndarray,
             basis_axes_array: jnp.ndarray,
             N: int) -> float:
    p_model = ensemble_probs_vectorized(params, partitions, all_s, basis_axes_array, N)
    # Avoid log(0) with epsilon
    ce = -jnp.sum(data_probs * jnp.log(p_model + 1e-12), axis=1)
    return jnp.mean(ce)

def compute_detailed_metrics(rho_est, rho_target):
    p_est = jnp.real(jnp.trace(rho_est @ rho_est))
    p_tgt = jnp.real(jnp.trace(rho_target @ rho_target))
    overlap = jnp.real(jnp.trace(rho_est @ rho_target))
    fid_hs = overlap / (jnp.sqrt(p_est * p_tgt) + 1e-12)
    return fid_hs, p_est, p_tgt

def print_rho_comparison(rho_est, rho_tgt, title):
    print(f"\n{'-'*30}\nMATRIX DUMP: {title}\n{'-'*30}")
    diff = np.array(rho_tgt - rho_est)
    abs_re = np.abs(np.real(diff))
    abs_im = np.abs(np.imag(diff))
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        print("\n[Error] |Re(Target - Learned)|:")
        print(abs_re)
        print("\n[Error] |Im(Target - Learned)|:")
        print(abs_im)
    print(f"{'-'*30}\n")

def train_ensemble(key,
                   N: int,
                   partitions: Optional[List[List[int]]],
                   cfg: Dict,
                   data_probs: jnp.ndarray,
                   basis_axes_array: jnp.ndarray,
                   all_s: jnp.ndarray,
                   rho_target: jnp.ndarray):
    """
    Training loop using OPTAX for better stability/performance than manual Adam.
    """
    K = cfg["ensemble_rank_K"]
    params = init_ensemble_params(key, K, N, partitions, cfg)
    
    # OPTAX Optimizer
    optimizer = optax.adam(learning_rate=cfg["lr"])
    opt_state = optimizer.init(params)
    
    # JIT-compiled update step
    @jit
    def update_step(params, opt_state):
        loss, grads = value_and_grad(nll_loss)(params, partitions, data_probs, all_s, basis_axes_array, N)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    log_every = cfg["log_every"]
    num_epochs = cfg["epochs"]
    
    loss_hist, fid_hist, pur_hist, steps_hist = [], [], [], []
    
    p_tgt_initial = float(jnp.real(jnp.trace(rho_target @ rho_target)))
    print(f">> CONFIG: Noise={cfg['noise_model']} (p={cfg['p_noise']}) | Rank K={K}")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'Fidelity':<8} | {'Purity (Est/Tgt)':<18}")
    print("-" * 50)

    # Main Loop
    for i in range(1, num_epochs + 1):
        params, opt_state, loss_val = update_step(params, opt_state)
        
        if i % log_every == 0:
            rho_est = reconstruct_rho(params, partitions, all_s)
            fid, p_est, p_tgt = compute_detailed_metrics(rho_est, rho_target)
            
            print(f"{i:<6d} | {loss_val:<8.4f} | {float(fid):<8.4f} | {float(p_est):.3f} / {float(p_tgt):.3f}")

            loss_hist.append(float(loss_val))
            fid_hist.append(float(fid))
            pur_hist.append(float(p_est))
            steps_hist.append(i)

    # Final result dump
    if N <= 4:
        print_rho_comparison(rho_est, rho_target, "Final Result")

    return steps_hist, loss_hist, fid_hist, pur_hist, (fid, p_est, p_tgt)


# =====================================================================
# 6. FILE NAMING 
# =====================================================================

def resolve_partitions(N, ans_cfg):
    mode = ans_cfg.get("partition_mode", "full_sep")
    if mode == "full_sep":
        if ans_cfg.get("name", "").startswith("Unconstrained"):
            return None
        return [[i for i in range(N)]]
    elif mode == "explicit":
        return ans_cfg["partitions"]
    else:
        return None

def config_suffix(cfg: Dict) -> str:
    """
    Legacy filename generator to match original code exactly.
    """
    # bases tag
    bases_list = cfg.get("measurement_bases", [])
    if cfg.get("random_bases_per_shot", False):
        bases_tag = "rand-" + "".join(str(b) for b in bases_list)
    else:
        bases_tag = "".join(str(b) for b in bases_list)

    # shots tag
    shots_pb = cfg.get("shots", 0)
    shots_tag = "inf" if shots_pb is None or shots_pb <= 0 else str(shots_pb)

    # nbases tag (calculated in experiment loop)
    nbases = cfg.get("nbases_effective", None)
    if nbases is not None:
        nbases_tag = f"_nbases-{nbases}"
    else:
        nbases_tag = ""

    # shpb nominal tag
    shpb_nom = cfg.get("shots_per_basis_nominal", None)
    if shpb_nom is None:
        shpb_nom = "inf" if shots_pb is None or shots_pb <= 0 else shots_pb
    shpb_tag = f"_shpb-{shpb_nom}"

    # noise and rank
    noise_tag = f"_noise-{cfg['noise_model']}_p{cfg['p_noise']}"
    rank_tag = f"_K-{cfg['ensemble_rank_K']}"

    return f"N{cfg['N']}_bases-{bases_tag}_shots-{shots_tag}{nbases_tag}{shpb_tag}{noise_tag}{rank_tag}"

def target_case_tag(case_cfg: Dict) -> str:
    kind = case_cfg["kind"]
    if kind == "GHZ": return "GHZ"
    elif kind == "W": return "W"
    else: return kind

def save_measurement_data(N: int,
                          measurements: List[Dict],
                          case_cfg: Dict,
                          cfg: Dict,
                          out_dir: str = MEAS_DIR) -> None:
    if not measurements:
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    case_tag = target_case_tag(case_cfg)
    suffix = config_suffix(cfg)
    filename = f"measdata_ensemble_{case_tag}_{suffix}.txt"
    path = os.path.join(out_dir, filename)

    with open(path, "w") as f:
        header = "bitstring," + ",".join(f"b{i}" for i in range(N))
        f.write(header + "\n")
        for rec in measurements:
            outcome_int = rec["outcome_int"]
            basis_spec = rec["basis_spec"]
            basis_full = basis_spec if len(basis_spec) == N else basis_spec * N
            bitstring = format(outcome_int, f"0{N}b")
            bases_per_qubit = list(basis_full)
            row = ",".join([bitstring] + bases_per_qubit)
            f.write(row + "\n")

    print(f"    Saved measurement data to {os.path.abspath(path)}")


# =====================================================================
# 7. PLOTTING 
# =====================================================================

def plot_results_grid(results, cfg):
    nrows = len(results)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    suffix = config_suffix(cfg)
    
    # Exact title format from original
    title_meta = (
        f"Noise: {cfg['noise_model']} (p={cfg['p_noise']}) | "
        f"Bases: {cfg['measurement_bases']} | "
        f"Shots_per_basis: {cfg['shots']} | "
        f"Rank K={cfg['ensemble_rank_K']}"
    )

    for i, (case_name, ans_dict) in enumerate(results):
        ax_l, ax_f, ax_p = axes[i]
        tgt_pur = None
        for ans_name, data in ans_dict.items():
            steps, loss, fid, pur, finals = data
            f_end, p_est, p_tgt = finals
            if tgt_pur is None:
                tgt_pur = p_tgt
            label = f"{ans_name}\n(F={float(f_end):.3f})"
            ax_l.plot(steps, loss, label=label, lw=2)
            ax_f.plot(steps, fid, label=label, lw=2)
            ax_p.plot(steps, pur, label=label, lw=2)

        if tgt_pur is not None:
            ax_p.axhline(float(tgt_pur), c='k', ls='--', alpha=0.5, label="Target Purity")

        ax_l.set_title(f"{case_name}\nNLL Loss")
        ax_f.set_title("HS Fidelity")
        ax_p.set_title("Purity Tr[rho^2]")
        ax_f.set_ylim(-0.05, 1.05)
        ax_p.set_ylim(-0.05, 1.05)
        if i == 0:
            ax_l.legend(fontsize=8)
            ax_p.legend(fontsize=8)

    plt.suptitle(title_meta)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = FIGURES_DIR + f"/NQS_Ensemble_Grid_{suffix}.png"
    plt.savefig(filename, dpi=150)
    plt.show()

def plot_topology(N, ansatzes, cfg):
    n_ans = len(ansatzes)
    fig, axes = plt.subplots(1, n_ans, figsize=(4 * n_ans, 4))
    if n_ans == 1:
        axes = [axes]

    for ax, ans in zip(axes, ansatzes):
        parts = resolve_partitions(N, ans)
        if parts is None:
            parts = [[i for i in range(N)]]

        G = nx.Graph()
        for i in range(N):
            G.add_node(f"Q{i}", pos=(i, 0), color='#ADD8E6')
        for b_idx, block in enumerate(parts):
            h_node = f"H_blk{b_idx}"
            avg_x = sum(block) / len(block)
            G.add_node(h_node, pos=(avg_x, 1), color='#FA8072')
            for q in block:
                G.add_edge(f"Q{q}", h_node)

        pos = nx.get_node_attributes(G, 'pos')
        cols = [G.nodes[n]['color'] for n in G.nodes]
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=cols, node_size=600)
        ax.set_title(f"{ans['name']}\n(Rank K={cfg['ensemble_rank_K']})")
        ax.set_ylim(-0.5, 1.5)

    plt.tight_layout()
    plt.savefig(f"./figures/NQS_Topology_{config_suffix(cfg)}.png", dpi=150)
    plt.show()


# =====================================================================
# 8. MAIN EXPERIMENT LOOP
# =====================================================================

def run_experiment(cfg):
    print(f"\n>>> EXPERIMENT: {cfg['name']}")

    if cfg["loss_mode"] != "measurements":
        raise ValueError("Only measurement-based loss_mode is supported.")

    # Enforce shots consistency
    shots_pb = cfg.get("shots_per_basis", None)
    shots_cfg = cfg.get("shots", None)
    if shots_pb is None and shots_cfg is None:
        shots_pb = 0
        shots_cfg = 0
    elif shots_pb is None:
        shots_pb = shots_cfg
    elif shots_cfg is None:
        shots_cfg = shots_pb
    
    cfg["shots"] = shots_cfg
    cfg["shots_per_basis"] = shots_pb
    shots = shots_pb

    key = random.PRNGKey(cfg["seed"])
    N = cfg["N"]
    all_s = get_all_configs(N)
    results = []

    # 1. Iterate Targets
    for case in cfg["target_cases"]:
        print(f"\n=== Target: {case['name']} ===")
        rho_target = build_target_rho(N, case, cfg["noise_model"], cfg["p_noise"])

        # 2. Generate Data (Vectorized)
        data_probs, basis_specs, measurements = make_data_probs_from_rho(
            rho_target,
            N,
            cfg["measurement_bases"],
            shots=shots,
            seed=cfg["seed"],
            random_bases_per_shot=cfg.get("random_bases_per_shot", False),
            num_random_bases=cfg.get("num_random_bases", None),
        )

        # 3. Encode axes
        basis_axes_array = encode_basis_specs_to_axes(basis_specs, N)
        
        # Inject computed stats into cfg for filename generation
        cfg["nbases_effective"] = len(basis_specs)
        cfg["shots_per_basis_nominal"] = shots if shots > 0 else "inf"

        # 4. Save Data
        save_measurement_data(N, measurements, case, cfg)

        # 5. Train all ansatzes
        ansatz_results = {}
        for ans in cfg["ansatzes"]:
            print(f"\n--- Training {ans['name']} ---")
            parts = resolve_partitions(N, ans)
            key, k_train = random.split(key)
            out = train_ensemble(k_train, N, parts, cfg,
                                 data_probs, basis_axes_array, all_s, rho_target)
            ansatz_results[ans['name']] = out

        results.append((case["name"], ansatz_results))

    # 6. Plotting
    plot_results_grid(results, cfg)
    # plot_topology(N, cfg["ansatzes"], cfg)


def main():
    cfg_list = CONFIG_LIST if CONFIG_LIST else [CONFIG]
    for cfg in cfg_list:
        run_experiment(cfg)


if __name__ == "__main__":
    main()