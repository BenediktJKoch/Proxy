import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet

def generatate_cosine_core(n_sample: int, rng, beta_0: float, beta_a: float, delta: float, 
                                gamma: float, lam_z: float, lam_u: float, 
                                U_noise: float = 1, A_noise: float = 1,
                                Z_noise: float = 1, W_noise: float = 1):
    # Generate Random Noise
    U = rng.normal(0, U_noise, n_sample)
    e_w = rng.normal(0, W_noise, n_sample)
    e_z = rng.normal(0, Z_noise, n_sample)
    e_a = rng.normal(0, A_noise, n_sample)
    
    # Structural Equations
    W = delta * U + e_w
    Z = gamma * U + e_z
    A = lam_z * Z + lam_u * U + e_a
    Y = np.cos(beta_0 + beta_a * A + U) + 1
    return U, W, Z, A, Y, e_w, e_z, e_a


def generate_train_cosine_pv(n_sample: int, beta_0: float, beta_a: float, delta: float, 
                                  gamma: float, lam_z: float, lam_u: float, 
                                  U_noise: float = 1, A_noise: float = 1,
                                  Z_noise: float = 1, W_noise: float = 1,
                                  seed=42, **kwargs):
    rng = default_rng(seed=seed)
    
    U, W, Z, A, Y, e_w, e_z, e_a = generatate_cosine_core(n_sample, rng, beta_0, beta_a, delta, 
                                gamma, lam_z, lam_u, U_noise, A_noise, Z_noise, W_noise)
    
    return PVTrainDataSet(treatment=A[:, np.newaxis],
                          treatment_proxy=Z[:, np.newaxis],
                          outcome_proxy=W[:, np.newaxis],
                          outcome=Y[:, np.newaxis],
                          backdoor=None)


def cal_outcome(A, U, beta_0, beta_a):
    return np.cos(beta_0 + beta_a * A + U) + 1


def cal_structural(a: float, beta_0, beta_a, U_noise: float = 1):
    rng = default_rng(seed=42)
    U = rng.normal(0, U_noise, 10000)
    outcome = cal_outcome(a, U, beta_0, beta_a)
    return np.mean(outcome)


def generate_test_cosine_pv(a_start, a_end, beta_0, beta_a, U_noise: float = 1, **kwargs):
    a_steps = np.linspace(a_start, a_end, 100)
    treatment = np.array([cal_structural(a, beta_0, beta_a, U_noise) for a in a_steps])
    return PVTestDataSet(structural=treatment[:, np.newaxis],
                         treatment=a_steps[:, np.newaxis])
