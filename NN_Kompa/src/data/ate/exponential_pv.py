import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet

def generatate_exponential_core(n_sample: int, rng, beta_0: float, beta_a: float, beta_w: float,
                                delta_0: float, delta: float, gamma: float, lam_z: float, lam_u: float, 
                                U_noise: float = 1, A_noise: float = 1,
                                Z_noise: float = 1, W_noise: float = 1):
    # Generate Random Noise
    U = rng.normal(0, U_noise, n_sample)
    e_w = rng.normal(0, W_noise, n_sample)
    e_z = rng.normal(0, Z_noise, n_sample)
    e_a = rng.normal(0, A_noise, n_sample)
    
    # Structural Equations
    W = delta_0 + delta * U + e_w
    Z = gamma * U + e_z
    A = lam_z * Z + lam_u * U + e_a
    Y = cal_outcome(A, W, U, beta_0, beta_a, beta_w)
    return U, W, Z, A, Y, e_w, e_z, e_a


def generate_train_exponential_pv(n_sample: int, beta_0: float, beta_a: float, beta_w: float,  
                                  delta_0: float,  delta: float, gamma: float, lam_z: float, lam_u: float, 
                                  U_noise: float = 1, A_noise: float = 1,
                                  Z_noise: float = 1, W_noise: float = 1,
                                  seed=42, **kwargs):
    rng = default_rng(seed=seed)
    
    U, W, Z, A, Y, e_w, e_z, e_a = generatate_exponential__core(n_sample, rng, beta_0, beta_a, beta_w, delta_0, delta, 
                                gamma, lam_z, lam_u, U_noise, A_noise, Z_noise, W_noise)
    
    return PVTrainDataSet(treatment=A[:, np.newaxis],
                          treatment_proxy=Z[:, np.newaxis],
                          outcome_proxy=W[:, np.newaxis],
                          outcome=Y[:, np.newaxis],
                          backdoor=None)


def cal_outcome(A, W, U, beta_0, beta_a, beta_w):
    return np.exp(beta_0 + beta_a * A + beta_w * W + U)


def cal_structural(a, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise):
    rng = default_rng(seed=42)
    U = rng.normal(0, U_noise, 10000)
    e_w = rng.normal(0, W_noise, 10000)
    W = delta_0 + delta * U + e_w
    outcome = cal_outcome(a, W, U, beta_0, beta_a, beta_w)
    return np.mean(outcome)


def generate_test_exponential_pv(a_start, a_end, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise, **kwargs):
    a_steps = np.linspace(a_start, a_end, 100)
    treatment = np.array([cal_structural(a, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise) for a in a_steps])
    return PVTestDataSet(structural=treatment[:, np.newaxis],
                         treatment=a_steps[:, np.newaxis])

def calc_haw(A, W, e_w, beta_0, beta_a, beta_w, delta_0, delta, W_noise):
    
    var_e_w = W_noise**2
    delta_star = 1/delta
    beta_0_bar = beta_0 + beta_w*delta_0
    beta_w_star = beta_w + delta_star
    alpha = 0.5 * beta_w**2 * var_e_w - beta_w_star * delta_0 - 0.5*beta_w_star**2 * var_e_w
    
    
    haw = np.exp(beta_0_bar + alpha + beta_a * A + beta_w_star*W)
    ha = np.exp(beta_0_bar + alpha + beta_a * A) * np.mean(np.exp(beta_w_star*W))
    return haw, ha

def gen_haw(a, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise):
    rng = default_rng(seed=42)
    
    U = rng.normal(0, U_noise, 10000)
    e_w = rng.normal(0, W_noise, 10000)
    W = delta_0 + delta * U + e_w
    
    haw, ha = calc_haw_extended(a, W, e_w, beta_0, beta_a, beta_w, delta_0, delta, W_noise)
    
    return haw, ha

def ha_effect(a_start, a_end, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise):
    a_steps = np.linspace(a_start, a_end, 100)
    ha = np.array([gen_haw_extended(a, beta_0, beta_a, beta_w, delta_0, delta, U_noise, W_noise)[1] for a in a_steps])
    return a_steps, ha
