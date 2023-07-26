import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet

def generatate_mastourie_core(n_sample: int, rng, sigma: float = np.sqrt(3), beta: float = np.sqrt(0.05), y_shift: float = 0):
    
    # Structural Equations
    U2 = rng.uniform(-1, 2, n_sample)
    U1 = rng.uniform(0, 1, n_sample) - ((U2 > 0) & (U2 < 1)).astype(int)
    W1 = U1 + rng.uniform(-1, 1, n_sample)
    W2 = U2 + rng.normal(0, sigma, n_sample)
    Z1 = U1 + rng.normal(0, sigma, n_sample)
    Z2 = U2 + rng.uniform(-1, 1, n_sample)
    A = U2 + rng.normal(0, beta, n_sample)
    Y = cal_outcome(A, U1, U2, y_shift)
    return U1, U2, Z1, Z2, A, W1, W2, Y


def generate_train_mastourie_pv(n_sample: int, sigma: float = np.sqrt(3), beta: float = np.sqrt(0.05), y_shift: float = 0,
                             seed=42, **kwargs):
    
    rng = default_rng(seed=seed)
    U1, U2, Z1, Z2, A, W1, W2, Y = generatate_mastourie_core(n_sample, rng, sigma, beta, y_shift)
    
    return PVTrainDataSet(treatment=A[:, np.newaxis],
                          treatment_proxy=np.c_[Z1, Z2],
                          outcome_proxy=np.c_[W1, W2],
                          outcome=Y[:, np.newaxis],
                          backdoor=None,
                         )


def cal_outcome(A, U1, U2, y_shift):
    Y = U2 * np.cos(2 * (A + 0.3 * U1 + 0.2)) + y_shift # to break NN symmetry
    return Y

def cal_structural(a: float, y_shift: float = 0):
    rng = default_rng(seed=42)
    # U1, U2, Y
    U2 = rng.uniform(-1, 2, 10000)
    U1 = rng.uniform(0, 1, 10000) - ((U2 > 0) & (U2 < 1)).astype(int)
    outcome = cal_outcome(a, U1, U2, y_shift)
    return np.mean(outcome)

def generate_test_mastourie_pv(a_start, a_end, n_steps: int = 100, y_shift: float = 0, **kwargs):
    a_steps = np.linspace(a_start, a_end, n_steps)
    struc = np.array([cal_structural(a, y_shift) for a in a_steps])
    return PVTestDataSet(treatment=a_steps[:, np.newaxis],
                         structural=struc[:, np.newaxis],
                         )
