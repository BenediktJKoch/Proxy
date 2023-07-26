import json
import click
from pathlib import Path
from shutil import make_archive
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from numpy.random import default_rng
from typing import Tuple, Optional, Dict, Any
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import operator
from sklearn.preprocessing import StandardScaler
import ast
import json

from src.experiment import experiments, get_run_func

from src.utils import grid_search_dict
from src.utils.kernel_func import ColumnWiseGaussianKernel, AbsKernel, BinaryKernel, GaussianKernel
from src.utils.jax_utils import Hadamard_prod, mat_mul, mat_trans, modif_kron, cal_loocv_emb, cal_loocv_alpha, \
    stage2_weights

from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet

from src.models.PMMR.model_val import PMMRModel
from src.models.kernelPV.model import get_kernel_func, KernelPVModel
from src.models.NMMR.NMMR_trainers import NMMR_Trainer_DemandExperiment, NMMR_Trainer_dSpriteExperiment, \
    NMMR_Trainer_RHCExperiment, NMMR_Trainer_MastourieExperiment

from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet, split_train_data, PVTrainDataSetTorch, \
    PVTestDataSetTorch, RHCTestDataSetTorch

def PMMR_loss(pred_bridge, target, kernel_matrix): 
    residual = target - pred_bridge
    n = residual.shape[0]
    K = kernel_matrix

    loss = (residual.T @ K @ residual) / (n ** 2)

    return loss[0][0]

def cal_val_err_pmmr(lam, model_param, data_name, train_data, val_data, preprocessor):
    # copy model_param so we dont change it
    model_param_val = dict(model_param)
    model_param_val["lam1"] = lam
    
    # fit model on train 
    model_train = PMMRModel(**model_param_val)
    K_AZX_train = model_train.fit(train_data, data_name)
    
    # fit model on val to get Kernel Matrix for loss
    model_val = PMMRModel(**model_param_val)
    K_AZX_val = model_val.fit(val_data, data_name)
    
    # predict h(a, w) (bridge fct) on val_set for (A, W) with model_train param & eval model
    pred_bridge = model_train.predict_bridge(val_data.treatment, val_data.outcome_proxy)
    pred_bridge = preprocessor.postprocess_for_prediction(pred_bridge)
    val_loss = PMMR_loss(pred_bridge, val_data.outcome, K_AZX_val)
    return val_loss

def pmmr_exp(data_config_in: Dict[str, Any], model_param: Dict[str, Any],
             one_mdl_dump_dir: Path,
             random_seed: int = 42, verbose: int = 0):
    
    data_config = dict(data_config_in)
    data_config_val = dict(data_config)
    
    # small hold out set?
#     data_config_val["n_sample"] = int(data_config["n_sample"] * 0.11)
#     data_config["n_sample"] = int(data_config["n_sample"] * 0.9) 
    
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_org = generate_train_data_ate(data_config=data_config_val, rand_seed=random_seed+1)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    
    data_name = data_config.get("name", None)
    
    # grid search
    lam1_candidate_list = model_param["lam1"]
    grid_search = dict(
        [(lam1_candi, cal_val_err_pmmr(lam1_candi, model_param, data_name, train_data, val_data, preprocessor)) for lam1_candi in lam1_candidate_list]
    )
    lam1_opt, loss = min(grid_search.items(), key=operator.itemgetter(1))
    
    # fit optimal model on train data
    model_param_opt = dict(model_param)
    model_param_opt["lam1"] = lam1_opt
    model = PMMRModel(**model_param_opt)
    model.fit(train_data, data_config["name"])
    
    # predict bridge fct
    pred_bridge = model.predict_bridge(val_data.treatment, val_data.outcome_proxy)
    pred_bridge = preprocessor.postprocess_for_prediction(pred_bridge)
    
    # predict true effect
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)
    
    mse = np.mean((pred - test_data.structural) ** 2)
    mae = np.mean(np.abs(pred - test_data.structural))
    
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    
    return train_data, test_data, val_data, pred, mse, mae, pred_bridge, lam1_opt

def kpv_exp(data_config: Dict[str, Any], model_param: Dict[str, Any],
            one_mdl_dump_dir: Path,
            random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = KernelPVModel(**model_param)
    model.fit(train_data, data_config["name"])
    
    pred, lams_opt = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)
    
    mse = np.mean((pred - test_data.structural) ** 2)
    mae = np.mean(np.abs(pred - test_data.structural))
    
    return train_data, test_data, pred, mse, mae, lams_opt

def NMMR_exp(data_config: Dict[str, Any], model_config: Dict[str, Any],
                    one_mdl_dump_dir: Path,
                    random_seed: int = 42, verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate data
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data = generate_test_data_ate(data_config=data_config)

    # convert datasets to Torch (for GPU runtime)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    data_name = data_config.get("name", None)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    
    # perform grid_search
    candidate_list = []
    if len(dict(grid_search_dict(model_config))) > 1:
        for config in grid_search_dict(model_config):
    
            # retrieve the trainer for this experiment
            if data_name in ['exponential', 'demand', 'cosine', 'linear']:
                trainer_grid = NMMR_Trainer_DemandExperiment(data_config, config[1], random_seed, one_mdl_dump_dir)
            elif data_name in ['mastourie']:
                trainer_grid = NMMR_Trainer_MastourieExperiment(data_config, config[1], random_seed, one_mdl_dump_dir)
        
            # train model
            model_grid, val_loss = trainer_grid.train(train_t, val_data_t, verbose)
            candidate_list.append((str(config[1]), np.array(val_loss)))
        
        # extract optimal model
        candidate_dict = dict(candidate_list)
        config_opt, loss = min(candidate_dict.items(), key=operator.itemgetter(1))
        config_opt = ast.literal_eval(config_opt)
    else:
        config_opt = list(dict(grid_search_dict(model_config)).values())[0]
    
    # fit optimal model
    if data_name in ['exponential', 'demand', 'cosine', 'linear']:
        trainer_opt = NMMR_Trainer_DemandExperiment(data_config, config_opt, random_seed, one_mdl_dump_dir)
    elif data_name in ['mastourie']:
        trainer_opt = NMMR_Trainer_MastourieExperiment(data_config, config_opt, random_seed, one_mdl_dump_dir)
        
    # train optimal model
    model_opt, val_loss = trainer_opt.train(train_t, val_data_t, verbose)
    

    # prepare test data on the gpu
    if trainer_opt.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()
        val_data_t = val_data_t.to_gpu()

    E_wx_hawx = trainer_opt.predict(model_opt, test_data_t, val_data_t)

    pred = E_wx_hawx.detach().numpy()

    np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
    
    mse = np.mean((pred - test_data.structural) ** 2)
    mae = np.mean(np.abs(pred - test_data.structural))
    
    
    return train_data, test_data, pred, mse, mae, config_opt

# def run_exp(model_config, data_config, sd_list, verbose = 2):    
#     exp_name = data_config["name"]
#     n_sample = data_config["n_sample"]
    
#     # extract time and name
#     now = datetime.datetime.now()
#     time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
#     model_name = model_config["name"]
    
#     # create path to save experiments
#     dump_dir = Path.cwd().joinpath('dumps')
#     exp_dir = dump_dir.joinpath(f"experiment_{exp_name}_n_sample_{n_sample}")
#     model_dir = exp_dir.joinpath(f"model_{model_name}")
#     time_dir = model_dir.joinpath(f"time_{time_string}")
#     if not os.path.exists(dump_dir):
#         os.mkdir(dump_dir)
#     if not os.path.exists(exp_dir):
#         os.mkdir(exp_dir)
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     if not os.path.exists(time_dir):
#         os.mkdir(time_dir)
        
        
#     # run experiment
#     pred_all = []
#     mse_all = []
#     mae_all = []
#     param_all = []
#     for sd in sd_list:
#         if model_name == "pmmr":
#             model_typ = model_name
#             train_data, test_data, val_data, pred, mse, mae, pred_bridge, param_opt = pmmr_exp(data_config_in=data_config, model_param=model_config, one_mdl_dump_dir = time_dir, random_seed = sd, verbose=verbose)
#             print(f"Finished with sd: {sd}, lam1:{param_opt}")
        
#         elif model_name == "kpv":
#             model_typ = model_name
#             train_data, test_data, pred, mse, mae, param_opt = kpv_exp(data_config=data_config, model_param=model_config, one_mdl_dump_dir = time_dir, random_seed = sd, verbose=verbose)
#             print(f"Finished with sd: {sd}, lam1/lam2:{param_opt}")
        
#         elif model_name == "nmmr":
#             loss_name = model_config["loss_name"][0]
#             model_typ = f"{model_name}_{loss_name}" 
#             train_data, test_data, pred, mse, mae, param_opt = NMMR_exp(data_config=data_config, model_config=model_config, one_mdl_dump_dir = time_dir, random_seed=sd, verbose = verbose)
#             print(f"Finished with sd: {sd}, lam1:{param_opt}")
            
#         else:
#             raise ValueError
            
#         pred_all.append(pred)
#         mse_all.append(mse)
#         mae_all.append(mae)
#         param_all.append(param_opt)
    
#     pred_all = np.squeeze(np.array(pred_all)).T

#     if len(sd_list) == 1:
#         pred_avg = pred_all
#     else:
#         pred_avg = np.mean(pred_all, axis = 1)
#     mse_avg = np.mean(mse_all)
#     mae_avg = np.mean(mae_all)
    
#     # for Mastourie plot
#     error = np.array(test_data.structural - pred_all)
#     std = np.std(error, axis = 1).reshape(-1, 1)
#     error_abs = np.array(np.abs(test_data.structural - pred_all))
#     std_abs = np.std(error_abs, axis = 1).reshape(-1, 1)
    
#     results = {
#         'model': model_typ, 
#         'pred_avg': pred_avg,
#         'train_data': train_data,
#         'test_data': test_data,
#         'pred_all': pred_all,
#         'mse_avg': mse_avg,
#         'mae_avg': mae_avg,
#         'mse_all': mse_all,
#         'mae_all': mae_all,
#         'param_opt': param_all,
#         'sd_list': sd_list,
#         'error': error,
#         'std': std,
#         'error_abs': error_abs,
#         'std_abs': std_abs,
#         'model_config': model_config,
#         'data_config': data_config,
#     }
    
#     df_results = pd.DataFrame.from_dict(results, orient='index')
    
#     sd_concat = '_'.join(map(str, sd_list))
#     if model_name in ["pmmr", "kpv"]:
#         df_results.to_csv(time_dir.joinpath(f"results_{sd_concat}.csv"), index=True)
#         np.save(time_dir.joinpath(f"pred_all_{sd_concat}.txt"), pred_all)
#     elif model_name == "nmmr":
#         df_results.to_csv(time_dir.joinpath(f"results_{loss_name}_{sd_concat}.csv"), index=True)
#         np.save(time_dir.joinpath(f"pred_all_{loss_name}_{sd_concat}.txt"), pred_all)
        
#     return results

def create_dir_path(config):
    # extract time and name
    now = datetime.datetime.now()
    time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = config["data"]["name"]
    n_sample = config["data"]["n_sample"]
    model_name = config["model"]["name"]
    
    if model_name == "nmmr":
        loss_name = config["model"]["loss_name"][0]
        model_typ = f"{model_name}_{loss_name}"
    else:
        model_typ = model_name
    
    # create path to save experiments
    path = Path.cwd() / 'dumps' / f"experiment_{exp_name}_n_sample_{n_sample}" / f"model_{model_typ}" / f"time_{time_string}"
    os.makedirs(path, exist_ok=True)
    
    return path, model_typ

def run_exp(model_config, data_config, sd_list, verbose = 2):
    # create right path
    config = {"model": model_config, "data": data_config}
    path, model_typ = create_dir_path(config)
    
    # get name
    model_name = model_config["name"]
    
    # run experiment
    pred_all, mse_all, mae_all, param_all = [], [], [], []
    for sd in sd_list:
        if model_name == "pmmr":
            train_data, test_data, val_data, pred, mse, mae, pred_bridge, param_opt = pmmr_exp(data_config_in=data_config, model_param=model_config, one_mdl_dump_dir = path, random_seed = sd, verbose=verbose)
            print(f"Finished with sd: {sd}, lam1:{param_opt}")
        
        elif model_name == "kpv":
            train_data, test_data, pred, mse, mae, param_opt = kpv_exp(data_config=data_config, model_param=model_config, one_mdl_dump_dir = path, random_seed = sd, verbose=verbose)
            print(f"Finished with sd: {sd}, lam1/lam2:{param_opt}")
        
        elif model_name == "nmmr":
            train_data, test_data, pred, mse, mae, param_opt = NMMR_exp(data_config=data_config, model_config=model_config, one_mdl_dump_dir = path, random_seed=sd, verbose = verbose)
            
        else:
            raise ValueError
        
        pred_all.append(pred)
        mse_all.append(mse)
        mae_all.append(mae)
        param_all.append(param_opt)
    
    pred_all = np.squeeze(np.array(pred_all)).T
    
    # create average
    if len(sd_list) == 1:
        pred_avg = pred_all
    else:
        pred_avg = np.mean(pred_all, axis = 1)
    mse_avg = np.mean(mse_all)
    mae_avg = np.mean(mae_all)
    
    # create error and std
    error = np.array(test_data.structural - pred_all)
    std = np.std(error, axis = 1).reshape(-1, 1)
    error_abs = np.array(np.abs(test_data.structural - pred_all))
    std_abs = np.std(error_abs, axis = 1).reshape(-1, 1)
    
    results = {
        'model': model_typ, 
        'pred_avg': pred_avg,
        'train_data': train_data,
        'test_data': test_data,
        'pred_all': pred_all,
        'mse_avg': mse_avg,
        'mae_avg': mae_avg,
        'mse_all': mse_all,
        'mae_all': mae_all,
        'param_opt': param_all,
        'sd_list': sd_list,
        'error': error,
        'std': std,
        'error_abs': error_abs,
        'std_abs': std_abs,
        'model_config': model_config,
        'data_config': data_config,
    }
    
    # save results
    df_results = pd.DataFrame.from_dict(results, orient='index')
    
    sd_concat = '_'.join(map(str, sd_list))
    if model_name in ["pmmr", "kpv"]:
        df_results.to_csv(path.joinpath(f"results_{sd_concat}.csv"), index=True)
        np.save(path.joinpath(f"pred_all_{sd_concat}.txt"), pred_all)
    elif model_name == "nmmr":
        df_results.to_csv(path.joinpath(f"results_{sd_concat}.csv"), index=True)
        np.save(path.joinpath(f"pred_all_{sd_concat}.txt"), pred_all)
        
    return results


# def save_data(df_save, data_dic):
#     exp_name = data_dic["name"]
#     n_sample = data_dic["n_sample"]
    
#     # extract time and name
#     now = datetime.datetime.now()
#     time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    
#     # create path to save experiments
#     save_dir = Path.cwd().joinpath('saves')
#     exp_dir = save_dir.joinpath(f"experiment_{exp_name}_n_sample_{n_sample}")
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     if not os.path.exists(exp_dir):
#         os.mkdir(exp_dir)
#     df_save.to_csv(exp_dir.joinpath(f"save_time_{time_string}.csv"), index=False)

def save_data(df_save, data_dic):
    exp_name = data_dic["name"]
    n_sample = data_dic["n_sample"]
    
    # extract time and name
    now = datetime.datetime.now()
    time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    # create path to save experiments
    exp_dir = Path.cwd() / 'saves' / f"experiment_{exp_name}_n_sample_{n_sample}"
    
    # create directories if they do not exist
    os.makedirs(exp_dir, exist_ok=True)

    # save the dataframe to csv
    df_save.to_csv(exp_dir / f"save_time_{time_string}.csv", index=False)

    