from src.utility import plot_optimization_status, plot_policy_loss
from config.device_info import config_benchmark, config_liquid
from src.rl_parallel.ppo_parallel import train_ppo_parallel

import argparse, os, warnings, pickle, torch
import numpy as np
import torch.multiprocessing as mp

mp.set_start_method('spawn', True)

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="Tokamak design optimization based on parallel advantage RL ")
    
    # tag for labeling the optimization process
    parser.add_argument("--tag", type = str, default = "")
    
    # Select blanket type: liquid / solid
    parser.add_argument("--blanket_type", type = str, default = "solid", choices = ['liquid','solid'])
    
    # GPU allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # PPO setup
    parser.add_argument("--buffer_size", type = int, default = 4)
    parser.add_argument("--num_episode", type = int, default = 10000)
    parser.add_argument("--verbose", type = int, default = 10000)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--gamma", type = float, default = 0.999)
    parser.add_argument("--eps_clip", type = float, default = 0.2)
    parser.add_argument("--entropy_coeff", type = float, default = 0.05)
    
    # Reward setup
    parser.add_argument("--w_cost", type = float, default = 0.1)
    parser.add_argument("--w_tau", type = float, default = 0.1)
    parser.add_argument("--w_beta", type = float, default = 0.5)
    parser.add_argument("--w_density", type = float, default = 0.5)
    parser.add_argument("--w_q", type = float, default = 1.0)
    parser.add_argument("--w_bs", type = float, default = 1.0)
    parser.add_argument("--w_i", type = float, default = 1.5)
    parser.add_argument("--cost_r", type = float, default = 1.0)
    parser.add_argument("--tau_r", type = float, default = 1.0)
    parser.add_argument("--a", type = float, default = 1.0)
    parser.add_argument("--reward_fail", type = float, default = -1.0)
    
    # Parallel RL setup
    parser.add_argument("--n_workers", type = int, default = 8)
    
    # Visualization
    parser.add_argument("--smoothing_temporal_length", type = int, default = 16)
    
    args = vars(parser.parse_args()) 

    return args
    
if __name__ == "__main__":
    
    # torch device state
    print("=============== Device setup ===============")
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())
    print("torch version : ", torch.__version__)
    
    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(args['gpu_num'])
    else:
        device = 'cpu'
    
    if args['blanket_type'] == 'liquid':
        config = config_liquid
    else:
        config = config_benchmark
        
    args_policy = {
        "input_dim":19 + 9,
        "mlp_dim":64,
        "n_actions":9,
        'std':0.5
    }
        
    args_reward = {
        "w_cost" : args['w_cost'],
        "w_tau" : args['w_tau'],
        "w_beta" : args['w_beta'],
        "w_density" : args['w_density'],
        "w_q" : args['w_q'],
        "w_bs" : args['w_bs'],
        "w_i" : args['w_i'],
        "cost_r" : args['cost_r'],
        "tau_r" : args['tau_r'],
        "a" : args['a'],
        "reward_fail" : args['reward_fail']
    }    
    
    n_workers = args['n_workers']
    
    # loss function for critic network
    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')
    
    # directory
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    if len(args['tag']) > 0:
        tag = "PPO-parallel_{}_{}".format(args['blanket_type'], args['tag'])
    else:
        tag = "PPO-parallel_{}".format(args['blanket_type'])
        
    save_best = "./weights/{}_best.pt".format(tag)
    save_last = "./weights/{}_last.pt".format(tag)
    save_result = "./results/params_search_{}.pkl".format(tag)
    
    # Design optimization
    print("============ Design optimization ===========")
    train_ppo_parallel(
        n_workers,
        args['buffer_size'],
        args_reward,
        args_policy,
        args['lr'],
        value_loss_fn,
        args['gamma'],
        args['eps_clip'],
        args['entropy_coeff'],
        device,
        args['num_episode'],
        args['verbose'],
        save_best,
        save_last
    )