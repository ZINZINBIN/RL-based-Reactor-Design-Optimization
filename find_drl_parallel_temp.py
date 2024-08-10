from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from src.env import Enviornment
from src.rl.ppo import train_ppo, ActorCritic, ReplayBufferPPO
from src.rl.reward import RewardSender
from src.utility import plot_optimization_status, plot_policy_loss
from config.device_info import config_benchmark, config_liquid
from src.rl_parallel.runners import Runners, EmulatorRunner
from src.rl_parallel.paac import create_environment

import argparse, os, warnings, pickle, torch
import numpy as np

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="Tokamak design optimization based on parallel advantage RL ")
    
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
    
    # Parallel RL setup
    
    
    args = vars(parser.parse_args()) 

    return args

# torch device state
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())
print("torch version : ", torch.__version__)
    
if __name__ == "__main__":
    
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
        
    args_reward = {
        "w_cost":0.1,
        "w_tau": 0.1,
        "w_beta": 0.5,
        "w_density":0.5,
        "w_q" : 1.0,
        "w_bs" : 1.0,
        "w_i" : 1.0,
        "cost_r" : 1.0,
        "tau_r" : 1.0,
        "a" : 1.0,
        "reward_fail" : -1.0
    }    
    
    workers = 16
    
    init_action = {
        'betan':config['betan'],
        'k':config['k'],
        'epsilon' : config['epsilon'],
        'electric_power' : config['electric_power'],
        'T_avg' : config['T_avg'],
        'B0' : config['B0'],
        'H' : config['H'],
        "armour_thickness" : config['armour_thickness'],
        "RF_recirculating_rate": config['RF_recirculating_rate'],
    }
    
        
    emulators = np.asarray([create_environment(args_reward) for _ in range(workers * 4)])
    init_state = emulators[0].init_state
    init_action = emulators[0].init_action
    
    init_state = np.array([init_state[key] for key in init_state.keys()] + [init_action[key] for key in init_action.keys()])
    init_state = np.repeat(init_state.reshape(1,-1), len(emulators), axis = 0)
    
    init_action = np.array([init_action[key] for key in init_action.keys()])
    init_action = np.repeat(init_action.reshape(1,-1), len(emulators), axis = 0)
    
    # variables: [('state': np.asarray(state : [s1,s2,...])),('action':np.asarray((action:[a1,a2,...]))),('next_state':...),('reward'),('done'),('prob_a')]
    variables = [
        (init_state),
        (init_action),
        (np.zeros((len(emulators), 19 + 9), dtype = np.float32)),
        (np.asarray([0 for emulator in emulators], dtype=np.uint8)),
        (np.asarray([0 for emulator in emulators], dtype=np.uint8)),
        (np.zeros((len(emulators), 9), dtype = np.float32)),
    ]
    
    runner = Runners(emulators, workers, variables)
    runner.init_env()
    runner.start()
    
    # update next action 
    runner.update_environments()
    
    # buffer.barrier()
    runner.wait_updated()
    
    # policy and value network
    policy_network = ActorCritic(input_dim = 19 + 9, mlp_dim = 64, n_actions = 9, std = 0.25)
    
    # gpu allocation
    policy_network.to(device)
    
    # optimizer    
    policy_optimizer = torch.optim.RMSprop(policy_network.parameters(), lr = args['lr'])
    
    # loss function for critic network
    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')