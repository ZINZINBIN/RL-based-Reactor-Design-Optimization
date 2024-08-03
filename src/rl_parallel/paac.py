'''
    Paper: Efficient parallel methods for deep reinforcement learning
    - paac.py from https://github.com/Alfredvc/paac/blob/master/paac.py
    - Parallel framework for Efficient Deep RL (ActorCritic Version)
'''
from src.rl.ppo import update_policy, ActorCritic, ReplayBufferPPO
from src.env import Enviornment
from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from src.rl.reward import RewardSender
from src.rl_parallel.runners import Runners, EmulatorRunner
from config.device_info import config_benchmark
from config.search_space_info import search_space
from typing import Optional, Dict
from tqdm.auto import tqdm

import numpy as np
import multiprocessing as mp
from multiprocessing import Queue

# pytorch framework
import torch
import torch.nn as nn

def create_environment(
    args_reward:Dict,
    ):
    
    # load reference state 
    config = config_benchmark
    
    profile = Profile(
        nu_T = config["nu_T"],
        nu_p = config["nu_p"],
        nu_n = config["nu_n"],
        n_avg = config["n_avg"], 
        T_avg = config["T_avg"], 
        p_avg = config['p_avg']
    )
    
    source = CDsource(
        conversion_efficiency = config['conversion_efficiency'],
        absorption_efficiency = config['absorption_efficiency'],
    )
    
    tokamak = Tokamak(
        profile,
        source,
        betan = config['betan'],
        Q = config['Q'],
        k = config['k'],
        epsilon = config['epsilon'],  
        tri = config['tri'],
        thermal_efficiency = config['thermal_efficiency'],
        electric_power = config['electric_power'],
        armour_thickness = config['armour_thickness'],
        armour_density = config['armour_density'],
        armour_cs = config['armour_cs'],
        maximum_wall_load = config['maximum_wall_load'],
        maximum_heat_load = config['maximum_heat_load'],
        shield_density = config['shield_density'],
        shield_depth = config['shield_depth'],
        shield_cs = config['shield_cs'],
        Li_6_density = config['Li_6_density'],
        Li_7_density = config['Li_7_density'],
        slowing_down_cs= config['slowing_down_cs'],
        breeding_cs= config['breeding_cs'],
        E_thres = config['E_thres'],
        pb_density = config['pb_density'],
        scatter_cs_pb=config['cs_pb_scatter'],
        multi_cs_pb=config['cs_pb_multi'],
        B0 = config['B0'],
        H = config['H'],
        maximum_allowable_J = config['maximum_allowable_J'],
        maximum_allowable_stress = config['maximum_allowable_stress'],
        RF_recirculating_rate= config['RF_recirculating_rate'],
        flux_ratio = config['flux_ratio']
    )
    
    reward_sender = RewardSender(
        w_cost = args_reward['w_cost'],
        w_tau = args_reward['w_tau'],
        w_beta = args_reward['w_beta'],
        w_density = args_reward['w_density'],
        w_q = args_reward['w_q'],
        w_bs = args_reward['w_bs'],
        w_i = args_reward['w_i'],
        cost_r = args_reward['cost_r'],
        tau_r = args_reward['tau_r'],
        a = args_reward['a']
    )
    
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
    
    init_state = tokamak.get_design_performance()
    env = Enviornment(tokamak, reward_sender, init_state, init_action)
    return env

def train_ppo_parallel(
    memory : ReplayBufferPPO, 
    num_workers:int,
    num_envs:int,
    args_reward:Dict,
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu",
    num_episode : int = 10000,  
    verbose : int = 8,
    save_best : Optional[str] = None,
    save_last : Optional[str] = None,
):
    if device is None:
        device = "cpu"
    
    best_reward = 0
    reward_list = []
    loss_list = []
    
    emulators = [create_environment(args_reward) for _ in range(num_envs)]
    runners = Runners(EmulatorRunner, emulators, num_workers, variables = None)
    
    # initialize
    runners.init_env()
    
    # multiprocessing
    runners.start()
        
    for i_episode in range(num_episode):
        
        # Load shared trajectories
        shared_states, shared_actions, shared_next_states, shared_rewards, shared_done, shared_log_probs = runners.get_shared_variables()
        
        # update next action 
        runners.update_environments()
        
        # buffer.barrier()
        runners.wait_updated()
        
        # load shared memory
        memory.push(shared_states, shared_actions, shared_next_states, shared_rewards, shared_done, shared_log_probs)
        
        # Optimize the network's parameters
        if memory.__len__() >= memory.capacity:
            policy_loss = update_policy(
                memory, 
                policy_network,
                policy_optimizer,
                criterion,
                gamma,
                eps_clip,
                entropy_coeff,
                device
            )
            
            runners.init_env()
            loss_list.append(policy_loss.detach().cpu().numpy())
                
        # save weights
        torch.save(policy_network.state_dict(), save_last)
              
    print("RL training process clear....!")
    
    return