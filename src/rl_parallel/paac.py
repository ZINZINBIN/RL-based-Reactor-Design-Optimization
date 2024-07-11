'''
    Paper: Efficient parallel methods for deep reinforcement learning
    - paac.py from https://github.com/Alfredvc/paac/blob/master/paac.py
    - Parallel framework for Efficient Deep RL (ActorCritic Version)
'''
from src.rl.ppo import update_policy, ActorCritic
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
        beta_r = args_reward['beta_r'],
        q_r = args_reward['q_r'],
        n_r = args_reward['n_r'],
        f_r = args_reward['f_r'],
        i_r = args_reward['i_r'],
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
    num_workers:int,
    num_envs:int,
    args_reward:Dict,
    memory : Queue, 
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
    
        if env.current_state is None:
            state = env.init_state
            ctrl = env.init_action
        else:
            state = env.current_state
            ctrl = env.current_action
            
        state_tensor = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
    
        policy_network.eval()
        action_tensor, entropy, log_probs, value = policy_network.sample(state_tensor.to(device))
        action = action_tensor.detach().squeeze(0).cpu().numpy()
        
        ctrl_new = {
            'betan':action[0],
            'k':action[1],
            'epsilon' : action[2],
            'electric_power' : action[3],
            'T_avg' : action[4],
            'B0' : action[5],
            'H' : action[6],
            "armour_thickness":action[7],
            "RF_recirculating_rate":action[8],
        }
        
        state_new, reward, done, _ = env.step(ctrl_new)
        
        if state_new is None:
            continue
    
        reward_list.append(reward)
        reward = torch.tensor([reward])
        
        next_state_tensor = np.array([state_new[key] for key in state_new.keys()] + [ctrl_new[key] for key in ctrl_new.keys()])
        next_state_tensor = torch.from_numpy(next_state_tensor).unsqueeze(0).float()
        
        # memory에 transition 저장
        memory.push(state_tensor, action_tensor, next_state_tensor, reward, done, log_probs)

        # update state
        env.current_state = state_new
        env.current_action = ctrl_new
            
        # update policy
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
            
            env.current_state = None
            env.current_action = None
            
            loss_list.append(policy_loss.detach().cpu().numpy())
                
        if i_episode % verbose == 0:
            
            print(r"| episode:{} | reward : {} | tau : {:.3f} | beta limit : {} | q limit : {} | n limit {} | f_bs limit : {} | ignition : {} | cost : {:.3f}".format(
                i_episode+1, env.rewards[-1], env.taus[-1], env.beta_limits[-1], env.q_limits[-1], env.n_limits[-1], env.f_limits[-1], env.i_limits[-1], env.costs[-1]
            ))
            
            # env.tokamak.print_info(None)

        # save weights
        torch.save(policy_network.state_dict(), save_last)
        
        if env.rewards[-1] > best_reward:
            best_reward = env.rewards[-1]
            best_episode = i_episode
            torch.save(policy_network.state_dict(), save_best)

    print("RL training process clear....!")
    
    result = {
        "control":env.actions,
        "state":env.states,
        "reward":env.rewards,
        "tau":env.taus,
        "beta_limit" : env.beta_limits,
        "q_limit" : env.q_limits,
        "n_limit" : env.n_limits,
        "f_limit" : env.f_limits,
        "i_limit" : env.i_limits,
        "cost" : env.costs,
        "loss" : loss_list
    }
    
    return result