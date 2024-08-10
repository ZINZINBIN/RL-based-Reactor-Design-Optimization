'''
    Paper: Efficient parallel methods for deep reinforcement learning
    - paac.py from https://github.com/Alfredvc/paac/blob/master/paac.py
    - Parallel framework for Efficient Deep RL (ActorCritic Version)
'''
from src.rl.ppo import ActorCritic
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

class GlobalMemory:
    def __init__(self, buffer_size:int, state_dim:int, action_dim:int, n_emulator:int):
        
        self.states = np.zeros((buffer_size, n_emulator, state_dim), dtype = np.float32)
        self.actions = np.zeros((buffer_size, n_emulator, action_dim), dtype = np.float32)
        
        self.next_states = np.zeros((buffer_size, n_emulator, state_dim), dtype = np.float32)
        self.rewards = np.zeros((buffer_size, n_emulator), dtype = np.float32)
        self.probs_a = np.zeros((buffer_size, n_emulator, action_dim), dtype = np.float32)   
        
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_emulator = n_emulator
        
    def get_buffer_size(self):
        return self.buffer_size
    
def initialize_memory(emulators):
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

def update_policy(
    memory:GlobalMemory, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu"
    ):
    
    policy_network.train()

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = 'none') # Huber Loss for critic network
    
    states = memory.states
    actions = memory.actions
    next_states = memory.next_states
    probs_a = memory.probs_a
    rewards = memory.rewards
    
    # Multi-step version reward: Monte Carlo estimate
    rewards_list = []
    discounted_reward = np.zeros((rewards.shape[1:]))
    
    for t in reversed(range(memory.get_buffer_size())):
        reward_t = rewards[t]
        discounted_reward = reward_t + (gamma * discounted_reward)
        rewards_list.insert(0, discounted_reward)
        
    expected_rewards = np.concatenate(rewards_list, axis = 0).reshape(-1,)
    reward_batch = torch.cat(expected_rewards).float().to(device)
    
    state_batch = torch.cat(states.reshape(-1, memory.state_dim)).float().to(device)
    action_batch = torch.cat(actions.reshape(-1, memory.action_dim)).float().to(device)
    prob_a_batch = torch.cat(probs_a.reshape(-1, memory.action_dim)).float().to(device) # pi_old
    
    next_state_batch = torch.cat(next_states.reshape(-1, memory.state_dim)).float().to(device)
    
    policy_optimizer.zero_grad()
    
    _, _, next_log_probs, next_value = policy_network.sample(next_state_batch)
    action, entropy, log_probs, value = policy_network.sample(state_batch)
    
    td_target = reward_batch.view_as(next_value) + gamma * next_value
    
    delta = td_target - value        
    ratio = torch.exp(log_probs - prob_a_batch.detach())
    surr1 = ratio * delta
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * delta
    loss = -torch.min(surr1, surr2) + criterion(value, td_target) - entropy_coeff * entropy
    loss = loss.mean()
    loss.backward()

    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1)
        
    policy_optimizer.step()
    
    return loss
    

def train_ppo_parallel(
    memory : GlobalMemory, 
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
    
    # Load shared trajectories
    shared_states, shared_actions, shared_rewards = runners.get_shared_variables()
        
    i_episode = 0
    
    while i_episode <= num_episode:
        
        # Get new action from policy network
        policy_network.eval()
        
        for t in range(memory.get_buffer_size()):
            
            # memory update : state 
            memory.states[t] = shared_states
        
            state_tensor = torch.from_numpy(shared_states)
            action_tensor, entropy, log_probs, value = policy_network.sample(state_tensor.to(device))
            
            for z in range(action_tensor.size()[0]):
                shared_actions[z] = action_tensor[z].detach().cpu().numpy()
                
            # memory update : next_state, reward, action, prob_a
            memory.next_states[t] = shared_states
            memory.rewards[t] = shared_rewards
            memory.actions[t] = action_tensor
            memory.probs_a[t] = log_probs
            
            # update next action 
            runners.update_environments()
            
            i_episode += 1
            
            # buffer.barrier()
        runners.wait_updated()
        
        # Optimize the network's parameters
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