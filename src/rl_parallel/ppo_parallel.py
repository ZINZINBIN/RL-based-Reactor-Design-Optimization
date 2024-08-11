import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
from typing import Optional, Dict, List

from src.env import Enviornment
from src.rl.ppo import ActorCritic
from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from src.rl.reward import RewardSender
from config.device_info import config_benchmark
from config.search_space_info import search_space

from torch.distributions import Normal
import os, pickle
from collections import namedtuple

# Message type creator
MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])
MsgRewardInfo = namedtuple('MsgRewardInfo', ['agent', 'episode', 'reward'])
MsgMaxReached = namedtuple('MsgMaxReached', ['agent', 'reached'])

# action range for policy network
default_action_range = search_space

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

class ReplayBufferPPO:
    def __init__(self, max_timestep : int, n_emulator:int, state_dim:int, action_dim:int, device : str):
        self.max_timestep = max_timestep
        self.n_emulator = n_emulator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.states = torch.zeros((max_timestep, n_emulator, state_dim)).to(device).share_memory_()
        self.actions = torch.zeros((max_timestep, n_emulator, action_dim)).to(device).share_memory_()
        
        self.next_states = torch.zeros((max_timestep, n_emulator, state_dim)).to(device).share_memory_()
        self.rewards = torch.zeros((max_timestep, n_emulator)).to(device).share_memory_()
        self.probs_a = torch.zeros((max_timestep, n_emulator, action_dim)).to(device).share_memory_()
        
    def get(self):
        return self.states, self.actions, self.next_states, self.rewards, self.probs_a
    
    def initialize(self):
        max_timestep = self.max_timestep
        n_emulator = self.n_emulator
        state_dim = self.state_dim
        action_dim = self.action_dim
        device = self.device
        
        self.states = torch.zeros((max_timestep, n_emulator, state_dim)).to(device).share_memory_()
        self.actions = torch.zeros((max_timestep, n_emulator, action_dim)).to(device).share_memory_()
        self.next_states = torch.zeros((max_timestep, n_emulator, state_dim)).to(device).share_memory_()
        self.rewards = torch.zeros((max_timestep, n_emulator)).to(device).share_memory_()
        self.probs_a = torch.zeros((max_timestep, n_emulator, action_dim)).to(device).share_memory_()
    
    def clear(self):
        self.states.cpu()
        self.actions.cpu()
        self.next_states.cpu()
        self.rewards.cpu()
        self.probs_a.cpu()
        
        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.probs_a = None
    
def update_policy(
    memory:ReplayBufferPPO, 
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
    
    # state, action, next_states, probs_a : (max_timesteps, n_emulators, dims) => (max_timestep * n_emul, dims)
    states, actions, next_states, rewards, probs_a = memory.get()
    
    state_batch = states.view((memory.max_timestep * memory.n_emulator, memory.state_dim)).float() 
    action_batch = states.view((memory.max_timestep * memory.n_emulator, memory.action_dim)).float()
    next_state_batch = states.view((memory.max_timestep * memory.n_emulator, memory.state_dim)).float()
    prob_a_batch = states.view((memory.max_timestep * memory.n_emulator, memory.action_dim)).float()

    # Multi-step version reward: Monte Carlo estimate
    rewards_list = []
    discounted_reward = torch.zeros((memory.n_emulator)).to(device)
    
    for t in reversed(range(memory.max_timestep)):
        reward_t = rewards[t]
        discounted_reward = reward_t + (gamma * discounted_reward)
        rewards_list.insert(0, discounted_reward)
        
    reward_batch = torch.cat(rewards_list).float().to(device)
    
    # update policy through clip loss
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

class Agent(mp.Process):
    def __init__(
        self, 
        name : str, 
        memory:ReplayBufferPPO, 
        pipe:List, 
        max_timestep:int, 
        num_episode:int, 
        gamma:float, 
        verbose:int, 
        args_reward:Dict,
        policy_old:ActorCritic,
        device:str
        ):
        
        mp.Process.__init__(self,name = name)

        self.proc_id = name
        self.memory = memory
        self.pipe = pipe
        
        self.max_timestep = max_timestep
        self.num_episode = num_episode
        self.verbose = verbose
        
        self.gamma = gamma
        self.env = create_environment(args_reward)
        
        self.policy = policy_old
        self.device = device
        
    def convert_tensor(self, states, actions, rewards, next_states, probs_a):
        state_tensor = torch.cat(states).float()
        action_tensor = torch.cat(actions).float()
        reward_tensor = torch.cat(rewards).float()
        next_state_tensor = torch.cat(next_states).float()
        prob_a_tensor = torch.cat(probs_a).float()
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, prob_a_tensor
    
    def add_to_buffer(self, state_tensor, action_tensor, reward_tensor, next_state_tensor, prob_a_tensor):
        
        proc_id = int(self.name)
        
        self.memory.states[:, proc_id, :] = state_tensor
        self.memory.actions[:, proc_id, :] = action_tensor
        self.memory.rewards[:, proc_id] = reward_tensor
        self.memory.next_states[:, proc_id, :] = next_state_tensor
        self.memory.probs_a[:, proc_id, :] = prob_a_tensor
        
    def run(self):
        
        print("Agent {} started, Process ID {}".format(self.name, os.getpid()))
        
        states = []
        actions = []
        rewards = []
        probs_a = []
        next_states = []
        

        i_episode = 0
        local_timestep = 0
    
        while i_episode <= self.num_episode:
            
            if self.env.current_state is None:
                state = self.env.init_state
                ctrl = self.env.init_action
            else:
                state = self.env.current_state
                ctrl = self.env.current_action
        
            state_tensor = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
    
            self.policy.eval()
            
            action_tensor, entropy, log_probs, value = self.policy.sample(state_tensor.to(self.device))
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
        
            state_new, reward, done, _ = self.env.step(ctrl_new)
            
            reward = torch.tensor([reward])
            
            next_state_tensor = np.array([state_new[key] for key in state_new.keys()] + [ctrl_new[key] for key in ctrl_new.keys()])
            next_state_tensor = torch.from_numpy(next_state_tensor).unsqueeze(0).float()
        
            states.append(state_tensor)
            actions.append(action_tensor)
            rewards.append(reward)
            next_states.append(next_state_tensor)
            probs_a.append(log_probs)
            
            # update state
            self.env.current_state = state_new
            self.env.current_action = ctrl_new
            
            # update episode
            i_episode += 1
            local_timestep += 1
                 
            # update shared memory and policy
            if local_timestep >= self.max_timestep:
                
                # convert list to tensor
                states_tensor, actions_tensor, rewards_tensor, next_states_tensor, probs_a_tensor \
                = self.convert_tensor(states, actions, rewards, next_states, probs_a)
                
                # add to memory
                self.add_to_buffer(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, probs_a_tensor)
                
                msg = MsgUpdateRequest(int(self.proc_id), True)
                self.pipe.send(msg)
                msg = self.pipe.recv()
                
                self.env.current_state = None
                self.env.current_action = None
                
                # clean list
                states.clear()
                actions.clear()
                rewards.clear()
                next_states.clear()
                probs_a.clear()
                
                local_timestep = 0
                
def train_ppo_parallel(
    num_workers:int,
    buffer_size:int,
    args_reward:Dict,
    args_policy:Dict,
    lr : float, 
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
        
    # define policy network (new vs old)
    policy_network = ActorCritic(
        input_dim = args_policy['input_dim'], 
        mlp_dim = args_policy['mlp_dim'], 
        n_actions = args_policy['n_actions'], 
        std = args_policy['std']
    ).to(device)
    
    policy_network_old = ActorCritic(
        input_dim = args_policy['input_dim'], 
        mlp_dim = args_policy['mlp_dim'], 
        n_actions = args_policy['n_actions'], 
        std = args_policy['std']
    ).to(device).share_memory()
    
    policy_network_old.load_state_dict(policy_network.state_dict())
    
    policy_optimizer = torch.optim.RMSprop(policy_network.parameters(), lr)
    
    memory = ReplayBufferPPO(buffer_size, num_workers, args_policy['input_dim'], args_policy['n_actions'], device)
    
    agents = []
    pipes = []
    
    update_request = [False] * num_workers
    agent_completed = [False] * num_workers
    
    best_reward = 0
    reward_list = []
    loss_list = []
    
    update_iteration = 0
    log_iteration = 0
    
    # initialize subprocesses experience
    for agent_id in range(num_workers):
        p_start, p_end = mp.Pipe()
        agent = Agent(str(agent_id), memory, p_end, buffer_size, num_episode, gamma, verbose, args_reward, policy_network_old, device)
        agent.start()
        
        agents.append(agent)
        pipes.append(p_start)
    
    while True:
        for i, conn in enumerate(pipes):
            if conn.poll():
                msg = conn.recv()
    
                if type(msg).__name__ == "MsgMaxReached":
                    agent_completed[i] = True
                
                elif type(msg).__name__ == "MsgUpdateRequest":
                    update_request[i] = True
                    
                    if False not in update_request:
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
                        
                        update_request = [False] * num_workers
                        update_iteration += 1
                        msg = update_iteration
                    
                        for pipe in pipes:
                            pipe.send(msg)
            
        if False not in agent_completed:
            print("Parallelized RL optimization complete..!")
            break
        
    for agent in agents:
        agent.terminate()
            
    return None