import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.env import Enviornment
from torch.distributions import Normal
import os, pickle
from collections import namedtuple, deque
from typing import Optional

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action','next_state','reward','done','prob_a')
)

default_action_range = {
    "betan":[2.0, 3.5],
    "k" : [1.5, 2.5],
    "epsilon" : [2.0, 5.0],
    "electric_power" : [800, 1500],
    "T_avg" : [10, 25],
    "B0" : [10, 16],
    "H" : [1.0, 1.5],
    "armour_thickness": [0.01, 0.15],
    "RF_recirculating_rate":[0.05, 0.2],
}

class ReplayBufferPPO(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def get_trajectory(self):
        traj = [self.memory[idx] for idx in range(len(self.memory))]
        return traj
    
    def clear(self):
        self.memory.clear()
    
    def save_buffer(self, env_name : str, tag : str = "", save_path : Optional[str] = None):
        
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/', exist_ok=True)

        if save_path is None:
            save_path = "checkpoints/buffer_{}_{}".format(env_name, tag)
            
        print("Process : saving buffer to {}".format(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump(self.memory, f)
        
    def load_buffer(self, save_path : str):
        print("Process : loading buffer from {}".format(save_path))
        
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)

class ActorCritic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, action_range : Dict = default_action_range, std : float = 0.0):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        
        self.fc_pi = nn.Linear(mlp_dim // 2, n_actions)
        self.fc_v = nn.Linear(mlp_dim // 2, 1)
        
        self.action_range = action_range
        self.min_values = [action_range[key][0] for key in action_range.keys()]
        self.max_values = [action_range[key][1] for key in action_range.keys()]
        
        self.log_std = nn.Parameter(torch.ones(1, n_actions) * std)
        
    def forward(self, x : torch.Tensor):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        mu = self.fc_pi(x)
        mu = torch.clamp(mu, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))
        std = self.log_std.exp().expand_as(mu)
    
        dist = Normal(mu, std)
        value = self.fc_v(x)
        
        return dist, value
    
    def sample(self, x : torch.Tensor):
        dist, value = self.forward(x)
        xs = dist.rsample()
        action = torch.clamp(xs, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, entropy, log_probs, value
    
# update policy
def update_policy(
    memory : ReplayBufferPPO, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu",
    ):

    policy_network.train()

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction='mean') # Huber Loss for critic network
    
    transitions = memory.get_trajectory()
    memory.clear()
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, batch.next_state)
        ),
        device = device,
        dtype = torch.bool
    )
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    prob_a_batch = torch.cat(batch.prob_a).to(device)

    policy_optimizer.zero_grad()
    
    _, _, next_log_probs, next_value = policy_network.sample(non_final_next_states)
    action, entropy, log_probs, value = policy_network.sample(state_batch)
    
    td_target = reward_batch + gamma * next_value
    td_target = td_target.detach()
    
    delta = td_target - value        
    ratio = torch.exp(log_probs - prob_a_batch)
    surr1 = ratio * delta
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * delta
    loss = -torch.min(surr1, surr2) + criterion(value, td_target) - entropy_coeff * entropy
    
    loss = loss.mean()
    loss.backward()

    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1)
        
    policy_optimizer.step()
    
    return loss

def train_ppo(
    env : Enviornment,
    memory : ReplayBufferPPO, 
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

    episode_durations = []
    
    best_reward = 0
    best_episode = 0
    best_action = None
    best_state = None
    reward_list = []
    
    for i_episode in tqdm(range(num_episode), desc = 'PPO algorithm training process'):
    
        state = env.init_state
        ctrl = env.init_action
        
        state_tensor = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
    
        policy_network.eval()
        action_tensor, entropy, log_probs, value = policy_network.sample(state_tensor.to(device))
        action_tensor = action_tensor.detach()
        action = action_tensor.squeeze(0).cpu().numpy()
        
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
        state = state_new
        ctrl = ctrl_new
            
        # update policy
        policy_loss = update_policy(
            memory, 
            policy_network, 
            policy_optimizer,
            criterion,
            gamma, 
            eps_clip,
            entropy_coeff,
            device,
        )
                
        if i_episode % verbose == 0:
            print(r"| episode:{} | reward : {} | tau : {:.3f} | beta limit : {} | q limit : {} | n limit {} | f_bs limit : {} | ignition : {} | cost : {:.3f}".format(
                i_episode+1, env.rewards[-1], env.taus[-1], env.beta_limits[-1], env.q_limits[-1], env.n_limits[-1], env.f_limits[-1], env.i_limits[-1], env.costs[-1]
            ))
            env.tokamak.print_info(None)

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
        "cost" : env.costs
    }
    
    return result