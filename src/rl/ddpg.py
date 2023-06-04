import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc, time
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.rl.buffer import ReplayBuffer, Transition
from src.env import Enviornment

default_action_range = {
    "betan":[2.5, 3.0],
    "k" : [1.5, 2.5],
    "epsilon" : [3.0, 5.0],
    "electric_power" : [1000, 1500],
    "T_avg" : [10, 100],
    "B0" : [13, 16],
    "H" : [1.0, 1.5]
}

class Actor(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, action_range : Dict = default_action_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.fc4 = nn.Linear(mlp_dim // 2, n_actions)
        
        self.action_range = action_range
        self.min_values = [action_range[key][0] for key in action_range.keys()]
        self.max_values = [action_range[key][1] for key in action_range.keys()]
        
    def forward(self, x : torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.clamp(x, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))
        return x
    
# Critic
class Critic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_actions, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.fc4 = nn.Linear(mlp_dim // 2, 1)
        
    def forward(self, state:torch.Tensor, action : torch.Tensor):
        x = torch.cat([state, action], dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class OUNoise:
    def __init__(self, n_action : int, mu : float = 0, theta : float = 0.1, max_sigma : float = 0.1, min_sigma : float = 0.05):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        
        self.action_dim = n_action
        
        self.state = torch.empty_like(torch.ones((1, self.action_dim)))
        
        self.reset()

    def reset(self):
        self.state = torch.ones((1,self.action_dim), dtype = torch.float32) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx  = self.theta * (self.mu - x) + self.sigma * torch.from_numpy(np.random.randn(1, self.action_dim).astype(np.float32))
        self.state = x + dx
        return self.state
    
    def get_action(self, action:torch.Tensor, t : float = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1, t / 1000)
        return action + ou_state.to(action.device)
    
# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    value_optimizer : torch.optim.Optimizer,
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2
    ):

    policy_network.train()
    value_network.train()
    
    target_policy_network.eval()
    target_value_network.eval()

    if len(memory) < batch_size:
        return None, None

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction='mean') # Huber Loss for critic network
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 경우의 mask
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

    state_batch_ = state_batch.detach().clone()

    # update value network
    # Q = r + r'* Q'(s_{t+1}, J(a|s_{t+1}))
    # Loss[y - Q] -> update value network
    value_network.train()
    next_q_values = torch.zeros((batch_size,1), device = device)
    next_q_values[non_final_mask] = target_value_network(non_final_next_states, target_policy_network(non_final_next_states).detach()).detach()
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q_values
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value).detach()
    
    q_values = value_network(state_batch, action_batch)
    value_loss = criterion(q_values, bellman_q_values)

    value_optimizer.zero_grad()
    value_loss.backward()    
    
    # gradient clipping for value_network
    for param in value_network.parameters():
        param.grad.data.clamp_(-1,1) 
        
    value_optimizer.step()

    # update policy network 
    # sum of Q-value -> grad Q(s,a) * grad J(a|s) -> update policy
    value_network.eval()
    policy_loss = value_network(state_batch_, policy_network(state_batch_))
    policy_loss = -policy_loss.mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    # gradient clipping for policy_network
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
    
    policy_optimizer.step()
    
    # target network soft tau update
    for target_param, param in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    for target_param, param in zip(target_policy_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    return value_loss.item(), policy_loss.item()

def train_ddpg(
    env : Enviornment,
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    policy_optimizer : torch.optim.Optimizer,
    value_optimizer : torch.optim.Optimizer,
    value_loss_fn :Optional[nn.Module] = None,
    batch_size : int = 64, 
    search_size : int = 16,
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 10000,  
    verbose : int = 8,
    save_best : Optional[str] = None,
    save_last : Optional[str] = None,
    ):

    value_network.train()
    policy_network.train()

    target_value_network.eval()
    target_policy_network.eval()

    if device is None:
        device = "cpu"

    episode_durations = []
    
    best_reward = 0
    best_episode = 0
    best_action = None
    best_state = None
    reward_list = []
    
    ou_noise = OUNoise(n_action = 7)
    
    for i_episode in tqdm(range(num_episode), desc = 'DDPG algorithm training process'):
    
        if i_episode == 0:
            state = env.init_state
            ctrl = env.init_action
            
        ou_noise.reset()

        for t in range(search_size):
            
            state_tensor = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
        
            policy_network.eval()
            action_tensor = policy_network(state_tensor.to(device)).detach()
            action_tensor = ou_noise.get_action(action_tensor, i_episode)
            action = action_tensor.detach().squeeze(0).cpu().numpy()
            
            ctrl_new = {
                'betan':action[0],
                'k':action[1],
                'epsilon' : action[2],
                'electric_power' : action[3],
                'T_avg' : action[4],
                'B0' : action[5],
                'H' : action[6]
            }
            
            state_new, reward, done, _ = env.step(ctrl_new)
        
            reward_list.append(reward)
            reward = torch.tensor([reward])
            
            next_state_tensor = np.array([state_new[key] for key in state_new.keys()] + [ctrl_new[key] for key in ctrl_new.keys()])
            next_state_tensor = torch.from_numpy(next_state_tensor).unsqueeze(0).float()
            
            # memory에 transition 저장
            memory.push(state_tensor, action_tensor, next_state_tensor, reward, done)

            # update state
            state = state_new
            ctrl = ctrl_new
            
            ou_noise.evolve_state()
            
        # update policy
        value_loss, policy_loss = update_policy(
            memory,
            policy_network,
            value_network,
            target_policy_network,
            target_value_network,
            value_optimizer,
            policy_optimizer,
            value_loss_fn,
            batch_size,
            gamma,
            device,
            min_value,
            max_value,
            tau
        )
                
        if i_episode % verbose == 0:
            print(r"| episode:{} | reward : {} | tau : {:.3f} | beta limit : {} | q limit : {} | n limit {} | f_bs limit : {}".format(
                i_episode+1, env.rewards[-1], env.taus[-1], env.beta_limits[-1], env.q_limits[-1], env.n_limits[-1], env.f_limits[-1]
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
        "f_limit" : env.f_limits
    }
    
    env.close()

    return result