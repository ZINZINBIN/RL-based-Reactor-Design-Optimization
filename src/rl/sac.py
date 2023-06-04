import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.rl.buffer import ReplayBuffer, Transition
from src.env import Enviornment
from torch.distributions import Normal

default_action_range = {
    "betan":[2.5, 3.0],
    "k" : [1.5, 2.5],
    "epsilon" : [3.0, 5.0],
    "electric_power" : [1000, 1500],
    "T_avg" : [10, 100],
    "B0" : [13, 16],
    "H" : [1.0, 1.5]
}

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, action_range : Dict = default_action_range, log_std_min : float = -10.0, log_std_max : float = -2.0):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.fc_mean = nn.Linear(mlp_dim // 2, n_actions)
        self.fc_std = nn.Linear(mlp_dim // 2, n_actions)
        
        self.action_range = action_range
        self.min_values = [action_range[key][0] for key in action_range.keys()]
        self.max_values = [action_range[key][1] for key in action_range.keys()]
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, x : torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mean(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, min = self.log_std_min, max = self.log_std_max)
        
        mu = torch.clamp(mu, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))
        return mu, log_std
    
    def sample(self, x : torch.Tensor):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        
        normal_dist = Normal(mu, std)
        xs = normal_dist.rsample()
        action = torch.clamp(xs, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))
        
        log_probs = normal_dist.log_prob(xs) - torch.log(1 - action.pow(2) + 1e-3)
        
        # compute entropy with conserving action dimension
        entropy = -log_probs.sum(dim=1, keepdim = True)
        
        return action, entropy, torch.tanh(mu)
    
# Critic
class QNetwork(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int):
        super(QNetwork, self).__init__()
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
    
class TwinnedQNetwork(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int):
        super(TwinnedQNetwork, self).__init__()
        self.Q1 = QNetwork(input_dim, mlp_dim, n_actions)
        self.Q2 = QNetwork(input_dim, mlp_dim, n_actions)
        
    def forward(self, state : torch.Tensor, action : torch.Tensor):
        
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        
        return q1, q2
    
# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    q_network : nn.Module, 
    target_q_network : nn.Module,
    target_entropy : torch.Tensor,
    log_alpha : torch.Tensor,
    q1_optimizer : torch.optim.Optimizer,
    q2_optimizer : torch.optim.Optimizer,
    policy_optimizer : torch.optim.Optimizer,
    alpha_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2
    ):

    policy_network.train()
    q_network.train()
    
    target_q_network.eval()

    if len(memory) < batch_size:
        return None, None, None

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

    alpha = log_alpha.exp()
    q1, q2 = q_network(state_batch, action_batch)
    
    with torch.no_grad():
        next_action_batch, next_entropy, _ = policy_network.sample(non_final_next_states)
        next_q1, next_q2 = target_q_network(non_final_next_states, next_action_batch)
        
        next_q = torch.zeros((batch_size,1), device = device)
        next_q[non_final_mask] = torch.min(next_q1, next_q2) + alpha.to(device) * next_entropy
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value)
    
    q1_loss = criterion(q1, bellman_q_values)
    q1_optimizer.zero_grad()
    q1_loss.backward()
    
    for param in q_network.Q1.parameters():
        param.grad.data.clamp_(-1,1) 
            
    q1_optimizer.step()
    
    q2_loss = criterion(q2, bellman_q_values)
    q2_optimizer.zero_grad()
    q2_loss.backward()
    
    for param in q_network.Q2.parameters():
        param.grad.data.clamp_(-1,1) 
        
    q2_optimizer.step()
    
    # step 2. update policy weights
    action_batch_sampled, entropy, _ = policy_network.sample(state_batch)
    q1, q2 = q_network(state_batch, action_batch)
    q = torch.min(q1, action_batch_sampled)
    
    policy_loss = torch.mean(q + entropy * alpha.to(device)) * (-1)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
        
    policy_optimizer.step()
    
    # step 3. adjust temperature
    entropy_loss = (-1) * torch.mean(log_alpha.to(device) * (target_entropy.to(device) - entropy).detach())
    alpha_optimizer.zero_grad()
    entropy_loss.backward()
    alpha_optimizer.step()
    
    # step 4. update target network parameter
    for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return q1_loss.item(), q2_loss.item(), policy_loss.item()

def train_sac(
    env : Enviornment,
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    q_network : nn.Module, 
    target_q_network : nn.Module,
    target_entropy : nn.Module,
    log_alpha : torch.Tensor,
    policy_optimizer : torch.optim.Optimizer,
    q1_optimizer : torch.optim.Optimizer,
    q2_optimizer : torch.optim.Optimizer,
    alpha_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
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


    if device is None:
        device = "cpu"

    episode_durations = []
    
    best_reward = 0
    best_episode = 0
    best_action = None
    best_state = None
    reward_list = []
    
    for i_episode in tqdm(range(num_episode), desc = 'SAC algorithm training process'):
    
        if i_episode == 0:
            state = env.init_state
            ctrl = env.init_action
        
        for t in range(search_size):
            
            state_tensor = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()
        
            policy_network.eval()
            action_tensor, _, _ = policy_network.sample(state_tensor.to(device))
            action_tensor = action_tensor.detach()
            action = action_tensor.squeeze(0).cpu().numpy()
            
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
            
        # update policy
        q1_loss, q2_loss, policy_loss = update_policy(
            memory,
            policy_network,
            q_network, 
            target_q_network,
            target_entropy,
            log_alpha,
            q1_optimizer,
            q2_optimizer,
            policy_optimizer,
            alpha_optimizer,
            criterion,
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