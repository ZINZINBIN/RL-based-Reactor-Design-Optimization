import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Dict, Callable, List

from src.design.env import Enviornment
from src.config.search_space_info import search_space, state_space

from collections import namedtuple, deque
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done','prob_a')
)

default_action_range = search_space
default_state_range = state_space

class ReplayBuffer:
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def get_trajectory(self):
        traj = [self.memory[idx] for idx in range(len(self.memory))]
        return traj
    
    def clear(self):
        self.memory.clear()

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, action_range : Dict = default_action_range, state_range : Dict = default_state_range):
        super(GaussianPolicy, self).__init__()

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_mu = nn.Linear(mlp_dim // 2, n_actions)
        self.fc_log_std = nn.Linear(mlp_dim // 2, n_actions)

        self.action_range = action_range
        self.state_range = state_range
        self.min_values = [action_range[key][0] for key in action_range.keys()]
        self.max_values = [action_range[key][1] for key in action_range.keys()]

        self.input_min_values = [state_range[key][0] for key in state_range.keys()] 
        self.input_max_values = [state_range[key][1] for key in state_range.keys()] 

    def forward(self, x : torch.Tensor):

        # normalization
        x = (x - torch.Tensor(self.input_min_values).to(x.device)) / (torch.Tensor(self.input_max_values).to(x.device) - torch.Tensor(self.input_min_values).to(x.device))
        x -= 0.5
        x *= 2

        x = F.tanh(self.fc1(self.norm1(x)))
        x = F.tanh(self.fc2(self.norm2(x)))
        x = F.tanh(self.fc3(self.norm3(x)))

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist

    def sample(self, x : torch.Tensor):
        dist = self.forward(x)
        xs = dist.rsample()
        
        ys = torch.tanh(xs)

        # rescale action range
        action = (0.5 + 0.5 * xs) * (torch.Tensor(self.max_values).to(x.device) - torch.Tensor(self.min_values).to(x.device)) + torch.Tensor(self.min_values).to(x.device)

        # action bounded for stable learning + valid design parameter
        action = torch.clamp(action, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))

        log_probs = dist.log_prob(xs) - torch.log((1-ys.pow(2)) + 1e-6)
        entropy = dist.entropy().mean()

        return action, entropy, log_probs

    def suggest(self, state:Dict, device:str):
        state_tensor = torch.from_numpy(self._dict2arr(state, state_space.keys())).unsqueeze(0).float()
        action_tensor, _, _, _ = self.sample(state_tensor.to(device))
        action = action_tensor.detach().squeeze(0).cpu().numpy()
        ctrl = self._arr2dict(action, search_space.keys())
        return ctrl

    def _dict2arr(self, x: Dict, keys: List):
        y = []

        for key in keys:
            v = x[key]

            if type(v) == np.array:
                y.append(v.item())
            else:
                y.append(v)

        return np.array(y)

    def _arr2dict(self, x: np.array, keys: List):

        y = {}
        for idx, key in enumerate(keys):
            v = x[idx]
            y[key] = v

        return y

class QNetwork(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, state_range : Dict = default_state_range):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_v = nn.Linear(mlp_dim // 2, 1)

        self.state_range = state_range

        self.input_min_values = [state_range[key][0] for key in state_range.keys()] 
        self.input_max_values = [state_range[key][1] for key in state_range.keys()] 

    def forward(self, x : torch.Tensor):

        # normalization
        x = (x - torch.Tensor(self.input_min_values).to(x.device)) / (torch.Tensor(self.input_max_values).to(x.device) - torch.Tensor(self.input_min_values).to(x.device))
        x -= 0.5
        x *= 2

        x = F.tanh(self.fc1(self.norm1(x)))
        x = F.tanh(self.fc2(self.norm2(x)))
        x = F.tanh(self.fc3(self.norm3(x)))

        v = self.fc_v(x)
        return v

    def _dict2arr(self, x: Dict, keys: List):
        y = []

        for key in keys:
            v = x[key]

            if type(v) == np.array:
                y.append(v.item())
            else:
                y.append(v)

        return np.array(y)

    def _arr2dict(self, x: np.array, keys: List):

        y = {}
        for idx, key in enumerate(keys):
            v = x[idx]
            y[key] = v

        return y

# update policy
def update_policy(
    memory: ReplayBuffer,
    p_network: GaussianPolicy,
    p_optimizer: torch.optim.Optimizer,
    q1_network: QNetwork,
    q1_optimizer: torch.optim.Optimizer,
    q2_network: QNetwork,
    q2_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    alpha:float = 0.1,
    gamma: float = 0.99,
    device: Optional[str] = "cpu",
):

    if device is None:
        device = "cpu"

    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = 'none') # Huber Loss for critic network

    transitions = memory.get_trajectory()
    memory.clear()

    batch = Transition(*zip(*transitions))

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).float().to(device)
    reward_batch = torch.cat(batch.reward).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device)

    with torch.no_grad():
        next_action, _, next_log_prob = p_network.sample(non_final_next_states)
        q1 = q1_network(non_final_next_states)
        q2 = q2_network(non_final_next_states)
        q = torch.min(q1, q2) - alpha * next_log_prob
        q_target = reward_batch +  gamma * q

    q1_loss = torch.mean(F.mse_loss(q1_network(state_batch) - q_target))
    q2_loss = torch.mean(F.mse_loss(q2_network(state_batch) - q_target))

    q1_optimizer.zero_grad()
    q1_loss.backward()
    
    for param in q1_network.parameters():
        param.grad.data.clamp_(-1,1)
    
    q1_optimizer.step()

    q2_optimizer.zero_grad()
    q2_loss.backward()
    
    for param in q2_network.parameters():
        param.grad.data.clamp_(-1,1)
        
    q2_optimizer.step()

    p_network.train()

    action, entropy, log_prob = p_network.sample(state_batch)
    q1 = q1_network(state_batch)
    q2 = q2_network(state_batch)    

    p_loss = (-1) * torch.mean(q1 + entropy * alpha)
    
    p_optimizer.zero_grad()
    p_loss.backward()
    
    for param in p_network.parameters():
        param.grad.data.clamp_(-1,1)
        
    p_optimizer.step()
    
    return q1_loss.item(), q2_loss.item(), p_loss.item()


# Evaluation of the design performance
def evaluate_single_process(env:Enviornment, ctrl:Dict, objective:Callable, constraint:Callable):
    state = env.step(ctrl)
    return objective(state), constraint(state), state

# batch-evaluation of the design performance for multi-core process
def evaluate_batch(env:Enviornment, ctrl_batch:List, objective:Callable, constraint:Callable):
    return [evaluate_single_process(env, ctrl, objective, constraint) for ctrl in ctrl_batch]

def search_param_space(
    env: Enviornment,
    objective: Callable,
    constraint: Callable,
    memory: ReplayBuffer,
    p_network: GaussianPolicy,
    p_optimizer: torch.optim.Optimizer,
    q1_network: QNetwork,
    q1_optimizer: torch.optim.Optimizer,
    q2_network: QNetwork,
    q2_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    alpha:float = 0.1,
    gamma: float = 0.99,
    device: Optional[str] = "cpu",
    save_best: Optional[str] = None,
    save_last: Optional[str] = None,
    num_episode: int = 10000,
    verbose: int = 100,
    n_grid: int = 10,
    n_proc: int = -1,
    sample_size: int = 1024,
):

    if n_proc == -1:
        n_proc = cpu_count()

    if device is None:
        device = "cpu"

    # Step 1. Initial samples generation for offline learning with PPO agent (Random sampling)
    init_params = {}

    for param in search_space.keys():

        p_min = search_space[param][0]
        p_max = search_space[param][1]
        ps = np.linspace(p_min, p_max, n_grid)
        indice = np.random.randint(0, n_grid, sample_size)
        init_params[param] = ps[indice]

    ctrls = []

    for idx in range(sample_size):
        ctrl = {}
        for key in search_space.keys():
            ctrl[key] = init_params[key][idx]
        ctrls.append(ctrl)

    batch_size = sample_size // n_proc
    batch_ctrls = [ctrls[i : i + batch_size] for i in range(0, sample_size, batch_size)]

    evals_batches = Parallel(n_jobs=n_proc)(delayed(evaluate_batch)(env, batch, objective, constraint) for batch in batch_ctrls)

    evals = [item for sublist in evals_batches for item in sublist]
    fs, gs, states = zip(*evals)

    fs = np.array(fs)
    gs = np.array(gs)

    # Reward-weighted regression for offline training of the policy network
    reward = torch.from_numpy(fs).reshape(-1).to(device)

    init_state = p_network._dict2arr(env.init_state, state_space.keys())
    init_state_tensor = torch.from_numpy(init_state).unsqueeze(0).float().repeat(len(reward),1)

    action_tensor = [torch.from_numpy(p_network._dict2arr(ctrl, search_space.keys())).unsqueeze(0).float() for ctrl in ctrls]
    action_tensor = torch.concatenate(action_tensor, dim = 0).to(device)

    pred_action_tensor, _, _, _ = p_network.sample(init_state_tensor.to(device))

    l2 = torch.sum((action_tensor - pred_action_tensor) ** 2, dim = 1)
    weighted_l2 = torch.sum(reward * l2) * (-1)

    p_optimizer.zero_grad()
    weighted_l2.backward()
    p_optimizer.step()

    # Step 2. Online training step
    best_reward = 0
    reward_list = []
    loss_list = []

    for i_episode in tqdm(range(num_episode)):

        state = env.step(ctrl)
        f, g = objective(state), constraint(state)

        state_tensor = p_network._dict2arr(state, state_space.keys())
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()

        p_network.eval()
        action_tensor, _, log_probs, _ = p_network.sample(state_tensor.to(device))
        action = action_tensor.detach().squeeze(0).cpu().numpy()

        ctrl_new = p_network._arr2dict(action, search_space.keys())
        state_new = env.step(ctrl_new)

        if state_new is None:
            continue

        reward_list.append(f)
        reward = torch.tensor([f])

        next_state_tensor = p_network._dict2arr(state_new, state_space.keys())
        next_state_tensor = torch.from_numpy(next_state_tensor).unsqueeze(0).float()

        # save the component into trajectory
        memory.push(state_tensor, action_tensor, next_state_tensor, reward, False, log_probs)

        # update state
        env.current_state = state_new
        env.current_action = ctrl_new

        # update policy
        if memory.__len__() >= memory.capacity:
            policy_loss = update_policy(
                memory,
                p_network,
                p_optimizer,
                q1_network,
                q1_optimizer,
                q2_network,
                q2_optimizer,
                criterion,
                alpha,
                gamma,
                device,
            )

            env.current_state = None
            env.current_action = None

            loss_list.append(policy_loss.detach().cpu().numpy())

        if i_episode % verbose == 0:
            print(
                r"| episode:{} | cost : {:.3f} | tau : {:.3f} | b limit : {} | q limit : {} | n limit {} | f limit : {} | i limit : {} ".format(
                    i_episode + 1,
                    env.costs[-1],
                    env.taus[-1],
                    env.b_limits[-1],
                    env.q_limits[-1],
                    env.n_limits[-1],
                    env.f_limits[-1],
                    env.i_limits[-1],
                )
            )

        # save weights
        if save_last is not None:
            torch.save(p_network.state_dict(), save_last)

        if reward_list[-1] > best_reward:
            best_reward = reward_list[-1]

            if save_best is not None:
                torch.save(p_network.state_dict(), save_best)

    result = {
        "control": env.actions,
        "state": env.states,
        "tau": env.taus,
        "b_limit": env.b_limits,
        "q_limit": env.q_limits,
        "n_limit": env.n_limits,
        "f_limit": env.f_limits,
        "i_limit": env.i_limits,
        "cost": env.costs,
    }

    return result
