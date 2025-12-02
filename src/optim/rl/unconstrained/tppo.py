import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

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

class ActorCritic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, action_range : Dict = default_action_range, std : float = 0.0, state_range : Dict = default_state_range):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_pi = nn.Linear(mlp_dim // 2, n_actions)
        self.fc_v = nn.Linear(mlp_dim // 2, 1)

        self.action_range = action_range
        self.state_range = state_range
        self.min_values = [action_range[key][0] for key in action_range.keys()]
        self.max_values = [action_range[key][1] for key in action_range.keys()]

        self.input_min_values = [state_range[key][0] for key in state_range.keys()] # + [action_range[key][0] for key in action_range.keys()]
        self.input_max_values = [state_range[key][1] for key in state_range.keys()] # + [action_range[key][1] for key in action_range.keys()]

        self.log_std = nn.Parameter(torch.ones(1, n_actions) * std)

    def forward(self, x : torch.Tensor):

        # normalization
        x = (x - torch.Tensor(self.input_min_values).to(x.device)) / (torch.Tensor(self.input_max_values).to(x.device) - torch.Tensor(self.input_min_values).to(x.device))
        x -= 0.5
        x *= 2

        x = F.tanh(self.fc1(self.norm1(x)))
        x = F.tanh(self.fc2(self.norm2(x)))
        x = F.tanh(self.fc3(self.norm3(x)))

        mu = self.fc_pi(x)
        std = self.log_std.exp().expand_as(mu)

        dist = Normal(mu, std)
        value = self.fc_v(x)

        return dist, value

    def sample(self, x : torch.Tensor):
        dist, value = self.forward(x)
        xs = dist.rsample()

        # rescale action range
        action = (0.5 + 0.5 * xs) * (torch.Tensor(self.max_values).to(x.device) - torch.Tensor(self.min_values).to(x.device)) + torch.Tensor(self.min_values).to(x.device)

        # action bounded for stable learning + valid design parameter
        action = torch.clamp(action, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))

        log_probs = dist.log_prob(xs)
        entropy = dist.entropy().mean()

        return action, entropy, log_probs, value
    
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


# update policy
def update_policy(
    memory: ReplayBuffer,
    policy_network: ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    gamma: float = 0.99,
    eps_clip: float = 0.1,
    entropy_coeff: float = 0.1,
    device: Optional[str] = "cpu",
    kl_delta: float = 0.025,
    rb_alpha: float = 0.1,
):

    policy_network.train()

    if device is None:
        device = "cpu"

    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = 'none') # Huber Loss for critic network

    transitions = memory.get_trajectory()
    memory.clear()
    batch = Transition(*zip(*transitions))

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device) # pi_old

    # Multi-step version reward: Monte Carlo estimate
    rewards = []
    discounted_reward = 0
    for reward in reversed(batch.reward):
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)

    reward_batch = torch.cat(rewards).float().to(device)

    policy_optimizer.zero_grad()

    _, _, next_log_probs, next_value = policy_network.sample(non_final_next_states)
    action, entropy, log_probs, value = policy_network.sample(state_batch)

    td_target = reward_batch.view_as(next_value) + gamma * next_value

    delta = td_target - value        
    ratio = torch.exp(log_probs - prob_a_batch.detach())

    # TP-RB loss
    surr1 = ratio * delta
    kl = kl_divergence(
        Normal(log_probs.exp().mean(), log_probs.exp().std()),
        Normal(prob_a_batch.detach().exp().mean(), prob_a_batch.detach().exp().std()),
    )

    surr2 = torch.where((kl >= kl_delta), -rb_alpha * ratio, ratio) * delta
    loss = (-1) * torch.min(surr1, surr2) + criterion(value, td_target) - entropy_coeff * entropy

    loss = loss.mean()
    loss.backward()

    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1)

    policy_optimizer.step()

    return loss


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
    policy_network: ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    gamma: float = 0.99,
    eps_clip: float = 0.1,
    entropy_coeff: float = 0.1,
    device: Optional[str] = "cpu",
    save_best: Optional[str] = None,
    save_last: Optional[str] = None,
    num_episode: int = 10000,
    verbose: int = 100,
    n_grid: int = 10,
    n_proc: int = -1,
    sample_size: int = 1024,
    kl_delta: float = 0.025,
    rb_alpha: float = 0.1,
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

    init_state = policy_network._dict2arr(env.init_state, state_space.keys())
    init_state_tensor = torch.from_numpy(init_state).unsqueeze(0).float().repeat(len(reward),1)

    action_tensor = [torch.from_numpy(policy_network._dict2arr(ctrl, search_space.keys())).unsqueeze(0).float() for ctrl in ctrls]
    action_tensor = torch.concatenate(action_tensor, dim = 0).to(device)

    pred_action_tensor, _, _, _ = policy_network.sample(init_state_tensor.to(device))

    l2 = torch.sum((action_tensor - pred_action_tensor) ** 2, dim = 1)
    weighted_l2 = torch.sum(reward * l2) * (-1)

    policy_optimizer.zero_grad()
    weighted_l2.backward()
    policy_optimizer.step()

    # Step 2. Online training step
    best_reward = 0
    reward_list = []
    loss_list = []

    for i_episode in tqdm(range(num_episode)):

        state = env.step(ctrl)
        f, g = objective(state), constraint(state)

        state_tensor = policy_network._dict2arr(state, state_space.keys())
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()

        policy_network.eval()
        action_tensor, _, log_probs, _ = policy_network.sample(state_tensor.to(device))
        action = action_tensor.detach().squeeze(0).cpu().numpy()

        ctrl_new = policy_network._arr2dict(action, search_space.keys())
        state_new = env.step(ctrl_new)

        if state_new is None:
            continue

        reward_list.append(f)
        reward = torch.tensor([f])

        next_state_tensor = policy_network._dict2arr(state_new, state_space.keys())
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
                policy_network, 
                policy_optimizer,
                criterion,
                gamma, 
                eps_clip,
                entropy_coeff,
                device,
                kl_delta,
                rb_alpha
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
            torch.save(policy_network.state_dict(), save_last)

        if reward_list[-1] > best_reward:
            best_reward = reward_list[-1]

            if save_best is not None:
                torch.save(policy_network.state_dict(), save_best)

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
