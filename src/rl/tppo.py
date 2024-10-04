# Truly Proximal Policy Optimization
# Enhanced clipping function with Trust region optimization and Roll back function
# github: https://github.com/wisnunugroho21/reinforcement_learning_truly_ppo

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, Dict
from src.env import Enviornment
from config.search_space_info import search_space, state_space
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from collections import namedtuple, deque

from src.rl.ppo import ActorCritic, ReplayBufferPPO, Transition

# update policy
def update_policy(
    memory: ReplayBufferPPO,
    policy_network: ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    gamma: float = 0.99,
    entropy_coeff: float = 0.1,
    device: Optional[str] = "cpu",
    kl_delta:float = 0.025,
    rb_alpha:float = 0.1,
):

    policy_network.train()

    if device is None:
        device = "cpu"

    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction="none")  # Huber Loss for critic network

    transitions = memory.get_trajectory()
    memory.clear()
    batch = Transition(*zip(*transitions))

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    ).to(device)

    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device)  # pi_old

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
    surr = ratio * delta
    kl = kl_divergence(Normal(log_probs.exp().mean(), log_probs.exp().std()), Normal(prob_a_batch.exp().mean(), prob_a_batch.exp().std()))
    
    tr_rb_loss = torch.min(
        torch.where(
            (kl >= kl_delta),
            - rb_alpha * ratio,
            ratio
        ) * delta,
        surr
    )
    
    # tr_rb_loss = torch.where(
    #     (kl >= kl_delta) & (ratio > 1),
    #     surr - entropy,
    #     surr
    # )
    
    loss = (
        (-1) * tr_rb_loss + criterion(value, td_target) - entropy_coeff * entropy
    )
    
    loss = loss.mean()
    loss.backward()

    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)

    policy_optimizer.step()

    return loss


def train_tppo(
    env: Enviornment,
    memory: ReplayBufferPPO,
    policy_network: ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    gamma: float = 0.99,
    kl_delta: float = 0.025,
    rb_alpha: float = 0.1,
    entropy_coeff: float = 0.1,
    device: Optional[str] = "cpu",
    num_episode: int = 10000,
    verbose: int = 8,
    save_best: Optional[str] = None,
    save_last: Optional[str] = None,
):

    if device is None:
        device = "cpu"

    best_reward = 0
    reward_list = []
    loss_list = []

    for i_episode in tqdm(
        range(num_episode), desc="Truly PPO algorithm for design optimization"
    ):

        if env.current_state is None:
            state = env.init_state
            ctrl = env.init_action
        else:
            state = env.current_state
            ctrl = env.current_action

        state_tensor = np.array(
            [state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()]
        )
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).float()

        policy_network.eval()
        action_tensor, entropy, log_probs, value = policy_network.sample(
            state_tensor.to(device)
        )
        action = action_tensor.detach().squeeze(0).cpu().numpy()

        ctrl_new = {
            "betan": action[0],
            "k": action[1],
            "epsilon": action[2],
            "electric_power": action[3],
            "T_avg": action[4],
            "B0": action[5],
            "H": action[6],
            "armour_thickness": action[7],
            "RF_recirculating_rate": action[8],
        }

        state_new, reward, done, _ = env.step(ctrl_new)

        if state_new is None:
            continue

        reward_list.append(reward)
        reward = torch.tensor([reward])

        next_state_tensor = np.array(
            [state_new[key] for key in state_new.keys()]
            + [ctrl_new[key] for key in ctrl_new.keys()]
        )
        next_state_tensor = torch.from_numpy(next_state_tensor).unsqueeze(0).float()

        # memory에 transition 저장
        memory.push(
            state_tensor, action_tensor, next_state_tensor, reward, done, log_probs
        )

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
                entropy_coeff,
                device,
                kl_delta,
                rb_alpha,
            )

            env.current_state = None
            env.current_action = None

            loss_list.append(policy_loss.detach().cpu().numpy())

        if i_episode % verbose == 0:

            print(
                r"| episode:{} | reward : {} | tau : {:.3f} | beta limit : {} | q limit : {} | n limit {} | f_bs limit : {} | ignition : {} | cost : {:.3f}".format(
                    i_episode + 1,
                    env.rewards[-1],
                    env.taus[-1],
                    env.beta_limits[-1],
                    env.q_limits[-1],
                    env.n_limits[-1],
                    env.f_limits[-1],
                    env.i_limits[-1],
                    env.costs[-1],
                )
            )

            env.tokamak.print_info(None)

        # save weights
        torch.save(policy_network.state_dict(), save_last)

        if env.rewards[-1] > best_reward:
            best_reward = env.rewards[-1]
            best_episode = i_episode
            torch.save(policy_network.state_dict(), save_best)

    print("RL training process clear....!")

    result = {
        "control": env.actions,
        "state": env.states,
        "reward": env.rewards,
        "tau": env.taus,
        "beta_limit": env.beta_limits,
        "q_limit": env.q_limits,
        "n_limit": env.n_limits,
        "f_limit": env.f_limits,
        "i_limit": env.i_limits,
        "cost": env.costs,
        "loss": loss_list,
    }

    return result
