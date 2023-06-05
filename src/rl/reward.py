from typing import Dict
import math
import numpy as np

class RewardSender:
    def __init__(
        self,
        w_tau : float, 
        w_beta : float, 
        w_density : float, 
        w_q : float, 
        w_bs : float,
        w_i : float,
        tau_r : float = 1.0,
        beta_r : float = 1.0,
        q_r : float = 1.0,
        n_r : float = 1.0,
        f_r : float = 1.0,
        i_r : float = 1.0,
        ):
        self.w_tau = w_tau
        self.w_beta = w_beta
        self.w_density = w_density
        self.w_q = w_q
        self.w_bs = w_bs
        self.w_i = w_i
        
        self.tau_r = tau_r
        self.beta_r = beta_r
        self.q_r = q_r
        self.n_r = n_r
        self.f_r = f_r
        self.i_r = i_r
        
    def _compute_sigmoid(self, x):
        return math.exp(x) / (math.exp(x)+1)
        
    def _compute_reward(self, state:Dict):
        
        tau = state['tau']
        beta = state['beta']
        beta_max = state['beta_troyon']
        n = state['n']
        n_g = state['n_g']
        q = state['q']
        q_kink = state['q_kink']
        f_NC = state['f_NC']
        f_BS = state['f_BS']
        n_tau = state['n_tau']
        n_tau_lower = state['n_tau_lower']
        
        reward_tau = self._compute_sigmoid(tau / self.tau_r)
        reward_beta = np.heaviside(1-beta/beta_max, 0.5) * self._compute_sigmoid(beta / self.beta_r)
        reward_q = np.heaviside(q/q_kink - 1, 0.5) * self._compute_sigmoid(q / self.q_r)
        reward_n = np.heaviside(1-n/n_g, 0.5) * self._compute_sigmoid(n/self.n_r)
        reward_f = np.heaviside(f_NC / f_BS - 1, 0.5) * self._compute_sigmoid(f_NC / f_BS / self.f_r)
        reward_i = np.heaviside(n_tau / n_tau_lower - 1, 0.5) * self._compute_sigmoid(n_tau / n_tau_lower / self.i_r)
        
        reward = reward_tau * self.w_tau + reward_beta * self.w_beta + reward_q * self.w_q + reward_n * self.w_density + reward_f * self.w_bs + reward_i * self.w_i
        return reward
        
    def __call__(self, state : Dict):
        reward = self._compute_reward(state)
        return reward