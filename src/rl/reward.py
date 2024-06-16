from typing import Dict, Literal
import math
import numpy as np


class RewardSender:
    def __init__(
        self,
        w_cost : float,
        w_tau : float, 
        w_beta : float, 
        w_density : float, 
        w_q : float, 
        w_bs : float,
        w_i : float,
        cost_r : float = 1.0,
        tau_r : float = 1.0,
        beta_r : float = 1.0,
        q_r : float = 1.0,
        n_r : float = 1.0,
        f_r : float = 1.0,
        i_r : float = 1.0,
        ):
        self.w_cost = w_cost
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
        self.cost_r = cost_r
        
        # Reward engineering
        # If the designed device has inplausible state, the reward should handle this case
        self.reward_fail = -5.0
        
        self.valid_beta = [0,10]
        self.valid_q = [0,5]
        self.valid_n = [0,5]
        
    def _compute_sigmoid(self, x):
        return math.exp(x) / (math.exp(x)+1)
    
    def _compute_stability_reward(self, x, scale, method : Literal['soft', 'hard'] = 'soft'):
        
        reward = self._compute_sigmoid(x / scale)
        
        # case 1 : hard constraint
        # reward_beta *= np.heaviside(1-beta/beta_max, 0)
        
        # case 2 : soft constraint
        reward *= self._compute_sigmoid(4.0 * (1-x))
        return None
    
    def _compute_performance_reward(self, x):
        return None
        
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
        cost = state['cost']
        
        # quantitive metrics
        reward_tau = self._compute_sigmoid(tau / self.tau_r)
        reward_cost = self._compute_sigmoid(self.cost_r / cost)
        
        # quantitive + conditional reward
        # (1) range: [v_min, v_max] (physically valid)
        # (2) stability / criteria : (ex) q > q_kink
        # (3) quantitive: good to be larger
        
        # stability
        # Troyon beta limit
        reward_beta = self._compute_sigmoid(beta / beta_max / self.beta_r)
        
        # case 1 : hard constraint
        # reward_beta *= np.heaviside(1-beta/beta_max, 0)
        
        # case 2 : soft constraint
        reward_beta *= self._compute_sigmoid(4.0 * (1-beta/beta_max))
        
        reward_beta += self.reward_fail * (
             np.heaviside(beta - self.valid_beta[1], 0) + np.heaviside(self.valid_beta[0] - beta, 0)
        )
        
        # kink instability
        reward_q =  self._compute_sigmoid(q / q_kink / self.q_r)
        
        # case 1 : hard constraint
        # reward_q *= np.heaviside(q/q_kink - 1, 0)
        
        # case 2 : soft constraint
        reward_q *= self._compute_sigmoid(4.0 * (q/q_kink - 1))
        
        reward_q += self.reward_fail * (
             np.heaviside(q - self.valid_q[1], 0) + np.heaviside(self.valid_q[0] - q, 0)
        )
        
        # Greenwald density
        reward_n = self._compute_sigmoid(n/n_g/self.n_r)
        
        # case 1 : hard constraint
        # reward_n *= np.heaviside(1-n/n_g, 0)
        
        # case 2 : soft constraint
        reward_n *= self._compute_sigmoid(4.0 * (1-n/n_g))
        
        reward_n += self.reward_fail * (
             np.heaviside(n - self.valid_n[1], 0) + np.heaviside(self.valid_n[0] - n, 0)
        )
        
        # achievable bootstrap fraction for steady-state operation
        # case 1: f_nc must be larger than f_bs (strong constraint)
        # reward_f = np.heaviside(f_NC / f_BS - 1, 0.5) * self._compute_sigmoid(f_NC / f_BS / self.f_r)
        
        # case 2: soft constraint
        reward_f = self._compute_sigmoid(f_NC / f_BS / self.f_r)
        
        # ignition condition
        # case 1: n_tau must be larger than criteria
        # reward_i = np.heaviside(n_tau / n_tau_lower - 1, 0.5) * self._compute_sigmoid(n_tau / n_tau_lower / self.i_r)
        
        # case 2: firstly, it is good as larger than 1
        reward_i = self._compute_sigmoid(n_tau / n_tau_lower / self.i_r)
        
        reward = reward_tau * self.w_tau + reward_cost * self.w_cost + reward_beta * self.w_beta + reward_q * self.w_q + reward_n * self.w_density + reward_f * self.w_bs + reward_i * self.w_i
        reward /= (self.w_tau + self.w_cost + self.w_beta + self.w_q + self.w_density + self.w_bs + self.w_i)
        
        return reward
    
    def _compute_reward_dict(self, state:Dict):
        
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
        cost = state['cost']
        
        reward_tau = self._compute_sigmoid(tau / self.tau_r)
        reward_cost = self._compute_sigmoid(self.cost_r / cost)
        
        # stability
        # Troyon beta limit
        reward_beta =  self._compute_sigmoid(beta / beta_max / self.beta_r)
        
        # case 1 : hard constraint
        # reward_beta *= np.heaviside(1-beta/beta_max, 0)
        
        # case 2 : soft constraint
        reward_beta *= self._compute_sigmoid(4.0 * (1-beta/beta_max))
        
        reward_beta += self.reward_fail * (
             np.heaviside(beta - self.valid_beta[1], 0) + np.heaviside(self.valid_beta[0] - beta, 0)
        )
        
        # kink instability
        reward_q =  self._compute_sigmoid(q / q_kink / self.q_r)
        
        # case 1 : hard constraint
        # reward_q *= np.heaviside(q/q_kink - 1, 0)
        
        # case 2 : soft constraint
        reward_q *= self._compute_sigmoid(4.0 * (q/q_kink - 1))
        
        reward_q += self.reward_fail * (
             np.heaviside(q - self.valid_q[1], 0) + np.heaviside(self.valid_q[0] - q, 0)
        )
        
        # Greenwald density
        reward_n = self._compute_sigmoid(n/n_g/self.n_r)
        
        # case 1 : hard constraint
        # reward_n *= np.heaviside(1-n/n_g, 0)
        
        # case 2 : soft constraint
        reward_n *= self._compute_sigmoid(4.0 * (1-n/n_g))
        
        reward_n += self.reward_fail * (
             np.heaviside(n - self.valid_n[1], 0) + np.heaviside(self.valid_n[0] - n, 0)
        )
        
        # achievable bootstrap fraction for steady-state operation
        # case 1: f_nc must be larger than f_bs (strong constraint)
        # reward_f = np.heaviside(f_NC / f_BS - 1, 0.5) * self._compute_sigmoid(f_NC / f_BS / self.f_r)
        
        # case 2: soft constraint
        reward_f = self._compute_sigmoid(f_NC / f_BS / self.f_r)
        
        # ignition condition
        # case 1: n_tau must be larger than criteria
        # reward_i = np.heaviside(n_tau / n_tau_lower - 1, 0.5) * self._compute_sigmoid(n_tau / n_tau_lower / self.i_r)
        
        # case 2: firstly, it is good as larger than 1
        reward_i = self._compute_sigmoid(n_tau / n_tau_lower / self.i_r)
        
        reward = reward_tau * self.w_tau + reward_cost * self.w_cost + reward_beta * self.w_beta + reward_q * self.w_q + reward_n * self.w_density + reward_f * self.w_bs + reward_i * self.w_i
        reward /= (self.w_tau + self.w_cost + self.w_beta + self.w_q + self.w_density + self.w_bs + self.w_i)
        
        reward_dict = {
            "total":reward,
            "tau":reward_tau,
            "cost":reward_cost,
            "beta":reward_beta,
            "q":reward_q,
            "density":reward_n,
            "fbs":reward_f,
        }
        
        return reward_dict
        
    def __call__(self, state : Dict):
        reward = self._compute_reward(state)
        return reward