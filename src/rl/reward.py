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
        a : float = 3.0,
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
        self.reward_fail = -2.0
        self.a = a
        
        self.valid_beta = [0,10]
        self.valid_q = [0,5]
        self.valid_n = [0,5]
        
    def _compute_tanh(self, x):
        return math.tanh(x)
    
    def _compute_stability_reward(self, x, x_scale, x_min, x_max, a : float = 3.0):
        xn = x / x_scale
        xnl = x_min / x_scale
        xrl = x_max / x_scale
        
        if xn < xrl:
            reward = self._compute_tanh(a * (xn - xnl))
        else:
            reward = self._compute_tanh(a * (2 * xrl - xnl - xn))
            
        return reward
    
    def _compute_performance_reward(self, x, x_scale, x_min, x_max, a : float = 3.0):
        xn = x / x_scale
        xnl = x_min / x_scale
        reward = self._compute_tanh(a * (xn - xnl))
        return reward
        
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
        reward_tau = self._compute_performance_reward(tau, self.tau_r, 0, None, self.a)
        reward_cost = self._compute_performance_reward(1 / cost, 1 / self.cost_r, 0.5 * 1 / self.cost_r, None, self.a)
        
        '''
            quantitive + conditional reward
            (1) range: [v_min, v_max] (physically valid)
            (2) stability / criteria : (ex) q > q_kink
            (3) quantitive: good to be larger
        '''
        
        # stability
        # Troyon beta limit
        reward_beta = self._compute_stability_reward(beta, self.beta_r, 0.5 * beta_max, beta_max, self.a)
        
        reward_beta += self.reward_fail * (
             np.heaviside(beta - self.valid_beta[1], 0) + np.heaviside(self.valid_beta[0] - beta, 0)
        )
        
        # kink instability
        reward_q = self._compute_stability_reward(1 / q, 1 / self.q_r, 0.5 * 1 / q_kink, 1 / q_kink, self.a)
        
        reward_q += self.reward_fail * (
             np.heaviside(q - self.valid_q[1], 0) + np.heaviside(self.valid_q[0] - q, 0)
        )
        
        # Greenwald density   
        reward_n = self._compute_stability_reward(n, self.n_r, 0.5 * n_g, n_g, self.a)
        
        reward_n += self.reward_fail * (
             np.heaviside(n - self.valid_n[1], 0) + np.heaviside(self.valid_n[0] - n, 0)
        )
        
        # achievable bootstrap fraction for steady-state operation
        reward_f = self._compute_performance_reward(f_NC / f_BS, self.f_r, 0.5, None, self.a)
        
        # ignition condition
        reward_i = self._compute_performance_reward(n_tau, self.i_r, n_tau_lower, None, self.a)
        
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
        
        # quantitive metrics
        reward_tau = self._compute_performance_reward(tau, self.tau_r, 0, None, self.a)
        reward_cost = self._compute_performance_reward(1 / cost, 1 / self.cost_r, 0.5 * 1 / self.cost_r, None, self.a)
                
        # stability
        # Troyon beta limit
        reward_beta = self._compute_stability_reward(beta, self.beta_r, 0.5 * beta_max, beta_max, self.a)
        
        reward_beta += self.reward_fail * (
             np.heaviside(beta - self.valid_beta[1], 0) + np.heaviside(self.valid_beta[0] - beta, 0)
        )
        
        # kink instability
        reward_q = self._compute_stability_reward(1 / q, 1 / self.q_r, 0.5 * 1 / q_kink, 1 / q_kink, self.a)
        
        reward_q += self.reward_fail * (
             np.heaviside(q - self.valid_q[1], 0) + np.heaviside(self.valid_q[0] - q, 0)
        )
        
        # Greenwald density   
        reward_n = self._compute_stability_reward(n, self.n_r, 0.5 * n_g, n_g, self.a)
        
        reward_n += self.reward_fail * (
             np.heaviside(n - self.valid_n[1], 0) + np.heaviside(self.valid_n[0] - n, 0)
        )
        
        # achievable bootstrap fraction for steady-state operation
        reward_f = self._compute_performance_reward(f_NC / f_BS, self.f_r, 0.5, None, self.a)
        
        # ignition condition
        reward_i = self._compute_performance_reward(n_tau, self.i_r, n_tau_lower, None, self.a)
        
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