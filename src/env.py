import gym
import numpy as np
from src.device import Tokamak
from src.rl.reward import RewardSender

class Enviornment(gym.Env):
    def __init__(self, tokamak : Tokamak, reward_sender : RewardSender, init_state, init_action):
        self.tokamak = tokamak
        self.reward_sender = reward_sender
        
        # states and actions list
        self.actions = []
        self.states = []
        self.rewards = []
        
        self.taus = []
        self.beta_limits = []
        self.q_limits = []
        self.n_limits = []
        self.f_limits = []
        self.i_limits = []
        
        self.done = False
        self.current_state = None
        self.current_action = None
        
        self.init_state = init_state
        self.init_action = init_action
    
    def step(self, action):
        
        # scaling
        action['electric_power'] *= 10 ** 6
        
        # tokamak variable parameter update
        try:
            self.tokamak.update_design(action['betan'], action['k'], action['epsilon'], action['electric_power'], action['T_avg'], action['B0'], action['H'])
            state = self.tokamak.get_design_performance()
            reward = self.reward_sender(state)
            
        except:
            state = None
            reward = None
            
        if state is None or reward is None:
            return None, None, None, None
            
        # update state
        self.current_action = action
        self.current_state = state
        
        # save values
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        
        is_beta_limit = state['beta'] / state['beta_troyon'] < 1
        is_q_limit = state['q'] / state['q_kink'] > 1
        is_n_limit = state['n'] / state['n_g'] < 1
        is_f_limit = state['f_NC'] / state['f_BS'] > 1
        is_i_limit = state['n_tau'] / state['n_tau_lower'] > 1
        
        self.taus.append(state['tau'])
        self.beta_limits.append(is_beta_limit)
        self.q_limits.append(is_q_limit)
        self.n_limits.append(is_n_limit)
        self.f_limits.append(is_f_limit)
        self.i_limits.append(is_i_limit)
        
        return state, reward, False, {}
    
    def reset(self):
        self.current_action = None
        self.current_state = None
    
    def close(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.taus.clear()
        self.beta_limits.clear()
        self.q_limits.clear()
        self.n_limits.clear()
        self.f_limits.clear()
        self.i_limits.clear()
        
        self.current_action = None
        self.current_state = None