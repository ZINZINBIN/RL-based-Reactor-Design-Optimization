import numpy as np
from src.design.device import Tokamak
from src.config.search_space_info import search_space

class Environment:
    def __init__(self, tokamak: Tokamak, init_state, init_action):
        self.tokamak = tokamak

        # states and actions list
        self.actions = []
        self.states = []
        self.rewards = []

        # optimization status
        self.optim_status = {}

        # Objectives
        self.taus = []      # Energy confinement
        self.costs = []     # Cost parameter
        self.Qs = []        # Q-factor

        # Operational constraints
        self.b_limits = []  # Beta limit
        self.q_limits = []  # Kink instability
        self.n_limits = []  # Greenwald density limit
        self.f_limits = []  # Bootstrap current limit
        
        # Ignition condition
        self.i_limits = [] 

        self.done = False
        self.current_state = None
        self.current_action = None

        self.init_state = init_state
        self.init_action = init_action

    def step(self, action, save:bool=True):
        
        # scaling
        if (action["electric_power"] <= search_space["electric_power"][1] and action["electric_power"] >= search_space["electric_power"][0]):
            action["electric_power"] *= 10**6

        # tokamak variable parameter update
        try:
            self.tokamak.update_design(
                action["betan"],
                action["k"],
                action["epsilon"],
                action["electric_power"],
                action["T_avg"],
                action["B0"],
                action["H"],
                action["armour_thickness"],
                action["RF_recirculating_rate"],
            )

            state = self.tokamak.get_design_performance()

        except:
            state = None
        
        if state is None:
            is_b_limit = 0
            is_q_limit = 0
            is_n_limit = 0
            is_f_limit = 0
            is_i_limit = 0

        else:
            is_b_limit = 1 if state["beta"] / state["beta_troyon"] < 1 else 0
            is_q_limit = 1 if state["q"] / state["q_kink"] > 1 else 0
            is_n_limit = 1 if state["n"] / state["n_g"] < 1 else 0
            is_f_limit = 1 if state["f_NC"] / state["f_BS"] > 1 else 0
            is_i_limit = 1 if state["n_tau"] / state["n_tau_lower"] > 1 else 0

        # update state
        self.current_action = action
        self.current_state = state
        
        if save:
            # save values
            self.actions.append(action)
            self.states.append(state)

            tau = state["tau"] if state is not None else None
            cost = state["cost"] if state is not None else None
            Q = state["Q"] if state is not None else None

            self.taus.append(tau)
            self.costs.append(cost)
            self.Qs.append(Q)

            self.b_limits.append(is_b_limit)
            self.q_limits.append(is_q_limit)
            self.n_limits.append(is_n_limit)
            self.f_limits.append(is_f_limit)
            self.i_limits.append(is_i_limit)
    
        return state

    def reset(self):
        self.current_action = None
        self.current_state = None

    def close(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.taus.clear()
        self.costs.clear()
        self.Qs.clear()
        self.b_limits.clear()
        self.q_limits.clear()
        self.n_limits.clear()
        self.f_limits.clear()
        self.i_limits.clear()

        self.current_action = None
        self.current_state = None