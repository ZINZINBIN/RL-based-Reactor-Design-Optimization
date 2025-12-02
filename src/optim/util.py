from typing import List, Dict, Optional
import numpy as np

def objective(state:Optional[Dict], w:List = [1.0, 1.0, 1.0], lamda:List = [0.25, 0.25, 0.25, 0.25, 0.25], discretized:bool = True):
        
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    if state is None:
        return 0

    if type(lamda) == List:
        lamda = np.array(lamda)

    if type(w) == List:
        w = np.array(w)
    
    if np.sum(lamda) > 0:
        lamda /= np.sum(lamda)
    
    if np.sum(w) > 0:
        w /= np.sum(w)

    tau = state["tau"]
    cost = state["cost"]
    Q = state['Q']

    tau_ref = 0.944
    cost_ref = 1.003
    Q_ref = 10.38
    
    score = np.array([tau / tau_ref, cost_ref / cost, Q / Q_ref])
    score = sigmoid(score)
    
    feasibility = constraint(state, discretized=discretized)
    total = np.inner(score, w) + np.inner(feasibility, lamda)

    return total

def constraint(state:Optional[Dict], discretized:bool = True):
    # constraint: C(x) < 1
    if state is None:
        b_ratio = 1
        q_ratio = 1
        n_ratio = 1
        f_ratio = 1
        i_ratio = 1

    else:
        b_ratio = state["beta"] / state["beta_troyon"]
        q_ratio = state["q_kink"] / state["q"]
        n_ratio = state["n"] / state["n_g"]
        f_ratio = state["f_BS"] / state["f_NC"]
        i_ratio = state["n_tau_lower"] / state["n_tau"]

    if discretized:
        is_b_limit = 1 if b_ratio < 1 else 0
        is_q_limit = 1 if q_ratio < 1 else 0
        is_n_limit = 1 if n_ratio < 1 else 0
        is_f_limit = 1 if f_ratio < 1 else 0
        is_i_limit = 1 if i_ratio < 1 else 0
        return np.array([is_b_limit, is_q_limit, is_n_limit, is_f_limit, is_i_limit])

    else:
        return np.array([b_ratio, q_ratio, n_ratio, f_ratio, i_ratio])
