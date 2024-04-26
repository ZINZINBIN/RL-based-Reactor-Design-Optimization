import numpy as np
import random
from itertools import count, product
from tqdm.auto import tqdm
from typing import Optional, Dict
from src.env import Enviornment
from config.search_space_info import search_space

def search_param_space(
    env : Enviornment,
    num_episode : int = 10000,  
    verbose : int = 8,
    n_grid : int = 32
):

    param_space = {}
    
    for param in search_space.keys():
        
        p_min = search_space[param][0]
        p_max = search_space[param][1]
        
        ps = np.linspace(p_min, p_max, n_grid)
        random.shuffle(ps)
        
        param_space[param] = ps
        
    for i_episode, values in enumerate(product(*param_space.values())):
        
        ctrl = dict(zip(param_space.keys(), values))
        
        state, reward, done, _ = env.step(ctrl)
        
        if state is None:
            continue
            
        env.current_state = state
        env.current_action = ctrl
                
        if i_episode % verbose == 0:
            print(r"| episode:{} | reward : {} | tau : {:.3f} | beta limit : {} | q limit : {} | n limit {} | f_bs limit : {} | ignition : {} | cost : {:.3f}".format(
                i_episode+1, env.rewards[-1], env.taus[-1], env.beta_limits[-1], env.q_limits[-1], env.n_limits[-1], env.f_limits[-1], env.i_limits[-1], env.costs[-1]
            ))
            env.tokamak.print_info(None)
            
        if i_episode >= num_episode:
            break

    print("Grid search process clear....!")
    
    result = {
        "control":env.actions,
        "state":env.states,
        "reward":env.rewards,
        "tau":env.taus,
        "beta_limit" : env.beta_limits,
        "q_limit" : env.q_limits,
        "n_limit" : env.n_limits,
        "f_limit" : env.f_limits,
        "i_limit" : env.i_limits,
        "cost" : env.costs
    }
    
    return result