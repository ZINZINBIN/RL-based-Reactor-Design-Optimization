import numpy as np
import random
from itertools import product
from multiprocessing import cpu_count, Process, Queue
from typing import Optional, Callable, List
from src.design.env import Enviornment
from src.config.search_space_info import search_space

# For single cpu processing
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
        state = env.step(ctrl)

        if state is None:
            continue

        env.current_state = state
        env.current_action = ctrl

        if i_episode % verbose == 0:
            print(
                r"| episode:{} | cost: {:.3f} | tau: {:.3f} | Q: {:.3f} | b limit: {} | q limit:{} | n limit: {} | f limit: {} | i limit: {} ".format(
                    i_episode + 1,
                    env.costs[-1],
                    env.taus[-1],
                    env.Qs[-1],
                    env.b_limits[-1],
                    env.q_limits[-1],
                    env.n_limits[-1],
                    env.f_limits[-1],
                    env.i_limits[-1],
                )
            )

        if i_episode >= num_episode:
            break

    result = {
        "control":env.actions,
        "state":env.states,
        "tau":env.taus,
        "Q":env.Qs,
        "b_limit" : env.b_limits,
        "q_limit" : env.q_limits,
        "n_limit" : env.n_limits,
        "f_limit" : env.f_limits,
        "i_limit" : env.i_limits,
        "cost" : env.costs
    }

    return result


# For multi-cpu processing
def search_param_space_multi_cpu(
    search_param_space_single_process: Callable,
    num_episode: int = 10000,
    n_grid: int = 32,
    n_proc:int = -1,
):

    if n_proc == -1:
        n_proc = cpu_count()
    
    param_space = {}

    for param in search_space.keys():

        p_min = search_space[param][0]
        p_max = search_space[param][1]

        ps = np.linspace(p_min, p_max, n_grid)
        indice = np.random.randint(0,n_grid, num_episode)
        param_space[param] = ps[indice]

    combinations = []

    for idx in range(num_episode):
        ctrl = {}
        for key in search_space.keys():
            ctrl[key] = param_space[key][idx]
        combinations.append(ctrl)

    print(f"Starting Grid-search with {n_proc} processes on {len(combinations)} combinations...")

    chunk_size = len(combinations) // n_proc
    chunks = [combinations[i:i+chunk_size] for i in range(0, len(combinations), chunk_size)]

    if len(chunks) > n_proc:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    queue = Queue()
    processes = []
    
    for i in range(n_proc):
        param_list = chunks[i]
        
        p = Process(target=search_param_space_single_process, args=(param_list, queue))
        p.start()
        processes.append(p)

    combined = {
        "control": [],
        "state": [],
        "tau": [],
        "b_limit": [],
        "q_limit": [],
        "n_limit": [],
        "f_limit": [],
        "i_limit": [],
        "cost": [],
    }

    for _ in range(n_proc):
        
        result = queue.get()
        
        for k in combined.keys():
            combined[k].extend(result[k])

    for p in processes:
        p.join()
    
    return combined
