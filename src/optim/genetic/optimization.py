import numpy as np
from typing import Callable, List, Dict
from tqdm import tqdm
from src.design.env import Environment
from src.config.search_space_info import search_space, state_space
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Seed for reproducibility
np.random.seed(42)

ctrl_keys = search_space.keys()
state_keys = state_space.keys()

ctrl_keys_list = list(ctrl_keys)

def evaluate_single_process(env:Environment, ctrl:Dict, objective:Callable, constraint:Callable):
    state = env.step(ctrl, False)
    return objective(state, discretized = True), constraint(state), state

def evaluate_batch(env:Environment, ctrl_batch:List, objective:Callable, constraint:Callable):
    return [evaluate_single_process(env, ctrl, objective, constraint) for ctrl in ctrl_batch]

# genetic algorithm for design optimization
def search_param_space(
    env: Environment,
    objective:Callable,
    constraint:Callable,
    num_episode: int = 10000,
    verbose: int = 100,
    pop_size: int = 64,
    mutation_rate: float = 0.2,
    mutation_sigma: float = 0.25,
    num_parents: int = 32,
    n_proc:int = -1,
    n_update:int = 5,
    sample_size:int = 1000,
):

    init_params = {}

    # trajectory
    traj_actions = []
    traj_states = []
    traj_taus = []
    traj_b_limits = []
    traj_q_limits = []
    traj_n_limits = []
    traj_f_limits = []
    traj_i_limits = []
    traj_costs = []
    traj_Qs = []

    if n_proc == -1:
        n_proc = cpu_count()

    # Gaussian sampling
    for param in search_space.keys():
        p_min = search_space[param][0]
        p_max = search_space[param][1]

        mu = 0.5 * (p_min + p_max)
        sig = (p_max - p_min) * 0.5
        sample = mu + sig * np.random.randn(sample_size)
        init_params[param] = np.clip(sample, p_min, p_max)

    combinations = []

    for idx in range(sample_size):
        ctrl = {}
        for key in search_space.keys():
            ctrl[key] = init_params[key][idx]
        combinations.append(ctrl)

    batch_size = sample_size // n_proc
    batched_combination = [combinations[i:i+batch_size] for i in range(0, sample_size, batch_size)]

    evals_batches = Parallel(n_jobs = n_proc)(delayed(evaluate_batch)(env, batch, objective, constraint) for batch in batched_combination)
    evals = [item for sublist in evals_batches for item in sublist]
    fs, gs, states = zip(*evals)

    fs = np.array(fs)
    gs = np.array(gs)

    # Save offline stage
    for state, action, g in zip(states, combinations, gs):
        traj_states.append(state)
        traj_actions.append(action)
        traj_b_limits.append(g[0])
        traj_q_limits.append(g[1])
        traj_n_limits.append(g[2])
        traj_f_limits.append(g[3])
        traj_i_limits.append(1 if state["n_tau"] / state["n_tau_lower"] > 1 else 0)
        traj_costs.append(state["cost"])
        traj_Qs.append(state["Q"])
        traj_taus.append(state["tau"])

    # Select random samples
    selected_indices = np.random.choice(np.arange(len(fs)), size = pop_size, replace = False)
    states, combinations, fs, gs = [states[idx] for idx in selected_indices], [combinations[idx] for idx in selected_indices], fs[selected_indices], gs[selected_indices]

    best_scores = []

    # Redefine batch size for online optimization
    batch_size = pop_size // n_proc

    for i_episode in tqdm(range(num_episode)):

        # create population with multicore process
        batched_combination = [combinations[i:i+batch_size] for i in range(0, pop_size, batch_size)]

        evals_batches = Parallel(n_jobs = n_proc)(delayed(evaluate_batch)(env, batch, objective, constraint) for batch in batched_combination)
        evals = [item for sublist in evals_batches for item in sublist]
        fs, gs, states = zip(*evals)

        fs = np.array(fs)
        gs = np.array(gs)

        # Save selected state
        idx = np.random.randint(pop_size)
        
        state = states[idx]
        g = gs[idx]
        action = combinations[idx]

        traj_costs.append(state['cost'])
        traj_taus.append(state['tau'])
        traj_Qs.append(state['Q'])
        traj_b_limits.append(g[0])
        traj_q_limits.append(g[1])
        traj_n_limits.append(g[2])
        traj_f_limits.append(g[3])
        traj_i_limits.append(1 if state["n_tau"] / state["n_tau_lower"] > 1 else 0)
        traj_actions.append(action)
        traj_states.append(state)

        best_scores.append(fs.max())

        # greedy selection
        if i_episode % n_update == 0:
            indice = np.argsort(fs)[-num_parents:]
        else:
            indice = np.random.choice(np.arange(len(fs)), size = num_parents, replace=False)

        selected = [combinations[idx] for idx in indice]

        # cross-over to generate offspring
        offspring = []

        for _ in range(pop_size - num_parents):

            alpha = np.random.rand()

            p1 = selected[np.random.randint(num_parents)]
            p2 = selected[np.random.randint(num_parents)]

            child = {}

            for key in search_space.keys():
                child[key] = alpha * p1[key] + (1-alpha) * p2[key]

            offspring.append(child)

        # Mutation of the offspring
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                for key in ctrl_keys_list:
                    p_min = search_space[key][0]
                    p_max = search_space[key][1]

                    scale = (p_max - p_min) / 2

                    offspring[i][key] += scale * np.random.normal(0, mutation_sigma)
                    offspring[i][key] = np.clip(offspring[i][key], p_min, p_max)

        # Concatenate the offspring and parent
        combinations = selected + offspring

        if i_episode % verbose == 0:
            print(
                r"| episode:{} | cost: {:.3f} | tau: {:.3f} | Q: {:.3f} | b limit: {} | q limit:{} | n limit: {} | f limit: {} | i limit: {} ".format(
                    i_episode + 1,
                    traj_costs[-1],
                    traj_taus[-1],
                    traj_Qs[-1],
                    traj_b_limits[-1],
                    traj_q_limits[-1],
                    traj_n_limits[-1],
                    traj_f_limits[-1],
                    traj_i_limits[-1],
                )
            )

        if i_episode >= num_episode:
            break

    result = {
        "control": traj_actions[:num_episode],
        "state": traj_states[:num_episode],
        "tau": traj_taus[:num_episode],
        "Q": traj_Qs[:num_episode],
        "b_limit": traj_b_limits[:num_episode],
        "q_limit": traj_q_limits[:num_episode],
        "n_limit": traj_n_limits[:num_episode],
        "f_limit": traj_f_limits[:num_episode],
        "i_limit": traj_i_limits[:num_episode],
        "cost": traj_costs[:num_episode],
    }

    return result
