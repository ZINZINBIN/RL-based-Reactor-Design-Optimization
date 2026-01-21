import numpy as np
from typing import Callable, List, Dict, Union
from tqdm import tqdm
from src.design.env import Environment
from src.optim.particle.optimizer import ParticleSwarmOptimizer
from src.config.search_space_info import search_space, state_space
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Seed for reproducibility
np.random.seed(42)

ctrl_keys = search_space.keys()
state_keys = state_space.keys()

state_keys_list = list(state_keys)
ctrl_keys_list = list(ctrl_keys)

def dict2arr(x:Dict, keys:List):
    y = []
    for key in keys:
        y.append(x[key].item())
    return np.array(y)

def arr2dict(x:np.array, keys:List):
    y = {}
    for i, key in enumerate(keys):
        y[key] = x[i]
    return y

class DesignOptimizer:
    def __init__(
        self, 
        obj:Callable,
        n_ptls: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
        ):
        
        self.state_keys = state_keys_list
        self.ctrl_keys = ctrl_keys_list
        
        bounds = []
        
        for param in search_space.keys():
            p_min = search_space[param][0]
            p_max = search_space[param][1]
            bounds.append([p_min, p_max])
            
        bounds = np.array(bounds)
        self.optim = ParticleSwarmOptimizer(obj, bounds, n_ptls, w, c1, c2)
        
        self.optim.initialize_particles(len(search_space))
        
    def update(self):
        self.optim.update_v()
        self.optim.update_x()
    
    def register_batch(self, states:List[Dict], ctrls:List[Dict], fs:np.ndarray, gs:np.ndarray):
        
        xs = []
        
        for ctrl in ctrls:
            xs.append(dict2arr(ctrl, self.ctrl_keys).reshape(1,-1))

        xs = np.vstack(xs)
        self.optim.register(xs, None, fs)
        
    def extract_batch(self):
        xs = []
        for ptl in self.optim.ptls:
            x = arr2dict(ptl.x, self.ctrl_keys)
            xs.append(x)
        return xs


def evaluate_single_process(env:Environment, ctrl:Dict, objective:Callable, constraint:Callable):
    state = env.step(ctrl)
    return objective(state), constraint(state), state

def evaluate_batch(env:Environment, ctrl_batch:List, objective:Callable, constraint:Callable):
    return [evaluate_single_process(env, ctrl, objective, constraint) for ctrl in ctrl_batch]

def search_param_space(
    env: Environment,
    objective: Callable,
    constraint: Callable,
    num_episode: int = 10000,
    verbose: int = 100,
    n_proc:int = 4,
    n_particles: int = 128,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
):  

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

    f_obj = lambda x : objective(env.step(arr2dict(x, ctrl_keys_list), save = False), discretized=False)

    # Particle swarm optimizer
    optimizer = DesignOptimizer(f_obj, n_particles, w, c1, c2)

    # Step 1. Initial samples generation for training GP in the Optimizer (Random sampling)
    init_params = {}

    # Gaussian sampling
    for param in search_space.keys():
        p_min = search_space[param][0]
        p_max = search_space[param][1]

        mu = 0.5 * (p_min + p_max)
        sig = (p_max - p_min) * 0.5
        sample = mu + sig * np.random.randn(n_particles)
        init_params[param] = np.clip(sample, p_min, p_max)

    ctrls = []

    for idx in range(n_particles):
        ctrl = {}
        for key in search_space.keys():
            ctrl[key] = init_params[key][idx]
        ctrls.append(ctrl)

    batch_size = n_particles // n_proc
    batch_ctrls = [ctrls[i : i + batch_size] for i in range(0, n_particles, batch_size)]

    evals_batches = Parallel(n_jobs=n_proc)(
        delayed(evaluate_batch)(env, batch, objective, constraint)
        for batch in batch_ctrls
    )

    evals = [item for sublist in evals_batches for item in sublist]
    fs, gs, states = zip(*evals)

    fs = np.array(fs)
    gs = np.array(gs)

    # Register initial samples to the optimizer
    optimizer.register_batch(states, ctrls, fs, gs)

    # update particle
    optimizer.update()

    # Step 2. Online optimization and suggestion of the design parameters
    for i_episode in tqdm(range(num_episode)):

        ctrls = optimizer.extract_batch()
        batch_ctrls = [ctrls[i : i + batch_size] for i in range(0, n_particles, batch_size)]

        evals_batches = Parallel(n_jobs=n_proc)(
            delayed(evaluate_batch)(env, batch, objective, constraint)
            for batch in batch_ctrls
        )

        evals = [item for sublist in evals_batches for item in sublist]
        fs, gs, states = zip(*evals)

        fs = np.array(fs)
        gs = np.array(gs)

        for state, action, g in zip(states, ctrls, gs):

            if state is None:
                continue

            else:
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
                
        # Register evaluated samples to the optimizer
        optimizer.register_batch(states, ctrls, fs, gs)

        # update particle
        optimizer.update()

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
