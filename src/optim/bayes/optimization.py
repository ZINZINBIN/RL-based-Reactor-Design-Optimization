import numpy as np
from typing import Callable, List, Dict, Union
from tqdm import tqdm
from src.design.env import Environment
from src.optim.bayes.optimizer import BayesianOptimizer
from sklearn.gaussian_process.kernels import Kernel
from src.config.search_space_info import search_space, state_space
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Seed for reproducibility
np.random.seed(42)

ctrl_keys = search_space.keys()
state_keys = state_space.keys()

state_keys_list = list(state_keys)
ctrl_keys_list = list(ctrl_keys)

class DesignOptimizer:
    def __init__(self, kernel:Kernel, buffer_size:int = 512, xi:float = 0.01, n_restart:int = 16):

        self.state_keys = state_keys_list
        self.ctrl_keys = ctrl_keys_list

        bounds = []

        for param in search_space.keys():
            p_min = search_space[param][0]
            p_max = search_space[param][1]
            bounds.append([p_min, p_max])

        bounds = np.array(bounds)

        self.optim = BayesianOptimizer(kernel, bounds, buffer_size, xi, n_restart)

    def update(self):
        self.optim.update()

    def register_batch(self, states:List[Dict], ctrls:List[Dict], fs:np.ndarray, gs:np.ndarray):

        X_sample = []
        
        for ctrl in ctrls:
            X_sample.append(self._dict2arr(ctrl, self.ctrl_keys).reshape(1,-1))

        X_sample = np.vstack(X_sample)
        Y_sample = fs

        self.optim.register(X_sample, Y_sample)

    def register(self, state:Dict, ctrl:Dict, f:Union[float, np.array], g:np.array):

        X_sample = self._dict2arr(ctrl, self.ctrl_keys)
        Y_sample = f
        
        self.optim.register(X_sample, Y_sample)

    def suggest(self):
        x_best = self.optim.suggest()
        x_best = self._arr2dict(x_best, self.ctrl_keys)
        return x_best

    def _dict2arr(self, x:Dict, keys:List):
        y = []

        for key in keys:
            v = x[key]
            y.append(v.item())

        return np.array(y)

    def _arr2dict(self, x:np.array, keys:List):

        y = {}
        for idx, key in enumerate(keys):
            v = x[idx]
            y[key] = v

        return y

# Evaluation of the design performance
def evaluate_single_process(env: Environment, ctrl: Dict, objective: Callable, constraint: Callable):
    state = env.step(ctrl)
    return objective(state, discretized=False), constraint(state, discretized = False), state

# batch-evaluation of the design performance for multi-core process
def evaluate_batch(env:Environment, ctrl_batch:List, objective:Callable, constraint:Callable):
    return [evaluate_single_process(env, ctrl, objective, constraint) for ctrl in ctrl_batch]

# Search optimial design parameters using Bayesian optimization
def search_param_space(
    env: Environment,
    optimizer: DesignOptimizer,
    objective: Callable,
    constraint: Callable,
    num_episode: int = 10000,
    verbose: int = 100,
    n_proc: int = -1,
    sample_size: int = 1024,
):

    # Step 1. Initial samples generation for training GP in the Optimizer (Random sampling)
    init_params = {}

    # Gaussian sampling
    for param in search_space.keys():
        p_min = search_space[param][0]
        p_max = search_space[param][1]

        mu = 0.5 * (p_min + p_max)
        sig = (p_max - p_min) * 0.5
        sample = mu + sig * np.random.randn(sample_size)
        init_params[param] = np.clip(sample, p_min, p_max)

    ctrls = []

    for idx in range(sample_size):
        ctrl = {}
        for key in search_space.keys():
            ctrl[key] = init_params[key][idx]
        ctrls.append(ctrl)

    if n_proc == -1:
        n_proc = cpu_count()

    batch_size = sample_size // n_proc
    batch_ctrls = [ctrls[i : i + batch_size] for i in range(0, sample_size, batch_size)]

    evals_batches = Parallel(n_jobs=n_proc)(
        delayed(evaluate_batch)(env, batch, objective, constraint)
        for batch in batch_ctrls
    )

    evals = [item for sublist in evals_batches for item in sublist]
    fs, gs, states = zip(*evals)

    fs = np.array(fs)
    gs = np.array(gs)

    # save offline stage
    for state, action, g in zip(states, ctrls, gs):

        if state is None:
            continue

        else:
            env.costs.append(state["cost"])
            env.taus.append(state["tau"])
            env.Qs.append(state["Q"])
            env.b_limits.append(g[0])
            env.q_limits.append(g[1])
            env.n_limits.append(g[2])
            env.f_limits.append(g[3])
            env.i_limits.append(1 if state["n_tau"] / state["n_tau_lower"] > 1 else 0)
            env.actions.append(action)
            env.states.append(state)

    # Step 2. Offline Bayesian optimization
    optimizer.register_batch(states, ctrls, fs, gs)
    optimizer.update() 

    # Step 3. Online optimization and suggestion of the design parameters
    for i_episode in tqdm(range(num_episode)):

        ctrl = optimizer.suggest()
        state = env.step(ctrl)

        f, g = objective(state, discretized=False), constraint(state, discretized=False)

        if state is None:
            continue

        env.current_state = state
        env.current_action = ctrl

        # register info to optimizer
        optimizer.register(state, ctrl, f, g)

        # update the GP regressor for EI approximation
        optimizer.update()

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

    result = {
        "control": env.actions[:num_episode],
        "state": env.states[:num_episode],
        "tau": env.taus[:num_episode],
        "Q": env.Qs[:num_episode],
        "b_limit": env.b_limits[:num_episode],
        "q_limit": env.q_limits[:num_episode],
        "n_limit": env.n_limits[:num_episode],
        "f_limit": env.f_limits[:num_episode],
        "i_limit": env.i_limits[:num_episode],
        "cost": env.costs[:num_episode],
    }

    return result
