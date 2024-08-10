from typing import Optional
from config.search_space_info import search_space
import os, pickle
from collections import namedtuple, deque

import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double

Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done','prob_a')
)

default_action_range = search_space

class SharedReplayBuffer(object):
    
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}
    
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def get_trajectory(self):
        traj = [self.memory[idx] for idx in range(len(self.memory))]
        return traj
    
    def clear(self):
        self.memory.clear()
    
    def save_buffer(self, env_name : str, tag : str = "", save_path : Optional[str] = None):
        
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/', exist_ok=True)

        if save_path is None:
            save_path = "checkpoints/buffer_{}_{}".format(env_name, tag)
            
        print("Process : saving buffer to {}".format(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump(self.memory, f)
        
    def load_buffer(self, save_path : str):
        print("Process : loading buffer from {}".format(save_path))
        
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)