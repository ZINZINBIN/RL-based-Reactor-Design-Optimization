import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
from tqdm.auto import tqdm
from typing import Optional, Dict
from src.env import Enviornment
from config.search_space_info import search_space
from torch.distributions import Normal
import os, pickle
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done','prob_a')
)

default_action_range = search_space

class SharedReplayBuffer(object):
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