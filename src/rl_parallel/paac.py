'''
    Paper: Efficient parallel methods for deep reinforcement learning
    - paac.py from https://github.com/Alfredvc/paac/blob/master/paac.py
    - Parallel framework for Efficient Deep RL (ActorCritic Version)
'''
from src.rl.ppo import update_policy, ActorCritic
from src.env import Enviornment
from config.search_space_info import search_space
from multiprocessing import Process

class PAACRunner(Process):
    def __init__(self):
        pass