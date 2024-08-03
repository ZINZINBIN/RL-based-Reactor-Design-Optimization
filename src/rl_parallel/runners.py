import numpy as np
import os
from src.env import Enviornment
from multiprocessing import Queue, Process
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double
from typing import List
from collections import namedtuple

# variables: [(np.asarray(state : [s1,s2,...])), (np.asarray(reward : [r1,r2,...])), np.asarray((action:[a1,a2,...]))]
class EmulatorRunner(Process):
    def __init__(self, id:int, envs:List[Enviornment], variables, queue, barrier:Queue):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.envs = envs
        self.variables = variables
        self.queue = queue
        self.barrier = barrier
    
    def init_env(self):
        for env in self.envs:
            env.reset()    
    
    def run(self):
        super(EmulatorRunner, self).run()
        self._run()
        
    def _run(self):
        
        for i, (env, action) in enumerate(zip(self.envs, self.variables[-1])):    
                  
            ctrl = {
                'betan':action[0],
                'k':action[1],
                'epsilon' : action[2],
                'electric_power' : action[3],
                'T_avg' : action[4],
                'B0' : action[5],
                'H' : action[6],
                "armour_thickness":action[7],
                "RF_recirculating_rate":action[8],
            }
            
            state, reward, _, _ = env.step(ctrl)
            state = np.array([state[key] for key in state.keys()] + [ctrl[key] for key in ctrl.keys()])
            
            self.variables[0][0] = state
            self.variables[1][0] = reward
            
            print("Process ID:{} / Parent ID: {} run".format(os.getpid(), os.getppid()))
            
        self.barrier.put(True)

class Runners(object):
    
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, emulators, workers:int, variables):
        self.variables = [self._get_shared(var) for var in variables]
        self.workers = workers
        self.queues = [Queue() for _ in range(workers)]
        self.barrier = Queue()
        
        self.runners = [
            EmulatorRunner(i, emulators, vars, self.queues[i], self.barrier) for i, (emulators, vars) in 
            enumerate(zip(np.split(emulators, workers), zip(*[np.split(var, workers) for var in self.variables])))
        ]
        
    def init_env(self):
        for runner in self.runners:
            runner.init_env()

    def _get_shared(self, array:np.asarray):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """
        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for r in self.runners:
            r.start()

    def stop(self):
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.variables

    def update_environments(self):
        for queue in self.queues:
            queue.put(True)

    def wait_updated(self):
        for wd in range(self.workers):
            self.barrier.get()