import gym

class Enviornment(gym.Env):
    def step(self):
        return NotImplementedError("self.step(*args)")
    
    def reset(self):
        return NotImplementedError("self.reset(*args)")
    
    def close(self):
        return NotImplementedError("self.close(*args)")
    
    def render(self):
        return NotImplementedError("self.render(*args)")