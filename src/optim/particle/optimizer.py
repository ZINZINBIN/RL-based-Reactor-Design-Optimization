import numpy as np
from typing import Callable, Tuple, Optional

class Particle:
    def __init__(self, x: np.ndarray, v: np.ndarray, f:np.ndarray):
        self.x = x
        self.v = v
        self.f = f
        self.best_x = x.copy()
        self.best_f = float("inf") * (-1)

class ParticleSwarmOptimizer:
    def __init__(
        self,
        obj:Callable,
        bounds: np.ndarray,
        n_ptls: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):  
        self.obj = obj
        self.n_ptls = n_ptls
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.ptls = []
        self.global_best_x = None
        self.global_best_f = float("inf") * (-1)

    def initialize_particles(self, dims: int):

        for _ in range(self.n_ptls):
            x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=dims)
            v = np.random.uniform(-1, 1, dims)
            f = self.obj(x)
            ptl = Particle(x, v, f)

            ptl.best_x = x.copy()
            ptl.best_f = f

            if self.global_best_f == float("inf") * (-1) or f > self.global_best_f:
                self.global_best_x = x.copy()
                self.global_best_f = f

            self.ptls.append(ptl)

    def register(self, xs:np.ndarray, vs:Optional[np.ndarray], fs:Optional[np.array]):

        for idx in range(self.n_ptls):
            self.ptls[idx].x = xs[idx]

            if vs is not None:
                self.ptls[idx].v = vs[idx]

            if fs is not None:
                self.ptls[idx].f = fs[idx]

            else:
                self.ptls[idx].f = self.obj(xs[idx])

            if fs is not None:

                if self.ptls[idx].best_f == float("inf") * (-1) or fs[idx] > self.ptls[idx].best_f:
                    self.ptls[idx].best_x = xs[idx].copy()
                    self.ptls[idx].best_f = fs[idx]

                if self.global_best_f == float("inf") * (-1) or fs[idx] > self.global_best_f:
                    self.global_best_x = xs[idx].copy()
                    self.global_best_f = fs[idx]

    def update(self):

        for ptl in self.ptls:

            r1 = np.random.rand(len(ptl.x))
            r2 = np.random.rand(len(ptl.x))

            cognitive = self.c1 * r1 * (ptl.best_x - ptl.x)
            social = self.c2 * r2 * (self.global_best_x - ptl.x)

            ptl.v = self.w * ptl.v + cognitive + social
            ptl.x = np.clip(ptl.x + ptl.v, self.bounds[:,0], self.bounds[:,1])
            ptl.f = self.obj(ptl.x)

            if ptl.best_f == float("inf") * (-1) or ptl.f > ptl.best_f:
                ptl.best_x = ptl.x.copy()
                ptl.best_f = ptl.f

                if self.global_best_f == float("inf") * (-1) or ptl.f > self.global_best_f:
                    self.global_best_x = ptl.x.copy()
                    self.global_best_f = ptl.f

    def update_x(self):
        
        for ptl in self.ptls:
            ptl.x = np.clip(ptl.x + ptl.v, self.bounds[:,0], self.bounds[:,1])

    def update_v(self):
        
        for ptl in self.ptls:

            r1 = np.random.rand(len(ptl.x))
            r2 = np.random.rand(len(ptl.x))

            cognitive = self.c1 * r1 * (ptl.best_x - ptl.x)
            social = self.c2 * r2 * (self.global_best_x - ptl.x)

            ptl.v = self.w * ptl.v + cognitive + social

if __name__ == "__main__":

    def test_f(x):
        return np.sum(x**2) * (-1)
    
    n_dim = 4
    n_ptls = 100
    n_episode = 50

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    optim = ParticleSwarmOptimizer(test_f, bounds, n_ptls = n_ptls, w = 0.7, c1 = 1.0, c2 = 1.0)
    optim.initialize_particles(n_dim)

    best_y = np.inf * (-1)
    best_x = None
    best_iter = 0

    for i_episode in range(n_episode):

        optim.update()
        
        x_next = optim.global_best_x
        y_next = optim.global_best_f

        if y_next > best_y:
            best_y = y_next
            best_x = x_next
            best_iter = i_episode + 1

        print("iter:{} | best x:{} | best y:{} | next x:{} | next y:{}".format(i_episode + 1, best_x, best_y, x_next, y_next))

    print("\nBest iter:{} | best solution:{}".format(best_iter, best_x))
    print("Function value at best solution:", best_y)
