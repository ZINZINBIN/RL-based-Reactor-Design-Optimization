import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from warnings import catch_warnings, simplefilter

class BayesOpt:
    def __init__(self, kernel:Kernel, bounds:np.ndarray, buffer_size:int = 512, xi:float = 0.01, n_restart:int = 16):
        self.buffer_size = buffer_size
        self.n_restart = n_restart
        self.xi = xi
        self.bounds = bounds
        self._gp = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", normalize_y=True)

        self.X_sample = None
        self.Y_sample = None

    def predict(self, X:np.ndarray):
        with catch_warnings():
            simplefilter("ignore")
            mu, sig = self._gp.predict(X, return_std = True)
        return mu, sig

    def EI(self, X:np.ndarray, X_sample:np.ndarray, xi:float = 0.01):
        mu, sig = self.predict(X)
        mu_sample, _ = self.predict(X_sample)

        mu_sample_best = np.max(mu_sample)

        with np.errstate(divide="warn"):
            imp = mu - mu_sample_best - xi
            z = imp / (sig + 1e-9)
            ei = imp * norm.cdf(z) + sig * norm.pdf(z)
            ei[sig == 0.0] = 0.0

        return ei

    def register(self, X:np.ndarray, Y:np.ndarray):

        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if self.X_sample is None:
            self.X_sample = X
            self.Y_sample = Y
        else:
            self.X_sample = np.vstack([self.X_sample, X])
            self.Y_sample = np.vstack([self.Y_sample, Y])

        if len(self.X_sample) > self.buffer_size:
            indice = np.argsort(self.Y_sample.ravel())[-self.buffer_size:]            
            self.X_sample = self.X_sample[indice]
            self.Y_sample = self.Y_sample[indice]

    def update(self):
        with catch_warnings():
            simplefilter("ignore")
            self._gp.fit(self.X_sample, self.Y_sample)

    def _obj(self, X:np.ndarray):
        return (-1) * self.EI(X.reshape(-1, self.X_sample.shape[1]), self.X_sample, self.xi)

    def suggest(self):

        val_min = None
        x_min = None
        dim = self.X_sample.shape[1]

        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_restart, dim)):
            res = minimize(self._obj, x0=x0, bounds=self.bounds, method='L-BFGS-B') 

            if val_min is None:
                val_min = res.fun
                x_min = res.x

            elif res.fun < val_min:
                val_min = res.fun
                x_min = res.x           

        return x_min

if __name__ == "__main__":

    def test_f(x):
        return np.sum(x**2) * (-1)

    n_episode = 50
    xi = 0.1
    buffer_size = 64
    n_restart = 32

    n_update = 1

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) + Matern(length_scale=1.0, nu=2.5)

    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(50, 4))
    Y_sample = np.array([test_f(x) for x in X_sample])

    optim = BayesOpt(kernel, bounds, buffer_size, xi, n_restart)

    optim.register(X_sample, Y_sample)
    optim.update()

    best_y = np.inf * (-1)
    best_x = None

    for i_episode in range(n_episode):

        x_next = optim.suggest()
        y_next = np.array([test_f(x_next)])

        optim.register(x_next, y_next)

        if i_episode % n_update == 0:
            optim.update()

        if y_next > best_y:
            best_y = y_next
            best_x = x_next

        print("iter:{} | best x:{} | best y:{} | next x:{} | next y:{}".format(i_episode + 1, best_x, best_y, x_next, y_next))

    print("\nBest solution found:", best_x)
    print("Function value at best solution:", best_y)
