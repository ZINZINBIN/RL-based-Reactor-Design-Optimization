import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from warnings import catch_warnings, simplefilter

class BayesianOptimizer:
    def __init__(self, kernel:Kernel, bounds:np.ndarray, buffer_size:int = 512, xi:float = 0.01, n_restart:int = 16):
        self.buffer_size = buffer_size
        self.n_restart = n_restart
        self.xi = xi
        self.bounds = bounds
        self._gp = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", normalize_y=True)

        self.X_sample = None
        self.Y_sample = None

    def predict(self, X:np.ndarray):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        with catch_warnings():
            simplefilter("ignore")
            mu, sig = self._gp.predict(X, return_std = True)

        return mu, sig

    def EI(self, X:np.ndarray, xi:float = 0.01):
        mu, sig = self.predict(X)

        # Method 01. mu_sample from prediction with X_sample
        # mu_sample, _ = self.predict(self.X_sample)
        # mu_sample_best = np.max(mu_sample)

        # Method 02. mu_sample from registered samples
        mu_sample_best = np.max(self.Y_sample)

        with np.errstate(divide="warn"):
            imp = mu - mu_sample_best - xi
            z = imp / (sig + 1e-9)
            ei = imp * norm.cdf(z) + sig * norm.pdf(z)
            # ei[sig < 1e-9] = 0.0

        return ei

    def UCB(self, X:np.ndarray, kappa:float = 2.0):
        # Upper confidence bound
        mu, sig = self.predict(X)
        return mu + kappa * sig

    def register(self, X:np.ndarray, Y:np.ndarray):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if self.X_sample is None:
            self.X_sample = X.copy()
            self.Y_sample = Y.copy()

        else:
            self.X_sample = np.vstack([self.X_sample, X.copy()])
            self.Y_sample = np.vstack([self.Y_sample, Y.copy()])

        if len(self.X_sample) > self.buffer_size:
            indice = np.argsort(self.Y_sample.ravel())[-self.buffer_size:]            
            self.X_sample = self.X_sample[indice]
            self.Y_sample = self.Y_sample[indice]

    def update(self):
        with catch_warnings():
            simplefilter("ignore")
            self._gp.fit(self.X_sample, self.Y_sample)

    def _obj(self, X:np.ndarray):
        dim = self.X_sample.shape[1]
        return self.EI(X.reshape(-1, dim), self.xi)

    def suggest(self):

        dim = self.X_sample.shape[1]

        # Method 01. normal distribution around the best sample
        mu = self.X_sample[np.argmax(self.Y_sample)]
        std = (self.bounds[:, 1] - self.bounds[:, 0]) / 4.0
        X_sample = np.clip(np.random.normal(mu, std, size=(self.n_restart, dim)), self.bounds[:, 0], self.bounds[:, 1])

        # Method 02. uniform distribution
        # X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size = (self.n_restart, dim))

        Y_sample = self._obj(X_sample)

        idx = np.argmax(Y_sample)
        x_min = X_sample[idx]
        return x_min

if __name__ == "__main__":

    def test_f(x):
        return np.sum(x**2) * (-1)

    n_dim = 2
    n_sample = 100

    n_episode = 100
    xi = 0.01
    buffer_size = 128
    n_restart = 50

    n_update = 1

    bounds = np.array([[-1.0, 1.0] for _ in range(n_dim)])
    kernel = Matern(length_scale=1.0, nu=2.5)

    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_sample, n_dim))
    Y_sample = np.array([test_f(x) for x in X_sample])

    optim = BayesianOptimizer(kernel, bounds, buffer_size, xi, n_restart)

    optim.register(X_sample, Y_sample)
    optim.update()

    best_y = np.inf * (-1)
    best_x = None
    best_iter = 0

    for i_episode in range(n_episode):

        x_next = optim.suggest()
        y_next = np.array([test_f(x_next)])

        optim.register(x_next, y_next)

        if i_episode % n_update == 0:
            optim.update()

        if y_next > best_y:
            best_y = y_next
            best_x = x_next
            best_iter = i_episode + 1

        print("iter:{} | best x:{} | best y:{} | next x:{} | next y:{}".format(i_episode + 1, best_x, best_y, x_next, y_next))

    print("\nBest iter:{} | best solution:{}".format(best_iter, best_x))
    print("Function value at best solution:", best_y)
