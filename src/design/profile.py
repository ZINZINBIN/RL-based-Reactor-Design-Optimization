import numpy as np

class Profile:
    def __init__(
        self, 
        nu_T : float,
        nu_p : float,
        nu_n : float,
        n_avg : float, 
        T_avg : float, 
        p_avg : float
        ):

        # profile parameters
        self.nu_T = nu_T
        self.nu_p = nu_p
        self.nu_n = nu_n

        # global information
        self.n_avg = n_avg
        self.T_avg = T_avg
        self.p_avg = p_avg

    def _profile(self, nu:float, avg:float, n:int):
        r_a = np.linspace(0,1,n)
        return avg * (1+nu) * (1-r_a ** 2) ** nu

    def compute_p(self, radius : float, a_minor : float):
        p = self.p_avg * (1 + self.nu_p) * (1 - (radius / a_minor) ** 2) ** self.nu_p
        return p

    def compute_p_profile(self, n : int = 64):
        return self._profile(self.nu_p, self.p_avg, n)

    def compute_p_total(self, n : int = 64):
        p_profile = self.compute_p_profile(n)
        return np.mean(p_profile)

    def compute_n(self, radius : float, a_minor : float):
        n = self.n_avg * (1 + self.nu_n) * (1 - (radius / a_minor) ** 2) ** self.nu_n
        return n

    def compute_n_profile(self, n : int = 64):
        return self._profile(self.nu_n, self.n_avg, n)

    def compute_n_total(self, n : int = 64):
        n_profile = self.compute_n_profile(n)
        return np.mean(n_profile)

    def compute_T(self, radius : float, a_minor : float):
        T = self.T_avg * (1 + self.nu_T) * (1 - (radius / a_minor) ** 2) ** self.nu_T
        return T

    def compute_T_profile(self, n : int = 64):
        return self._profile(self.nu_T, self.T_avg, n)

    def compute_T_total(self, n : int = 64):
        T_profile = self.compute_T_profile(n)
        return np.mean(T_profile)
