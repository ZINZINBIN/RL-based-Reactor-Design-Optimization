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
    
    def compute_p(self, radius : float, a_minor : float):
        p = self.p_avg * (1 + self.nu_p) * (1 - (radius / a_minor) ** 2) ** self.nu_p
        return p
        
    def compute_p_profile(self, n : int = 64):
        r_a = np.linspace(0,1,n)
        p_profile = self.p_avg * (1 + self.nu_p) * (1 - (r_a) ** 2) ** self.nu_p
        return p_profile
    
    def compute_p_total(self, n : int = 64):
        p_profile = self.compute_p_profile(n)
        p_total = sum(p_profile) / n
        return p_total
    
    def compute_n(self, radius : float, a_minor : float):
        n = self.n_avg * (1 + self.nu_n) * (1 - (radius / a_minor) ** 2) ** self.nu_n
        return n
        
    def compute_n_profile(self, n : int = 64):
        r_a = np.linspace(0,1,n)
        n_profile = self.n_avg * (1 + self.nu_n) * (1 - (r_a) ** 2) ** self.nu_n
        return n_profile
    
    def compute_n_total(self, n : int = 64):
        n_profile = self.compute_n_profile(n)
        n_total = sum(n_profile) / n
        return n_total
    
    def compute_T(self, radius : float, a_minor : float):
        T = self.T_avg * (1 + self.nu_T) * (1 - (radius / a_minor) ** 2) ** self.nu_T
        return T

    def compute_T_profile(self, n : int = 64):
        r_a = np.linspace(0,1,n)
        T_profile = self.T_avg * (1 + self.nu_T) * (1 - (r_a) ** 2) ** self.nu_T
        return T_profile
    
    def compute_T_total(self, n : int = 64):
        T_profile = self.compute_T_profile(n)
        T_total = sum(T_profile) / n
        return T_total