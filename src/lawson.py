import numpy as np
import math

class Lawson:
    def __init__(self):
        self.Q_dt = 17.6 * 10 ** 6
        self.A_br = 1.6 * 10 ** (-38) * (1.6 * 10 ** (-19)) ** (-1)
        self.A_cyc = 6.3 * 10 ** (-20) * (1.6 * 10 ** (-19)) ** (-1)
        
        self.fc = 3.5 / 17.6
        
    def compute_avg_cross_section_v(self, T : float):
        k0 = -60.4593
        k1 = 6.1371
        k2 = -0.8609
        k3 = 0.0356
        k4 = -0.0045
        sig_v = math.exp(k0 + k1 * math.log(T) + k2 * math.log(T) ** 2 + k3 * math.log(T) ** 3 + k4 * math.log(T) ** 4)
        return sig_v
    
    def compute_n_tau_lower_bound(self, T : float, n : float, B : float, psi : float):
        denominator = self.fc * self.Q_dt * self.compute_avg_cross_section_v(T) / 4 - self.A_br * (T * 10 ** 3) ** 0.5 - psi * self.A_cyc * B ** 2 * (T * 10 ** 3) / n
        return 3 * T * 10 ** 3 / denominator
    
    def compute_n_tau_T_lower_bound(self, T : float, n : float, B : float, psi : float):
        denominator = self.fc * self.Q_dt * self.compute_avg_cross_section_v(T) / 4 - self.A_br * (T * 10 ** 3) ** 0.5 - psi * self.A_cyc * B ** 2 * (T * 10 ** 3) / n
        return 3 * T ** 2 * 10 ** 3 / denominator
    
    def compute_n_tau_Q_lower_bound(self, T : float, n : float, B : float, psi : float, Q : float):
        denominator = (1 / Q + self.fc) * self.Q_dt * self.compute_avg_cross_section_v(T) / 4 - self.A_br * (T * 10 ** 3) ** 0.5 - psi * self.A_cyc * B ** 2 * (T * 10 ** 3) / n
        return 3 * T * 10 ** 3 / denominator
    
    def compute_n_tau_T_Q_lower_bound(self, T : float, n : float, B : float, psi : float, Q : float):
        denominator = (1 / Q + self.fc) * self.Q_dt * self.compute_avg_cross_section_v(T) / 4 - self.A_br * (T * 10 ** 3) ** 0.5 - psi * self.A_cyc * B ** 2 * (T * 10 ** 3) / n
        return 3 * T ** 2 * 10 ** 3 / denominator