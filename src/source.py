import numpy as np

class CDsource:
    def __init__(
        self, 
        conversion_efficiency : float, 
        absorption_efficiency : float, 
        w_pe : float, 
        w_pi : float,
        B0 : float, 
        eb : float,
        ):
        
        self.conversion_efficiency = conversion_efficiency
        self.absorption_efficiency = absorption_efficiency
        self.w_pe = w_pe
        self.w_pi = w_pi
        
        self.B0 = B0
        self.eb = eb
        
        self.me = 9.11 * 10 ** (-31)
        self.e = 1.6 * 10 ** (-19)
        
    def compute_B(self, rho : float):
        B = self.B0 / (1 + self.eb * rho)
        return B
        
    def compute_W_e(self, rho : float):
        B = self.compute_B(rho)
        W_e = self.e * B / self.me
        return W_e
        
    def compute_w_LH(self, rho : float):
        W_e = self.compute_W_e(rho)
        w_LH = self.w_pi / (1 + (self.w_pe / W_e ) ** 2) ** 0.5
        return w_LH
        
    def compute_CD_efficiency(self):
        
        w_LH = self.compute_w_LH(0.8)
        w = 2 * w_LH
        index_parallel = self.w_pe / self.W_e + (1 + (self.w_pe / self.W_e ) ** 2) ** 0.5 * (1 - (w_LH / w) ** 2) ** 0.5
        
        eta_CD = 1.2 / index_parallel ** 2
        return eta_CD