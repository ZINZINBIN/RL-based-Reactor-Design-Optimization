class CDsource:
    def __init__(
        self, 
        conversion_efficiency : float, 
        absorption_efficiency : float, 
        ):

        self.conversion_efficiency = conversion_efficiency
        self.absorption_efficiency = absorption_efficiency

        self.mi = 1.67 * 10 ** (-27)
        self.me = 9.11 * 10 ** (-31)
        self.e = 1.6 * 10 ** (-19)
        self.eps = 8.85 * 10 ** (-12)

    def update_plasma_frequency(self, n_avg : float):
        self.w_pe = (self.e ** 2 * n_avg / self.eps / self.me) ** 0.5
        self.w_pi = (self.e ** 2 * n_avg / self.eps / self.mi) ** 0.5

    def update_eb(self, a : float, b : float, Rc : float):
        self.eb = a / Rc

    def update_B0(self, B0 : float):
        self.B0 = B0

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

    def compute_index_parallel(self, rho:float):
        w_LH = self.compute_w_LH(rho)
        W_e = self.compute_W_e(rho)
        w = 2 * w_LH
        index_parallel = self.w_pe / W_e + (1 + (self.w_pe / W_e ) ** 2) ** 0.5 * (1 - (w_LH / w) ** 2) ** 0.5

        return index_parallel

    def compute_CD_efficiency(self):
        index_parallel = self.compute_index_parallel(0.8)
        eta_CD = 1.2 / index_parallel ** 2
        return eta_CD

    def compute_I_CD(self, R0 : float, P_CD : float, n_avg : float):
        eta_CD = self.compute_CD_efficiency()
        I_CD = eta_CD * P_CD / R0 / n_avg * 10 ** 14
        return I_CD
