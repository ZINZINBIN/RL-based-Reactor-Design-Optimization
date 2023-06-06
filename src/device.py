import numpy as np
import math
import matplotlib.pyplot as plt
from src.profile import Profile
from src.source import CDsource
from src.lawson import Lawson
from config.neutron_info import *
from typing import Dict, Optional

class Blanket:
    def __init__(self, density_6 : float, density_7 : float, slowing_down_cs : float, breeding_cs : float, E_thres : float, density_pb : float, scatter_cs_pb : float, multi_cs_pb : float):
        self.density_6 = density_6
        self.density_7 = density_7
        self.slownig_down_cs = slowing_down_cs
        self.breeding_cs = breeding_cs    
        self.E_thres = E_thres
        self.lamda_s_li = 1 / density_7 / slowing_down_cs
        
        self.density_pn = density_pb
        self.scatter_cs_pb = scatter_cs_pb
        self.multi_cs_pb = multi_cs_pb
        
        if density_pb is None:
            self.lamda_s_pb = None
            self.lamda_m_pb = None
            self.lamda_s = self.lamda_s_li
        else:
            self.lamda_s_pb = 1 / density_pb / scatter_cs_pb
            self.lamda_m_pb = 1 / density_pb / multi_cs_pb
            self.lamda_s = (self.lamda_s_li ** (-1) + self.lamda_s_pb ** (-1)) ** (-1)
        
    def compute_lamda_br(self, E : float):
        breeding_cs = self.breeding_cs * math.sqrt(self.E_thres / E)
        lamda_br = 1 / self.density_6 / breeding_cs
        return lamda_br
    
    def compute_neutron_energy(self, E_input : float, x : float):
        E_output = math.exp(-x/self.lamda_s) * E_input
        return E_output

    def compute_neutron_flux(self, in_flux : float, in_energy:float, x : float):    
        alpha_B = 1 / (2 * self.lamda_s * self.density_6 * self.breeding_cs) * math.sqrt(in_energy / self.E_thres)
        flux = in_flux * math.exp((1-math.exp(x/2/self.lamda_s)) / alpha_B)
        
        if self.lamda_m_pb:
            flux = self.compute_multiplier_effect(flux, x)
        
        return flux

    def compute_desire_depth(self, in_energy : float, in_flux : float, out_flux : float):
        alpha_B = 1 / (2 * self.lamda_s * self.density_6 * self.breeding_cs) * math.sqrt(in_energy / self.E_thres)
        depth = self.lamda_s * math.log(1+alpha_B * math.log(in_flux / out_flux))
        return depth
    
    def compute_multiplier_effect(self, in_flux : float, x : float):
        flux = in_flux * math.exp(math.log(2) * 0.5 * x / self.lamda_m_pb)
        return flux

# ShieldingModule : Shield / First wall / Vaccuum Chamber 
class Shielding:
    def __init__(self, depth : float, density : float, cs : float, max_heat_load : float, max_neutron_flux : float):
        # property
        self.depth = depth
        self.cs = cs
        
        # maximum load
        self.max_heat_load = max_heat_load
        self.max_neutron_flux = max_neutron_flux
        
        # density and mean free path
        self.density = density
        self.lamda_s = 1 / density / cs
        
    def compute_neutron_energy(self, E_input : float, x : float):
        E_output = math.exp(-x/self.lamda_s) * E_input
        return E_output
    
    def compute_neutron_flux(self, in_flux : float, in_energy : float, x : float):
        flux = in_flux * math.exp(-x / self.lamda_s)
        return flux

class VaccumVessel:
    def __init__(self, depth : float, max_heat_load : float, max_neutron_flux : float, material : str):
        self.depth = depth
        self.max_heat_load = max_heat_load
        self.max_neutron_flux = max_neutron_flux
        self.material = material
        
# TF coil thickness
class TFcoil:
    def __init__(self, a : float, b : float, R0 : float, B0 : float, maximum_allowable_J : float, maximum_allowable_stress : float):
        self.a = a
        self.b = b
        self.R0 = R0
        self.eps_b = (a + b) / R0
        
        self.B0 = B0 * (1 - self.eps_b)
        
        self.maximum_allowable_J = maximum_allowable_J
        self.maximum_allowable_stress = maximum_allowable_stress
    
    def compute_mechanical_thickness(self):
        mu = math.pi * 4 * 10 ** (-7)
        alpha_m = self.B0 ** 2 / mu / self.maximum_allowable_stress * (2*self.eps_b / (1+self.eps_b) + 0.5 * math.log((1+self.eps_b) / (1-self.eps_b)))
        c_m = self.R0 * (1 - self.eps_b - math.sqrt((1-self.eps_b) ** 2 - alpha_m))
        return c_m
    
    def compute_superconducting_thickness(self):
        mu = math.pi * 4 * 10 ** (-7)
        alpha_j = 2 * self.B0 / mu / self.R0 / self.maximum_allowable_J
        c_j = self.R0 * (1 - self.eps_b - math.sqrt((1-self.eps_b)**2 - alpha_j))
        return c_j
    
    def compute_thickness(self):
        return self.compute_mechanical_thickness() + self.compute_superconducting_thickness()

# Plasma core with vaccuum vessel
class Core:
    def __init__(self, profile : Profile, k : float, epsilon : float, tri : float):
        # profiles for temperature, density and pressure
        self.profile = profile
        
        # shape parameters
        self.k = k
        self.epsilon = epsilon
        self.tri = tri
        
        # geometry
        self.Rc = None
        self.a = None
        
    def compute_avg_cross_section_v(self, T : float):
        k0 = -60.4593
        k1 = 6.1371
        k2 = -0.8609
        k3 = 0.0356
        k4 = -0.0045
        sig_v = math.exp(k0 + k1 * math.log(T) + k2 * math.log(T) ** 2 + k3 * math.log(T) ** 3 + k4 * math.log(T) ** 4)
        return sig_v

    def compute_fusion_power_density(self):
        En = 14.1
        Ea = 3.6
        p = self.profile.compute_p_total(n=64)
        T = self.profile.compute_t_total(n=64)
        sig_v = self.compute_avg_cross_section_v(T)
        Sp = 1 / 16 * (En + Ea) * p ** 2 * sig_v / T ** 2
        return Sp
    
    def compute_neutron_flux(self):
        N = self.profile.compute_n_total(n=64)
        T = self.profile.compute_T_total(n=64)
        sig_v = self.compute_avg_cross_section_v(T)
        
        volume = self.Rc * math.pi * 2 * math.pi * self.a ** 2
        surface = 4 * math.pi * self.Rc * self.a * math.sqrt((1+self.k**2)/2)
        dt_rate = 0.25 * N ** 2 * sig_v * volume
        flux = dt_rate / surface
        return flux
    
    def compute_desire_geometry(self, thermal_efficiency : float, maximum_wall_load : float, electric_power : float):
        En = 14.1
        Ef = 22.4
        Ra = 0.25 / math.pi ** 2 * En / Ef * electric_power / thermal_efficiency / maximum_wall_load * math.sqrt(2 / (1 + self.k ** 2))
        R_desire = math.sqrt(Ra * self.epsilon)
        a_desire =  R_desire / self.epsilon
        
        self.Rc = R_desire
        self.a = a_desire
        
        return R_desire, a_desire
    
    def compute_core_volume(self):
        volume = self.Rc * math.pi * 2 * math.pi * self.a ** 2 * self.k
        return volume
    
        
class Tokamak:
    def __init__(
        self, 
        profile : Profile, 
        source : CDsource,
        betan : float,
        Q : float,
        k : float, 
        epsilon : float, 
        tri : float, 
        thermal_efficiency : float, 
        electric_power : float,
        armour_thickness : float,
        armour_density : float, 
        armour_cs : float,
        maximum_wall_load : float,
        maximum_heat_load : float,
        shield_density : float,
        shield_depth : float,
        shield_cs : float,
        Li_6_density : float, 
        Li_7_density : float, 
        slowing_down_cs : float,
        breeding_cs : float,
        E_thres : float,
        pb_density : float,
        scatter_cs_pb : float,
        multi_cs_pb : float,
        B0 : float, 
        H : float,
        maximum_allowable_J : float,
        maximum_allowable_stress : float,
        RF_recirculating_rate : float,
        flux_ratio : float,
        ):
        
        self.profile = profile
        self.source = source
        self.lawson = Lawson()
        self.betan = betan
        self.Q = Q
        self.k = k
        self.epsilon = epsilon
        self.tri = tri
        self.thermal_efficiency = thermal_efficiency
        self.electric_power = electric_power
        self.RF_recirculating_rate = RF_recirculating_rate
        self.flux_ratio = flux_ratio
        
        self.B0 = B0
        
        self.H = H
        
        # material properties
        self.maximum_allowable_J = maximum_allowable_J
        self.maximum_allowable_stress = maximum_allowable_stress
        self.maximum_wall_load = maximum_wall_load
        self.maximum_heat_load = maximum_heat_load
        
        # core
        self.core = Core(profile, k, epsilon, tri)
        R, a = self.core.compute_desire_geometry(thermal_efficiency, maximum_wall_load, electric_power)
        self.Rc = R
        self.a = a
        
        # update profile with exact values
        self.update_p_avg()
        self.update_n_avg()
        
        # armour
        self.armour = Shielding(armour_thickness, armour_density, armour_cs, None, maximum_wall_load)
        self.armour_thickness = armour_thickness
        
        # blanket
        self.blanket = Blanket(Li_6_density, Li_7_density, slowing_down_cs, breeding_cs, E_thres, pb_density, scatter_cs_pb, multi_cs_pb)
        in_flux = self.core.compute_neutron_flux()
        in_energy = self.armour.compute_neutron_energy(14.1, armour_thickness)
        
        # armour shielding
        in_flux = self.armour.compute_neutron_flux(in_flux, in_energy, armour_thickness)
        out_flux = in_flux * flux_ratio
        
        self.blanket_thickness = self.blanket.compute_desire_depth(in_energy, in_flux, out_flux)
        
        # shield
        self.shield = Shielding(shield_depth, shield_density, shield_cs, maximum_heat_load, maximum_wall_load)
        self.shield_depth = shield_depth        
        
        # coil
        self.coil = TFcoil(a, self.blanket_thickness, R, B0, maximum_allowable_J, maximum_allowable_stress)
        self.coil_thickness = self.coil.compute_thickness()
        
        self.total_thickness = self.coil_thickness + self.blanket_thickness + self.armour_thickness + self.a + self.shield_depth
        
        # update current source 
        self.source.update_B0(self.B0 * (1-(self.a + self.blanket_thickness + self.shield_depth) / self.Rc))
        self.source.update_plasma_frequency(self.profile.compute_n(0.8 * self.a, self.a))
        self.source.update_eb(self.a, self.blanket_thickness, self.Rc)
        
    def update_design(self, betan : float, k : float, epsilon : float, electric_power : float, T_avg : float, B0 : float, H : float, armour_thickness : float, RF_recirculating_rate : float):
        
        # update T_avg
        self.profile.T_avg = T_avg

        self.betan = betan
        self.k = k
        self.epsilon = epsilon
        self.electric_power = electric_power
        self.B0 = B0
        
        self.H = H
        
        self.armour_thickness = armour_thickness
        self.RF_recirculating_rate = RF_recirculating_rate
        
        # core
        self.core.profile = self.profile
        self.core.k = k
        self.core.epsilon = epsilon
        R, a = self.core.compute_desire_geometry(self.thermal_efficiency, self.maximum_wall_load, electric_power)
        self.Rc = R
        self.a = a
        
        # update profile with exact values
        self.update_p_avg()
        self.update_n_avg()
        
        # armour shielding
        in_flux = self.core.compute_neutron_flux()
        in_energy = self.armour.compute_neutron_energy(14.1, self.armour_thickness)
        
        in_flux = self.armour.compute_neutron_flux(in_flux, in_energy, self.armour_thickness)
        out_flux = in_flux * self.flux_ratio
        
        # blanket
        self.blanket_thickness = self.blanket.compute_desire_depth(in_energy, in_flux, out_flux)
        
        # coil
        self.coil.a = a
        self.coil.b = self.blanket_thickness
        self.coil.eps_b = (a+self.blanket_thickness) / R
        self.coil.B0 = B0 * (1-self.coil.eps_b)
        self.coil_thickness = self.coil.compute_thickness()
        
        self.total_thickness = self.coil_thickness + self.blanket_thickness + self.armour_thickness + self.a + self.shield_depth
        
        # update current source 
        self.source.update_B0(self.B0 * (1-(self.a + self.blanket_thickness + self.shield_depth) / self.Rc))
        self.source.update_plasma_frequency(self.profile.compute_n(0.8 * self.a, self.a))
        self.source.update_eb(self.a, self.blanket_thickness, self.Rc)
        
    def update_p_avg(self):
        fp = self.profile.compute_p_profile(n=65)[:-1] / self.profile.p_avg
        fT = self.profile.compute_T_profile(n=65)[:-1] / self.profile.T_avg
        
        sig_v = np.array([self.core.compute_avg_cross_section_v(T) for T in self.profile.compute_T_profile(n=65)[:-1]])
        rho = np.linspace(0,1,65)[:-1]
        drho = rho[1] - rho[0]
        integral = sum(fp ** 2 / fT ** 2 * sig_v *rho * drho) * 2
        Ef = 22.4 * 10 ** 6 * 1.6 * 10 ** (-19)
        const = 16 * self.electric_power / self.core.compute_core_volume() / self.thermal_efficiency / Ef
        p_avg = self.profile.T_avg * math.sqrt(const / integral) * 1.6 * 10 ** (-19) * 10 ** 3
        
        self.profile.p_avg = p_avg
        
    def update_n_avg(self):
        fp = self.profile.compute_p_profile(n=65)[:-1] / self.profile.p_avg
        fn = self.profile.compute_n_profile(n=65)[:-1] / self.profile.n_avg
        fT = self.profile.compute_T_profile(n=65)[:-1] / self.profile.T_avg
        n_avg = sum(fp) / sum(fn * fT) * 0.5 * self.profile.p_avg / self.profile.T_avg / (1.6 * 10 ** (-19) * 10 ** 3)
        self.profile.n_avg = n_avg
        
    def compute_beta(self):
        B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        beta = 2 * self.profile.p_avg * 4 * math.pi * 10 ** (-7) / B ** 2
        return beta
        
    def compute_TBR(self):
        sig_v = self.core.compute_avg_cross_section_v(self.profile.compute_T_total(n=64))
        Ndt = self.profile.compute_n_total(n=64)
        dt_rate = 0.25 * Ndt ** 2 * sig_v * self.core.compute_core_volume()
        
        En = 14.1
        
        in_flux = self.core.compute_neutron_flux()
        in_flux = self.armour.compute_neutron_flux(in_flux, None, self.armour_thickness)
        
        En = self.armour.compute_neutron_energy(En, self.armour_thickness)
        const = self.blanket.breeding_cs * self.blanket.density_6 * 2 * self.blanket.lamda_s
        
        def _compute_neutron_flux_E(E):
            return in_flux * np.exp((-1) * const * (np.sqrt(self.blanket.E_thres / E) - np.sqrt(self.blanket.E_thres / En)))
        
        def _compute_neutron_energy(x):
            if self.blanket.density_pn is not None:
                const_m = np.exp(-x / self.blanket.lamda_m_pb)
            else:
                const_m = 1
            return En * np.exp((-1) * x / self.blanket.lamda_s) * const_m
        
        x_arr = np.linspace(0, self.blanket_thickness, 64)
        E_arr = _compute_neutron_energy(x_arr)
        
        if self.blanket.density_pn is not None:
            multiplier = 1.7
        else:
            multiplier = 1
        
        tr_generation_arr = self.blanket.breeding_cs * self.blanket.density_6 * np.sqrt(self.blanket.E_thres / E_arr) * _compute_neutron_flux_E(E_arr) * multiplier
        tr_generation_arr += 20 * 10 ** (-31) * self.blanket.density_7 * _compute_neutron_flux_E(E_arr)
        
        tr_generation_arr *= math.sqrt((1 + self.k ** 2) / 2) * 2 * math.pi * (x_arr + self.a + self.armour_thickness) * self.Rc * math.pi * 2 * (x_arr[1] - x_arr[0]) * self.k
        tr_generation = np.sum(tr_generation_arr)
        TBR = tr_generation / dt_rate
        
        return TBR
    
    def compute_confinement_time(self):
        Ef = 22.4
        Ea = 3.6
        tau = Ef / Ea * 1.5 * self.core.compute_core_volume() * self.profile.p_avg * self.thermal_efficiency / self.electric_power
        return tau
    
    def compute_Ip(self):
        Ef = 22.4
        Ea = 3.6
        tau = self.compute_confinement_time()
        B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        n_ = self.profile.n_avg / 10 ** 20
        A = 2.5
        Ip = 7.98 * tau ** 1.08 * (Ea / Ef * self.electric_power / self.thermal_efficiency / 10 ** 6) ** 0.74
        Ip /= self.H ** 1.08 * self.Rc ** 1.49 * self.a ** 0.62 * self.k ** 0.84 * n_ ** 0.44 * B ** 0.16 * A ** 0.2
        return Ip    
    
    def compute_q(self):
        Ip = self.compute_Ip()
        B = B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        q = 2 * math.pi * self.a ** 2 * B * (1 + self.k ** 2)
        q /= 4 * math.pi * 10 ** (-7) * self.Rc * Ip * 10 ** 6
        return q
    
    def compute_parallel_heat_flux(self):
        Ef = 22.4
        Ea = 3.6
        Pa = self.electric_power / self.thermal_efficiency * Ea / Ef
        B = B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        Q = Pa * B / self.Rc * 10 ** (-6)
        return Q
    
    def compute_greenwald_density(self):
        Ip = self.compute_Ip()
        ng = Ip / math.pi / self.a ** 2 * 10 ** 20
        return ng
    
    def compute_troyon_beta(self):
        B = B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        beta_max = self.betan * self.compute_Ip() / self.a / B
        return beta_max
    
    def compute_bootstrap_fraction(self):
        P_CD = self.electric_power * self.RF_recirculating_rate
        I_CD = self.source.compute_I_CD(self.Rc, P_CD, self.profile.n_avg)
        Ip = self.compute_Ip()
        f_bs = 1 - I_CD / Ip
        return f_bs
    
    def compute_NC_bootstrap_fraction(self):
        a_hat = self.a * self.k ** 0.5
        Ip = self.compute_Ip()
        
        rho = np.linspace(0,1,64)
        
        p_profile = self.profile.compute_p_profile(64)
        n_profile = self.profile.compute_n_profile(64)
        T_profile = self.profile.compute_T_profile(64)
        
        drho = rho[1] - rho[0]
        
        pprime = (p_profile[1:] - p_profile[:-1]) / drho
        nprime = (n_profile[1:] - n_profile[:-1]) / drho
        Tprime = (T_profile[1:] - T_profile[:-1]) / drho
        
        rho = (rho[1:] + rho[:-1]) / 2
        p_profile = (p_profile[1:] + p_profile[:-1]) / 2
        n_profile = (n_profile[1:] + n_profile[:-1]) / 2
        T_profile = (T_profile[1:] + T_profile[:-1]) / 2
        
        x = rho ** 2.25
        alpha = 2.53
        mu = 4 * math.pi * 10 ** (-7)
        
        Jtor = Ip / math.pi / a_hat ** 2 * (9/8) * rho ** 0.25 * (alpha ** 2 * (1-x) * np.exp(alpha * x) / (np.exp(alpha) - 1 - alpha))
        Bp = mu * Ip / 2 / math.pi / a_hat * (1 / rho) * (((1 + alpha - alpha * x) * np.exp(alpha * x) - 1 - alpha)/ (np.exp(alpha) - 1 - alpha)) * 10 ** 6
        
        JB = -2.44 * (rho * a_hat / self.Rc) ** 0.5 * (p_profile / Bp) * (1 / n_profile * nprime + 0.055 / T_profile * Tprime) / a_hat
        
        IB = 2 * sum(JB * rho * drho) * self.a ** 2 * self.k * math.pi * 10 ** (-6)
        f_NC = IB / Ip
        return f_NC
    
    def check_ignition(self):
        T_operation = self.profile.T_avg
        tau_operation = self.compute_confinement_time()
        
        B = self.B0 * (1 - (self.a + self.blanket_thickness)/self.Rc)
        n = self.profile.n_avg 
        
        psi = 10 ** (-2)
        n_tau_lower_bound = self.lawson.compute_n_tau_lower_bound(T_operation, n, B, psi) * 10 ** (-20)
        n *= 10 ** (-20)
        n_tau = n * tau_operation
        
        is_ignition = True if n_tau > n_tau_lower_bound else False
        return is_ignition, n_tau, n_tau_lower_bound
    
    def compute_cost_params(self):
        Vb = 2 * math.pi ** 2 * self.Rc * ((self.a + self.blanket_thickness) * (self.a * self.k + self.blanket_thickness) - self.k * self.a ** 2)
        Vtf = 4 * math.pi * self.coil_thickness * (2 * self.Rc - 2 * self.a - 2 * self.blanket_thickness - 2 * self.coil_thickness) * ((1 + self.k) * self.a + 2 * self.blanket_thickness + self.coil_thickness)
        cost = (Vb + Vtf) / self.electric_power * 10 ** 6
        return cost
        
    def print_lawson_criteria(self, filename : str):
        
        tau_operation = self.compute_confinement_time()
        T_operation = self.profile.T_avg
        
        T = np.linspace(6, 100, 64, endpoint=False)
        n = self.profile.n_avg
        B = self.B0 * (1 - (self.a + self.blanket_thickness)/self.Rc)
        
        psi = 10 ** (-2)
        
        n_tau = [self.lawson.compute_n_tau_lower_bound(t, n, B, psi) * 10 ** (-20) for t in T]
        n_tau_5 = [self.lawson.compute_n_tau_Q_lower_bound(t, n, B, psi, 5) * 10 ** (-20) for t in T]
        n_tau_Q = [self.lawson.compute_n_tau_Q_lower_bound(t, n, B, psi, self.Q) * 10 ** (-20) for t in T]
        n_tau_break = [self.lawson.compute_n_tau_Q_lower_bound(t, n, B, psi, 1) * 10 ** (-20) for t in T]
        n *= 10 ** (-20)
        
        fig, ax = plt.subplots(1,1, figsize = (8,6))
        ax.plot(T, n_tau, "k", label = "Lawson criteria (Ignition)")
        ax.plot(T, n_tau_5, "r", label = "Lawson criteria (Q=5)")
        ax.plot(T, n_tau_Q, "b", label = "Lawson criteria (Q={})".format(self.Q))
        ax.plot(T, n_tau_break, "g", label = "Lawson criteria (Breakeven)")
        ax.scatter(T_operation, tau_operation * n, c = 'r', label = 'Tokamak design')
    
        ax.set_xlabel("T(unit : keV)")
        ax.set_ylabel("$(N\\tau_E)_{dt}(unit:10^{20}s * m^{-3})$")
        ax.set_xlim([5,100])
        ax.set_ylim([0,10])
        ax.legend()
        fig.tight_layout()
        plt.savefig(filename)
        
    def print_profile(self, filename :str):
        n = 64
        r_a = np.linspace(0,1,n)
        p_profile = self.profile.compute_p_profile(n) / self.profile.p_avg
        T_profile = self.profile.compute_T_profile(n) / self.profile.T_avg
        n_profile = self.profile.compute_n_profile(n) / self.profile.n_avg
        
        fig, ax = plt.subplots(1,1, figsize = (8,6))
        ax.plot(r_a, p_profile, "k", label = "p-profile")
        ax.plot(r_a, T_profile, "r", label = "T-profile")
        ax.plot(r_a, n_profile, "b", label = "n-profile")
    
        ax.set_xlabel("Normalized radius")
        ax.set_ylabel("Normalized quantity")
        ax.set_xlim([0,1.0])
        ax.set_ylim([0,3.0])
        ax.legend()
        fig.tight_layout()
        plt.savefig(filename)
        
    def print_info(self, filename : Optional[str] = None):
        
        print("\n================================================")
        print("============= Tokamak design info ==============")
        print("================================================\n")
        print("=============== Structure design ===============\n")
        print("| CS coil | TF coil | shield | blanket | 1st wall ) core ) 1st wall ) blanket ) shield ) TF coil")
        print("\n================ Geometric info ================")
        print("| Major radius R : {:.3f} m".format(self.Rc))
        print("| Minor radius a : {:.3f} m".format(self.a))
        print("| Armour : {:.3f} m".format(self.armour_thickness))
        print("| Blanket : {:.3f} m".format(self.blanket_thickness))
        print("| Shield : {:.3f} m".format(self.shield_depth))
        print("| TF coil : {:.3f} m".format(self.coil_thickness))
        print("| total thickness : {:.3f} m".format(self.total_thickness))
        print("============== Physical parameters ==============")
        print("| Magnetic field : {:.3f}".format(self.B0))
        print("| Elongation : {:.3f}".format(self.k))
        print("| Aspect ratio : {:.3f}".format(self.epsilon))
        print("| Thermal efficiency : {:.3f}".format(self.thermal_efficiency))
        print("| Electric power : {:.3f} MW".format(self.electric_power / 10**6))
        print("| TBR : {:.3f}".format(self.compute_TBR()))
        print("| beta : {:.3f}".format(self.compute_beta() * 100))
        print("| tau : {:.3f} s".format(self.compute_confinement_time()))
        print("| Ip : {:.3f} MA".format(self.compute_Ip()))
        print("| q : {:.3f}".format(self.compute_q()))
        print("| f_bs : {:.3f}".format(self.compute_bootstrap_fraction()))
        print("| Q-parallel : {:.2f} MW-T/m".format(self.compute_parallel_heat_flux()))
        print("| T_avg : {:.2f} keV".format(self.profile.T_avg))
        print("| n_avg : {:.2f}x10^20 #/m^3".format(self.profile.n_avg / 10 ** 20))
        
        beta = self.compute_beta()
        beta_troyon = self.compute_troyon_beta()
        b_check = "O" if beta * 100 < beta_troyon else "X"
        
        q = self.compute_q()
        q_kink = 2
        q_check = "O" if q > q_kink else "X"
        
        n = self.profile.n_avg / 10 ** 20
        ng = self.compute_greenwald_density() / 10 ** 20
        n_check = "O" if n < ng else "X"
        
        f_NC = self.compute_NC_bootstrap_fraction()
        f_bs = self.compute_bootstrap_fraction()
        bs_check = "O" if f_NC > f_bs else "X"
        
        is_ignition, n_tau, n_tau_criteria = self.check_ignition()
        ignition_check = "O" if is_ignition else "X"
        
        print("=============== Operation limit ================")
        print("| Greenwald density : {:.3f}, operation density : {:.3f} | {}".format(ng, n, n_check))
        print("| q-kink : {:.3f}, operation q : {:.3f} | {}".format(q_kink, q, q_check))
        print("| Troyon beta : {:.3f}, operation beta : {:.3f} | {}".format(beta_troyon, beta * 100, b_check))
        print("| Neoclassical f_bs : {:.3f}, operation f_bs : {:.3f} | {}".format(f_NC, f_bs, bs_check))
        print("| Lawson nTau : {:.3f} , operation nTau: {:.3f} | {}".format(n_tau_criteria, n_tau, ignition_check))
        print("| Cost params : {:.3f}".format(self.compute_cost_params()))
        print("================================================")
        
        if filename:
            with open(filename, "w") as f:
                f.write("\n================================================")
                f.write("\n============= Tokamak design info ==============")
                f.write("\n================================================\n")
                f.write("\n=============== Structure design ===============\n")
                f.write("\n| CS coil | TF coil | shield | blanket | 1st wall ) core ) 1st wall ) blanket ) shield ) TF coil")
                f.write("\n================ Geometric info ================")
                f.write("\n| Major radius R : {:.3f} m".format(self.Rc))
                f.write("\n| Minor radius a : {:.3f} m".format(self.a))
                f.write("\n| Armour : {:.3f} m".format(self.armour_thickness))
                f.write("\n| Blanket : {:.3f} m".format(self.blanket_thickness))
                f.write("\n| Shield : {:.3f} m".format(self.shield_depth))
                f.write("\n| TF coil : {:.3f} m".format(self.coil_thickness))
                f.write("\n| total thickness : {:.3f} m".format(self.total_thickness))
                f.write("\n============== Physical parameters ==============")
                f.write("\n| Magnetic field : {:.3f} T".format(self.B0))
                f.write("\n| Elongation : {:.3f}".format(self.k))
                f.write("\n| Aspect ratio : {:.3f}".format(self.epsilon))
                f.write("\n| Thermal efficiency : {:.3f}".format(self.thermal_efficiency))
                f.write("\n| Electric power : {:.3f} MW".format(self.electric_power / 10**6))
                f.write("\n| TBR : {:.3f}".format(self.compute_TBR()))
                f.write("\n| beta : {:.3f}".format(self.compute_beta() * 100))
                f.write("\n| tau : {:.3f} s".format(self.compute_confinement_time()))
                f.write("\n| Ip : {:.3f} MA".format(self.compute_Ip()))
                f.write("\n| q : {:.3f}".format(self.compute_q()))
                f.write("\n| f_bs : {:.3f}".format(self.compute_bootstrap_fraction()))
                f.write("\n| Q-parallel : {:.2f} MW-T/m".format(self.compute_parallel_heat_flux()))
                f.write("\n| T_avg : {:.2f} keV".format(self.profile.T_avg))
                f.write("\n| n_avg : {:.2f}x10^20 #/m^3".format(self.profile.n_avg / 10 ** 20))
                f.write("\n=============== Operation limit ================")
                f.write("\n| Greenwald density : {:.3f}, operation density : {:.3f} | {}".format(ng, n, n_check))
                f.write("\n| q-kink : {:.3f}, operation q : {:.3f} | {}".format(q_kink, q, q_check))
                f.write("\n| Troyon beta : {:.3f}, operation beta : {:.3f} | {}".format(beta_troyon, beta * 100, b_check))
                f.write("\n| Neoclassical f_bs : {:.3f}, operation f_bs : {:.3f} | {}".format(f_NC, f_bs, bs_check))
                f.write("\n| Lawson nTau : {:.3f} , operation n*Tau: {:.3f} | {}".format(n_tau_criteria, n_tau, ignition_check))
                f.write("\n| Cost params : {:.3f}".format(self.compute_cost_params()))
                f.write("\n================================================")
                
    def get_design_performance(self):
        
        _, n_tau, n_tau_lower = self.check_ignition()
        
        result = {
            "R" : self.Rc,
            "a" : self.a,
            "blanket_thickness" : self.blanket_thickness,
            "coil_thickness" : self.coil_thickness,
            "n" : self.profile.n_avg / 10 ** 20,
            "TBR" : self.compute_TBR(),
            "beta" : self.compute_beta() * 100,
            "tau" : self.compute_confinement_time(),
            "Ip" : self.compute_Ip(),
            "q" : self.compute_q(),
            "f_BS" : self.compute_bootstrap_fraction(),
            "Q_parallel" : self.compute_parallel_heat_flux(),
            "n_g" : self.compute_greenwald_density() / 10 ** 20,
            "q_kink" : 2,
            "beta_troyon" : self.compute_troyon_beta(),
            "f_NC" : self.compute_NC_bootstrap_fraction(),
            "n_tau" : n_tau,
            "n_tau_lower" : n_tau_lower,
            "cost" : self.compute_cost_params()
        }
        
        return result
    
    def print_overall_performance(self, filename : str):
        
        # original point
        result = self.get_design_performance()
        a_origin = self.a
        eps_origin = self.epsilon
        H_origin = self.H
        B_origin = self.B0
        T_avg_origin = self.profile.T_avg
        electric_power_origin = self.electric_power
        k_origin = self.k
        betan_origin = self.betan
        
        
        n_limit_origin = result['n'] / result['n_g']
        f_limit_origin = result['f_NC'] / result['f_BS']
        q_limit_origin = result['q_kink'] / result['q']
        b_limit_origin = result['beta'] / result['beta_troyon']
        
        fig, axes = plt.subplots(2,2, figsize = (12,12))
        axes = axes.ravel()
        
        n_points = 200
        
        # variable : a
        eps_list = np.linspace(2.5, 5.0, n_points)
    
        n_limits = []
        b_limits = []
        q_limits = []
        f_limits = []
        a_list = []
        
        for eps in eps_list:
            try:
                self.update_design(betan_origin, k_origin, eps, electric_power_origin, T_avg_origin, B_origin, H_origin, self.armour_thickness, self.RF_recirculating_rate)
                result = self.get_design_performance()
                n_limits.append(result['n'] / result['n_g'])
                b_limits.append(result['beta'] / result['beta_troyon'])
                q_limits.append(result['q_kink'] / result['q'])
                f_limits.append(result['f_BS'] / result['f_NC'])
                a_list.append(result['a'])
            except:
                continue
        
        axes[0].plot(a_list, n_limits, 'k', label = '$n/n_G$')
        axes[0].plot(a_list, b_limits, 'r', label = '$\\beta/\\beta_T$')
        axes[0].plot(a_list, f_limits, 'b', label = '$f_{BS}/f_{NC}$')
        axes[0].plot(a_list, q_limits, 'g', label = '$q_K/q$')
        axes[0].axvline(a_origin, 0, 1, c='k', label = 'Designed tokamak')
        axes[0].set_xlabel("a[m]")
        axes[0].set_ylim([0, 2.0])
        axes[0].legend(loc = 'upper right')
        axes[0].fill_between(a_list, 1, 2.0, facecolor = 'gray')
        
        # variable : H
        H_list_ = np.linspace(1.0, 1.5, n_points)
        
        n_limits = []
        b_limits = []
        q_limits = []
        f_limits = []
        H_list = []
        
        for H in H_list_:
            try:
                self.update_design(betan_origin, k_origin, eps_origin, electric_power_origin, T_avg_origin, B_origin, H, self.armour_thickness, self.RF_recirculating_rate)
                result = self.get_design_performance()
                n_limits.append(result['n'] / result['n_g'])
                b_limits.append(result['beta'] / result['beta_troyon'])
                q_limits.append(result['q_kink'] / result['q'])
                f_limits.append(result['f_BS'] / result['f_NC'])
                H_list.append(H)
            except:
                continue
        
        axes[1].plot(H_list, n_limits, 'k', label = '$n/n_G$')
        axes[1].plot(H_list, b_limits, 'r', label = '$\\beta/\\beta_T$')
        axes[1].plot(H_list, f_limits, 'b', label = '$f_{BS}/f_{NC}$')
        axes[1].plot(H_list, q_limits, 'g', label = '$q_K/q$')
        axes[1].axvline(H_origin, 0, 1, c='k', label = 'Designed tokamak')
        axes[1].set_xlabel("H")
        axes[1].set_ylim([0, 2.0])
        axes[1].legend(loc = 'upper right')
        axes[1].fill_between(H_list, 1, 2.0, facecolor = 'gray')
        
        # variable : B
        B_list_ = np.linspace(10, 20, n_points)
        
        n_limits = []
        b_limits = []
        q_limits = []
        f_limits = []
        B_list = []
        
        for B in B_list_:
            try:
                self.update_design(betan_origin, k_origin, eps_origin, electric_power_origin, T_avg_origin, B, H_origin, self.armour_thickness, self.RF_recirculating_rate)
                result = self.get_design_performance()
                n_limits.append(result['n'] / result['n_g'])
                b_limits.append(result['beta'] / result['beta_troyon'])
                q_limits.append(result['q_kink'] / result['q'])
                f_limits.append(result['f_BS'] / result['f_NC'])
                B_list.append(B)
            except:
                continue
        
        axes[2].plot(B_list, n_limits, 'k', label = '$n/n_G$')
        axes[2].plot(B_list, b_limits, 'r', label = '$\\beta/\\beta_T$')
        axes[2].plot(B_list, f_limits, 'b', label = '$f_{BS}/f_{NC}$')
        axes[2].plot(B_list, q_limits, 'g', label = '$q_K/q$')
        axes[2].axvline(B_origin, 0, 1, c='k', label = 'Designed tokamak')
        axes[2].set_xlabel("$B_{max}[T]$")
        axes[2].set_ylim([0, 2.0])
        axes[2].legend(loc = 'upper right')
        axes[2].fill_between(B_list, 1, 2.0, facecolor = 'gray')
        
        # variable : output
        output_list_ = np.linspace(500, 2000, n_points)
        
        n_limits = []
        b_limits = []
        q_limits = []
        f_limits = []
        output_list = []
        
        for output in output_list_:
            try:
                self.update_design(betan_origin, k_origin, eps_origin, output * 10 ** 6, T_avg_origin, B_origin, H_origin, self.armour_thickness, self.RF_recirculating_rate)
                result = self.get_design_performance()
                n_limits.append(result['n'] / result['n_g'])
                b_limits.append(result['beta'] / result['beta_troyon'])
                q_limits.append(result['q_kink'] / result['q'])
                f_limits.append(result['f_BS'] / result['f_NC'])
                output_list.append(output)
            except:
                continue
        
        if len(output_list) == 0:
            print("(Warning) | overall performance can not be computed")
            return
        
        axes[3].plot(output_list, n_limits, 'k', label = '$n/n_G$')
        axes[3].plot(output_list, b_limits, 'r', label = '$\\beta/\\beta_T$')
        axes[3].plot(output_list, f_limits, 'b', label = '$f_{BS}/f_{NC}$')
        axes[3].plot(output_list, q_limits, 'g', label = '$q_K/q$')
        axes[3].axvline(electric_power_origin / 10 ** 6, 0, 1, c='k', label = 'Designed tokamak')
        axes[3].set_xlabel("$P_E[MW]$")
        axes[3].set_ylim([0, 2.0])
        axes[3].legend(loc = 'upper right')
        axes[3].fill_between(output_list, 1, 2.0, facecolor = 'gray')
        
        # save file
        fig.tight_layout()
        plt.savefig(filename)