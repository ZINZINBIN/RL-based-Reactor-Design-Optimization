import numpy as np
import math
from src.profile import Profile
from typing import Dict

class Blanket:
    def __init__(self, density_6 : float, density_7 : float, slowing_down_cs : float, breeding_cs : float, E_thres : float):
        self.density_6 = density_6
        self.density_7 = density_7
        self.slownig_down_cs = slowing_down_cs
        self.breeding_cs = breeding_cs    
        self.E_thres = E_thres
        self.lamda_s = 1 / density_7 / slowing_down_cs
        
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
        return flux

    def compute_desire_depth(self, in_energy : float, in_flux : float, out_flux : float):
        alpha_B = 1 / (2 * self.lamda_s * self.density_6 * self.breeding_cs) * math.sqrt(in_energy / self.E_thres)
        depth = self.lamda_s * math.log(1+alpha_B * math.log(in_flux / out_flux))
        return depth

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
        B0 : float, 
        H : float,
        maximum_allowable_J : float,
        maximum_allowable_stress : float,
        ):
        
        self.profile = profile
        self.k = k
        self.epsilon = epsilon
        self.tri = tri
        self.thermal_efficiency = thermal_efficiency
        self.electric_power = electric_power
        
        self.B0 = B0
        
        self.H = H
        
        # core
        self.core = Core(profile, k, epsilon, tri)
        R, a = self.core.compute_desire_geometry(thermal_efficiency, maximum_wall_load, electric_power)
        self.Rc = R
        self.a = a
        
        # armour
        self.armour = Shielding(armour_thickness, armour_density, armour_cs, None, maximum_wall_load)
        self.armour_thickness = armour_thickness
        
        # blanket
        self.blanket = Blanket(Li_6_density, Li_7_density, slowing_down_cs, breeding_cs, E_thres)
        in_flux = self.core.compute_neutron_flux()
        out_flux = in_flux * 10 ** (-5)
        in_energy = 14.1
        
        self.blanket_thickness = self.blanket.compute_desire_depth(in_energy, in_flux, out_flux)
        
        # shield
        self.shield = Shielding(shield_depth, shield_density, shield_cs, maximum_heat_load, maximum_wall_load)
        self.shield_depth = shield_depth        
        
        # coil
        self.coil = TFcoil(a, self.blanket_thickness, R, B0, maximum_allowable_J, maximum_allowable_stress)
        self.coil_thickness = self.coil.compute_thickness()
        
        self.total_thickness = self.coil_thickness + self.blanket_thickness + self.armour_thickness + self.a + self.shield_depth
        
        # update profile with exact values
        self.update_p_avg()
        self.update_n_avg()
        
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
        Eout = self.blanket.compute_neutron_energy(En, self.blanket_thickness)
        
        in_flux = self.core.compute_neutron_flux()
        const = self.blanket.breeding_cs * self.blanket.density_6 * 2 * self.blanket.lamda_s
        
        def _compute_neutron_flux_E(E):
            return in_flux * np.exp((-1) * const * (np.sqrt(self.blanket.E_thres / E) - np.sqrt(self.blanket.E_thres / En)))
        
        def _compute_neutron_energy(x):
            return En * np.exp((-1) * x / self.blanket.lamda_s)
        
        x_arr = np.linspace(self.armour_thickness, self.blanket_thickness + self.armour_thickness, 64)
        E_arr = _compute_neutron_energy(x_arr)
        
        tr_generation_arr = self.blanket.breeding_cs * self.blanket.density_6 * np.sqrt(self.blanket.E_thres / E_arr) * _compute_neutron_flux_E(E_arr)
        tr_generation_arr *= math.sqrt((1 + self.k ** 2) / 2) * 2 * math.pi * (x_arr + self.a) * self.Rc * math.pi * 2 * (x_arr[1] - x_arr[0]) * self.k
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
        q = 2 * math.pi * self.a ** 2 * B * (1 + self.k ** 2) * 0.5
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
        betan = 2.8
        B = B = self.B0 * (1 - (self.a + self.blanket_thickness + self.shield_depth) / self.Rc)
        beta_max = betan * self.compute_Ip() / self.a / B
        return beta_max
    
    def print_info(self):
        
        print("\n================================================")
        print("============= Tokamak design info ==============")
        print("================================================\n")
        print("=============== Structure design ===============\n")
        print("| CS coil | TF coil | shield | blanket | 1st wall ) core ) 1st wall ) blanket ) shield ) TF coil")
        print("\n================ Geometric info ================")
        print("| Major radius R : {:.3f}".format(self.Rc))
        print("| Minor radius a : {:.3f}".format(self.a))
        print("| Armour : {:.3f}".format(self.armour_thickness))
        print("| Blanket : {:.3f}".format(self.blanket_thickness))
        print("| Shield : {:.3f}".format(self.shield_depth))
        print("| TF coil : {:.3f}".format(self.coil_thickness))
        print("| total thickness : {:.3f}".format(self.total_thickness))
        print("================ Operation info ================")
        print("| Magnetic field : {:.3f}".format(self.B0))
        print("| Elongation : {:.3f}".format(self.k))
        print("| Aspect ratio : {:.3f}".format(self.epsilon))
        print("| Triangularity : {:.3f}".format(self.tri))
        print("| Thermal efficiency : {:.3f}".format(self.thermal_efficiency))
        print("| Electric power : {:.3f} MW".format(self.electric_power / 10**6))
        print("| TBR : {:.3f}".format(self.compute_TBR()))
        print("| beta : {:.3f}".format(self.compute_beta() * 100))
        print("| tau : {:.3f}s".format(self.compute_confinement_time()))
        print("| Ip : {:.3f}MA".format(self.compute_Ip()))
        print("| q : {:.3f}".format(self.compute_q()))
        print("| Q-parallel : {:.3f}MW-T/m".format(self.compute_parallel_heat_flux()))
        print("================================================")
        
    def check_operation_limit(self):
        
        beta = self.compute_beta()
        beta_troyon = self.compute_troyon_beta()
        b_check = "O" if beta < beta_troyon else "X"
        
        q = self.compute_q()
        q_kink = 2
        q_check = "O" if q > q_kink else "X"
        
        n = self.profile.n_avg / 10 ** 20
        ng = self.compute_greenwald_density() / 10 ** 20
        n_check = "O" if n < ng else "X"
        
        print("=============== Operation limit ================")
        print("| Greenwald density : {:.3f}, operation density : {:.3f} | {}".format(ng, n, n_check))
        print("| q-kink : {:.3f}, operation q : {:.3f} | {}".format(q_kink, q, q_check))
        print("| Troyon beta : {:.3f}, operation beta : {:.3f} | {}".format(beta_troyon, beta, b_check))
        print("================================================")