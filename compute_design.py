from src.design.device import Tokamak
from src.design.profile import Profile
from src.design.source import CDsource
from src.design.util import read_design
from src.config.device_info import config_benchmark
import argparse
import os

def parsing():
    parser = argparse.ArgumentParser(description="Reactor design computation code")
    parser.add_argument("--save_dir", type = str, default = "./results/design")
    parser.add_argument("--use_benchmark", type=bool, default=False)
    parser.add_argument("--algorithm", type=str, default="bayesian", choices=["genetic", "bayesian", "gridsearch", "crl", "rl"])
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    config = config_benchmark
    
    if args['use_benchmark']:
        tag = "benchmark"
    else:
        tag = args['algorithm']
        config_path = os.path.join("./results", args['algorithm'], "config.pkl")
        config = read_design(config_path)
    
    filepath = os.path.join(args['save_dir'], tag)

    # directory
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    profile = Profile(
        nu_T = config["nu_T"],
        nu_p = config["nu_p"],
        nu_n = config["nu_n"],
        n_avg = config["n_avg"], 
        T_avg = config["T_avg"], 
        p_avg = config['p_avg']
    )

    source = CDsource(
        conversion_efficiency = config['conversion_efficiency'],
        absorption_efficiency = config['absorption_efficiency'],
    )

    tokamak = Tokamak(
        profile,
        source,
        betan = config['betan'],
        Q = config['Q'],
        k = config['k'],
        epsilon = config['epsilon'],  
        tri = config['tri'],
        thermal_efficiency = config['thermal_efficiency'],
        electric_power = config['electric_power'],
        armour_thickness = config['armour_thickness'],
        armour_density = config['armour_density'],
        armour_cs = config['armour_cs'],
        maximum_wall_load = config['maximum_wall_load'],
        maximum_heat_load = config['maximum_heat_load'],
        shield_density = config['shield_density'],
        shield_depth = config['shield_depth'],
        shield_cs = config['shield_cs'],
        Li_6_density = config['Li_6_density'],
        Li_7_density = config['Li_7_density'],
        slowing_down_cs= config['slowing_down_cs'],
        breeding_cs= config['breeding_cs'],
        E_thres = config['E_thres'],
        pb_density = config['pb_density'],
        scatter_cs_pb=config['cs_pb_scatter'],
        multi_cs_pb=config['cs_pb_multi'],
        B0 = config['B0'],
        H = config['H'],
        maximum_allowable_J = config['maximum_allowable_J'],
        maximum_allowable_stress = config['maximum_allowable_stress'],
        RF_recirculating_rate= config['RF_recirculating_rate'],
        flux_ratio = config['flux_ratio']
    )

    # save file
    tokamak.print_info(os.path.join(filepath, "stat.txt"))
    tokamak.print_profile(os.path.join(filepath, "profile.png"))
    tokamak.print_design_configuration(os.path.join(filepath, "poloidal_design.png"))
    tokamak.print_lawson_criteria(os.path.join(filepath, "lawson.png"))
    tokamak.print_overall_performance(os.path.join(filepath, "overall.png"))
