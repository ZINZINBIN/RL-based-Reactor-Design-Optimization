from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from config.device_info import config_benchmark, config_by_rl, config_liquid
import argparse
import os

def parsing():
    parser = argparse.ArgumentParser(description="Compute the tokamak design and verify the lawson criteria")
    parser.add_argument("--save_dir", type = str, default = "./results")
    parser.add_argument("--tag", type = str, default = "project")
    parser.add_argument("--use_benchmark", type = bool, default = False)
    parser.add_argument('--use_rl', type = bool, default = False)
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    
    args = parsing()
    
    if args['use_benchmark']:
        args['tag'] = 'reference'
        config = config_benchmark
        
    elif args['use_rl']:
        config = config_by_rl 
        args['tag'] = 'ppo'
        
    else:
        config = config_liquid
    
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
    tokamak.print_info(os.path.join(args['save_dir'], "{}_stat.txt".format(args['tag'])))
    tokamak.print_profile(os.path.join(args['save_dir'], "{}_profile.png".format(args['tag'])))
    tokamak.print_design_configuration(os.path.join(args['save_dir'], "{}_poloidal_design.png".format(args['tag'])))
    tokamak.print_lawson_criteria(os.path.join(args['save_dir'], "{}_lawson.png".format(args['tag'])))
    tokamak.print_overall_performance(os.path.join(args['save_dir'], "{}_overall.png".format(args['tag'])))