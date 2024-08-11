from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from src.env import Enviornment
from src.rl.reward import RewardSender
from src.utility import plot_optimization_status, find_optimal_case
from src.gridsearch.brute_force import search_param_space
from config.device_info import config_benchmark, config_liquid
import pickle
import argparse, os, warnings

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="Tokamak design optimization based on Brute force algorithm")

    # Select blanket type: liquid / solid
    parser.add_argument("--blanket_type", type = str, default = "solid", choices = ['liquid','solid'])

    # Setup
    parser.add_argument("--num_episode", type = int, default = 10000)
    parser.add_argument("--verbose", type = int, default = 1000)
    parser.add_argument("--n_grid", type = int, default = 10)

    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":

    args = parsing()

    if args['blanket_type'] == 'liquid':
        config = config_liquid
    else:
        config = config_benchmark

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

    reward_sender = RewardSender(
        w_cost = 0.1,
        w_tau = 0.1,
        w_beta = 0.5,
        w_density=0.5,
        w_q = 1.0,
        w_bs = 1.0,
        w_i = 1.5,
        cost_r = 1.0,
        tau_r = 1.0,
        a = 1.0,
        reward_fail = -1.0
    )

    init_action = {
        'betan':config['betan'],
        'k':config['k'],
        'epsilon' : config['epsilon'],
        'electric_power' : config['electric_power'],
        'T_avg' : config['T_avg'],
        'B0' : config['B0'],
        'H' : config['H'],
        "armour_thickness" : config['armour_thickness'],
        "RF_recirculating_rate": config['RF_recirculating_rate'],
    }

    init_state = tokamak.get_design_performance()

    env = Enviornment(tokamak, reward_sender, init_state, init_action)

    # directory
    if not os.path.exists("./results"):
        os.makedirs("./results")

    tag = "gridsearch_{}".format(args['blanket_type'])
    save_result = "./results/params_search_{}.pkl".format(tag)

    # Design optimization
    print("============ Design optimization ============")
    result = search_param_space(
        env,
        args['num_episode'],
        args['verbose'],
        args['n_grid']
    )

    print("======== Logging optimization process ========")
    optimization_status = env.optim_status
    plot_optimization_status(optimization_status, args['verbose'], "./results/gridsearch_optimization")

    with open(save_result, 'wb') as file:
        pickle.dump(result, file)

    env.close()
    
    # save optimal design information
    find_optimal_case(result, {"save_dir":"./results", "tag":"gridsearch"})