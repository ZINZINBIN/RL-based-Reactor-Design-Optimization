from src.design.device import Tokamak
from src.design.profile import Profile
from src.design.source import CDsource
from src.design.env import Environment
from src.config.device_info import config_benchmark
from src.optim.bayes.optimization import search_param_space, DesignOptimizer
from src.optim.util import objective, constraint
from src.analysis.util import find_optimal_design
from src.design.util import save_design
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import pickle
import argparse, os, warnings

warnings.filterwarnings(action="ignore")

def parsing():
    parser = argparse.ArgumentParser(description="Tokamak design optimization based on Bayesian Optimization")

    # Setup
    parser.add_argument("--num_episode", type=int, default=10000)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--n_proc", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=256)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--n_restart", type=int, default=32)

    # directory
    parser.add_argument("--save_dir", type=str, default="./results/bayesian")

    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":

    args = parsing()
    config = config_benchmark

    profile = Profile(
        nu_T=config["nu_T"],
        nu_p=config["nu_p"],
        nu_n=config["nu_n"],
        n_avg=config["n_avg"],
        T_avg=config["T_avg"],
        p_avg=config["p_avg"],
    )

    source = CDsource(
        conversion_efficiency=config["conversion_efficiency"],
        absorption_efficiency=config["absorption_efficiency"],
    )

    tokamak = Tokamak(
        profile,
        source,
        betan=config["betan"],
        Q=config["Q"],
        k=config["k"],
        epsilon=config["epsilon"],
        tri=config["tri"],
        thermal_efficiency=config["thermal_efficiency"],
        electric_power=config["electric_power"],
        armour_thickness=config["armour_thickness"],
        armour_density=config["armour_density"],
        armour_cs=config["armour_cs"],
        maximum_wall_load=config["maximum_wall_load"],
        maximum_heat_load=config["maximum_heat_load"],
        shield_density=config["shield_density"],
        shield_depth=config["shield_depth"],
        shield_cs=config["shield_cs"],
        Li_6_density=config["Li_6_density"],
        Li_7_density=config["Li_7_density"],
        slowing_down_cs=config["slowing_down_cs"],
        breeding_cs=config["breeding_cs"],
        E_thres=config["E_thres"],
        pb_density=config["pb_density"],
        scatter_cs_pb=config["cs_pb_scatter"],
        multi_cs_pb=config["cs_pb_multi"],
        B0=config["B0"],
        H=config["H"],
        maximum_allowable_J=config["maximum_allowable_J"],
        maximum_allowable_stress=config["maximum_allowable_stress"],
        RF_recirculating_rate=config["RF_recirculating_rate"],
        flux_ratio=config["flux_ratio"],
    )

    init_action = {
        "betan": config["betan"],
        "k": config["k"],
        "epsilon": config["epsilon"],
        "electric_power": config["electric_power"],
        "T_avg": config["T_avg"],
        "B0": config["B0"],
        "H": config["H"],
        "armour_thickness": config["armour_thickness"],
        "RF_recirculating_rate": config["RF_recirculating_rate"],
    }

    init_state = tokamak.get_design_performance()

    env = Environment(tokamak, init_state, init_action)

    # directory
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    save_result = os.path.join(args["save_dir"], "params_search.pkl")

    # Design optimization
    print("============ Design optimization ============")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5)
    optimizer = DesignOptimizer(kernel, args['buffer_size'], args['xi'], args['n_restart'])

    result = search_param_space(
        env,
        optimizer,
        objective,
        constraint,
        args['num_episode'],
        args['verbose'],
        args['n_proc'],
        args['sample_size']
    )

    with open(save_result, "wb") as file:
        pickle.dump(result, file)

    optimal = find_optimal_design(result)
    
    if optimal is not None:
        save_design(optimal, args["save_dir"], "optimal_config.pkl")