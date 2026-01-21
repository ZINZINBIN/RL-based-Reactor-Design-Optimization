from src.design.device import Tokamak
from src.design.profile import Profile
from src.design.source import CDsource
from src.design.env import Environment
from src.config.device_info import config_benchmark
from src.optim.util import objective, constraint
from src.optim.rl.constrained.rcpo import search_param_space, ReplayBuffer, ActorCritic
from src.analysis.util import find_optimal_design
from src.design.util import save_design
import pickle, torch, argparse, os, warnings

warnings.filterwarnings(action="ignore")

def parsing():
    parser = argparse.ArgumentParser(description="Tokamak design optimization based on constrained reinforcement learning algorithm")

    # Setup
    parser.add_argument("--num_episode", type=int, default=10000)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--n_proc", type=int, default=4)

    parser.add_argument("--buffer_size", type=int, default=5)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--std", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_lamda", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=0.1)

    # directory
    parser.add_argument("--save_dir", type=str, default="./results/crl")

    args = vars(parser.parse_args())

    return args

# torch device state
print("=============== Device setup ===============")
print("torch cuda avaliable : ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("torch current gpu : ", torch.cuda.current_device())
    print("torch available gpus : ", torch.cuda.device_count())

    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()
else:
    print("torch current gpu : None")
    print("torch available gpus : None")
    print("CPU computation")

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

    # Setup
    memory = ReplayBuffer(capacity = args['buffer_size'])
    criterion = torch.nn.SmoothL1Loss(reduction="none")
    policy_network = ActorCritic(input_dim = 21, mlp_dim = args['mlp_dim'], n_actions = 9, std = args['std'])
    policy_optimizer = torch.optim.RMSprop(policy_network.parameters(), lr = args['lr'])

    log_lamda = torch.nn.Parameter(torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])), requires_grad=True)
    lamda_optimizer = torch.optim.SGD([log_lamda], lr=args["lr_lamda"])

    # Design optimization
    print("============ Design optimization ============")
    result = search_param_space(
        env,
        objective,
        constraint,
        memory,
        policy_network,
        policy_optimizer,
        log_lamda,
        lamda_optimizer,
        criterion,
        args['gamma'],
        args['eps_clip'],
        args['entropy_coeff'],
        "cpu",
        None,
        None,
        args['num_episode'],
        args['verbose'],
        args["n_proc"],
        args['sample_size']
    )

    with open(save_result, "wb") as file:
        pickle.dump(result, file)

    optimal = find_optimal_design(result)
    
    if optimal is not None:
        save_design(optimal, args["save_dir"], "optimal_config.pkl")
