from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource
from src.env import Enviornment
from src.rl.ppo import train_ppo, ActorCritic, ReplayBufferPPO
from src.rl.reward import RewardSender
from config.device_info import config_benchmark, config_liquid
import torch
import pickle
import argparse
import os

# torch cuda setting
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 
    
if __name__ == "__main__":
    
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
    
    reward_sender = RewardSender(
        w_cost = 0.5,
        w_tau = 0.25,
        w_beta = 1.0,
        w_density=1.0,
        w_q = 0.5,
        w_bs = 0.5,
        w_i = 1.0,
        cost_r = 1.0,
        tau_r = 1.0,
        beta_r = 1.0,
        q_r = 1.0,
        n_r = 1.0,
        f_r = 1.0,
        i_r = 1.0
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
    
    # policy and value network
    policy_network = ActorCritic(input_dim = 19 + 9, mlp_dim = 128, n_actions = 9, std = 0.1)
    
    # gpu allocation
    policy_network.to(device)
    
    # optimizer    
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = 0.001)

    # loss function for critic network
    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    # memory
    memory = ReplayBufferPPO(1000)
    
    import numpy as np
    result = train_ppo(
        env, 
        memory,
        policy_network,
        policy_optimizer,
        value_loss_fn,
        0.995,
        0.1,
        0.1,
        device,
        10000,
        128,
        "./weights/ppo_best.pt",
        "./weights/ppo_last.pt",
    )
    
    with open('./results/params_search_ppo.pickle', 'wb') as file:
        pickle.dump(result, file)
        
    env.close()