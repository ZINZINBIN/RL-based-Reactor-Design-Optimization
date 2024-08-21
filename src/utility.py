import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from typing import List, Optional, Literal, Dict, Union
from config.device_info import config_benchmark

from src.device import Tokamak
from src.profile import Profile
from src.source import CDsource

def find_optimal_case(result:Dict, args:Dict):
    
    tau = [comp['tau'] for comp in result['state']]
    b_limit = result['beta_limit']
    q_limit = result['q_limit']
    n_limit = result['n_limit']
    f_limit = result['f_limit']
    i_limit = result['i_limit']
    
    tbr = np.array([result['state'][idx]['TBR'] for idx in range(len(tau))])
    T = np.array([result['control'][idx]['T_avg'] for idx in range(len(tau))])

    tau = np.array(tau)
    b_limit = np.array(b_limit)
    q_limit = np.array(q_limit)
    n_limit = np.array(n_limit)
    f_limit = np.array(f_limit)
    i_limit = np.array(i_limit)

    indices = np.where(((b_limit == 1) * (q_limit == 1) * (n_limit == 1) * (f_limit == 1) * (tbr >= 1)) == 1)
    print("# of optimal cases : {}".format(len(indices[0])))
    
    if len(indices[0]) == 0:
        print("No optimal cases found...!")
        return
    
    # Cost parameter
    cost = np.array([result['state'][idx]['cost'] for idx in indices[0]])

    # Choose the minimum cost case
    arg_min = np.argmin(cost)
    arg_min = indices[0][arg_min]

    for key in result['state'][arg_min].keys():
        print("{} : {:.3f}".format(key, result['state'][arg_min][key]))
        
    for key in result['control'][arg_min].keys():
        print("{} : {:.3f}".format(key, result['control'][arg_min][key]))
        
    # add config
    config = config_benchmark.copy()
    
    for key in result['control'][arg_min].keys():
        config[key] = result['control'][arg_min][key]
        
    for key in result['state'][arg_min].keys():
        config[key] = result['state'][arg_min][key]
  
    # print overall performance
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
    
    del tokamak
    del source
    del profile
    

def temperal_average(X:np.array, Y:np.array, k:int):
    
    clip_length = X.shape[0] // k
    
    X_mean = np.zeros(clip_length)
    
    Y_mean = np.zeros(clip_length)
    Y_lower = np.zeros(clip_length)
    Y_upper = np.zeros(clip_length)
    
    for i in range(clip_length):
        
        idx_start = i * k 
        idx_end = (i+1) * k 
        
        if idx_end >= X.shape[0]:
            idx_end = X.shape[0] - 1
            
        X_mean[i] = int(np.mean(X[idx_start:idx_end]))
        Y_mean[i] = np.mean(Y[idx_start:idx_end])
        Y_lower[i] = np.min(Y[idx_start:idx_end])
        Y_upper[i] = np.max(Y[idx_start:idx_end])
    
    return X_mean, Y_mean, Y_lower, Y_upper

def plot_policy_loss(
    loss_list:List, 
    buffer_size : int, 
    temporal_length:int = 8, 
    save_dir : Optional[str] = None,
    ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    loss = np.repeat(np.array(loss_list).reshape(-1,1), repeats = buffer_size, axis = 1).reshape(-1,)
    episode = np.array(range(1, len(loss)+1, 1))
    
    # clip the invalid value
    loss = np.clip(loss, a_min = -2.0, a_max = 5.0)
    
    # temperal average
    x_mean, loss_mean, loss_lower, loss_upper = temperal_average(episode, loss, temporal_length)
    
    fig = plt.figure(figsize = (8,4))
    
    clr = plt.cm.Purples(0.9)
    
    plt.plot(x_mean, loss_mean, c = 'r', label = '$<loss_t>$')
    plt.fill_between(x_mean, loss_lower, loss_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
    
    plt.xlabel("Episodes")
    plt.ylabel("Policy loss")
    plt.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_loss.png"), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    
    fig.clear()
    

# print the result of overall optimization process
def plot_optimization_status(
    optimization_status:Dict, 
    temporal_length:int = 8, 
    save_dir : Optional[str] = None,
    ):
    
    '''
        optimization_status: Dict[key, value]
        - key: obj-1, obj-2, .... (ex) q95, fbs, beta, ...., total
        - value: List type of reward with respect to each episode
        
        smoothing_k : n_points for moving average process
        smoothing_method: backward or center
    '''
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if temporal_length == 1:
        print("buffer size = 1 | the default value 8 is automatically selected")
        temporal_length = 8
    
    for idx, key in enumerate(optimization_status.keys()):
        
        obj_reward = np.array(optimization_status[key])
        episode = np.array(range(1, len(obj_reward)+1, 1))
        
        x_mean, obj_reward_mean, obj_reward_lower, obj_reward_upper = temperal_average(episode, obj_reward, temporal_length)
        
        fig = plt.figure(figsize = (8,4))
       
        clr = plt.cm.Purples(0.9)
       
        plt.plot(x_mean, obj_reward_mean, c = 'r', label = '$<r_t>$:{}'.format(key))
        plt.fill_between(x_mean, obj_reward_lower, obj_reward_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
        
        plt.xlabel("Episodes")
        plt.ylabel("Reward:{}".format(key))
        plt.legend(loc = 'upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "reward_history_{}.png".format(key)), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
        
        fig.clear()