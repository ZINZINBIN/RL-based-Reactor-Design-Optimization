import pickle, os
from typing import Dict
from src.config.device_info import config_benchmark

params = config_benchmark.keys()

def read_design(filepath:str):
    
    if not os.path.exists(filepath):
        print("Error | {} configuration file doesn't exist".format(filepath))
        return None

    with open(filepath, "rb") as f:
        config = pickle.load(f)

    new_config = {}

    for key in params:
        if key not in config.keys():
            new_config[key] = config_benchmark[key]
        else:
            new_config[key] = config[key]
            
    print("Load design config: complete")

    return config

def save_design(config:Dict, savepath:str, filename:str):

    save_config = {}
    filepath = os.path.join(savepath, filename)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for key in params:
        if key not in config.keys():
            save_config[key] = config_benchmark[key]
        else:
            save_config[key] = config[key]

    with open(filepath, "wb") as f:
        pickle.dump(save_config, f)
        
    print("Save design config: complete")
