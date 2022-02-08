import yaml
import os

def read_config_file(filepath = '../src/config_train_DR.yaml', verbose=False):
    with open(filepath) as file:
        yaml_data = yaml.safe_load(file)

    if(verbose):
        print(yaml_data)
    
    return yaml_data