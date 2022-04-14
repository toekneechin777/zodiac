import yaml

DEFAULT_TRAIN = 'configs/train_default.yaml'

def parse_train_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    with open(DEFAULT_TRAIN) as file:
        default_config = yaml.load(file, Loader=yaml.FullLoader)

    # Set default if not specified
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
    
    return config
