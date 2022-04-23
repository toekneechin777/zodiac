import yaml
import logging
import glob
import os

from downloads import attempt_download

DEFAULT_TRAIN = 'configs/train_default.yaml'

level = logging.INFO
LOGGER = logging.getLogger("train")
LOGGER.setLevel(level)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s: %(message)s"))
handler.setLevel(level)
LOGGER.addHandler(handler)

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

def load_weights(models):
    weights = []
    for model in models:
        for model_yaml in models[model]:
            with open(model_yaml) as file:
                model_config = yaml.load(file, Loader= yaml.FullLoader)
            
            if(not os.path.isfile(model_config['weights'])):
                LOGGER.info(f'Downloading weights...')
                if(attempt_download(model, model_config['weights'])):
                    weights.append(model_config['weights'])
            
    return weights

 
def get_latest_run(search_dir = './experiments/'):
    last_list = glob.glob(f'{search_dir}/**/latest*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''
    
