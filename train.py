import os
import sys
import argparse

from pathlib import Path

from utils.general import parse_train_config, get_latest_run, load_weights, init_seeds, check_dataset, LOGGER

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train(config, weights):
    save_dir, epochs, batch_size = Path(config['save-dir']), config['epochs'], config['batch-size']

    # Where to output weights
    weights_dir = save_dir / 'weights'
    last, best = weights_dir / 'last.pt', weights_dir / 'best.pt'

    # Load Hyperparameters
    hyp = config['hyperparameters']

    # Things to Implement:
    # Evolving Hyperparameters
    # DDP (Distributed Data Parallel)

    #Config
    cuda = config['device-type'] != 'cpu'
    init_seeds(1+RANK)
    check_dataset(config['data'])

    # Model

    # Freeze Layers

    # Image Size

    # Batch Size

    # Optimizer

    # Scheduler

    # Resume

    # Data Train Loader

    # Start Training


def main(args):
    config = parse_train_config(args.train_config)
    LOGGER.info(f"Config: {config}")

    # Check for Resume Training if possible
    if config['resume']:
        checkpoint = config['resume'] if isinstance(config['resume'], str) else get_latest_run()
        assert os.path.isfile(checkpoint), '[ERROR] resume option does not exist'
        LOGGER.info(f"Resume training from {checkpoint}")
        weights = checkpoint
    # Otherwise load default weights
    else:
        weights = load_weights(config['models'])

    train(config, weights)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, required=True, help='the path to train parameter config')
    args = parser.parse_args()

    main(args)