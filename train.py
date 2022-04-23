import os
import sys
import argparse

from pathlib import Path

from utils.general import parse_train_config, get_latest_run, load_weights, LOGGER

def train(config, weights):
    save_dir, epochs, batch_size = Path(config['save-dir']), config['epochs'], config['batch-size']

    weights_dir = save_dir / 'weights'
    last, best = weights_dir / 'last.pt', weights_dir / 'best.pt'

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
    
    # Things to Implement:
    # Evolving Hyperparameters
    # DDP (Distributed Data Parallel)

    train(config, weights)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, required=True, help='the path to train parameter config')
    args = parser.parse_args()

    main(args)