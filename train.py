
import argparse
from utils import parse_train_config

def main(args):
    config = parse_train_config(args.train_config)

    print(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, required=True, help='the path to train parameter config')
    args = parser.parse_args()

    main(args)