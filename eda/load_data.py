import os
import yaml
import argparse
import fiftyone as fo
import fiftyone.zoo as foz

def parse_config(config_path):
    with open(config_path) as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)['Augmentation']
    
    overall_prob = config_data['OverallProb']
    flips = config_data['Flips']
    rotation = config_data['Rotation']
    color = config_data['Color']

    return overall_prob, flips, rotation, color


def main(args):
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=config['LabelTypes'],
        classes=config['Classes'],
        max_samples=1000,
    )

    val_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=config['LabelTypes'],
        classes=config['Classes'],
        max_samples=100,
    )

    if config['Action'] == 'explore':
        session = fo.launch_app(train_dataset)

        session.wait()
    elif config['Action'] == 'download':
        try:
            os.mkdir('data')
        except:
            print('Folder already existed')

        train_dataset.export(
            dataset_type=fo.types.COCODetectionDataset,
            export_dir="./data/train",
            label_field="ground_truth",
        )

        val_dataset.export(
            dataset_type=fo.types.COCODetectionDataset,
            export_dir="./data/val",
            label_field="ground_truth",
        )
    else:
        print('[ERROR] Invalid Action Type')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate and Download Dataset')
    parser.add_argument('--config', metavar='path', required=True, help='the path to the config yaml file')
    args = parser.parse_args()
    main(args)