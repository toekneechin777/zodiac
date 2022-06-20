# ZODIAC

### Zoo of Object Detection models In A Click

One Click Deploy for Training and Inference of Object Detection Models

```
conda create -n zodiac python=3
conda activate zodiac
pip3 install -r requirements.txt
```

 ### Download Dataset

 To download a sample of the coco dataset go to the eda folder and run:

```
python3 load_data.py tiny_coco.yaml
```

You need to make sure you have fiftyone installed which should be in the requirements.txt