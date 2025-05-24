# Our-paper
This is the official implementation of the paper "CCS-TR: Automated Complex Construction Scenes Detection via a Novel Lightweight Transformer-based method"


## Modules
Relevant improved modules implementation are in `./model.py`.


## Data
The data we used are SODA and VisDrone2019 datasets (public datasets), which can also be acquired from this [google_drive](./)


## Train
```
pip install -r requirements.txt
python main.py --batch 16 --epoch 400
```

## Inference/Val
```
python val.py --checkpoint "your trained weights"
python inference.py --checkpoint "your trained weights"
```

## Note
Relevent source will be released after acceptance.
