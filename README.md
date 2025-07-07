# Automated Detection of Complex Construction Scenes Using a Lightweight Transformer-based Method
This is the implementation of the paper "Automated Detection of Complex Construction Scenes Using a Lightweight Transformer-based Method"

<figure style="text-align: center;">
  <img src="./configs/framework.jpg" alt="img" style="width: 80%; height: auto; max-width: 100%;">
  <figcaption>Overall view of the Complex Construction Scenes Transformer (CSS-TR)</figcaption>
</figure>


# Installation
This code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Please install the required dependencies, and you can also refer to their preparation tutorials as a reference.

## Modules
Relevant improved modules implementation are in `./modules.py`.


## Data prepation
Some data prepation scripts can be found in `utils/data_process`.  
The data we used are SODA and VisDrone2019 datasets (public datasets).    
Relevant weights of the proposed model can be acquired from this [google_drive](https://drive.google.com/drive/folders/1Qq_Ks48a_tCyX_gPF9LKJ-SXY-aVdF7K?usp=drive_link).  


## Train
```
pip install -r requirements.txt
python train.py
```

## Val
```
python val.py
```

## Citation
If our repo helps your work, please cite:
```
@article{XIAO2025106330,
title = {Automated detection of complex construction scenes using a lightweight transformer-based method},
journal = {Automation in Construction},
volume = {177},
pages = {106330},
year = {2025},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2025.106330},
url = {https://www.sciencedirect.com/science/article/pii/S092658052500370X},
author = {Hongru Xiao and Bin Yang and Yujie Lu and Wenshuo Chen and Songning Lai and Biaoli Gao},
keywords = {Transformer-based object detection, Complex construction scenes, Scale-isolate fusion attention, Instructive contrastive learning}
}
```
  
If you have any questions, please feel free to contact me at 'hongru_xiao@tongji.edu.cn'.
