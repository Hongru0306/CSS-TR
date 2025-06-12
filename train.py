import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/CSSTR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/SODA.yaml',
                cache=False,
                imgsz=640,
                epochs=450,
                batch=28,
                workers=4,
                device='0, 1',
                resume=False, # last.pt path
                project='runs/train',
                name='exp',
                )