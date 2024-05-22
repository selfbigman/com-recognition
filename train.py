import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s-ASF.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='E:/MYP/ultralytics-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # lr0 = 0.01,
                # momentum = 0.9,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='v8s-asf',
                )