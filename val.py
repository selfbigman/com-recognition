import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/v8s/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='v8s',
              )