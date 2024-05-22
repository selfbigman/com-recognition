import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/v8s-goldyolo-asf3/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='v8s-goldyolo-asf',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )