import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    model = YOLO(r'C:\Users\Jaz05\Desktop\lunwenfudao\BIAPENet\BIAPENet\BIAPENet\improve_yolo\ultralytics\cfg\models\v8\impriove_yolo.yaml')
    # model.load('') # loading pretrain weights

    model.train(data=r'C:\Users\Jaz05\Desktop\lunwenfudao\BIAPENet\BIAPENet\BIAPENet\improve_yolo\yolo_3dshibie.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=1,
                # close_mosaic=10,
                workers=0,
                patience=0,
                device='0',
                # iou=0.5,
                optimizer='SGD', # using SGD
                resume=False, # 断点续训,YOLO初始化时选择last.pt
                # resume=r'', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
# tryIoU 0.9