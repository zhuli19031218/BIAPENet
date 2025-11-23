import torch

from ultralytics import YOLO
import torch.nn.functional as F

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"runs/detect/SeaPerson_weight/improved_best.pt")  # build a new model from scratch
    # model = YOLO(r"E:\cbd\ICCV\improve_yolo\runs\detect\TinyPerson_vsdenet_normal\weights\best.pt")  # load a pretrained model (recommended for training)
    #
    # # Use the model
    # # model.train(data=r"E:\cbd\ICCV\improve_yolo\ultralytics\cfg\datasets\SeaPerson.yaml")  # train the model
    metrics = model.val(data=r"E:\cbd\ICCV\improve_yolo\ultralytics\cfg\datasets\SeaPerson.yaml", split="val", save_json=False)  # evaluate model performance on the validation set
    # metrics = model.predict(source=r"E:\cbd\ICCV\datasets\SeaPerson\test\images\440.jpg", conf=0.25, save=True)
    # # results = model(r"E:\cbd\ICCV\datasets\VisDrone\test\images")  # predict on an image
    # # for i, result in enumerate(results):
    # #     result.save(filename='result_{0}.jpg'.format(i))  # save to disk
    #
    # # path = model.export(format="onnx")  # export the model to ONNX format
    # # results = model.train(resume=True)
    # pred = model.predict(r"E:\cbd\ICCV\improve_yolo\1.mp4")

