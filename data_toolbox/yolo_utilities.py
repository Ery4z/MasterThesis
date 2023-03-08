import matplotlib
import matplotlib.pyplot as plt
import torch

def get_yolo():
    '''At the time of writign this a bug happen after importing yolo (impossible to plot with matplotlib) this is the fix'''
        

    b = plt.get_backend()
    model = torch.hub.load("ultralytics/yolov5", "yolov5s",verbose=False)
    matplotlib.use(b)
    return model

global yolo 
yolo = get_yolo()