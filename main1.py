import cv2
import numpy as np
import torch
import time
import torchvision.transforms.functional as TF
from PIL import Image
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 
model.conf = 0.80
f="E:/Hackathon/HACKTRICS/t2"
cap = cv2.VideoCapture(0)
a=0
j=0
while True:
    ret, frame = cap.read()
    results = model(frame)
    humans=results.pred[0] == 0
    cv2.imshow('YOLOv5', results.render()[0])
    if True in humans[0:5]:
        for i, (pred, pred_label) in enumerate(zip(results.xyxy[0], results.names[0])):
            x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            obj_crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f+"/v1"+str(j)+".jpg", obj_crop)
            cv2.waitKey(1000)
            j+=1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
