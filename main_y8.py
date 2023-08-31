from ultralytics import YOLO
import cv2
import cvzone
import math
from tools import *

DET_TIMES = 5
INF = math.inf

classNames = ['apple','banana','orange','carrot','cucumber','grapes',
'lettuce','lime','tomato','watermelon']

supply = {      # min max
    'apple':     (INF, 0),
    'banana':    (INF, 0),
    'orange':    (INF, 0),
    'carrot':    (INF, 0),
    'cucumber':  (INF, 0),
    'grapes':    (INF, 0),
    'lettuce':   (INF, 0),
    'lime':      (INF, 0),
    'tomato':    (INF, 0),
    'watermelon': (INF, 0)
}

def detecty8(img_path, model_name='yolov8n.pt', detec_num=0):
    local_supply = {
        'apple': 0,
        'banana': 0,
        'orange': 0,
        'carrot':   0,
        'cucumber': 0,
        'grapes':   0,
        'lettuce':  0,
        'lime':     0,
        'tomato':   0,
        'watermelon': 0
    }

    model = YOLO(model_name)
    results = model(img_path)
    np_img = cv2.imread(img_path)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # print(box)
            cls = int(box.cls[0])
            np_img = draw(np_img, r.names, box.xyxy[0], box.conf[0], cls)
            print(r.names[cls])
            local_supply[r.names[cls]] +=1

    save_img(f'det_img{detec_num}.jpg', np_img)

    for el in local_supply:
        supply[el] = (min(supply[el][0], local_supply[el]), max(supply[el][1], local_supply[el]))

def detecty8_save(address, i):
    #take photo
    img = take_photo(address)
    img_path = save_img(f'original{i}.jpg', img)
    #detec
    detecty8(img_path, detec_num=i)
    #save
    save('.', supply)

for i in range(DET_TIMES):
    detecty8_save(0, i)