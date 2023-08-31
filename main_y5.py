import torch
import csv
import cvzone
import cv2
import math
import os
from tools import *

DET_TIMES = 5
root_path = os.getcwd()
print(root_path)


supply = {      # min max
    'apple':        (0, 0),
    'banana':       (0, 0),
    'orange':       (0, 0),
    'carrot':       (0, 0),
    'cucumber':     (0, 0),
    'grapes':       (0, 0),
    'lettuce':      (0, 0),
    'lime':         (0, 0),
    'tomato':       (0, 0),
    'watermelon':   (0, 0)
}


def detecty5(path_to_yolo5, img_path, model_name='yolov5n.pt', detec_num = 0):
    local_supply = {
        'apple':        0,
        'banana':       0,
        'orange':       0,
        'carrot':       0,
        'cucumber':     0,
        'grapes':       0,
        'lettuce':      0,
        'lime':         0,
        'tomato':       0,
        'watermelon':   0
    }

    model = torch.hub.load( path_to_yolo5, 'custom',
                            path='models\\' + model_name,
                            source='local',
                            force_reload=True
                            )

    results = model(img_path)
    np_img = cv2.imread(img_path)

    for r in results.tolist():
        for box in r.xyxy[0]:
            #draw
            clas = box[5]
            np_img = draw(np_img, r.names, box)
            #update supply
            print(r.names[int(clas)])
            local_supply[r.names[int(clas)]] += 1
    # cv2.imwrite(f'img{index}.jpg',np_img)
    save_img(f'det_img{detec_num}.jpg', np_img)

    for el in local_supply:
        supply[el] = (min(supply[el][0], local_supply[el]), max(supply[el][1], local_supply[el]))

    return local_supply

def detecty5_and_save(model_name, i):
    #take photo
    img = take_photo(0)
    img_path = save_img(f'original{i}.jpg', img)
    #detec
    res = detecty5('D:\project\codes\supply_managment\yolov5', img_path, model_name=model_name, detec_num=i)
    #save
    save('.', supply)

for i in range(DET_TIMES):
    print(supply)
    detecty5_and_save('best.pt', i)