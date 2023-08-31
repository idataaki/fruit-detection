# python inference.py -m models\yolov5_9cls.pt -n 3

import argparse
import math
import torch
import csv
import cvzone
import cv2
import math
import os
from tools import *
from ultralytics import YOLO

root = os.getcwd()
INF = math.inf
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
#-------------------------------------v5----------------------------------------
def detecty5(path_to_yolo5, img_path, model_name='yolov5n.pt', detec_num = 0):
    local_supply = {'apple':0,'banana':0,'orange':0,'carrot':0,
        'cucumber':0,'grapes':0,'lettuce':0,'lime':0,'tomato':0,'watermelon':0}

    model = torch.hub.load( path_to_yolo5, 'custom', path = model_name,
                            source ='local',force_reload = True)

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

    save_img(f'det_img{detec_num}.jpg', np_img)

    for el in local_supply:
        supply[el] = (min(supply[el][0], local_supply[el]), max(supply[el][1], local_supply[el]))

def detecty5_and_save(model_name, i):
    #take photo
    img = take_photo(0)
    img_path = save_img(f'original{i}.jpg', img)
    #detec
    res = detecty5(f'{root}\yolov5', img_path, model_name=model_name, detec_num=i)
    #save
    save('.', supply)
#-------------------------------------------------------------------------------

#-------------------------------------v8----------------------------------------
classNames = ['apple','banana','orange','carrot','cucumber','grapes',
'lettuce','lime','tomato','watermelon']

def detecty8(img_path, model_name='yolov8n.pt', detec_num=0):
    local_supply = {'apple':0,'banana':0,'orange':0,'carrot':0,
        'cucumber':0,'grapes':0,'lettuce':0,'lime':0,'tomato':0,'watermelon':0}

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

def detecty8_save(model_path, i):
    #take photo
    img = take_photo(0)
    img_path = save_img(f'original{i}.jpg', img)
    #detec
    detecty8(img_path, model_name=model_path, detec_num=i)
    #save
    save('.', supply)
#-------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("fruit detection inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="path to model",
    )
    parser.add_argument(
        "-n",
        "--nphoto",
        type=int,
        help="number of photos to be taken",
    )
    args = parser.parse_args()
    return args


def main(opt):
    model_path = opt.model
    if 'v5' in model_path:
        for i in range(int(opt.nphoto)):
            detecty5_and_save(model_path, i)
    elif 'v8' in model_path:
        for i in range(int(opt.nphoto)):
            detecty8_save(model_path, i)
    render = f"python {root}\\html\\render.py"
    cd = f"cd {root}\html"
    server = "python -m http.server 9000"
    os.system(f'cmd /k "{render} & {cd} & {server}"')


if __name__ == "__main__":
    options = get_args()
    main(options)