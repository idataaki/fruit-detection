import torch
import csv
import cvzone
import cv2
import math
import os


# draw a rec 
def draw(img, res_names, box, conf=0.0, clsn=0.0):
    if len(box) == 4:
        x1, y1, x2, y2 = box
    else:
        x1, y1, x2, y2, conf, clsn = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    cvzone.cornerRect(img, (x1, y1, w, h))
    # Confidence
    conf = math.ceil((conf* 100)) / 100
    # Class Name
    clsn = int(clsn)
    cvzone.putTextRect(img, f'{res_names[clsn]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
    return img 

# take a photo
def take_photo(address):
    cap = cv2.VideoCapture(address)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.read()
    print("WILL BE TAKEN IN ...")
    for i in range(3):
        print("\t", i+1)
        cv2.waitKey(1000)
    success, img = cap.read()
    return img

def save_img(img_path, img):
    cv2.imwrite(img_path, img)
    return img_path

def save(output, supply):
    with open(output+'\database.csv', 'w') as db:
        csvr = csv.writer(db, delimiter=',')
        for key in supply:
            csvr.writerow([key, supply[key][0], supply[key][1]])

