from main import take_photo, save
import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

# COCO classes
CLASSES = [
    'carrot',
    'cucumber',
    'grapes',
    'lettuce',
    'lime'  ,
    'tomato',
    'watermelon'
]

supply = {      # min max
    'apple': (0, 0),
    'banana': (0, 0),
    'orange': (0, 0),
    'carrot':   (0, 0),
    'cucumber': (0, 0),
    'grapes':   (0, 0),
    'lettuce':  (0, 0),
    'lime':     (0, 0),
    'tomato':   (0, 0),
    'watermelon': (0, 0)
}

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

def detect(photo_path):
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
    im = Image.open(photo_path)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    for p in probas[keep]:
        local_supply[CLASSES[p.argmax()]] += 1
    for el in local_supply:
        supply[el] = (min(supply[el][0], local_supply[el]), max(supply[el][1], local_supply[el]))

        
  # convert boxes from [0; 1] to image scales
#   bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
#   plot_results(im, probas[keep], bboxes_scaled)

def detect_save():
    img_path = take_photo(0)
    detect(img_path)
    save('.')

# checkpoint = torch.hub.load_state_dict_from_url(
#                 url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
#                 map_location='cpu',
#                 check_hash=True)

# torch.save(checkpoint,
#                'detr-r50_no-class-head.pth')

# model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)