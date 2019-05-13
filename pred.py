import torch
import torch.nn as nn
from torch.autograd import Variable
from data import BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import math
import time
import argparse
import numpy as np
import cv2
import PIL.Image
import PIL.ImageDraw


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Prediction')
parser.add_argument('--model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('infile', nargs='*')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)


class ResultImage:
    def __init__(self, img, class_list):
        self.pil_img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.class_list = class_list

    def show_img(self):
        self.pil_img.show()

    _COLORS = np.array(
            [[0,0,1], [0,1,1], [0,1,0], [1,1,0], [1,0,0], [1,0,1], [0,0,1]],
            dtype=int)

    def _get_color(self, class_idx):
        i = class_idx % 6
        a = class_idx // 6
        if a == 0:
            r = 0
        else:
            div = 2 ** math.floor(math.log2(a))
            r = (a - div + 0.5) / div
        col_i = self._COLORS[i]
        col_j = self._COLORS[i + 1]
        return tuple((((1 - r) * col_i + r * col_j) * 255).astype(int))

    def add_box(self, label, det):
        img_w = self.pil_img.width
        img_h = self.pil_img.height
        class_idx = self.class_list.index(label)
        mask = PIL.Image.new("L", self.pil_img.size, 255)
        mask_draw = PIL.ImageDraw.Draw(mask)
        mask_draw.rectangle(
                (det[1] * img_w + 0.5, det[2] * img_h + 0.5,
                 det[3] * img_w + 0.5, det[4] * img_h + 0.5),
                fill = 192)

        filler = PIL.Image.new("RGB", self.pil_img.size,
                               self._get_color(class_idx))
        self.pil_img = PIL.Image.composite(self.pil_img, filler, mask)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if args.cuda:
        net = net.cuda()

    for fn in args.infile:
        print(fn)
        img = cv2.imread(fn)
        h, w = img.shape[:2]
        img_f, _1, _2 = BaseTransform(300, dataset_mean)(img)
        img_f = img_f[:,:,(2, 1, 0)]
        im = torch.from_numpy(img_f).permute(2, 0, 1)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        start_time = time.time()
        detections = net(x).data
        detect_time = time.time() - start_time

        result = ResultImage(img, labelmap)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            for det in dets:
                if det[0] < args.threshold:
                    continue
                print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.
                        format(labelmap[j - 1], det[0],
                               det[1] * w + 1, det[2] * h + 1,
                               det[3] * w + 1, det[4] * h + 1))
                result.add_box(labelmap[j - 1], det)

        result.show_img()
        print('done prediction: {:.3f}s'.format(detect_time))
