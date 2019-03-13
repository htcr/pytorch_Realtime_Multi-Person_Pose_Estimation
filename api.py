import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from network.rtpose_vgg import get_model
from network.post import decode_pose, decode_pose_fg
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat


class OpenPoseForeGroundMarker(object):
    def __init__(self, max_size=600):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        weight_name = os.path.join(cur_dir, 'pose_model_scratch.pth')
        assert os.path.exists(weight_name), 'open pose model not found at {}'.format(weight_name)
        self.model = get_model('vgg19')
        state_dict = torch.load(weight_name)
        # remove 'module.' prefix
        state_dict = {k[7:] : v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()
        self.max_size=max_size

    def get_working_size(self, oh, ow):
        max_edge = max(oh, ow)
        scale = float(self.max_size) / max_edge
        nh, nw = scale * oh, scale * ow
        nh, nw = int(nh), int(nw)
        return nh, nw

    def infer_fg(self, img):
        """
        img: BGR image of shape (H, W, C)
        returns: binary mask image of shape (H, W), 255 for fg, 0 for bg
        """
        ori_h, ori_w = img.shape[0:2]
        new_h, new_w = self.get_working_size(ori_h, ori_w)
        img = cv2.resize(img, (new_w, new_h))
        
        # Get results of original image
        multiplier = get_multiplier(img)

        with torch.no_grad():
            orig_paf, orig_heat = get_outputs(
                multiplier, img, self.model,  'rtpose')
                
            # Get results of flipped image
            swapped_img = img[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                    self.model, 'rtpose')

            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(
                orig_heat, flipped_heat, orig_paf, flipped_paf)
                    
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        to_plot, canvas, candidate, subset = decode_pose_fg(
            img, param, heatmap, paf)
        
        canvas = cv2.resize(canvas, (ori_w, ori_h))
        fg_map = canvas > 128
        canvas[fg_map] = 255
        canvas[~fg_map] = 0
        return canvas[:, :, 0]

