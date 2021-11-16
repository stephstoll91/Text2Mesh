# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

from utils import JointMapper
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False
from smplifyx import SmplifyX
from Text2Pose.helpers import load_config

def create_smplifyx(cfg: dict = None):
    return SmplifyX(cfg)

if __name__ == "__main__":
    cfg_file = '/home/stephanie/Documents/Chapter3/Code/Text2Pose/smplify-x/cfg_files/fit_smplx.yaml'
    cfg = load_config(cfg_file)
    smplifyx = create_smplifyx(cfg)

    print("hi")
