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

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from Pose2Mesh.smplifyx.utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with np.load(keypoint_fn + '_hp.npz') as keypoint_file:
        hp_x = keypoint_file['x']
        hp_y = keypoint_file['y']
        #hp_x = 2. * (hp_x - np.min(hp_x)) / np.ptp(hp_x) - 1
        #hp_y = (2. * (hp_y - np.min(hp_y)) / np.ptp(hp_y) - 1) #* (-1.0)
    with np.load(keypoint_fn + '_f.npz') as keypoint_file:
        f_x = keypoint_file['x']
        f_y = keypoint_file['y']
        #f_x = 2. * (f_x - np.min(f_x)) / np.ptp(f_x) - 1
        #f_y = (2. * (f_y - np.min(f_y)) / np.ptp(f_y) - 1) #* (-1.0)

    keypoints = []

    gender_pd = []
    gender_gt = []

    hpz = np.expand_dims(np.ones_like(hp_x), -1)
    hpxyz = np.concatenate((np.expand_dims(hp_x, -1), np.expand_dims(hp_y, -1), hpz), -1)
    fz = np.expand_dims(np.ones_like(f_x), -1)
    fxyz = np.concatenate((np.expand_dims(f_x, -1), np.expand_dims(f_y, -1), fz), -1)


    body_keypoints = np.zeros((25, 3),
                              dtype=np.float32)
    body_keypoints[:15, :] = hpxyz[:15, :]
    body_keypoints = body_keypoints.reshape([-1, 3])
    if use_hands:
        left_hand_keyp = np.zeros((21, 3),
                              dtype=np.float32)

        left_hand_keyp[:,:] = hpxyz[15:15+21, :]

        right_hand_keyp = np.zeros((21, 3),
                              dtype=np.float32)
        right_hand_keyp[:, :] = hpxyz[15 + 21:, :]

        body_keypoints = np.concatenate(
            [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
    if use_face:
        # TODO: Make parameters, 17 is the offset for the eye brows,
        # etc. 51 is the total number of FLAME compatible landmarks
        face_keypoints = fxyz.reshape([-1, 3])[17: 17 + 51, :]

        contour_keyps = np.array(
            [], dtype=body_keypoints.dtype).reshape(0, 3)
        if use_face_contour:
            contour_keyps = fxyz.reshape([-1, 3])[:17, :]

        body_keypoints = np.concatenate(
            [body_keypoints, face_keypoints, contour_keyps], axis=0)

        #body_keypoints[:, 0] = 2. * (body_keypoints[:, 0] - np.min(body_keypoints[:, 0])) / np.ptp(
        #    body_keypoints[:, 0]) - 1
        #body_keypoints[:, 1] = 2. * (body_keypoints[:, 1] - np.min(body_keypoints[:, 1])) / np.ptp(
        #    body_keypoints[:, 1]) - 1

    keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn)
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 'img': img}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)
