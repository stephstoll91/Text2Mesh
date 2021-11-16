from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple
import numpy as np
import torch

from smplifyx.smplifyx.utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(hp_kp, f_kp, use_hands=True, use_face=True,
                   use_face_contour=False):

    device = hp_kp.device
    hp_kp = hp_kp * 3 * 10 * 12 * 2
    f_kp = f_kp * 3 * 10 * 10

    hp_offset = torch.Tensor([250, 300]).to(device)
    hp_kp = hp_kp + torch.ones((50, 2)).to(device) * hp_offset

    f_offset = torch.Tensor([300, 100]).to(device)
    f_kp = f_kp + torch.ones((70, 2)).to(device) * f_offset

    pose = hp_kp[:8, :]
    missing = torch.cat((torch.ones((1, 2)).to(device) * torch.Tensor([350, 700]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([300, 700]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([300, 900]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([300, 1100]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([400, 700]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([400, 900]).to(device),
                              torch.ones((1, 2)).to(device) * torch.Tensor([400, 1100]).to(device)
                              ), 0)
    hp_kp = torch.cat((pose, missing, hp_kp[8:, :]), 0)

    hpz = torch.ones_like(hp_kp[:, :1]).to(device)
    hpxyz = torch.cat((hp_kp, hpz), -1)
    fz = torch.ones_like(f_kp[:, :1]).to(device)
    fxyz = torch.cat((f_kp, fz), -1)

    body_keypoints = torch.zeros((25, 3),
                              dtype=torch.float32).to(device)
    body_keypoints[:15, :] = hpxyz[:15, :]
    body_keypoints = body_keypoints.reshape([-1, 3])
    if use_hands:
        left_hand_keyp = torch.zeros((21, 3),
                              dtype=torch.float32).to(device)

        left_hand_keyp[:,:] = hpxyz[15:15+21, :]

        right_hand_keyp = torch.zeros((21, 3),
                              dtype=torch.float32).to(device)
        right_hand_keyp[:, :] = hpxyz[15 + 21:, :]

        body_keypoints = torch.cat(
            (body_keypoints, left_hand_keyp, right_hand_keyp), 0)
    if use_face:
        # TODO: Make parameters, 17 is the offset for the eye brows,
        # etc. 51 is the total number of FLAME compatible landmarks
        face_keypoints = fxyz.reshape([-1, 3])[17: 17 + 51, :]

        contour_keyps = torch.empty((0, 3), dtype=body_keypoints.dtype).to(device)
        if use_face_contour:
            contour_keyps = fxyz.reshape([-1, 3])[:17, :]

        body_keypoints = torch.cat(
            (body_keypoints, face_keypoints, contour_keyps), 0)

    return torch.unsqueeze(body_keypoints, 0)


class OpenPose:

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self,
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

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

    def read_item(self, hp_kp, f_kp):
        #img = np.zeros((650, 650, 3)).astype(np.float32)

        keypoints = read_keypoints(hp_kp, f_kp, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        output_dict = {'keypoints': keypoints}

        return output_dict

