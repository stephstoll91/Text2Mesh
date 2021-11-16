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

import torch
import numpy as np

import smplx

from Pose2Mesh.smplifyx.utils import JointMapper
from Pose2Mesh.smplifyx.cmd_parser import parse_config
from Pose2Mesh.smplifyx.data_parser import create_dataset
from Pose2Mesh.smplifyx.fit_single_frame import fit_single_frame

from Pose2Mesh.smplifyx.camera import create_camera
from Pose2Mesh.smplifyx.prior import create_prior

torch.backends.cudnn.enabled = False

from back_trans.signjoey.data import load_inference_data as load_back_trans_data
from back_trans.signjoey.data import make_data_iter as make_back_data_iter
from back_trans.signjoey.helpers import load_config

from Pose2Mesh.smplifyx.slt_runner import SltRunner


def run_inference(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.join(output_folder, 'test')
    # Store the arguments for the current experiment
    # conf_fn = osp.join(output_folder, 'conf.yaml')
    # with open(conf_fn, 'w') as conf_file:
    #     yaml.dump(args, conf_file)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')

    data_folder = args.pop('data_folder')

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', 1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    cfg_file = "/".join(data_folder.split('/')[:-1]) + "/config.yaml"

    cfg = load_config(cfg_file)

    slt_runner = SltRunner(cfg['backtrans'])
    dev_data_back, test_data_back = load_back_trans_data(cfg["backtrans"]["data"], slt_runner.slt.gls_vocab, slt_runner.slt.txt_vocab)

    valid_iter_back = make_back_data_iter(
        dataset=test_data_back, batch_size=1, batch_type="sentence",
        shuffle=False, train=False)

    batches = 0
    tmje = 0.0
    for valid_batch_back in iter(valid_iter_back):

        sequence = valid_batch_back.sequence

        sequence_folder = osp.join(data_folder, sequence[0])
        sequence_out_folder = osp.join(sequence_folder, 'smplx')

        try:
            dataset_obj = create_dataset(data_folder=sequence_folder, img_folder=img_folder, **args)
        except Exception as e:
            print(e)
            continue

        if not osp.exists(sequence_folder):
            continue

        result_folder = args.pop('result_folder', 'results')
        result_folder = osp.join(sequence_out_folder, result_folder)
        if not osp.exists(result_folder):
            os.makedirs(result_folder)

        mesh_folder = args.pop('mesh_folder', 'meshes')
        mesh_folder = osp.join(sequence_out_folder, mesh_folder)
        if not osp.exists(mesh_folder):
            os.makedirs(mesh_folder)

        out_img_folder = osp.join(sequence_out_folder, 'images')
        if not osp.exists(out_img_folder):
            os.makedirs(out_img_folder)

        joint_mapper = JointMapper(dataset_obj.get_model2data())

        model_params = dict(model_path=args.get('model_folder'),
                            joint_mapper=joint_mapper,
                            create_global_orient=True,
                            create_body_pose=not args.get('use_vposer'),
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
                            dtype=dtype,
                            **args)

        male_model = smplx.create(gender='male', **model_params)
        # SMPL-H has no gender-neutral model
        if args.get('model_type') != 'smplh':
            neutral_model = smplx.create(gender='neutral', **model_params)
        female_model = smplx.create(gender='female', **model_params)

        # Create the camera object
        focal_length = args.get('focal_length')
        camera = create_camera(focal_length_x=focal_length,
                               focal_length_y=focal_length,
                               dtype=dtype,
                               **args)

        if hasattr(camera, 'rotation'):
            camera.rotation.requires_grad = False

        use_hands = args.get('use_hands', True)
        use_face = args.get('use_face', True)

        body_pose_prior = create_prior(
            prior_type=args.get('body_prior_type'),
            dtype=dtype,
            **args)

        jaw_prior, expr_prior = None, None
        if use_face:
            jaw_prior = create_prior(
                prior_type=args.get('jaw_prior_type'),
                dtype=dtype,
                **args)
            expr_prior = create_prior(
                prior_type=args.get('expr_prior_type', 'l2'),
                dtype=dtype, **args)

        left_hand_prior, right_hand_prior = None, None
        if use_hands:
            lhand_args = args.copy()
            lhand_args['num_gaussians'] = args.get('num_pca_comps')
            left_hand_prior = create_prior(
                prior_type=args.get('left_hand_prior_type'),
                dtype=dtype,
                use_left_hand=True,
                **lhand_args)

            rhand_args = args.copy()
            rhand_args['num_gaussians'] = args.get('num_pca_comps')
            right_hand_prior = create_prior(
                prior_type=args.get('right_hand_prior_type'),
                dtype=dtype,
                use_right_hand=True,
                **rhand_args)

        shape_prior = create_prior(
            prior_type=args.get('shape_prior_type', 'l2'),
            dtype=dtype, **args)

        angle_prior = create_prior(prior_type='angle', dtype=dtype)

        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')

            camera = camera.to(device=device)
            female_model = female_model.to(device=device)
            male_model = male_model.to(device=device)
            if args.get('model_type') != 'smplh':
                neutral_model = neutral_model.to(device=device)
            body_pose_prior = body_pose_prior.to(device=device)
            angle_prior = angle_prior.to(device=device)
            shape_prior = shape_prior.to(device=device)
            if use_face:
                expr_prior = expr_prior.to(device=device)
                jaw_prior = jaw_prior.to(device=device)
            if use_hands:
                left_hand_prior = left_hand_prior.to(device=device)
                right_hand_prior = right_hand_prior.to(device=device)
        else:
            device = torch.device('cpu')

        # A weight for every joint of the model
        joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                           dtype=dtype)
        # Add a fake batch dimension for broadcasting
        joint_weights.unsqueeze_(dim=0)

        pose_embedding = None
        body_mean_pose = None
        joints_2d = []
        diff_gt_inf = 0.0
        joints_2d_frame_t = None
        model_out = None
        for idx, data in enumerate(dataset_obj):
            img = np.zeros((650, 650, 3))
            fn = data['fn']
            keypoints = data['keypoints']

            print('Processing: {}'.format(data['img_path']))

            curr_result_folder = osp.join(result_folder, fn)
            if not osp.exists(curr_result_folder):
                os.makedirs(curr_result_folder)
            curr_mesh_folder = osp.join(mesh_folder, fn)
            if not osp.exists(curr_mesh_folder):
                os.makedirs(curr_mesh_folder)
            for person_id in range(keypoints.shape[0]):
                if person_id >= max_persons and max_persons > 0:
                    continue

                curr_result_fn = osp.join(curr_result_folder,
                                          '{:03d}.pkl'.format(person_id))
                curr_mesh_fn = osp.join(curr_mesh_folder,
                                        '{:03d}.obj'.format(person_id))

                curr_img_folder = osp.join(sequence_out_folder, 'images')

                if not osp.exists(curr_img_folder):
                    os.makedirs(curr_img_folder)

                body_model = male_model
                out_img_fn = curr_img_folder + '/' + fn + '.png'

                if idx == 0:
                    args['maxiters'] = 15
                else:
                    args['maxiters'] = 5


                body_model, pose_embedding, body_mean_pose, camera, joints_2d_frame, joints_gt_frame, model_out, t_diff = fit_single_frame(img, keypoints,
                                 body_model=body_model,
                                 camera=camera,
                                 joint_weights=joint_weights,
                                 dtype=dtype,
                                 output_folder=output_folder,
                                 result_folder=curr_result_folder,
                                 out_img_fn=out_img_fn,
                                 result_fn=curr_result_fn,
                                 mesh_fn=curr_mesh_fn,
                                 shape_prior=shape_prior,
                                 expr_prior=expr_prior,
                                 body_pose_prior=body_pose_prior,
                                 left_hand_prior=left_hand_prior,
                                 right_hand_prior=right_hand_prior,
                                 jaw_prior=jaw_prior,
                                 angle_prior=angle_prior,
                                 idx=idx,
                                 pose_embedding=pose_embedding,
                                 body_mean_pose=body_mean_pose,
                                 j_diff=model_out,
                                 **args)

                # Re-generate as I-Frame
                if t_diff >= 20.0 and idx > 1:
                    body_model = male_model
                    pose_embedding = None
                    body_mean_pose = None
                    focal_length = args.get('focal_length')
                    camera = create_camera(focal_length_x=focal_length,
                                           focal_length_y=focal_length,
                                           dtype=dtype,
                                           **args)

                    if hasattr(camera, 'rotation'):
                        camera.rotation.requires_grad = False
                    camera = camera.to(device=device)

                    model_out = None
                    args['maxiters'] = 15
                    body_model, pose_embedding, body_mean_pose, camera, joints_2d_frame, joints_gt_frame, model_out, t_diff = fit_single_frame(
                        img, keypoints,
                        body_model=body_model,
                        camera=camera,
                        joint_weights=joint_weights,
                        dtype=dtype,
                        output_folder=output_folder,
                        result_folder=curr_result_folder,
                        out_img_fn=out_img_fn,
                        result_fn=curr_result_fn,
                        mesh_fn=curr_mesh_fn,
                        shape_prior=shape_prior,
                        expr_prior=expr_prior,
                        body_pose_prior=body_pose_prior,
                        left_hand_prior=left_hand_prior,
                        right_hand_prior=right_hand_prior,
                        jaw_prior=jaw_prior,
                        angle_prior=angle_prior,
                        idx=0,
                        pose_embedding=pose_embedding,
                        body_mean_pose=body_mean_pose,
                        j_diff=model_out,
                        **args)

                joints_2d.append(joints_2d_frame)
                if joints_2d_frame_t is not None:
                    j_diff = np.abs(np.mean(joints_2d_frame_t - joints_2d_frame))
                    print(j_diff)
                joints_2d_frame_t = joints_2d_frame
                if not np.isnan(joints_2d_frame).any():
                    diff_gt_inf += np.abs(np.mean(joints_gt_frame - joints_2d_frame))

        mje = diff_gt_inf / idx
        print("Mean Joint Error: {}".format(mje))

        batches += 1
        tmje += mje

        # for demo
        if batches >= 4:
            break

if __name__ == "__main__":
    args = parse_config()
    run_inference(**args)
