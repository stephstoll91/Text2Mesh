import smplx
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from smplifyx.smplifyx.utils import JointMapper
from smplifyx.smplifyx.camera import create_camera
from smplifyx.smplifyx.prior import create_prior
from smplifyx.smplifyx.on_fly_data_parser import create_dataset
import human_body_prior.tools.model_loader as hpm
from human_body_prior.models.vposer_model import VPoser
import smplifyx.smplifyx.fitting as fitting
from smplifyx.smplifyx.optimizers import optim_factory

from Text2Pose.helpers import load_config
from Text2Pose.constants import TARGET_PAD#

from smplifyx.smplifyx.raymond_lights import create_raymond_lights
import PIL.Image as pil_img


class SmplifyX(nn.Module):

    def __init__(self, args):
        super(SmplifyX, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.dtype = torch.float32
        self.dataset_obj = create_dataset(**args)
        self.joint_mapper = JointMapper(self.dataset_obj.get_model2data())
        # A weight for every joint of the model
        self.joint_weights = self.dataset_obj.get_joint_weights().to(device=self.device,
                                                           dtype=self.dtype)
        # Add a fake batch dimension for broadcasting
        self.joint_weights.unsqueeze_(dim=0)

        self.use_joints_conf = args.get('use_joints_conf')

        vposer_ckpt = args.get('vposer_ckpt')
        vposer, _ = hpm.load_model(vposer_ckpt, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        self.vposer = vposer.to(device=self.device)
        self.vposer.eval()

        model_params = dict(model_path=args.get('model_folder'),
                            joint_mapper=self.joint_mapper,
                            create_global_orient=True,
                            create_body_pose=False,
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
                            dtype=self.dtype,
                            **args)

        self.body_model = smplx.create(**model_params).to(device=self.device)

        self.init_joints_idxs = (9, 12, 2, 5)
        self.edge_indices = [(5, 12), (2, 9)]

        # Create the camera object
        self.camera = create_camera(**args)
        if hasattr(self.camera, 'rotation'):
            self.camera.rotation.requires_grad = False

        self.camera = self.camera.to(device=self.device)

        self.body_pose_prior = create_prior(
            prior_type=args.get('body_prior_type'),
            dtype=self.dtype,
            **args).to(device=self.device)

        self.jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=self.dtype,
            **args).to(device=self.device)
        self.expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=self.dtype, **args).to(device=self.device)

        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        self.left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=self.dtype,
            use_left_hand=True,
            **lhand_args).to(device=self.device)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        self.right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=self.dtype,
            use_right_hand=True,
            **rhand_args).to(device=self.device)

        self.shape_prior = create_prior(
            prior_type=args.get('shape_prior_type', 'l2'),
            dtype=self.dtype, **args).to(device=self.device)

        self.angle_prior = create_prior(prior_type='angle', dtype=self.dtype).to(device=self.device)

        self.model_type=args['model_type']
        self.optim_type=args['optim_type']
        self.ftol=args['ftol']
        self.gtol=args['gtol']
        self.lr=args['lr']
        self.left_shoulder_idx=2
        self.right_shoulder_idx=5
        self.side_view_thsh=25.
        self.maxiters=5

        self.out_dir = args['out_dir']

    def set_weights_dict(self, idx):
        # Weights used for the pose prior and the shape prior
        if idx <= 0:
            opt_weights_dict = {'data_weight': [1, 1, 1, 1, 1],
                                'body_pose_weight': [404.0, 404.0, 100.0, 100.0, 80.0],  # body_pose_prior_weights,
                                'shape_weight': [100.0, 50.0, 50.0, 30.0, 20.0]}  # shape_weights}

            opt_weights_dict['face_weight'] = [0.0, 0.0, 0.0, 0.1, 1.0]  # face_joints_weights
            opt_weights_dict['expr_prior_weight'] = [100.0, 50.0, 10.0, 5.0, 5.0]
            opt_weights_dict['jaw_prior_weight'] = [[4040.0, 40400.0, 40400.0],
                                                    [4040.0, 40400.0, 40400.0],
                                                    [574.0, 5740.0, 5740.0],
                                                    [470.8, 478.0, 4708.0],
                                                    [470.8, 4708.0, 4708.0]]  # jaw_pose_prior_weights

            opt_weights_dict['hand_weight'] = [0.0, 0.0, 0.0, 0.1, 2.0]
            opt_weights_dict['hand_prior_weight'] = [404.0, 404.0, 57.4, 30.0, 5.0]  # hand_pose_prior_weights


        else:
            opt_weights_dict = {'data_weight': [1, 1, 1, 1, 1],
                                'body_pose_weight': [404.0, 404.0, 100.0, 100.0, 80.0],  # body_pose_prior_weights,
                                'shape_weight': [100.0, 50.0, 50.0, 30.0, 20.0]}  # shape_weights}

            opt_weights_dict['face_weight'] = [0.0, 0.0, 0.0, 0.1, 2.0]  # face_joints_weights
            opt_weights_dict['expr_prior_weight'] = [100.0, 50.0, 10.0, 5.0, 1.0]
            opt_weights_dict['jaw_prior_weight'] = [[4040.0, 40400.0, 40400.0],
                                                    [4040.0, 40400.0, 40400.0],
                                                    [574.0, 5740.0, 5740.0],
                                                    [470.8, 478.0, 4708.0],
                                                    [470.8, 4708.0, 4708.0]]  # jaw_pose_prior_weights

            opt_weights_dict['hand_weight'] = [0.0, 0.0, 0.0, 0.1, 2.0]  # hand_joints_weights
            opt_weights_dict['hand_prior_weight'] = [404.0, 404.0, 57.4, 30.0, 10.0]  # hand_pose_prior_weights

        return opt_weights_dict

def run_batch(model, keyp, sgn_length, batch_num, out_fn):

    sequence_size = keyp.shape[0]
    funky_loss = False
    go_back = 0
    out_2d = torch.full((sequence_size, 100), TARGET_PAD).to(model.device)
    out_verts = np.zeros((sequence_size, 10475, 3))
    if keyp.shape[-1] > 100:
        keyp = keyp[:, :-1].reshape((sequence_size, -1, 2))
    else:
        keyp = keyp.reshape((sequence_size, -1, 2))

    # loop sequence
    sequence = sgn_length.to(dtype=torch.long)
    model.maxiters = 15
    opt_weights_dict = model.set_weights_dict(0)
    for s in range(sequence):

        if s > 0:
            model.maxiters = 3
            opt_weights_dict = model.set_weights_dict(s)

        data = model.dataset_obj.read_item(keyp[s, :50, :], keyp[s, 50:, :])
        keypoint_data = data['keypoints']
        gt_joints = keypoint_data[:, :, :2]
        gt_joints = gt_joints.to(device=model.device, dtype=model.dtype)
        if model.use_joints_conf:
            joints_conf = keypoint_data[:, :, 2].reshape(1, -1)
            joints_conf = joints_conf.to(device=model.device, dtype=model.dtype)


        pose_embedding = torch.zeros([1, 32],
                                     dtype=model.dtype, device=model.device,
                                     requires_grad=True)

        keys = opt_weights_dict.keys()
        opt_weights = [dict(zip(keys, vals)) for vals in
                       zip(*(opt_weights_dict[k] for k in keys
                             if opt_weights_dict[k] is not None))]
        for weight_list in opt_weights:
            for key in weight_list:
                weight_list[key] = torch.tensor(weight_list[key],
                                                device=model.device,
                                                dtype=model.dtype)

        init_joints_idxs = torch.tensor(model.init_joints_idxs, device=model.device)

        if s <= 0:
            init_t = fitting.guess_init(model.body_model, gt_joints, model.edge_indices,
                                        use_vposer=True, vposer=model.vposer,
                                        pose_embedding=pose_embedding,
                                        model_type=model.model_type,
                                        focal_length=5000,
                                        dtype=model.dtype)

            camera_loss = fitting.create_loss('camera_init',
                                              trans_estimation=init_t,
                                              init_joints_idxs=init_joints_idxs,
                                              depth_loss_weight=1e2,
                                              dtype=model.dtype).to(device=model.device)
            camera_loss.trans_estimation[:] = init_t

            body_mean_pose = torch.zeros([1, 32],
                                         dtype=model.dtype)

        loss = fitting.create_loss(loss_type='smplify',
                                   joint_weights=model.joint_weights,
                                   vposer=model.vposer,
                                   pose_embedding=pose_embedding,
                                   body_pose_prior=model.body_pose_prior,
                                   shape_prior=model.shape_prior,
                                   angle_prior=model.angle_prior,
                                   expr_prior=model.expr_prior,
                                   left_hand_prior=model.left_hand_prior,
                                   right_hand_prior=model.right_hand_prior,
                                   jaw_prior=model.jaw_prior,
                                   pen_distance=None,
                                   search_tree=None,
                                   tri_filtering_module=None,
                                   dtype=model.dtype)

        loss = loss.to(device=model.device)
        with fitting.FittingMonitor(
                batch_size=1, visualize=False, maxiters=model.maxiters, model_type=model.model_type,
                ftol=float(model.ftol), gtol=float(model.gtol)) as monitor:
            img = np.zeros((650, 650, 3))

            H, W, _ = img.shape

            data_weight = 1000 / 650
            if s <= 0:
                camera_loss.reset_loss_weights({'data_weight': data_weight})
                model.body_model.reset_params(body_pose=body_mean_pose)

            shoulder_dist = torch.dist(gt_joints[:, model.left_shoulder_idx],
                                       gt_joints[:, model.right_shoulder_idx])
            try_both_orient = shoulder_dist.item() < model.side_view_thsh

            if s <= 0:
                with torch.no_grad():
                    model.camera.translation[:] = init_t.view_as(model.camera.translation)
                    model.camera.center[:] = torch.tensor([650, 650], dtype=model.dtype) * 0.5

            model.camera.translation.requires_grad = True

            camera_opt_params = [model.camera.translation, model.body_model.global_orient]

            camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
                camera_opt_params,lr=float(model.lr), max_iter=model.maxiters)

            if s <= 0:
                # The closure passed to the optimizer
                fit_camera = monitor.create_fitting_closure(
                    camera_optimizer, model.body_model, model.camera, gt_joints,
                    camera_loss, create_graph=camera_create_graph,
                    use_vposer=True, vposer=model.vposer,
                    pose_embedding=pose_embedding,
                    return_full_pose=False, return_verts=False)

                try:
                    # Step 1: Optimize over the torso joints the camera translation
                    # Initialize the computational graph by feeding the initial translation
                    # of the camera and the initial pose of the body model.
                    cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                            fit_camera,
                                                            camera_opt_params, model.body_model,
                                                            use_vposer=True,
                                                            pose_embedding=pose_embedding,
                                                            vposer=model.vposer,
                                                            joint_weights=model.joint_weights,
                                                            joints_conf=joints_conf)

                    if cam_init_loss_val is None:
                        raise Exception("Camera loss is None!")

                except Exception as e:
                    print(e)
                    funky_loss = True

            if try_both_orient:
                body_orient = model.body_model.global_orient.detach().cpu().numpy()
                flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                    cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
                flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

                flipped_orient = torch.tensor(flipped_orient,
                                              dtype=model.dtype,
                                              device=model.device).unsqueeze(dim=0)
                orientations = [body_orient, flipped_orient]

            else:
                orientations = [model.body_model.global_orient.detach().cpu().numpy()]

            results = []

            # Step 2: Optimize the full model
            final_loss_val = 0
            for or_idx, orient in enumerate(orientations):
                new_params = defaultdict(global_orient=orient,
                                         body_pose=body_mean_pose)
                if s <= 0:
                    model.body_model.reset_params(**new_params)
                    with torch.no_grad():
                        pose_embedding.fill_(0)

                for opt_idx, curr_weights in enumerate(opt_weights):
                    if s > 0 and opt_idx < 4:
                        continue

                    body_params = list(model.body_model.parameters())

                    final_params = list(
                        filter(lambda x: x.requires_grad, body_params))

                    final_params.append(pose_embedding)

                    body_optimizer, body_create_graph = optim_factory.create_optimizer(
                        final_params,lr=model.lr, max_iter=model.maxiters)

                    curr_weights['data_weight'] = data_weight
                    curr_weights['bending_prior_weight'] = (
                            3.17 * curr_weights['body_pose_weight'])
                    model.joint_weights[:, 25:67] = curr_weights['hand_weight']
                    model.joint_weights[:, 67:] = curr_weights['face_weight']
                    loss.reset_loss_weights(curr_weights)

                    closure = monitor.create_fitting_closure(
                        body_optimizer, model.body_model,
                        camera=model.camera, gt_joints=gt_joints,
                        joints_conf=joints_conf,
                        joint_weights=model.joint_weights,
                        loss=loss, create_graph=body_create_graph,
                        use_vposer=True, vposer=model.vposer,
                        pose_embedding=pose_embedding,
                        return_verts=True, return_full_pose=True)
                    try:
                        final_loss_val = monitor.run_fitting(
                            body_optimizer,
                            closure, final_params,
                            model.body_model,
                            pose_embedding=pose_embedding, vposer=model.vposer,
                            use_vposer=True)

                        if final_loss_val is None:
                            raise Exception("Body loss is None!")
                    except Exception as e:
                        print(e)
                        funky_loss = True
                        if s != 0:
                            go_back = 1

                # Get the result of the fitting process
                # Store in it the errors list in order to compare multiple
                # orientations, if they exist
                result = {'camera_' + str(key): val.detach().cpu().numpy()
                          for key, val in model.camera.named_parameters()}
                result.update({key: val.detach().cpu().numpy()
                               for key, val in model.body_model.named_parameters()})
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

                results.append({'loss': final_loss_val,
                                'result': result})


            body_pose = model.vposer.decode(
                pose_embedding)['pose_body'].contiguous().view(-1, 63)
            model_output = model.body_model(return_verts=True, body_pose=body_pose)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()

            j2d = keypoint_data
            j3d = model_output.joints
            j2d_b = j2d[:, :, 2].to(torch.bool)

            #j2d = j2d[:, j2d_b[0, :], :]
            j3d = j3d[:, j2d_b[0, :], :]

            def ptp(t, axis):
                return t.max(axis).values - t.min(axis).values

            j3d[:, :, 0] = (1100 * (j3d[:, :, 0] - torch.min(j3d[:, :, 0])) / ptp(j3d[:, :, 0], 1))
            j3d[:, :, 1] = (1100 * (j3d[:, :, 1] - torch.min(j3d[:, :, 1])) / ptp(j3d[:, :, 1], 1))

            hp_offset = torch.Tensor([250, 300]).to(model.device)
            joints = j3d[:, :57, :2] - (torch.ones((1, 57, 2)).to(model.device) * hp_offset)
            joints = joints[:, :, :2] / 10 / 12 / 2 / 3
            joints_pruned = torch.cat((joints[:, 0:8, :], joints[:, 8 + 7:, :]), 1)

            import trimesh

            out_mesh = trimesh.Trimesh(vertices, model.body_model.faces, process=False)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot)

            import pyrender
            #from mesh_viewer import MeshViewer

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')

            camera_center = model.camera.center.detach().cpu().numpy().squeeze()
            camera_transl = model.camera.translation.detach().cpu().numpy().squeeze()
            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            camera_transl[0] *= -1.0

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl

            camera_int = pyrender.camera.IntrinsicsCamera(
                fx=5000, fy=5000,
                cx=camera_center[0], cy=camera_center[1])
            scene.add(camera_int, pose=camera_pose)

            # Get the lights from the viewer
            light_nodes = create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)

            r = pyrender.OffscreenRenderer(viewport_width=650,
                                           viewport_height=650,
                                           point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0

            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)

            import os
            import os.path as osp
            out_dir_fn = model.out_dir + "/" + out_fn + '/smplx/'
            if not osp.exists(out_dir_fn):
                os.makedirs(out_dir_fn)

            img = pil_img.fromarray((output_img * 255).astype(np.uint8))
            img.save(out_dir_fn + "{:03d}".format(s) + ".png")

            #joints_gt = j2d[:, :57, :2] - (torch.ones((1, 57, 2)).to(self.device) * hp_offset)
            #joints_gt = joints_gt[:, :, :2] / 10 / 12 / 2 / 3

            #import matplotlib.pyplot as plt
            #plt.scatter(joints[:, :, 0].cpu().detach().numpy(), joints[:, :, 1].cpu().detach().numpy())
            #plt.show()
            #plt.scatter(joints_gt[:, :, 0].cpu().detach().numpy(), joints_gt[:, :, 1].cpu().detach().numpy())
            #plt.show()

            out_2d[s:s+1, :] = joints_pruned.reshape((1, 1, -1))
            out_verts[s, :, :] = vertices

    nans = torch.isnan(out_2d)
    if torch.any(nans[:s]):
        print("lerp alert!")
        if torch.all(nans[:s]):
            print("all nans!")
            out_2d = torch.full((sequence_size, 100), TARGET_PAD).to(model.device)

    torch.save(out_2d, 'batch_2D_' + str(batch_num) + '.pt')
    return #{'2d': out_2d, 'verts': out_verts}


def create_smplifyx():
    cfg_file = '/home/stephanie/Documents/Chapter3/Code/Text2Pose/smplifyx/cfg_files/fit_smplx.yaml'
    cfg = load_config(cfg_file)
    return SmplifyX(cfg)