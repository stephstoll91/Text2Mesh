import os
import os.path as osp
import pickle
import torch
import sys
import numpy as np

import smplx

from back_trans.signjoey.batch import Batch as BackBatch
from back_trans.signjoey.phoenix_utils.phoenix_cleanup import clean_phoenix_2014_trans
from back_trans.signjoey.metrics import wer_list, bleu, chrf, rouge

from back_trans.signjoey.data import load_inference_data as load_back_trans_data
from back_trans.signjoey.data import make_data_iter as make_back_data_iter
from back_trans.signjoey.helpers import load_config, make_logger

from Pose2Mesh.smplifyx.slt_runner import SltRunner

from Pose2Mesh.smplifyx.utils import JointMapper
from Pose2Mesh.smplifyx.cmd_parser import parse_config
from Pose2Mesh.smplifyx.data_parser import create_dataset

import human_body_prior.tools.model_loader as hpm
from human_body_prior.models.vposer_model import VPoser

from Pose2Mesh.smplifyx.camera import create_camera


def calc_bleu(**args):
    device = torch.device('cuda')
    output_folder = args.pop('output_folder')
    logger = make_logger(output_folder,'eval.log')
    output_folder = osp.join(output_folder, 'test')
    vposer_ckpt = args.get('vposer_ckpt')

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

    all_gls_outputs = []
    all_txt_outputs = []
    all_attention_scores = []

    gt_gls = []
    gt_txt = []

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

        if not osp.exists(sequence_out_folder):
            continue

        result_folder = args.pop('result_folder', 'results')
        result_folder = osp.join(sequence_out_folder, result_folder)

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

        body_model = smplx.create(gender='male', **model_params)
        body_model = body_model.to(device=device)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = hpm.load_model(vposer_ckpt, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        vposer = vposer.to(device=device)
        vposer.eval()

        # Create the camera object
        focal_length = args.get('focal_length')
        camera = create_camera(focal_length_x=focal_length,
                               focal_length_y=focal_length,
                               dtype=dtype,
                               **args)
        camera = camera.to(device=device)

        joints_2d = []
        joints_gt_2d = []
        diff_gt_inf = 0.0

        for idx, data in enumerate(dataset_obj):
            fn = data['fn']
            keypoints = data['keypoints']
            curr_result_folder = osp.join(result_folder, fn)

            curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(0))

            with open(curr_result_fn, "rb") as infile:
                bm = pickle.load(infile)
            body_pose = torch.Tensor(bm['body_pose']).to(device)
            body_pose = vposer.decode(
                body_pose)['pose_body'].contiguous().view(-1, 63)

            body_model.reset_params(**bm)
            model_output = body_model(return_verts=True, body_pose=body_pose)

            camera.center[:] = torch.tensor([650, 650], dtype=dtype) * 0.5
            camera.translation = torch.nn.Parameter(torch.Tensor(bm['camera_translation']).to(device))
            camera.rotation = torch.nn.Parameter(torch.Tensor(bm['camera_rotation']).to(device))
            j2d = keypoints
            j3d = camera(model_output.joints).cpu().detach().numpy()
            j2d_b = j2d[:, :, 2].astype(bool)
            j2d = j2d[:, j2d_b[0, :], :]
            j3d = j3d[:, j2d_b[0, :], :]

            j3d = np.concatenate((j3d[:, 0:8, :], j3d[:, 8 + 7:, :]), 1)
            j2d = np.concatenate((j2d[:, 0:8, :], j2d[:, 8 + 7:, :]), 1)

            hp_offset = [250, 300]

            joints = j3d[:, :50, :2] - (np.ones((1, 50, 2)) * hp_offset)
            joints = joints[:, :, :2] / 10 / 12 / 2 / 3

            joints_gt = j2d[:, :50, :2] - (np.ones((1, 50, 2)) * hp_offset)
            joints_gt = joints_gt[:, :, :2] / 10 / 12 / 2 / 3

            # import matplotlib.pyplot as plt
            # plt.scatter(joints[:, :, 0], joints[:, :, 1])
            # plt.scatter(joints_gt[:, :, 0], joints_gt[:, :, 1])
            # plt.show()

            joints_2d.append(joints)
            joints_gt_2d.append(joints_gt)

            diff_gt_inf += np.abs(np.mean(joints - joints_gt))

        mje = diff_gt_inf / idx
        print("Mean Joint Error: {}".format(mje))
        output = np.expand_dims(np.concatenate(joints_2d, axis=0), 0)
        output = torch.Tensor(np.reshape(output, (1, -1, 100))).to(device)

        valid_batch_back.sgn = (output[:, :, :100].cpu(), torch.Tensor([float(len(joints_2d))]))  # [:,:,:150]

        back_batch = BackBatch(torch_batch=valid_batch_back,
                               txt_pad_index=1,
                               sgn_dim=100,
                               is_train=False,
                               use_cuda=True,
                               future_prediction=0)

        (
            batch_gls_predictions,
            batch_txt_predictions,
            batch_attention_scores,
        ) = slt_runner.slt.run_batch(
            batch=back_batch,
            recognition_beam_size=slt_runner.eval_recognition_beam_size,
            translation_beam_size=slt_runner.eval_translation_beam_size,
            translation_beam_alpha=slt_runner.eval_translation_beam_alpha,
            translation_max_output_length=slt_runner.translation_max_output_length,
        )

        # decode back to symbols
        batch_gls = []
        batch_gls.extend(batch_gls_predictions if batch_gls_predictions is not None else [])
        decoded_gls = slt_runner.slt.gls_vocab.arrays_to_sentences(arrays=batch_gls)
        decoded_gt_gls = slt_runner.slt.gls_vocab.arrays_to_sentences(arrays=[valid_batch_back.gls[0].tolist()[0]])
        # Gloss clean-up function
        gls_cln_fn = clean_phoenix_2014_trans

        # Construct gloss sequences for metrics
        gls_ref = [gls_cln_fn(" ".join(t)) for t in decoded_gt_gls]
        gls_hyp = [gls_cln_fn(" ".join(t)) for t in decoded_gls]

        # GLS Metrics
        if len(gls_hyp) > 0:
            assert len(gls_ref) == len(gls_hyp)
            gls_wer_score = wer_list(hypotheses=gls_hyp, references=gls_ref)

        # decode back to symbols
        batch_txt = []
        batch_txt.extend(batch_txt_predictions if batch_txt_predictions is not None else [])
        decoded_txt = slt_runner.slt.txt_vocab.arrays_to_sentences(arrays=batch_txt)
        decoded_gt_txt = slt_runner.slt.txt_vocab.arrays_to_sentences(arrays=[valid_batch_back.txt[0].tolist()[0]])
        # evaluate with metric on full dataset
        join_char = " "
        # Construct text sequences for metrics
        txt_ref = [join_char.join(t) for t in decoded_gt_txt]
        txt_hyp = [join_char.join(t) for t in decoded_txt]

        assert len(txt_ref) == len(txt_hyp)

        # TXT Metrics
        txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
        txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
        txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        if len(txt_ref):
            logger.info("Text Ground Truth: %s\t"
                        "Text Back-Translation: %s\t",
                        txt_ref[0],
                        txt_hyp[0])

        logger.info(
            "Recognition Beam Size: %d\t"
            "Translation Beam Size: %d\t"
            "Translation Beam Alpha: %d\n\t"
            "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
            "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
            "CHRF %.2f\t"
            "ROUGE %.2f",
            slt_runner.eval_recognition_beam_size if slt_runner.do_recognition else -1,
            slt_runner.eval_translation_beam_size if slt_runner.do_translation else -1,
            slt_runner.eval_translation_beam_alpha if slt_runner.do_translation else -1,
            # WER
            gls_wer_score["wer"] if slt_runner.do_recognition else -1,
            gls_wer_score["del_rate"]
            if slt_runner.do_recognition
            else -1,
            gls_wer_score["ins_rate"]
            if slt_runner.do_recognition
            else -1,
            gls_wer_score["sub_rate"]
            if slt_runner.do_recognition
            else -1,
            # BLEU
            txt_bleu["bleu4"] if slt_runner.do_translation else -1,
            txt_bleu["bleu1"]
            if slt_runner.do_translation
            else -1,
            txt_bleu["bleu2"]
            if slt_runner.do_translation
            else -1,
            txt_bleu["bleu3"]
            if slt_runner.do_translation
            else -1,
            txt_bleu["bleu4"]
            if slt_runner.do_translation
            else -1,
            # Other
            txt_chrf if slt_runner.do_translation else -1,
            txt_rouge if slt_runner.do_translation else -1,
        )


        all_gls_outputs.extend(batch_gls_predictions if batch_gls_predictions is not None else [])
        all_txt_outputs.extend(batch_txt_predictions)

        all_attention_scores.extend(
            batch_attention_scores
            if batch_attention_scores is not None
            else []
        )

        gt_gls.append(valid_batch_back.gls[0].tolist()[0])
        gt_txt.append(valid_batch_back.txt[0].tolist()[0])

        logger.info(sequence_folder)

        batches += 1
        tmje += mje

    logger.info("Total Mean Joint Error: {}".format(tmje/batches))
    # decode back to symbols
    decoded_gls = slt_runner.slt.gls_vocab.arrays_to_sentences(arrays=all_gls_outputs)
    decoded_gt_gls = slt_runner.slt.gls_vocab.arrays_to_sentences(arrays=gt_gls)
    # Gloss clean-up function
    gls_cln_fn = clean_phoenix_2014_trans

    # Construct gloss sequences for metrics
    gls_ref = [gls_cln_fn(" ".join(t)) for t in decoded_gt_gls]
    gls_hyp = [gls_cln_fn(" ".join(t)) for t in decoded_gls]

    if len(gls_hyp) > 0:
        assert len(gls_ref) == len(gls_hyp)
        # GLS Metrics
        gls_wer_score = wer_list(hypotheses=gls_hyp, references=gls_ref)

    # decode back to symbols
    decoded_txt = slt_runner.slt.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
    decoded_gt_txt = slt_runner.slt.txt_vocab.arrays_to_sentences(arrays=gt_txt)
    # evaluate with metric on full dataset
    join_char = " "
    # Construct text sequences for metrics
    txt_ref = [join_char.join(t) for t in decoded_gt_txt]
    txt_hyp = [join_char.join(t) for t in decoded_txt]

    assert len(txt_ref) == len(txt_hyp)

    # TXT Metrics
    txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
    txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
    txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

    logger.info(
        "Recognition Beam Size: %d\t"
        "Translation Beam Size: %d\t"
        "Translation Beam Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        slt_runner.eval_recognition_beam_size if slt_runner.do_recognition else -1,
        slt_runner.eval_translation_beam_size if slt_runner.do_translation else -1,
        slt_runner.eval_translation_beam_alpha if slt_runner.do_translation else -1,
        # WER
        gls_wer_score["wer"] if slt_runner.do_recognition else -1,
        gls_wer_score["del_rate"]
        if slt_runner.do_recognition
        else -1,
        gls_wer_score["ins_rate"]
        if slt_runner.do_recognition
        else -1,
        gls_wer_score["sub_rate"]
        if slt_runner.do_recognition
        else -1,
        # BLEU
        txt_bleu["bleu4"] if slt_runner.do_translation else -1,
        txt_bleu["bleu1"]
        if slt_runner.do_translation
        else -1,
        txt_bleu["bleu2"]
        if slt_runner.do_translation
        else -1,
        txt_bleu["bleu3"]
        if slt_runner.do_translation
        else -1,
        txt_bleu["bleu4"]
        if slt_runner.do_translation
        else -1,
        # Other
        txt_chrf if slt_runner.do_translation else -1,
        txt_rouge if slt_runner.do_translation else -1,
    )


if __name__ == "__main__":
    args = parse_config()
    calc_bleu(**args)
