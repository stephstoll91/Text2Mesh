import numpy as np
from scipy import signal
import scipy
from scipy.interpolate import interp1d
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy import signal
import os
from os import listdir
from os.path import isfile, join
import io
import math
from sympy import *
from utils import *
import mat73
import matplotlib.pyplot as plt

from SignLanguageProcessing.ThreeDposeEstimator.demo import convert

import multiprocessing

def getSkeletalModelStructure():
    return (
        # 1st finger
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 3),

        # 2nd finger
        (0, 5, 0),
        (5, 6, 5),
        (6, 7, 6),
        (7, 8, 7),

        # 3rd finger
        (0, 9, 0),
        (9, 10, 9),
        (10, 11, 10),
        (11, 12, 11),

        # 4th finger
        (0, 13, 0),
        (13, 14, 13),
        (14, 15, 14),
        (15, 16, 15),

        # 5th finger
        (0, 17, 0),
        (17, 18, 17),
        (18, 19, 18),
        (19, 20, 19),
    )


def make_bones(joints):
    wrist = joints[:, 0, :]
    thumb = np.array([joints[:, 1, :] - wrist, joints[:, 1, :] - joints[:, 2, :], joints[:, 2, :] - joints[:, 3, :], joints[:, 3, :] - joints[:, 4, :]])
    index = np.array([joints[:, 5, :] - wrist, joints[:, 5, :] - joints[:, 6, :], joints[:, 6, :] - joints[:, 7, :], joints[:, 7, :] - joints[:, 8, :]])
    middle = np.array([joints[:, 9, :] - wrist, joints[:, 9, :] - joints[:, 10, :], joints[:, 10, :] - joints[:, 11, :], joints[:, 11, :] - joints[:, 12, :]])
    ring = np.array([joints[:, 13, :] - wrist, joints[:, 13, :] - joints[:, 14, :], joints[:, 14, :] - joints[:, 15, :],
              joints[:, 15, :] - joints[:, 16, :]])
    pinky = np.array([joints[:, 17, :] - wrist, joints[:, 17, :] - joints[:, 18, :], joints[:, 18, :] - joints[:, 19, :],
              joints[:, 19, :] - joints[:, 20, :]])

    bones = np.concatenate((wrist, thumb, index,  middle, ring, pinky), -1)
    return bones


def replace_nulls(joints):
    _, l = joints.shape

    # do preliminary check:
    z = np.where(joints <= 0.00000)[0]
    nz = np.where(joints > 0.00000)[0]
    if z.size > nz.size * 1.5:
        return np.asarray([])

    z_counter = 0
    for c in range(l):
        if z_counter == 2:
            z_counter = 0
            continue
        z_counter += 1

        indx = np.where(joints[:, c] <= 0.00000)[0]
        cindx = np.where(joints[:, c] > 0.00000)[0]
        try:
            if cindx.size == 0:
                raise Exception("nothing over zero")
            elif indx.size > 0:
                #case: indx starts at 0:
                if indx[0] == 0:
                    #fill with first occuring value
                    val = joints[:, c][cindx[0]]
                    joints[:, c][0:cindx[0]] = val
                #case: indx ends with 0
                if indx[-1] == len(joints) -1:
                    #fill with last occuring value
                    val = joints[:, c][cindx[-1]]
                    joints[:, c][cindx[-1]+1:] = val

                cindx = np.where(joints[:, c] > 0.00000)[0]
                indx = np.where(joints[:, c] <= 0.00000)[0]
                if indx.size > 0:
                    x = cindx
                    y = joints[cindx, c]
                    inter1 = interp1d(x, y, "linear")
                    newx = np.arange(len(joints[:, c]))
                    joints[:, c] = inter1(newx)

        except Exception as e:
            print(str(e))
            return np.asarray([])
    return joints


def objective(x, a, b, c):
    return a * x + b * x**2 + c


def check_bones(joints, orient):

    for f in range(joints.shape[0]):
        joint_frame = joints[f]
        joint_frame = np.reshape(joint_frame, (21, 2))[:, :]
        for j in range(1, 21, 4):
            finger = adjust_bone(np.concatenate((joint_frame[0:1, :], joint_frame[j:j+4, :]), 0), orient)
            joint_frame[j:j + 4] = finger[1:]
        joint_frame = np.reshape(joint_frame, (42,))
        joints[f] = joint_frame

    return joints


def adjust_bone(finger, right):
    thresh = - 1.0
    if finger[:, 0].all() > thresh and finger[:, 1].all() > thresh:

        # angles = np.arctan2((finger[:-1, 0] - finger[1:, 0]), (finger[:-1, 1] - finger[1:, 1])) * 180 / np.pi
        # if not right and (np.any(angles >= 180) or np.any(angles < 0)):
        #     finger[:, 0] = 0.0
        #     finger[:, 1] = 0.0
        #     return finger
        # elif right and (np.any(angles <= -180)):
        #     finger[:, 0] = 0.0
        #     finger[:, 1] = 0.0
        #     return finger

        # jdiff = abs(angles[:-1] - angles[1:])
        # if np.mean(jdiff) > 20:
        #     x = signal.savgol_filter(finger[:, 0], 5, 2)
        #     y = signal.savgol_filter(finger[:, 1], 5, 2)
        # else:
        #     x = signal.savgol_filter(finger[:, 0], 5, 1)
        #     y = signal.savgol_filter(finger[:, 1], 5, 1)
        #
        # finger[1:, 0] = x[1:]
        # finger[1:, 1] = y[1:]

        # angles = np.arctan2((finger[:-1, 0] - finger[1:, 0]), (finger[:-1, 1] - finger[1:, 1])) * 180 / np.pi
        # jdiff = abs(angles[:-1] - angles[1:])

        #center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
        length = np.sqrt(((finger[:-1, 0] - finger[1:, 0]) ** 2) + ((finger[:-1, 1] - finger[1:, 1]) ** 2)) / 2
        if np.max(length) > 20 or np.sum(length) < 5:
             finger[:, 0] = 0.0
             finger[:, 1] = 0.0

    return finger


def visualise(joints_ppl, joints_ppr, jointsl, jointsr, imgs_path, sequence_ID):
    if joints_ppl.shape[0] > 0 and joints_ppr.shape[0] > 0:
        out_path = "./pre_processed/"
        sequence_ID = out_path + sequence_ID
        if not os.path.exists(sequence_ID):
            os.makedirs(sequence_ID)

        sequence_ID_INF = sequence_ID + "_INF"
        if not os.path.exists(sequence_ID_INF):
            os.makedirs(sequence_ID_INF)
        for f in range(jointsl.shape[0]):

            jxl = jointsl[f, 0:42:2]
            jyl = jointsl[f, 1:42:2]
            jpxl = joints_ppl[f, 0:42:2]
            jpyl = joints_ppl[f, 1:42:2]

            jxr = jointsr[f, 0:42:2]
            jyr = jointsr[f, 1:42:2]
            jpxr = joints_ppr[f, 0:42:2]
            jpyr = joints_ppr[f, 1:42:2]

            jx = np.expand_dims(np.concatenate((jxl, jxr), 0), -1)
            jpx = np.expand_dims(np.concatenate((jpxl, jpxr), 0), -1)

            jy = np.expand_dims(np.concatenate((jyl, jyr), 0), -1)
            jpy = np.expand_dims(np.concatenate((jpyl, jpyr), 0), -1)

            fake_Bs = np.concatenate((jpx, jpy), -1)
            real_Bs = np.concatenate((jx, jy), -1)

            make_maps_and_imgs(fake_Bs, real_Bs, sequence_ID, sequence_ID_INF, f, imgs_path)

    return


def data_to_file(joints_ppl, joints_ppr, imgs_path, sequence_ID, base_dir):
    if joints_ppl.shape[0] > 0 and joints_ppr.shape[0] > 0:

        sequence_ID = base_dir + sequence_ID + "/"
        if not os.path.exists(sequence_ID):
            os.makedirs(sequence_ID)

        for f in range(joints_ppl.shape[0]):

            jpxl = joints_ppl[f, 0:42:2]
            jpyl = joints_ppl[f, 1:42:2]

            jpxr = joints_ppr[f, 0:42:2]
            jpyr = joints_ppr[f, 1:42:2]

            jpx = np.expand_dims(np.concatenate((jpxl, jpxr), 0), -1)

            jpy = np.expand_dims(np.concatenate((jpyl, jpyr), 0), -1)

            fakes = np.concatenate((jpx, jpy), -1)

            make_maps_and_imgs(fakes, sequence_ID, f, imgs_path)
    return


def moving_avg(joints):
    w = 3

    s = np.r_[joints[w - 1:0:-1], joints, joints[-2:-w - 1:-1]]
    window = np.ones(w, 'd')
    joints = np.convolve(window / window.sum(), s, mode='valid')

    return joints[int(w/2):-int(w/2)]


def savgol_per_joint(joints):
    poly = 1
    w = 11
    joints = signal.savgol_filter(joints, w, poly, mode='mirror')
    return joints


def interpolate_joints(joints, count, orient, correct_bones=False):
    if joints.shape[0] > 0:
        jc = joints[:, 2:joints.shape[1]:3]
        mm = np.percentile(jc, 10, 0)
        bool_mask = jc < mm

        # jdiff = abs(joints[:-1] - joints[1:])
        # jdiff = np.concatenate((np.zeros((1, 63)), jdiff), 0)
        # jdiffm = np.percentile(jdiff, 95, 0)
        # bool_mask_1 = jdiff > jdiffm
        #
        # bool_mask_x = bool_mask_1[:, 0:63:3]
        # bool_mask_y = bool_mask_1[:, 1:63:3]
        # bool_mask_xy = bool_mask_x + bool_mask_y

        bool_mask = bool_mask #+ bool_mask_xy
        if bool_mask.any() and count >= 0:
            joints[:, 0:joints.shape[1]:3][bool_mask] = 0.0
            joints[:, 1:joints.shape[1]:3][bool_mask] = 0.0
            joints[:, 2:joints.shape[1]:3] = np.where(joints[:, 2:joints.shape[1]:3] >= mm, joints[:, 2:joints.shape[1]:3], mm)

            joints = replace_nulls(joints)
            if correct_bones:
                joints = check_bones(joints, orient)
            joints = interpolate_joints(joints, count -1, orient)
        else:
            print(str(count))
            if correct_bones:
                joints = check_bones(joints, orient)
            joints = replace_nulls(joints)
            #joints = np.apply_along_axis(savgol_per_joint, 0, joints)
            return joints

    return joints


def correct_finger(fingers_t, lb0_tm, lb1_tm, lb2_tm, lb3_tm):
    ej = 0

    lb0_t = np.sqrt((fingers_t[:, 1, 0] - fingers_t[:, 0, 0]) ** 2 + (fingers_t[:, 1, 1] - fingers_t[:, 0, 1]) ** 2)
    lb1_t = np.sqrt((fingers_t[:, 2, 0] - fingers_t[:, 1, 0]) ** 2 + (fingers_t[:, 2, 1] - fingers_t[:, 1, 1]) ** 2)
    lb2_t = np.sqrt((fingers_t[:, 3, 0] - fingers_t[:, 2, 0]) ** 2 + (fingers_t[:, 3, 1] - fingers_t[:, 2, 1]) ** 2)
    lb3_t = np.sqrt((fingers_t[:, 4, 0] - fingers_t[:, 3, 0]) ** 2 + (fingers_t[:, 4, 1] - fingers_t[:, 3, 1]) ** 2)

    for i in range(lb0_t.shape[0]):
        x0 = fingers_t[i, 0, 0]
        x1 = fingers_t[i, 1, 0]
        y0 = fingers_t[i, 0, 1]
        y1 = fingers_t[i, 1, 1]

        z0 = fingers_t[i, 0, 2]
        xtar, ytar, ztar = correct_bone(x0, x1, y0, y1, lb0_t[i], lb0_tm, z0)

        fingers_t[i, 1, 0] = xtar
        fingers_t[i, 1, 1] = ytar
        fingers_t[i, 1, 2] = ztar

        x0 = fingers_t[i, 1, 0]
        x1 = fingers_t[i, 2, 0]
        y0 = fingers_t[i, 1, 1]
        y1 = fingers_t[i, 2, 1]

        z0 = fingers_t[i, 1, 2]

        xtar, ytar, ztar = correct_bone(x0, x1, y0, y1, lb1_t[i], lb1_tm, z0)

        fingers_t[i, 2, 0] = xtar
        fingers_t[i, 2, 1] = ytar
        fingers_t[i, 2, 2] = ztar

        x0 = fingers_t[i, 2, 0]
        x1 = fingers_t[i, 3, 0]
        y0 = fingers_t[i, 2, 1]
        y1 = fingers_t[i, 3, 1]

        z0 = fingers_t[i, 2, 2]

        xtar, ytar, ztar = correct_bone(x0, x1, y0, y1, lb2_t[i], lb2_tm, z0)

        fingers_t[i, 3, 0] = xtar
        fingers_t[i, 3, 1] = ytar
        fingers_t[i, 3, 2] = ztar

        x0 = fingers_t[i, 3, 0]
        x1 = fingers_t[i, 4, 0]
        y0 = fingers_t[i, 3, 1]
        y1 = fingers_t[i, 4, 1]

        z0 = fingers_t[i, 3, 2]

        xtar, ytar, ztar = correct_bone(x0, x1, y0, y1, lb3_t[i], lb3_tm, z0)

        fingers_t[i, 4, 0] = xtar
        fingers_t[i, 4, 1] = ytar
        fingers_t[i, 4, 2] = ztar

    return fingers_t#, ej

def correct_bone(x0, x1, y0, y1, lb, lbm, z0):
    xt, yt, zt = symbols('xt yt zt')

    if lb > lbm:
        try:

            sol = linsolve([Eq(x0 + lb/lbm * (xt - x0), x1),
                  Eq(y0 + lb/lbm * (yt - y0), y1)], [xt, yt])

            sol0 = []
            sol0.append((sol.args[0][0], sol.args[0][1], 0.0))
        except Exception as e:
            print(e)
            sol0 = []
            sol0.append((x1, y1, 0.0))

        if isinstance(sol0, dict):
            sol_l = []
            for key, value in sol0.items():
                sol_l.append(value)

            sol_t = tuple(sol_l)
            sol_l = []
            sol_l.append(sol_t)
            sol0 = sol_l

        if not isinstance(sol0[0][-1], float):
            sol0l = list(sol0[0])
            sol0l[-1] = 0.0
            sol0[0] = tuple(sol0l)

    else:
        sol = []
        sol.append((x1, y1, 0.0))
        sol0 = sol

    #e0 = (x1 - sol0[0][0]) ** 2 + (y1 - sol0[0][1]) ** 2 + 0.000000000000000001

    #return sol[0][0], sol[0][1], sol[0][2], np.array([e0, e1, e2])[s]
    return sol0[0][0], sol0[0][1], sol0[0][2]#, e0


def correction(new_joints, c=3):

    for j in range(1, 21, 4):
        fingers_t = np.concatenate((new_joints[:, 0:1, :], new_joints[:, j:j + 4, :]), 1)
        lb0_t = np.sqrt((fingers_t[:, 1, 0] - fingers_t[:, 0, 0]) ** 2 + (fingers_t[:, 1, 1] - fingers_t[:, 0, 1]) ** 2)
        lb1_t = np.sqrt((fingers_t[:, 2, 0] - fingers_t[:, 1, 0]) ** 2 + (fingers_t[:, 2, 1] - fingers_t[:, 1, 1]) ** 2)
        lb2_t = np.sqrt((fingers_t[:, 3, 0] - fingers_t[:, 2, 0]) ** 2 + (fingers_t[:, 3, 1] - fingers_t[:, 2, 1]) ** 2)
        lb3_t = np.sqrt((fingers_t[:, 4, 0] - fingers_t[:, 3, 0]) ** 2 + (fingers_t[:, 4, 1] - fingers_t[:, 3, 1]) ** 2)

        lb0_tm = np.mean(lb0_t)
        lb1_tm = np.mean(lb1_t)
        lb2_tm = np.mean(lb2_t)
        lb3_tm = np.mean(lb3_t)

        for t in range(1):
            fingers_t = correct_finger(fingers_t, lb0_tm, lb1_tm, lb2_tm, lb3_tm)
            print(t)
            #print(ef)

        new_joints[:, 0:1, :] = fingers_t[:, 0:1, :]
        new_joints[:, j:j + 4, :] = fingers_t[:, 1:, :]

    return new_joints[:, :, :2]


def correction_pose(new_joints, c=3):

    return new_joints[:, :, :2]


if __name__ == "__main__":
    base_dir = "/home/stephanie/Documents/Chapter3/Code/"
    img_dir = "/home/stephanie/Documents/Chapter3/Code/ProgressiveTransformersSLP-master/data_phoenix/PHOENIX-2014-T/features/fullFrame-210x260px/test"

    data_mats = ['phoenix_data_with_confs_input_73_dev.mat']
    count = 0
    legs = []
    with open(base_dir + 'vocab.txt', "r", encoding="utf-8") as f:
        word_to_idx = dict(line.strip().split(' ') for line in f)
    for f in data_mats:
        sess = f.split("_")[-1].split(".")[0]
        gloss_file = open(sess + '.gloss', 'w', encoding="utf-8")
        skels_file = open(sess + '.skels', 'w', encoding='utf-8')
        files_file = open(sess + '.files', 'w', encoding="utf-8")

        with io.open(base_dir + sess + '.corpus.csv', "r", encoding='utf-8', errors='surrogateescape') as annot:
            annotations = annot.readlines()
        data = mat73.loadmat(base_dir + f)
        input = data['input']
        k = len(input['hand_l'])

        for l in range(0, k):
            name = input['name'][l]
            matching = [s for s in annotations if name in s]
            if matching == []:
                print("help")
            else:
                hl = input['hand_l'][l]
                hr = input['hand_r'][l]
                pose = input['pose'][l]
                face = input['face'][l]
                path = sess + "/" + name

                pose = pose[:, :24]

                seq2d = np.hstack((pose, hl, hr))
                p = multiprocessing.Process(
                    target=convert,
                    args=(seq2d, face,)
                )
                p.start()
                p.join()
                #pose_hands3d, face2d = convert(seq2d, face)
                data = np.load('tmp.npz')
                pose_hands3d = data['pose_hands3d']

                face2d = data['face2d']
                gloss_seq = matching[0].split("|")[-2].strip()

                y_line = ""
                for h in range(len(pose_hands3d)):
                    #plt.scatter(pose_hands3d[h], pose_hands3d[h])
                    input_line = np.hstack((pose_hands3d[h], face2d[h])) / 3.0
                    c = h / len(pose_hands3d)
                    ys = np.array2string(input_line, separator=' ').strip('[').strip(']').replace('\n', '')
                    ys = ys + ' ' + str(c) + '  '
                    y_line = y_line + ys

                gloss_file.write(gloss_seq + '\n')
                skels_file.write(y_line + '\n')
                files_file.write(path + '\n')

        gloss_file.close()
        skels_file.close()
        files_file.close()



