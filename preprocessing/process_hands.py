# coding=utf-8
import numpy as np
import math
import scipy.io as sio
#from scipy.spatial import procrustes
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tsaug.src.tsaug import TimeWarp, AddNoise
import io

def apply_procrustes(ref, now, l):

    transformed = []
    now_frame = np.reshape(now[0], (-1, 2))
    disp, transformed_points, rot_scale_trans = procrustes(ref, now_frame)

    for p in range(l):
        frame = np.reshape(now[p], (-1, 2))

        # c — Translation component
        # T — Orthogonal rotation and reflection component
        # b — Scale component
        # Z = b * Y * T + c;

        frame_trans = np.dot(rot_scale_trans['scale'], frame)

        frame_trans = np.dot(frame_trans, rot_scale_trans['rotation'])

        frame_trans = frame_trans + rot_scale_trans['translation']

        # plt.figure()
        # plt.scatter(ref[:, 0], ref[:, 1])
        # plt.scatter(transformed_points[:, 0], transformed_points[:, 1])
        # plt.scatter(frame_trans[:, 0], frame_trans[:, 1])
        # plt.gca().invert_yaxis()
        # plt.savefig('./norms/' + str(p) + '.png')
        # plt.close()

        transformed.append(frame_trans.flatten())

    return np.asarray(transformed), rot_scale_trans


def procrustes(X, Y, scaling=False, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1
python -m pip install --upgrade pip
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def normalise(_input, _l):
    #load reference skeleton
    ref_pose = np.load('/home/steph/Documents/Chapter2/Code/smile-tmp/Mollica-ref-pose.npy')[0]
    #ref_face = np.load('Mollica-ref-face.npy')[0]
    #ref_hl = np.load('Mollica-ref-hand-left.npy')[0]
    #ref_hr = np.load('Mollica-ref-hand-right.npy')[0]

    ref_pose = np.reshape(ref_pose, (-1, 2))
    #ref_face = np.reshape(ref_face, (-1, 2))
    #ref_hl = np.reshape(ref_hl, (-1, 2))
    #ref_hr = np.reshape(ref_hr, (-1, 2))

    pose = _input[0, _l]['pose']
    face = _input[0,_l]['face']
    hl = _input[0,_l]['hand_l']
    hr = _input[0, _l]['hand_r']

    poseT, rst = apply_procrustes(ref_pose, pose, len(pose))

    faceT = []
    hlT = []
    hrT = []
    for p in range(len(pose)):
        frame = np.reshape(face[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        faceT.append(frame_trans.flatten())

        frame = np.reshape(hl[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        hlT.append(frame_trans.flatten())

        frame = np.reshape(hr[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        hrT.append(frame_trans.flatten())



    # faceT = apply_procrustes(ref_face, face, len(face))
    # hlT = apply_procrustes(ref_hl, hl, len(hl))
    # hrT = apply_procrustes(ref_hr, hr, len(hr))


    _input[0, _l]['pose'] = poseT
    _input[0, _l]['face'] = np.asarray(faceT)
    _input[0, _l]['hand_l'] = np.asarray(hlT)
    _input[0, _l]['hand_r'] = np.asarray(hrT)

    return _input

def moving_avg(joints):
    w = 3
    je = np.repeat(joints[0], w + 1)
    js = np.repeat(joints[-1], w + 1)
    joints = np.concatenate((je, joints, js), 0)
    joints = np.convolve(joints, np.ones(w), 'valid') / w
    return joints[w:-w]

def interpolate_joints(joints, legs=None):

    _, l = joints.shape

    # do preliminary check:
    z = np.where(joints <= 0.00000)[0]
    nz = np.where(joints > 0.00000)[0]
    if z.size > nz.size * 1.2:
        return np.asarray([]), legs

    for c in range(l):
        indx = np.where(joints[:, c] <= 0.00000)[0]
        cindx = np.where(joints[:, c] > 0.00000)[0]
        try:
            if cindx.size == 0:
                raise
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
                    inter1 = interp1d(x, y)
                    newx = np.arange(len(joints[:, c]))
                    joints[:, c] = inter1(newx)

        except:
            print("Lt. Dan!!")
            joints[:, c] = legs[c]


    if joints.all() > 0.0000 and l <= 28:
       legs = np.mean(joints, axis=0)

    joints = np.apply_along_axis(moving_avg, 0, joints)
    hh = np.min(joints)
    print(str(hh))
    return joints, legs

base_dir = "/home/stephanie/Documents/Chapter3/Code/"

data_mats = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
data_mats = [f for f in data_mats if f.endswith('.mat')]

data_mats = ['phoenix_data_input_dev.mat']
count = 0
legs = []
with open(base_dir + 'vocab.txt', "r", encoding="utf-8") as f:
    word_to_idx = dict(line.strip().split(' ') for line in f)
for f in data_mats:
    inputs = []
    outputs = []
    labels_in = []
    labels_out = []
    paths = []

    sess = f.split("_")[-1].split(".")[0]

    with io.open(base_dir + sess + '.corpus.csv', "r", encoding='utf-8', errors='surrogateescape') as annot:
        annotations = annot.readlines()

    data = sio.loadmat(base_dir + f)
    input = data['input']
    _,k = input.shape

    for l in range(0, k):

        name = input[0, l]['name']
        matching = [s for s in annotations if name[0] in s]
        if matching == []:
            print("help")
        else:
            pose = input[0, l]['pose']
            face = input[0, l]['face']
            hl = input[0, l]['hand_l']
            hr = input[0, l]['hand_r']
            path = sess + "/" + input[0, l]['name'][0]

            print(path)
            if pose != []:
                pose, legs = interpolate_joints(pose, legs)
                face, _ = interpolate_joints(face)
                hl, _ = interpolate_joints(hl)
                hr, _ = interpolate_joints(hr)

                if pose.size > 0 and face.size > 0 and hl.size > 0 and hr.size > 0:
                    gloss_seq = matching[0].split("|")[-2].strip()
                    gloss_list = gloss_seq.split(" ")
                    gloss_list = [string for string in gloss_list if string != ""]
                    gloss_id_list = []

                    for g in gloss_list:
                        try:
                            gloss_id_list.append(word_to_idx[g])
                        except:
                            print("key error")
                            gloss_id_list.append(word_to_idx["unk"])

                    while len(gloss_id_list) < 30:
                        gloss_id_list.append(-1)
                    #seq_inputs = []
                    #seq_outputs = []
                    #seq_labels_in = []
                    #seq_labels_out = []

                    # aug = np.hstack((pose, face, hl, hr))
                    # augmenter = (TimeWarp() + AddNoise())
                    # all_aug = augmenter.augment(aug.T).T
                    #
                    # pose_aug = all_aug[:, 0:28]
                    # face_aug = all_aug[:, 28:168]
                    # hl_aug = all_aug[:, 168:210]
                    # hr_aug = all_aug[:, 210:]

                    label_line_in = []
                    label_line_out = []

                    for h in range(len(pose)-1):


                        framep = np.reshape(pose[h], (-1, 2))
                        facep = np.reshape(face[h], (-1, 2))
                        hlp = np.reshape(hl[h], (-1, 2))
                        hrp = np.reshape(hr[h], (-1, 2))

                        if h == 0:
                            input_line = np.hstack((hl[h], hr[h], hl[h] - hl[h], hr[h] - hr[h]))

                        else:
                            input_line = np.hstack((hl[h], hr[h], hl[h] - hl[h-1], hr[h] - hr[h-1]))

                        output_line = np.hstack((hl[h + 1], hr[h + 1], hl[h + 1] - hl[h], hr[h + 1] - hr[h]))

                        label_line_in = np.hstack((h, h/len(pose), np.asarray(gloss_id_list).astype(np.float)))

                        inputs.append(input_line)
                        outputs.append(output_line)
                        labels_in.append(np.asarray(label_line_in).astype(np.float))
                        paths.append(path)

                    input_line = output_line
                    output_line = np.hstack((hl[h + 1], hr[h + 1],
                                             hl[h + 1] - hl[h + 1], hr[h + 1] - hr[h + 1]))
                    label_line_in = np.hstack((h+1, 1.0, np.asarray(gloss_id_list).astype(np.float)))
                    inputs.append(input_line)
                    outputs.append(output_line)
                    labels_in.append(np.asarray(label_line_in).astype(np.float))
                    paths.append(path)

                    count += 1

                    # for h in range(len(pose_aug) - 1):
                    #
                    #     framep = np.reshape(pose_aug[h], (-1, 2))
                    #     facep = np.reshape(face_aug[h], (-1, 2))
                    #     hlp = np.reshape(hl_aug[h], (-1, 2))
                    #     hrp = np.reshape(hr_aug[h], (-1, 2))
                    #
                    #     if h == 0:
                    #         input_line = np.hstack((hl_aug[h], hr_aug[h],
                    #                                 hl_aug[h] - hl_aug[h], hr_aug[h] - hr_aug[h]))
                    #
                    #     else:
                    #         input_line = np.hstack((hl_aug[h], hr_aug[h],
                    #                                 hl[h] - hl[h - 1], hr[h] - hr[h - 1]))
                    #
                    #     output_line = np.hstack((hl[h + 1], hr[h + 1],
                    #                              hl[h + 1] - hl[h], hr[h + 1] - hr[h]))
                    #
                    #     label_line_in = np.hstack((h, h / len(pose_aug), np.asarray(gloss_id_list).astype(np.float)))
                    #
                    #     inputs.append(input_line)
                    #     outputs.append(output_line)
                    #     labels_in.append(np.asarray(label_line_in).astype(np.float))
                    #     paths.append(path)
                    #
                    # input_line = output_line
                    # output_line = np.hstack((hl[h + 1], hr[h + 1],
                    #                          hl[h + 1] - hl[h + 1], hr[h + 1] - hr[h + 1]))
                    # label_line_in = np.hstack((h + 1, 1.0, np.asarray(gloss_id_list).astype(np.float)))
                    # inputs.append(input_line)
                    # outputs.append(output_line)
                    # labels_in.append(np.asarray(label_line_in).astype(np.float))
                    # paths.append(path)

                    # count += 1

    L_d = np.asarray(labels_in)[:, 1:]

    H = np.asarray(labels_in)[:, 0]
    seq_borders = list(np.where(H == 0.0))
    seq_borders_d = np.repeat(seq_borders[0], 2)
    seq_borders = seq_borders_d[1:]
    seq_borders = np.append(seq_borders, int(H.size))
    seq_borders = seq_borders.reshape((-1, 2))

    Xmean, Xstd = np.asarray(inputs).mean(axis=0), np.asarray(inputs).std(axis=0)
    Ymean, Ystd = np.asarray(outputs).mean(axis=0), np.asarray(outputs).std(axis=0)
    #Lmean, Lstd = np.asarray(labels_in)[:, 1:].mean(axis=0), np.asarray(labels_in)[:, 1:].std(axis=0)
    Lmean, Lstd = np.asarray(labels_in)[:, 2:].mean(axis=0), np.asarray(labels_in)[:, 2:].std(axis=0)
    #Lomean, Lostd = Lo.mean(axis=0), Lo.std(axis=0)

    for i in range(Xstd.size):
        if (Xstd[i] == 0):
            Xstd[i] = 1
    for i in range(Ystd.size):
        if (Ystd[i] == 0):
            Ystd[i] = 1
    for i in range(Lstd.size):
        if (Lstd[i] == 0):
            Lstd[i] = 1

    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    #labels_in = np.asarray(labels_in)[:, 1:]
    labels_in = np.asarray(labels_in)
    inputs = (inputs - Xmean) / Xstd
    outputs = (outputs - Ymean) / Ystd
    #labels_in[:, 2:] = (labels_in[:, 2:] - Lmean) / Lstd
    labels_in = labels_in[:, 1:]
    Xdim = inputs.shape[1]
    Ydim = outputs.shape[1]
    label_dim = labels_in.shape[1]

    samples = inputs.shape[0]
    np.savez("/home/stephanie/Documents/Chapter3/Code/phoenix14t_hands_smooth_" + sess + ".npz",
             input=np.asarray(inputs), output=np.asarray(outputs), label_in=np.asarray(labels_in), L_d=L_d, seq_borders=seq_borders,
             Xmean=Xmean, Xstd=Xstd, Ymean=Ymean, Ystd=Ystd, Lmean=Lmean, Lstd=Lstd, Xdim=Xdim, Ydim=Ydim, label_dim=label_dim, samples=samples)
    
    with open("/home/stephanie/Documents/Chapter3/Code/phoenix14t_hands_smooth_" + sess + ".txt", 'w') as f:
        for item in paths:
            f.write("%s\n" % item)
