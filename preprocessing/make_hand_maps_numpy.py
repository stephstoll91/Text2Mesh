import os
import json
from PIL import Image, ImageOps
import numpy as np
#from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from itertools import compress

xdim = 210
ydim = 260

def interp1d(p1, p2):
    if p1[0] <= p2[0]:
        pinterpx = np.arange(p1[0], p2[0] + 1, 1)
    else:
        pinterpx = np.arange(p1[0], p2[0] - 1, -1)
    if p1[1] <= p2[1]:
        pinterpy = np.arange(p1[1], p2[1] + 1, 1)
    else:
        pinterpy = np.arange(p1[1], p2[1] - 1, -1)

    length = max(pinterpx.shape[0], pinterpy.shape[0])

    if pinterpx.shape[0] < length:
        pinterpxs = np.zeros((length,), dtype=np.int64)
        for i in range(length):
            ii = i * pinterpx.shape[0] // length
            pinterpxs[i] = pinterpx[ii]
        pinterpx = pinterpxs
    if pinterpy.shape[0] < length:
        pinterpys = np.zeros((length,), dtype=np.int64)
        for i in range(length):
            ii = i * pinterpy.shape[0] // length
            pinterpys[i] = pinterpy[ii]
        pinterpy = pinterpys

    return pinterpx, pinterpy

def draw_subpart(xx, yy, hm, val):
    #xx = xx.tolist()
    #yy = yy.tolist()

    if len(xx) != len(yy):
        return hm
    for x, y in zip(xx, yy):
        if x > 0.0 and y > 0.0 and x < xdim and y < ydim:
                x = int(round(x))
                y = int(round(y))

                hm[y-2:y+2, x-2:x+2] = val

    if len(xx) < 2:
        return hm

    else:

        xxyy = np.vstack((np.array(xx), np.array(yy)))
        for pi in range(xxyy.shape[1] - 1):
            p1 = xxyy[:, pi]
            p2 = xxyy[:, pi + 1]
            pinterpx, pinterpy = interp1d(p1, p2)
            pinterpx = pinterpx.tolist()
            pinterpy = pinterpy.tolist()

            for x, y in zip(pinterpx, pinterpy):
                if x > 0.0 and y > 0.0 and x < xdim and y < ydim:
                    hm[y, x] = val

    return hm


def draw_hand(xh, yh, val, heat_map, from_gt, el):
    # fig = plt.figure()
    # plt.scatter(xh, yh)
    # plt.gca().invert_yaxis()
    # if from_gt:
    #     plt.savefig(str(el) + 'kps_from_gt.png')
    # else:
    #     plt.savefig(str(el) + 'kps_from_trans.png')
    # fig = plt.figure()
    # thumb
    x_t = [int(round(elem)) for elem in xh[0:5]]
    y_t = [int(round(elem)) for elem in yh[0:5]]

    # plt.scatter(x_t, y_t, c='b')
    # x_ts = savgol_filter(x_t[1:], 3, 1)
    # x_ts = np.insert(x_ts, 0, x_t[0], axis=0)
    # y_ts = savgol_filter(y_t[1:], 3, 1)
    # y_ts = np.insert(y_ts, 0, y_t[0], axis=0)
    # plt.plot(x_ts, y_ts, 'b')

    heat_map = draw_subpart(x_t, y_t, heat_map, val)

    # index
    x_i = [int(round(elem)) for elem in [xh[i] for i in (0, 5, 6, 7, 8)]]
    y_i = [int(round(elem)) for elem in [yh[i] for i in (0, 5, 6, 7, 8)]]

    # plt.scatter(x_i, y_i, c='g')
    # x_is = savgol_filter(x_i[1:], 3, 1)
    # x_is = np.insert(x_is, 0, x_i[0], axis=0)
    # y_is = savgol_filter(y_i[1:], 3, 1)
    # y_is = np.insert(y_is, 0, y_i[0], axis=0)
    # plt.plot(x_is, y_is, 'g')

    val += 1
    heat_map = draw_subpart(x_i, y_i, heat_map, val)

    # middle
    x_m = [int(round(elem)) for elem in [xh[i] for i in (0, 9, 10, 11, 12)]]
    y_m = [int(round(elem)) for elem in [yh[i] for i in (0, 9, 10, 11, 12)]]

    # plt.scatter(x_m, y_m, c='r')
    # x_ms = savgol_filter(x_m[1:], 3, 1)
    # x_ms = np.insert(x_ms, 0, x_m[0], axis=0)
    # y_ms = savgol_filter(y_m[1:], 3, 1)
    # y_ms = np.insert(y_ms, 0, y_m[0], axis=0)
    # plt.plot(x_ms, y_ms, 'r')

    val += 1
    heat_map = draw_subpart(x_m, y_m, heat_map, val)

    # ring
    x_r = [int(round(elem)) for elem in [xh[i] for i in (0, 13, 14, 15, 16)]]
    y_r = [int(round(elem)) for elem in [yh[i] for i in (0, 13, 14, 15, 16)]]

    # plt.scatter(x_r, y_r, c='y')
    # x_rs = savgol_filter(x_r[1:], 3, 1)
    # x_rs = np.insert(x_rs, 0, x_r[0], axis=0)
    # y_rs = savgol_filter(y_r[1:], 3, 1)
    # y_rs = np.insert(y_rs, 0, y_r[0], axis=0)
    # plt.plot(x_rs, y_rs, 'y')

    val += 1
    heat_map = draw_subpart(x_r, y_r, heat_map, val)

    # pinky
    x_p = [int(round(elem)) for elem in [xh[i] for i in (0, 17, 18, 19, 20)]]
    y_p = [int(round(elem)) for elem in [yh[i] for i in (0, 17, 18, 19, 20)]]

    # plt.scatter(x_p, y_p, c='m')
    # x_ps = savgol_filter(x_p[1:], 3, 1)
    # x_ps = np.insert(x_ps, 0, x_p[0], axis=0)
    # y_ps = savgol_filter(y_p[1:], 3, 1)
    # y_ps = np.insert(y_ps, 0, y_p[0], axis=0)
    # plt.plot(x_ps, y_ps, 'm')
    # plt.gca().invert_yaxis()
    # if from_gt:
    #     plt.savefig(str(el) + "kps_and_lines_from_gt.png")
    # else:
    #     plt.savefig(str(el) + "kps_and_lines_from_trans.png")

    val += 1
    heat_map = draw_subpart(x_p, y_p, heat_map, val)
    return heat_map

#from https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
def smoothListGaussian(list, degree=5):
    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weightGauss = []

    for i in range(window):
        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return np.asarray(smoothed)


def make_heatmaps(hx, hy, from_gt=False, name=".", el=0, left=False):

    val = 200

    hand_map = np.zeros((ydim, xdim), dtype=np.uint8)
    hand_map = draw_hand(hx, hy, val, hand_map, from_gt, el)

    img = Image.fromarray(hand_map).resize((256, 256)) #ImageOps.mirror()

    if left:
        orient = "_left"
    else:
        orient = "_right"

    # if from_gt:
    #     img.save(name + "/" + '{:03}'.format(el) + orient + "gt.png")
    # else:
    img.save(name + "/" + '{:03}'.format(el) + orient + ".png")

    return img

