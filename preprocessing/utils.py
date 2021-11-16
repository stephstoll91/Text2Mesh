import numpy as np
from PIL import Image, ImageOps
import os
from make_hand_maps_numpy import make_heatmaps
import json


def remove_global_context(uvis, vvis):

    vsz = 260
    usz = 210
    minsz = vsz
    maxsz = usz

    umin = min(uvis)
    vmin = min(vvis)
    umax = max(uvis)
    vmax = max(vvis)

    B = round(2.2 * max([umax - umin, vmax - vmin]))

    us = 0
    ue = usz - 1

    vs = 0
    ve = vsz - 1

    umid = umin + (umax - umin) / 2
    vmid = vmin + (vmax - vmin) / 2

    if (B < minsz - 1):

        us = round(max(0, umid - B / 2))
        ue = us + B

        if (ue > usz - 1):
            d = ue - (usz - 1)
            ue = ue - d
            us = us - d

        vs = round(max(0, vmid - B / 2))
        ve = vs + B

        if (ve > vsz - 1):
            d = ve - (vsz - 1)
            ve = ve - d
            vs = vs - d

    if (B >= minsz - 1):

        B = minsz - 1
        if usz == minsz:
            vs = round(max(0, vmid - B / 2))
            ve = vs + B

            if (ve > vsz - 1):
                d = ve - (vsz - 1)
                ve = ve - d
                vs = vs - d

        if vsz == minsz:
            us = round(max(0, umid - B / 2))
            ue = us + B

            if (ue > usz - 1):
                d = ue - (usz - 1)
                ue = ue - d
                us = us - d

    us = int(us) if us >= 0 else 0
    vs = int(vs) if vs >= 0 else 0
    ue = int(ue) if ue >= 0 else 0
    ve = int(ve) if ve >= 0 else 0

    if (ue - us) > 0 and (ve - vs) > 0:
        uvis = (uvis - us) * (209.0 / (ue - us))
        vvis = (vvis - vs) * (259.0 / (ve - vs))

    return uvis, vvis, ue, us, ve, vs


def remove_gc_np(sample):
    hl = sample[0:42]
    hr = sample[42:]

    hxgl = hl[0:84:2]
    hygl = hl[1:84:2]
    hxgr = hr[0:84:2]
    hygr = hr[1:84:2]

    hxl, hyl, uel, usl, vel, vsl = remove_global_context(hxgl, hygl)
    hxr, hyr, uer, usr, ver, vsr = remove_global_context(hxgr, hygr)

    left_hand = {}
    right_hand = {}

    left_hand['x'] = hxl
    left_hand['y'] = hyl
    left_hand['ue'] = uel
    left_hand['us'] = usl
    left_hand['ve'] = vel
    left_hand['vs'] = vsl

    right_hand['x'] = hxr
    right_hand['y'] = hyr
    right_hand['ue'] = uer
    right_hand['us'] = usr
    right_hand['ve'] = ver
    right_hand['vs'] = vsr

    return left_hand, right_hand


def make_maps(fake_Bs, real_Bs, sequence_ID, f):  # b*l*f

    fakes = {}
    reals = {}

    inter_f = [val for pair in zip(list(fake_Bs[:, 0]), list(fake_Bs[:, 1])) for val in pair]
    inter_r = [val for pair in zip(list(real_Bs[:, 0]), list(real_Bs[:, 1])) for val in pair]

    sample_fake = np.array(inter_f)
    sample_real = np.array(inter_r)
    left_fake, right_fake = remove_gc_np(sample_fake)
    left_real, right_real = remove_gc_np(sample_real)
    fakes['vals_left'] = left_fake
    fakes['vals_right'] = right_fake
    reals['vals_left'] = left_real
    reals['vals_right'] = right_real

    hand_maps_left_fake = make_heatmaps(left_fake['x'], left_fake['y'], False, sequence_ID, f, True)
    hand_maps_right_fake = make_heatmaps(right_fake['x'], right_fake['y'], False, sequence_ID, f, False)
    hand_maps_left_real = make_heatmaps(left_real['x'], left_real['y'], True, sequence_ID, f, True)
    hand_maps_right_real = make_heatmaps(right_real['x'], right_real['y'], True, sequence_ID, f, False)

    return

def make_imgs(left, right, path, fnum, image_path):
    left_us = left['us']
    left_ue = left['ue']
    left_vs = left['vs']
    left_ve = left['ve']

    right_vs = right['vs']
    right_ve = right['ve']
    right_us = right['us']
    right_ue = right['ue']

    session = path.split("/")[-2]
    full_session = os.path.join(image_path, session)
    fi = 'images' + '{:04}'.format(fnum + 1) + '.png'

    image = Image.open(os.path.join(full_session, fi))

    if (left_ue - left_us) > 0 and (left_ve - left_vs) > 0:
        img_left_real = image.resize((256, 256), box=(left_us, left_vs, left_ue, left_ve))
    else:
        img_left_real = image.resize((256, 256))

    if (right_ue - right_us) > 0 and (right_ve - right_vs) > 0:
        img_right_real = image.resize((256, 256), box=(right_us, right_vs, right_ue, right_ve))
    else:
        img_right_real = image.resize((256, 256))

    img_left_real.save(path + "/" + '{:03}'.format(fnum) + "_left" + ".png")
    img_right_real.save(path + "/" + '{:03}'.format(fnum) + "_right" + ".png")

    return img_left_real, img_right_real


def make_maps_and_imgs(fake_Bs, outpath, f, image_path, sani=False): # b*l*f

    fakes = {}
    inter_f = [val for pair in zip(list(fake_Bs[:, 0]), list(fake_Bs[:, 1])) for val in pair]
    sample_fake = np.array(inter_f)
    left_fake, right_fake = remove_gc_np(sample_fake)
    fakes['vals_left'] = left_fake
    fakes['vals_right'] = right_fake

    outpath_A = outpath + "A"
    outpath_B = outpath + "B"
    outpath_C = outpath + "C"

    if not os.path.exists(outpath_A):
        os.makedirs(outpath_A)
    if not os.path.exists(outpath_B):
        os.makedirs(outpath_B)
    if not os.path.exists(outpath_C):
        os.makedirs(outpath_C)

    with open(outpath_C + "/" + '{:03}'.format(f) + "_left.json", 'w') as fp:
        left_fake['x'] = left_fake['x'].tolist()
        left_fake['y'] = left_fake['y'].tolist()
        json.dump(left_fake, fp)
    with open(outpath_C + "/" + '{:03}'.format(f) + "_right.json", 'w') as fp:
        right_fake['x'] = right_fake['x'].tolist()
        right_fake['y'] = right_fake['y'].tolist()
        json.dump(right_fake, fp)

    hand_maps_left_fake = make_heatmaps(left_fake['x'], left_fake['y'], False, outpath_A, f, True)
    hand_maps_right_fake = make_heatmaps(right_fake['x'], right_fake['y'], False, outpath_A, f, False)

    img_left_fake, img_right_fake = make_imgs(left_fake, right_fake, outpath_B, f, image_path)

    if sani:
        sequence_ID_sanity = outpath + "/SANI"
        if not os.path.exists(sequence_ID_sanity):
            os.makedirs(sequence_ID_sanity)

        img_left_blend = ImageOps.Image.blend(hand_maps_left_fake.convert('RGB'), img_left_fake, 0.5)
        img_left_blend.save(sequence_ID_sanity + "/try_" + '{:03}'.format(f) + "_left.png")
        img_right_blend = ImageOps.Image.blend(hand_maps_right_fake.convert('RGB'), img_right_fake, 0.5)
        img_right_blend.save(sequence_ID_sanity + "/try_" + '{:03}'.format(f) + "_right.png")
    return