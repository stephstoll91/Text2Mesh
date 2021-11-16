import numpy as np
import os

session = "train"

data_path = '/home/stephanie/Documents/Chapter3/Code/Text2Pose/data/'

data_dirs = sorted([d for d in os.listdir(data_path) if session in d])
xmeans = []
ymeans = []
xstds = []
ystds = []

for dd in data_dirs:
    abs_path = os.path.join(data_path, dd)
    if os.path.isdir(abs_path) and "hands" not in abs_path:
        Xdim = int(np.load(abs_path + '/Xdim.npy'))
        Ydim = int(np.load(abs_path + '/Ydim.npy'))
        Xmean = np.load(abs_path + '/Xmean.npy')[0:Xdim]
        Xstd = np.load(abs_path + '/Xstd.npy')[0:Xdim]
        Ymean = np.load(abs_path + '/Ymean.npy')[0:Ydim]
        Ystd = np.load(abs_path + '/Ystd.npy')[0:Ydim]

        xmeans.append(Xmean)
        xstds.append(Xstd)
        ymeans.append(Ymean)
        ystds.append(Ystd)

xmeans = np.mean(np.vstack(xmeans), axis=0)
ymeans = np.mean(np.vstack(ymeans), axis=0)
xstds = np.mean(np.vstack(xstds), axis=0)
ystds = np.mean(np.vstack(ystds), axis=0)

np.save('Xmean.npy', xmeans)
np.save('Ymean.npy', ymeans)
np.save('Xstd.npy', xstds)
np.save('Ystd.npy', ystds)

with open('/home/stephanie/Documents/Chapter3/Code/Text2Pose/vocab.txt', "r", encoding="utf-8") as f:
    idx_to_word = dict(line.strip().split(' ') for line in f)
    idx_to_word = {value: key for key, value in idx_to_word.items()}


gloss_file = open(session +'.gloss', 'w', encoding="utf-8")
skels_file = open(session +'.skels', 'w', encoding='utf-8')
files_file = open(session +'.files', 'w', encoding="utf-8")

for dd in data_dirs:
    abs_path = os.path.join(data_path, dd)
    if os.path.isdir(abs_path) and "hands" not in abs_path:
        Xdim = int(np.load(abs_path + '/Xdim.npy'))
        Ydim = int(np.load(abs_path + '/Ydim.npy'))
        Xmean = np.load(abs_path + '/Xmean.npy')[0:Xdim]
        Xstd = np.load(abs_path + '/Xstd.npy')[0:Xdim]
        Ymean = np.load(abs_path + '/Ymean.npy')[0:Ydim]
        Ystd = np.load(abs_path + '/Ystd.npy')[0:Ydim]

        label_dim = np.load(abs_path + '/label_dim.npy')

        X = np.load(abs_path + '/input.npy', 'c')[:, 0:Xdim]
        Y = np.load(abs_path + '/output.npy', 'c')[:, 0:Ydim]
        lab = np.load(abs_path + '/label_in.npy', 'c')
        L_den = np.load(abs_path + '/L_d.npy', 'c')
        Lmean = np.load(abs_path + '/Lmean.npy')
        Lstd = np.load(abs_path + '/Lstd.npy')

        seq_borders = np.load(abs_path + '/seq_borders.npy')

        with open(abs_path + '/' + dd + '.txt') as f:
            paths = f.readlines()

        for i in seq_borders:
            start = i[0]
            end = i[1]

            y = Y[start:end]
            g = lab[start:end]
            ff = paths[start:end]

            c = g[:, 0]

            gl = g[0, 1:]
            gl = gl[gl >= 0]
            gloss_line = []
            for gg in gl:
                gloss_line.append(idx_to_word[str(int(gg))])

            gloss_line = " ".join(gloss_line)

            ff_line = ff[0]

            y_line = ""
            for j in range(y.shape[0]):
                yy = y[j, :]
                yy = yy * Ystd + Ymean
                yy = ((yy - ymeans) / ystds) / 3
                ys = np.array2string(yy, separator=' ').strip('[').strip(']').replace('\n', '')
                ys = ys + ' ' + str(c[j]) + '  '
                y_line = y_line + ys

            gloss_file.write(gloss_line + '\n')
            skels_file.write(y_line + '\n')
            files_file.write(ff_line)