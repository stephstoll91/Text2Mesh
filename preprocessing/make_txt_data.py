import numpy as np

session = "test"
data_path = '/home/stephanie/Documents/Chapter3/Code/Text2Pose/data/phoenix14t_smooth_0' + session + '/'

Xdim = int(np.load(data_path + 'Xdim.npy'))
Ydim = int(np.load(data_path + 'Ydim.npy'))
label_dim = np.load(data_path + 'label_dim.npy')

X = np.load(data_path + 'input.npy', 'c')[:, 0:Xdim]
Y = np.load(data_path + 'output.npy', 'c')[:, 0:Ydim]
lab = np.load(data_path + 'label_in.npy', 'c')
L_den = np.load(data_path + 'L_d.npy', 'c')
Xmean = np.load(data_path + 'Xmean.npy')[0:Xdim]
Xstd = np.load(data_path + 'Xstd.npy')[0:Xdim]
Ymean = np.load(data_path + 'Ymean.npy')[0:Ydim]
Ystd = np.load(data_path + 'Ystd.npy')[0:Ydim]
Lmean = np.load(data_path + 'Lmean.npy')
Lstd = np.load(data_path + 'Lstd.npy')

seq_borders = np.load(data_path + 'seq_borders.npy')

with open(data_path + 'phoenix14t_smooth_0' + session + '.txt') as f:
    paths = f.readlines()

with open('/home/stephanie/Documents/Chapter3/Code/Text2Pose/vocab.txt', "r", encoding="utf-8") as f:
    idx_to_word = dict(line.strip().split(' ') for line in f)
    idx_to_word = {value: key for key, value in idx_to_word.items()}


gloss_file = open(session +'.gloss', 'w', encoding="utf-8")
skels_file = open(session +'.skels', 'w', encoding='utf-8')
files_file = open(session +'.files', 'w', encoding="utf-8")
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
        yy = y[j, :] / 3
        ys = np.array2string(yy, separator=' ').strip('[').strip(']').replace('\n', '')
        ys = ys + ' ' + str(c[j]) + '  '
        y_line = y_line + ys

    gloss_file.write(gloss_line + '\n')
    skels_file.write(y_line + '\n')
    files_file.write(ff_line)
    #print(y_line)

gloss_file.close()
skels_file.close()
files_file.close()
