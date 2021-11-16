from pre_processing import *


if __name__ == "__main__":
    base_dir = "/home/stephanie/Documents/Chapter3/Code/"
    img_dir = "/home/stephanie/Documents/Stoll_PHD_remote/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"

    #data_mats = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
    #data_mats = [f for f in data_mats if f.endswith('.mat')]

    data_mats = ['phoenix_data_with_confs_input_train.mat']
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
        data = mat73.loadmat(base_dir + f)
        #data = sio.loadmat(base_dir + f)
        input = data['input']
        #_, k = input.shape
        k = len(input['hand_l'])

        for l in range(0, k):

            #name = input[0, l]['name']
            name = input['name'][l]
            matching = [s for s in annotations if name in s]
            if matching == []:
                print("help")
            else:
                #hl = input[0, l]['hand_l']
                #hr = input[0, l]['hand_r']
                hl = input['hand_l'][l]
                hr = input['hand_r'][l]
                path = sess + "/" + name

                sequence_ID = base_dir + path + "/"
                print(sequence_ID)
                if not os.path.exists(sequence_ID):

                    hlpp = np.copy(hl)
                    hrpp = np.copy(hr)

                    hlpp = interpolate_joints(hlpp, 100, False)
                    hrpp = interpolate_joints(hrpp, 100, True)

                    if hlpp.size > 0 and hrpp.size > 0:

                        jx = hlpp[:, 0:63:3]
                        jy = hlpp[:, 1:63:3]
                        jz = np.zeros_like(jx)
                        jz.fill(0.2)

                        jx = np.expand_dims(jx, -1)
                        jy = np.expand_dims(jy, -1)
                        jz = np.expand_dims(jz, -1)

                        new_joints = np.concatenate((jx, jy, jz), -1)

                        hlpp = correction(new_joints, 3)

                        jx = hrpp[:, 0:63:3]
                        jy = hrpp[:, 1:63:3]
                        jz = np.zeros_like(jx)
                        jz.fill(0.2)

                        jx = np.expand_dims(jx, -1)
                        jy = np.expand_dims(jy, -1)
                        jz = np.expand_dims(jz, -1)

                        new_joints = np.concatenate((jx, jy, jz), -1)

                        hrpp = correction(new_joints, 3)

                        hlpp = np.reshape(hlpp, (hlpp.shape[0], -1))
                        hrpp = np.reshape(hrpp, (hrpp.shape[0], -1))

                    if hlpp.size > 0 and hrpp.size > 0:
                        data_to_file(hlpp, hrpp, img_dir, path, base_dir)
                        #print("pling!\n")
