
if __name__ == "__main__":
    base_dir = "/home/stephanie/Documents/Chapter3/Code/Text2Pose/back_trans/data/smooth_corr_all/"
    sessions = ['train', 'test', 'dev']

    for sess in sessions:
        skels_file_3D = open(base_dir + sess + '.skels', 'r', encoding='utf-8')
        skels_file_2D = open(base_dir + sess + '.skels2dHands', 'w', encoding='utf-8')
        i = 0

        for skel_line in skels_file_3D:
            i += 1

            # Strip away the "\n" at the end of the line
            skel_line = skel_line.strip()

            # Split target into joint coordinate values
            skel_line = skel_line.split(" ")
            skel_line = list(filter(None, skel_line))
            if len(skel_line) == 1:
                continue

            skel_frames = [skel_line[i:i + 291] for i in range(0, len(skel_line), 291)]
            skel_frames_2D = ""
            for skf in skel_frames:
                c = skf[-1]
                xyz = skf[0:-1]
                face = xyz[150:]
                xyz = xyz[24:-140]
                x = xyz[0::3]
                y = xyz[1::3]

                xy = [None] * (len(x) + len(y))
                xy[::2] = x
                xy[1::2] = y

                new_skf = xy #+ face

                new_skf = " ".join(new_skf) + ' ' + c + '  '
                skel_frames_2D = skel_frames_2D + new_skf

            skels_file_2D.write(skel_frames_2D + '\n')
        skels_file_2D.close()
        skels_file_3D.close()


