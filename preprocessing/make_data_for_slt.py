import io

base_dir = "/home/stephanie/Documents/Chapter3/Code/"

sess = "dev"

gloss_file = open(sess +'.gloss', 'w', encoding="utf-8")
text_file = open(sess +'.text', 'w', encoding="utf-8")
signer_file = open(sess +'.signer', 'w', encoding="utf-8")

with io.open(base_dir + sess + '.corpus.csv', "r", encoding='utf-8', errors='surrogateescape') as annot:
    annotations = annot.readlines()

with open(base_dir + sess +'.files', 'r', encoding="utf-8") as ff:
    files = ff.readlines()

for f in files:
    fs = f.split("/")[-1].strip() + "|"
    matching = [s for s in annotations if fs in s]
    if len(matching) != 1:
        print(fs)
    else:
        l = matching[0]
        ls = l.split("|")
        sig = ls[-3].strip()
        gls = ls[-2].strip()
        txt = ls[-1].strip() + " ."

        gloss_file.write(gls + '\n')
        signer_file.write(sig + '\n')
        text_file.write(txt + '\n')


gloss_file.close()
signer_file.close()
text_file.close()

