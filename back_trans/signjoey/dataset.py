# coding: utf-8
"""
Data module
"""
from torchtext.legacy import data
from torchtext.legacy.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import io
import numpy as np


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def load_dataset_from_txt_files(file_path):
    loaded_object = []

    names_path = file_path + 'files'
    signer_path = file_path + 'signer'
    gloss_path = file_path + 'gloss'
    text_path = file_path + 'text'
    sign_path = file_path + 'skels2d'

    with io.open(names_path, mode='r', encoding='utf-8') as names_file, \
            io.open(signer_path, mode='r', encoding='utf-8') as signer_file, \
            io.open(gloss_path, mode='r', encoding='utf-8') as gloss_file, \
            io.open(text_path, mode='r', encoding='utf-8') as text_file, \
            io.open(sign_path, mode='r', encoding='utf-8') as sign_file:

        for names_line, signer_line, gloss_line, text_line, sign_line in zip(names_file, signer_file, gloss_file, text_file, sign_file):

            # Strip away the "\n" at the end of the line
            names_line, signer_line, gloss_line, text_line, sign_line = names_line.strip(), signer_line.strip(), gloss_line.strip(), text_line.strip(), sign_line.strip()

            # Split target into joint coordinate values
            sign_line = sign_line.split(" ")
            sign_line = list(filter(None, sign_line))
            if len(sign_line) == 1:
                continue
            # Turn each joint into a float value
            sign_line = [(float(joint)) for joint in sign_line]
            # sign_frames = [sign_line[i:i + 241] for i in range(0, len(sign_line), 241)]
            #
            # sign_frames = [torch.unsqueeze(torch.FloatTensor(sf), 0) for sf in sign_frames]
            # sign_frames_t = torch.Tensor(len(sign_frames), 241)
            # torch.cat(sign_frames, out=sign_frames_t)
            # sign_frames_t = sign_frames_t[:, :-1]

            sign_frames = [sign_line[i:i + 241] for i in range(0, len(sign_line), 241)]

            sign_frames = [torch.unsqueeze(torch.FloatTensor(sf), 0) for sf in sign_frames]
            sign_frames_t = torch.Tensor(len(sign_frames), 241)
            torch.cat(sign_frames, out=sign_frames_t)
            sign_frames_t = sign_frames_t[:, :-141]


            #sign_frames_t = sign_frames_t * 3.0

            if not torch.any(sign_frames_t != sign_frames_t) and sign_frames_t.shape[-1] == 100:
                sample = {}
                sample['name'] = names_line
                sample['signer'] = signer_line
                sample['gloss'] = gloss_line
                sample['text'] = text_line
                sample['sign'] = sign_frames_t
                loaded_object.append(sample)
            else:
                print("error")

    return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            #tmp = load_dataset_file(annotation_file)
            tmp = load_dataset_from_txt_files(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
