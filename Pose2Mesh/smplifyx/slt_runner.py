from back_trans.signjoey.training import build_inference_model_for_backtrans
from back_trans.signjoey.vocabulary import SIL_TOKEN
from back_trans.signjoey.loss import XentLoss

import torch
import os


class SltRunner:
    def __init__(self, cfg):
        self.slt = build_inference_model_for_backtrans(cfg)
        chkpt = os.path.join(cfg['training']['model_dir'], 'best.ckpt')
        if chkpt is not None:
            checkpoint = torch.load(chkpt, map_location=lambda storage, loc: storage)
            self.slt.load_state_dict(checkpoint["model_state"])

        for p in self.slt.parameters():
            p.requires_grad = False
        self.slt.eval()

        self.do_recognition = (
            cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            cfg["training"].get("translation_loss_weight", 1.0) > 0.0
        )
        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=cfg["training"])
        else:
            self.recognition_loss_function = None
            self.recognition_loss_weight = 0.0
            self.eval_recognition_beam_size = 1
            self.gls_silence_token = 0

        if self.do_translation:
            self._get_translation_params(train_config=cfg["training"])

        self.slt.cuda()

    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.slt.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=1, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )