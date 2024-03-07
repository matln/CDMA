# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import sys
import torch
import logging
import kaldiio
import traceback
import torchaudio

import speakernet.utils.utils as utils
# from local.pytorch.dsbn import convert_dsbn
# from local.pytorch.dabn import convert_dabn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def extract(wav_paths, gpu_id, job_id, model_params_path, nnet_config, out_dir, replacements, counter, lock):
    try:
        model_blueprint, model_creation = utils.read_nnet_config(nnet_config, log=False)
        model = utils.import_module(model_blueprint, model_creation)
        state_dict = {k.replace("module.", ""): v for k, v in torch.load(model_params_path, map_location="cpu").items()}
        # convert_dabn(model)
        model.load_state_dict(state_dict, strict=False)

        # Select device
        model = utils.select_model_device(model, True, gpu_id=gpu_id, log=False)
        model.eval()

        vectors_wspecifier = f"ark,scp:{out_dir}/xvector.{job_id}.ark,{out_dir}/xvector.{job_id}.scp"

        replacements = replacements.split(":")
        if replacements[0] != "none":
            assert len(replacements) == 2

        with kaldiio.WriteHelper(vectors_wspecifier) as writer:
            for wav_path in wav_paths:
                seg = wav_path.strip().split(" ")
                if seg[1] == "sox":
                    utt = seg[0]
                    wavs = seg[2:-4]
                    sigs = []
                    for wav in wavs:
                        if replacements[0] == "none":
                            _sig, _ = torchaudio.load(wav)
                        else:
                            _sig, _ = torchaudio.load(
                                wav.replace(replacements[0], replacements[1])
                            )
                        sigs.append(_sig)
                    sig = torch.cat(sigs, dim=1)

                elif len(seg) == 2:
                    utt, wav = seg
                    # print(f"Process utterance {utt}")
                    if replacements[0] == "none":
                        sig, fs = torchaudio.load(wav.strip())
                    else:
                        sig, fs = torchaudio.load(
                            wav.replace(replacements[0], replacements[1])
                        )

                elif len(seg) == 4:
                    utt, wav, start, end = seg
                    # num_frames = int((float(end) - float(start)) * 16000)
                    # start = int(float(start) * 16000)
                    assert float(start) == int(start) and float(end) == int(end)
                    num_frames = int(end) - int(start)
                    start = int(start)
                    if replacements[0] == "none":
                        sig, fs = torchaudio.load(
                            wav,
                            num_frames=num_frames,
                            frame_offset=start,
                        )
                    else:
                        sig, fs = torchaudio.load(
                            wav.replace(replacements[0], replacements[1]),
                            num_frames=num_frames,
                            frame_offset=start,
                        )

                feats = model.get_feats(sig)
                embedding = model.extract_embedding(feats)
                writer(utt, embedding.numpy())

                with lock:
                    counter.value += 1
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
