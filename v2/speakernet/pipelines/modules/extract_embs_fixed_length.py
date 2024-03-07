# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import torch
import kaldiio
import warnings
import argparse
import traceback
import torchaudio
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

import speakernet.dataio.kaldi_io as kaldi_io
import speakernet.utils.utils as utils
import speakernet.utils.kaldi_common as kaldi_common


# Parse
parser = argparse.ArgumentParser(
    description="Extract embeddings form a piece of feats.scp, wav.scp or pipeline")


parser.add_argument("--nnet-config", type=str, default="",
                    help="This config contains model_blueprint and model_creation.")

parser.add_argument("--model-blueprint", type=str, default=None,
                    help="A *.py which includes the instance of nnet in this training.")

parser.add_argument("--model-creation", type=str, default=None,
                    help="A command to create the model class according to the class \
                        declaration in --model-path, such as using Xvector(40,2) to create \
                        a Xvector nnet.")

parser.add_argument("--nj", type=str, help="")

parser.add_argument("--use-gpu", type=str, default='true',
                    choices=["true", "false"],
                    help="If true, use GPU to extract embeddings.")

parser.add_argument("--gpu-id", type=str, default="",
                    help="Specify a fixed gpu, or select gpu automatically.")

parser.add_argument("--chunk-size", type=int, default=-1, help="")

parser.add_argument("--egs-type", type=str, default="chunk", help="")

parser.add_argument("--batch-size", type=int, default=64, help="")

parser.add_argument("--num-workers", type=int, default=2, help="")

parser.add_argument("--pin-memory", action=kaldi_common.StrToBoolAction, type=str,
                    default=False, choices=["true", "false"], help="")

parser.add_argument("--replacements", type=lambda kv: kv.split(":"),
                    help="For wav path")

parser.add_argument("model_path", metavar="model-path", type=str,
                    help="The model used to extract embeddings.")

parser.add_argument("egs_path", metavar="egs-path", type=str, help="")

parser.add_argument("vectors_wspecifier", metavar="vectors-wspecifier",
                    type=str, help="")


print(' '.join(sys.argv))

args = parser.parse_args()

if args.replacements[0] != "none":
    assert len(args.replacements) == 2
    replacements = {args.replacements[0]: args.replacements[1]}
else:
    replacements = {}

# Start

class ChunkEgs(Dataset):
    """
    """

    def __init__(
        self,
        egs_path,
        chunk_size=-1,
        egs_type="chunk",
        samplerate=16000,
        replacements={},
        delimiter=",",
        repl_field="wav_path",
        id_field="utt_id"
    ):
        """
        """
        assert egs_type == "chunk" or egs_type == "vector"
        assert egs_path != "" and egs_path is not None
        self.samplerate = samplerate

        if egs_type == "chunk":
            self.data = utils.load_data_csv(
                egs_path, replacements, repl_field=repl_field, id_field=id_field, delimiter=delimiter
            )
            self.data_ids = list(self.data.keys())

            if (
                "start_position" in self.data[self.data_ids[0]].keys()
                and "end_position" in self.data[self.data_ids[0]].keys()
            ):
                self.chunk_position = True
            elif (
                "start_position" not in self.data[self.data_ids[0]].keys()
                and "end_position" not in self.data[self.data_ids[0]].keys()
            ):
                self.chunk_position = False

            else:
                raise TypeError(
                    "Expected both start-position and end-position are exist in {}.".format(
                        egs_path
                    )
                )

        else:
            self.chunk_position = False
            self.utt2vector = {}
            with open(egs_path, "r") as r:
                lines = r.readlines()
                for line in lines:
                    utt, vector = line.strip().split(' ')
                    self.utt2vector[utt] = vector
            self.data_ids = list(self.utt2vector.keys())

        # It is important that using .astype(np.string_) for string object to avoid memeory leak 
        # when multi-threads dataloader are used.

        self.egs_type = egs_type
        self.chunk_size = chunk_size

    def __getitem__(self, index):
        if self.egs_type == "chunk":
            data_id = self.data_ids[index]
            data_point = self.data[data_id]
            wav_size = int(data_point["duration"] * self.samplerate)

            if self.chunk_position:
                sig, fs = utils.load_wavs(
                    data_point["wav_path"],
                    data_point["start_position"],
                    data_point["end_position"]
                )
            else:
                sig, fs = utils.load_wavs(data_point["wav_path"])

            sig = sig.squeeze(0)

            if self.chunk_position and wav_size < self.chunk_size:
                warnings.warn(f"wav_size {wav_size} < self.chunk_size {self.chunk_size}")
                pad_size = self.chunk_size - wav_size
                pad_start = np.random.randint(0, wav_size - pad_size)
                pad_chunk = sig[pad_start: pad_start + pad_size]
                sig = torch.cat((sig, pad_chunk))

            return data_id, sig

        else:
            data_id = self.data_ids[index]
            egs = kaldi_io.read_vec_flt(self.utt2vector[data_id])

            # Note, egs which is read from kaldi_io is read-only and 
            # use egs = np.require(egs, requirements=['O', 'W']) to make it writeable.
            # It avoids the problem "ValueError: assignment destination is read-only".
            # Note that, do not use inputs.flags.writeable = True when the version of numpy >= 1.17.
            egs = np.require(egs, requirements=['O', 'W'])

            return data_id, egs.T

    def __len__(self):
        return len(self.data_ids)


arkfile, scpfile = args.vectors_wspecifier.split('ark,scp:')[-1].split(',')
try:
    dataset = ChunkEgs(args.egs_path, args.chunk_size, args.egs_type, replacements=replacements)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

    # nnet_config include model_blueprint and model_creation
    if args.nnet_config != "":
        model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)
    elif args.model_blueprint is not None and args.model_creation is not None:
        model_blueprint = args.model_blueprint
        model_creation = args.model_creation
    else:
        raise ValueError("Expected nnet_config or (model_blueprint, model_creation) to exist.")

    model = utils.import_module(model_blueprint, model_creation)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)

    # Select device
    model = utils.select_model_device(model, args.use_gpu, gpu_id=args.gpu_id)

    model.eval()

    # if len(args.gpu_id.split(',')) > 1:
    #     model = model.module

    # with kaldi_io.open_or_fd(args.vectors_wspecifier, 'wb') as w:
    #     for (key, inputs) in tqdm(dataloader):
    #         embedding = model.extract_embedding(inputs)
    #         embedding = embedding.numpy()
    #         # for i in range(len(embedding)):
    #         #     kaldi_io.write_vec_flt(w, embedding[i], key=key[i])

    key2emb = {}
    for (keys, inputs) in dataloader:
        if args.online_training:
            inputs = model.get_feats(inputs)
        embedding = model.extract_embedding(inputs)
        embedding = embedding.numpy()
        for i in range(len(keys)):
            key2emb[keys[i]] = embedding[i]
            print("Process utterance for key {0}".format(keys[i]))
    kaldiio.save_ark(arkfile, key2emb, scp=scpfile)

except BaseException as e:
    if not isinstance(e, KeyboardInterrupt):
        traceback.print_exc()
    sys.exit(1)
