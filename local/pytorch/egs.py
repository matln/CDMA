# -*- coding:utf-8 -*-

import os, sys
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist


# subtools = '/data/lijianchen/workspace/sre/subtools'
subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))
from libs.support.prefetch_generator import BackgroundGenerator

from libs.egs.egs import ChunkEgs, BaseBunch, DataLoaderFast, get_info_from_egsdir

from .sampler import RandomSpeakerSampler, RandomSpeakerBatchSampler

# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer


class Bunch(BaseBunch):
    """BaseBunch:(trainset,[valid]).
    """

    def __init__(self, s_trainset, t_trainset=None, s_valid=None, use_fast_loader=False, max_prefetch=10,
                 batch_size=128, valid_batch_size=512, num_workers=0, pin_memory=False, num_chunks=8):
        """
        num_chunks: num of chunks per speaker per batch
        s_trainset: trainset of the source domain
        t_trainset: trainset of the target domain
        """

        num_gpu = 1
        self.num_chunks = num_chunks
        self.num_spks = int(batch_size / num_chunks)


        if t_trainset is not None:
            assert batch_size % num_chunks == 0
            # s_train_sampler = RandomSpeakerSampler(s_trainset, batch_size, num_chunks)
            # t_train_sampler = RandomSpeakerSampler(t_trainset, batch_size, num_chunks)
            # shuffle = False

            s_train_batch_sampler = RandomSpeakerBatchSampler(s_trainset, self.num_spks, num_chunks)
            t_train_batch_sampler = RandomSpeakerBatchSampler(t_trainset, self.num_spks, num_chunks)
        else:
            s_train_sampler = None
            shuffle = True


        if use_fast_loader:
            if t_trainset is not None:
                # TODO: 未知原因，同时用两个dataloaderfast，第二个的随机就不是固定的了
                # self.t_train_loader = DataLoaderFast(max_prefetch, t_trainset, batch_size=batch_size,
                #                                      shuffle=shuffle, num_workers=num_workers,
                #                                      pin_memory=pin_memory, drop_last=True,
                #                                      sampler=t_train_sampler)
                # self.t_train_loader = DataLoader(t_trainset, batch_size=batch_size, shuffle=shuffle,
                #                                  num_workers=num_workers, pin_memory=pin_memory,
                #                                  drop_last=True, sampler=t_train_sampler)

                self.s_train_loader = DataLoaderFast(max_prefetch, s_trainset, num_workers=num_workers,
                                                     pin_memory=pin_memory, batch_sampler=s_train_batch_sampler)

                self.t_train_loader = DataLoader(t_trainset, num_workers=num_workers, pin_memory=pin_memory,
                                                 batch_sampler=t_train_batch_sampler)
            else:
                self.train_loader = DataLoaderFast(max_prefetch, s_trainset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=num_workers,
                                                     pin_memory=pin_memory, drop_last=True,
                                                     sampler=s_train_sampler)
        else:
            if t_trainset is not None:
                self.s_train_loader = DataLoader(s_trainset, num_workers=num_workers, pin_memory=pin_memory,
                                                 batch_sampler=s_train_batch_sampler)
                self.t_train_loader = DataLoader(t_trainset, num_workers=num_workers, pin_memory=pin_memory,
                                                 batch_sampler=t_train_batch_sampler)
            else:
                self.train_loader = DataLoader(s_trainset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, pin_memory=pin_memory,
                                                 drop_last=True, sampler=s_train_sampler)

        if t_trainset is not None:
            self.num_batch_train = min(len(self.s_train_loader), len(self.t_train_loader))
        else:
            self.num_batch_train = len(self.train_loader)


        if self.num_batch_train <= 0:
            raise ValueError("Expected num_batch of trainset > 0. There are your egs info: num_gpu={}, num_batches={}, "
                             "batch-size={}, drop_last=True.\nNote: If batch-size > num_batches and drop_last is true, then it "
                             "will get 0 batch.".format(num_gpu, self.num_batch_train, batch_size))

        if s_valid is not None:
            valid_batch_size = min(valid_batch_size, len(s_valid))  # To save GPU memory

            if len(s_valid) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            # Do not use DataLoaderFast for valid for it increases the memory all the time when compute_valid_accuracy is True.
            # But I have not find the real reason.
            self.valid_loader = DataLoader(s_valid, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers,
                                           pin_memory=pin_memory, drop_last=False)

            self.num_batch_valid = len(self.valid_loader)
        else:
            self.valid_loader = None
            self.num_batch_valid = 0

    @classmethod
    def get_bunch_from_csv(cls, s_train_csv:str, t_train_csv:str=None, s_valid_csv:str=None, egs_params:dict={}, data_loader_params_dict:dict={}):
        egs_type = "chunk"
        if "egs_type" in egs_params.keys():
            egs_type = egs_params.pop("egs_type")
            if egs_type != "chunk" and egs_type != "vector":
                raise TypeError("Do not support {} egs now. Select one from [chunk, vector].".format(egs_type))

        s_trainset = ChunkEgs(s_train_csv, **egs_params, egs_type=egs_type)
        t_trainset = None
        if t_train_csv is not None:
            t_trainset = ChunkEgs(t_train_csv, **egs_params, egs_type=egs_type)

        s_valid = None
        if s_valid_csv != "" and s_valid_csv is not None:
            s_valid = ChunkEgs(s_valid_csv, egs_type=egs_type)
        return cls(s_trainset, t_trainset, s_valid, **data_loader_params_dict)

    @classmethod
    def get_bunch_from_egsdir(self, s_egsdir:str, t_egsdir:str=None, egs_params:dict={}, data_loader_params_dict:dict={}):
        train_csv_name = None
        s_train_csv_name = None
        valid_csv_name = None
        if "s_train_csv_name" in egs_params.keys():
            s_train_csv_name = egs_params.pop("s_train_csv_name")
        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        s_feat_dim, s_num_targets, s_train_csv, s_valid_csv = get_info_from_egsdir(
            s_egsdir, train_csv_name=s_train_csv_name, valid_csv_name=valid_csv_name)

        t_train_csv = None
        t_num_targets = 0
        if t_egsdir is not None:
            t_feat_dim, t_num_targets, t_train_csv, _ = get_info_from_egsdir(
                t_egsdir, train_csv_name=train_csv_name, valid_csv_name=valid_csv_name)
            assert s_feat_dim == t_feat_dim

        info = {"feat_dim": s_feat_dim, "num_targets": s_num_targets, "t_num_targets": t_num_targets}
        bunch = self.get_bunch_from_csv(s_train_csv, t_train_csv, s_valid_csv, egs_params, data_loader_params_dict)
        return bunch, info


if __name__ == '__main__':
    # import libs.support.utils as utils
    # utils.set_all_seed(1024, deterministic=True)
    data_loader_params_dict = {'batch_size': 128, 'num_workers': 0, 'use_fast_loader': True}
        # "exp/egs/fbank_64-voxceleb1o2_train-200-sequential",
    bunch, info = Bunch.get_bunch_from_egsdir(
        "exp/egs/fbank_64-voices_dev_label-200-sequential-novad",
        data_loader_params_dict=data_loader_params_dict)
    # sampler = RandomSpeakerSampler(bunch.trainset, 32, 4)
    for _ in range(20):
        # for i, (batch1, batch2) in enumerate(zip(bunch.s_train_loader, bunch.t_train_loader)):
        for i, batch1 in enumerate(bunch.train_loader):
            data1, label1 = batch1
            # data2, label2 = batch2
            # print(data1[0, 0, :5])
            # if i > 1:
            #     break
            # print(data.size())
            # print(label.size())
        print('+++++++++++++++')
