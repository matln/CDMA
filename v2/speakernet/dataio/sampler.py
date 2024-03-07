"""
Copyright 2022 Jianchen Li
"""

import torch
import numpy as np
from torch.utils.data import Sampler


class RandomCycleIter:
    def __init__(
        self, data, test_mode=False, generator=None, remove_duplicates=False, num_samples=4
    ):
        self.data_list = torch.tensor(list(data))
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        self.generator = generator
        self.remove_duplicates = remove_duplicates
        self.num_samples = num_samples

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                if self.remove_duplicates and self.length % self.num_samples != 0:
                    assert self.length >= self.num_samples
                    remaining_items = self.data_list[-(self.length % self.num_samples) :]
                    selected_items = self.data_list[: -(self.length % self.num_samples)]
                    self.data_list = torch.cat(
                        (
                            selected_items[
                                torch.randperm(len(selected_items), generator=self.generator)
                            ],
                            remaining_items[
                                torch.randperm(len(remaining_items), generator=self.generator)
                            ],
                        )
                    )
                else:
                    self.data_list = self.data_list[
                        torch.randperm(self.length, generator=self.generator)
                    ]
        return self.data_list[self.i].item()


class SpeakerAwareSampler(Sampler):
    def __init__(
        self, data_source, num_samples_cls=1, num_cls=1, generator=None, remove_duplicates=False
    ):
        self.data_source = data_source
        self.generator = generator
        self.num_samples_cls = num_samples_cls
        self.num_cls = num_cls
        self.remove_duplicates = remove_duplicates

        self.num_spks = len(np.unique(self.data_source))
        self.cls_data_list = [list() for _ in range(self.num_spks)]
        for i, label in enumerate(self.data_source):
            self.cls_data_list[label].append(i)
        self.num_samples = max([len(x) for x in self.cls_data_list]) * len(self.cls_data_list)

    def speaker_aware_sample_generator(self, cls_iter, data_iter_list, n, num_samples_cls=1):
        i = 0
        j = 0
        while i < n:

            # yield next(data_iter_list[next(cls_iter)])

            if j >= num_samples_cls:
                j = 0

            if j == 0:
                temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
                yield temp_tuple[j]
            else:
                yield temp_tuple[j]

            i += 1
            j += 1

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        # self.spk_iter = RandomCycleIter1(range(num_spks), num_cls)
        self.spk_iter = RandomCycleIter(
            range(self.num_spks),
            generator=generator,
            remove_duplicates=self.remove_duplicates,
            num_samples=self.num_cls,
        )
        # self.data_iter_list = [RandomCycleIter1(x, num_samples_cls) for x in cls_data_list]
        self.data_iter_list = [
            RandomCycleIter(
                x,
                generator=generator,
                remove_duplicates=self.remove_duplicates,
                num_samples=self.num_samples_cls,
            )
            for x in self.cls_data_list
        ]

        return self.speaker_aware_sample_generator(
            self.spk_iter, self.data_iter_list, self.num_samples, self.num_samples_cls
        )

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    from torch.utils.data import dataloader

    a = [0 for i in range(10)]
    a.extend([1 for i in range(10)])
    a.extend([2 for i in range(10)])
    a.extend([3 for i in range(10)])
    a.extend([4 for i in range(10)])
    generator = torch.Generator()
    generator.manual_seed(1024)
    sampler = SpeakerAwareSampler(a, num_samples_cls=3, generator=generator)
    for i in iter(sampler):
        print(i)
    print("------------------")
    generator.manual_seed(1024)
    for i in iter(sampler):
        print(i)
