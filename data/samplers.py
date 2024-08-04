import math
import random

from collections import defaultdict

from torch.utils.data.sampler import Sampler


class SamplePerClassSampler(Sampler):
    """Sample fixed number of images per class."""

    def __init__(self, image_records, batch_size, k, is_training):
        self.class_indexes = list(set([r[1] for r in image_records]))

        self.record_indexes_by_class_index = defaultdict(list)
        for i, r in enumerate(image_records):
            self.record_indexes_by_class_index[r[1]].append(i)

        self.record_index_groups = []
        for ci in self.class_indexes:
            ri = self.record_indexes_by_class_index[ci]
            for i in range(0, len(ri), k):
                scg = ri[i : (i + k)]
                self.record_index_groups.append(scg)

        self.batch_size = batch_size
        self.k = k
        self.num_batches = int(
            math.floor(len(image_records) / batch_size) if is_training else math.ceil(len(image_records) / batch_size)
        )

        self.is_training = is_training

    def __iter__(self):
        if self.is_training:
            for _ in range(self.num_batches):
                index_sample = []

                # Attempt to sample k images per class; if less than k images available for a class, get more classes.
                while len(index_sample) < self.batch_size:
                    permuted_class_indexes = random.sample(self.class_indexes, len(self.class_indexes))
                    for ci in permuted_class_indexes:
                        ri = self.record_indexes_by_class_index[ci]
                        sri = random.sample(ri, min(self.k, len(ri)))
                        index_sample.extend(sri)
                        if len(index_sample) >= self.batch_size:
                            break

                yield index_sample[: self.batch_size]

        else:
            permuted_record_index_groups = random.sample(self.record_index_groups, k=len(self.record_index_groups))
            permuted_record_indexes = [i for rig in permuted_record_index_groups for i in rig]

            i = 0
            for _ in range(self.num_batches):
                index_sample = []

                for _ in range(self.batch_size):
                    record_index = permuted_record_indexes[i]
                    index_sample.append(record_index)

                    i += 1
                    if i >= len(permuted_record_indexes):
                        break

                yield index_sample[: self.batch_size]

    def __len__(self):
        return self.num_batches


class TripletSampler(Sampler):
    """Sample randomly according to set of triplets."""

    def __init__(self, triplets, batch_size, is_training):
        self.triplets = triplets

        assert batch_size % 3 == 0
        self.num_triplets_in_batch = batch_size // 3
        rounding_fn = math.floor if is_training else math.ceil
        self.num_batches = int(rounding_fn(len(triplets) / self.num_triplets_in_batch))

        self.is_training = is_training

    def __iter__(self):
        if self.is_training:
            all_triplets = random.sample(self.triplets, k=len(self.triplets))
        else:
            all_triplets = self.triplets

        for b in range(self.num_batches):
            triplets = all_triplets[(b * self.num_triplets_in_batch) : ((b + 1) * self.num_triplets_in_batch)]
            indexes = [i for t in triplets for i in t]
            yield indexes

    def __len__(self):
        return self.num_batches
