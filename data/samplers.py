import math
import random

from torch.utils.data.sampler import Sampler


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
