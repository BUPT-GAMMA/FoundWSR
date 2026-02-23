import numpy as np
from torch.utils.data import Sampler

class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, drop_last=False, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        length_to_indices = {}
        for idx, L in enumerate(lengths):
            if L not in length_to_indices:
                length_to_indices[L] = []
            length_to_indices[L].append(idx)

        self.batches = []
        for L, indices in length_to_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)

        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)