import numpy as np
import torch


class SamplingMattersBatchMiner:
    def __init__(self):
        self.lower_cutoff = 0.5
        self.upper_cutoff = 1.4
        self.name = "distance"

    def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs, dim = batch.shape

        if distances is None:
            distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [], []
        labels_visited = []
        anchors = []

        tar_labels = labels if tar_labels is None else tar_labels

        for i in range(bs):
            neg = tar_labels != labels[i]
            pos = tar_labels == labels[i]

            anchors.append(i)

            if np.sum(pos) > 0:
                # Sample positives randomly
                if np.sum(pos) > 1:
                    pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))

            # Sample negatives by distance
            if np.sum(neg) > 0:
                q_d_inv = self.inverse_sphere_distances(dim, bs, distances[i], tar_labels, labels[i])
                negatives.append(np.random.choice(sel_d, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

    def inverse_sphere_distances(self, dim, bs, anchor_to_all_dists, labels, anchor_label):
        dists = anchor_to_all_dists

        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = (2.0 - float(dim)) * torch.log(dists) - (float(dim - 3) / 2) * torch.log(
            1.0 - 0.25 * (dists.pow(2))
        )
        log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))  # - max(log) for stability
        q_d_inv[np.where(labels == anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
        # errors where there are no available negatives (for high samples_per_class cases).
        # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

        q_d_inv = q_d_inv / q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.sqrt()


class RelaxedSamplingMattersBatchMiner:
    def __init__(self, args):
        # self.tau = 0.1  # -> virtually always select a positive in each triplet
        self.tau = 0.4  # -> high chance of not selecting a positive in each triplet

        self.lower_cutoff = 0.5
        self.upper_cutoff = np.sqrt(2.0)

    def __call__(self, source_embeddings, embeddings, _):
        n, dim = embeddings.shape
        n2 = n**2

        source_distances_squared = self.pdist(source_embeddings[:, 0, :].detach().cpu(), squared=True)
        source_pair_weights = torch.exp(-source_distances_squared / self.tau)
        source_pair_weights_adjusted = torch.clamp(source_pair_weights - torch.eye(n), min=0.0, max=None)

        distances = self.pdist(embeddings.detach().cpu()).clamp(min=self.lower_cutoff)

        flat_indexes = []
        for i in range(n):
            spwa = source_pair_weights_adjusted[i].numpy()
            # If no positive or no negative, do not use the current anchor.
            if (np.max(spwa) < 0.01) or (np.min(spwa) >= 0.01):
                continue

            # Randomly sample one positive.
            p = spwa / np.sum(spwa)
            j = np.random.choice(n, p=p)

            # Randomly sample one negative by distance.
            q_d_inv = self.inverse_sphere_distances(dim, distances[i], source_distances_squared[i])
            if np.sum(q_d_inv) == 0.0:
                positives = positives[:-1]
                continue
            p = q_d_inv / np.sum(q_d_inv)
            k = np.random.choice(n, p=p)

            flat_indexes.append(i * n2 + j * n + k)

        return torch.tensor(flat_indexes, device=embeddings.device)

    def inverse_sphere_distances(self, dim, distances, source_distances_squared):
        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = (2.0 - dim) * torch.log(distances) + (3.0 - dim) / 2.0 * torch.log(1.0 - 0.25 * distances.pow(2))

        sd2 = torch.clamp(source_distances_squared, min=self.lower_cutoff**2, max=self.upper_cutoff**2)
        offset = 0.5 * (2.0 - dim) * torch.log(sd2) + (3.0 - dim) / 2.0 * torch.log(1.0 - 0.25 * sd2)

        # adjusted for positivity of source labels
        log_q_d_inv -= offset

        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))

        return q_d_inv.numpy()

    def pdist(self, A, squared=False):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        if squared:
            return res
        return res.sqrt()
