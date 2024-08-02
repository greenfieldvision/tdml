import numpy as np

import tdml.model.detail as detail

from abc import ABC, abstractmethod
from collections import OrderedDict


class Metrics(ABC):
    @abstractmethod
    def compute(self, supervision, embeddings):
        raise NotImplementedError()

    def aggregate(self, all_metric_values):
        if len(all_metric_values) == 0:
            return {}

        metric_names = set(all_metric_values[0].keys())
        averaged_metric_values = {
            metric_name: np.mean([metric_values[metric_name] for metric_values in all_metric_values])
            for metric_name in metric_names
        }

        return averaged_metric_values

    def compute_global(self, all_supervision, all_embeddings):
        return self.compute(all_supervision, all_embeddings)


class MetricsClassSupervision(Metrics):
    NUM_NEIGHBORS_EFFICIENT_METRICS = 2

    def compute(self, supervision, embeddings):
        # If the supervision information is in one hot form, convert it to indexes.
        if len(supervision.shape) == 1:
            classes = supervision
        elif len(supervision.shape) == 3:
            classes = np.argmax(np.squeeze(supervision), axis=1)
        else:
            classes = None

        def valid_mean(x, valid):
            return np.sum(x) / np.sum(valid)

        valid, top_corrects = detail.calculate_efficient_ranking_metrics(
            classes, embeddings, self.NUM_NEIGHBORS_EFFICIENT_METRICS
        )

        return {"mP@1": valid_mean(top_corrects, valid)}


class MetricsTripletSupervision(Metrics):
    def __init__(self, directed):
        self.directed = directed

    def compute(self, _, embeddings):
        distances, _, _ = detail.calculate_similarities_groups_np(embeddings, 3, self.directed)

        if self.directed:
            d_01, d_02 = distances
            correct_triplets = d_02 > d_01
        else:
            d_01, d_02, d_12 = distances
            correct_triplets = (d_02 > d_01) & (d_12 > d_01)

        return {"FCT": np.mean(correct_triplets)}


class MetricsSourceEmbeddingSupervision(Metrics):
    SEED = 0
    NUM_RANDOM_TRIPLETS = 1_000_000

    def __init__(self):
        self.random_triplet_indexes = None

    def compute(self, supervision, embeddings):
        return {"fraction_same_triplet_relation": 0.0, "fraction_same_triplet_relation_thresholded": 0.0}

    def compute_global(self, all_source_embeddings, all_embeddings):
        if self.random_triplet_indexes is None:
            np.random.seed(self.SEED)

            n = len(all_source_embeddings)
            random_indexes = np.random.choice(n, size=3 * self.NUM_RANDOM_TRIPLETS, replace=True)
            self.random_triplet_indexes = [
                random_indexes[: self.NUM_RANDOM_TRIPLETS],
                random_indexes[self.NUM_RANDOM_TRIPLETS : (2 * self.NUM_RANDOM_TRIPLETS)],
                random_indexes[(2 * self.NUM_RANDOM_TRIPLETS) : (3 * self.NUM_RANDOM_TRIPLETS)],
            ]
        indexes_i, indexes_j, indexes_k = self.random_triplet_indexes

        source_distance_differences = []
        for e in range(all_source_embeddings.shape[1]):
            source_d_ij = np.linalg.norm(
                all_source_embeddings[indexes_i, e, :] - all_source_embeddings[indexes_j, e, :],
                axis=1,
            )
            source_d_ik = np.linalg.norm(
                all_source_embeddings[indexes_i, e, :] - all_source_embeddings[indexes_k, e, :],
                axis=1,
            )
            source_dd = source_d_ik - source_d_ij
            source_distance_differences.append(source_dd)
        avg_source_dd = np.mean(source_distance_differences, axis=0)

        d_ij = np.linalg.norm(all_embeddings[indexes_i, :] - all_embeddings[indexes_j, :], axis=1)
        d_ik = np.linalg.norm(all_embeddings[indexes_i, :] - all_embeddings[indexes_k, :], axis=1)
        dd = d_ik - d_ij

        # FSTR
        triplet_relation_same = ((avg_source_dd > 0.0) & (dd > 0.0)) | ((avg_source_dd < 0.0) & (dd < 0.0))
        fstr = np.sum(triplet_relation_same.astype(np.float32)) / (
            np.sum((avg_source_dd != 0.0).astype(np.float32)) + 1e-12
        )

        # FSTRT
        relevant = (avg_source_dd > 0.2) | (avg_source_dd < -0.2)
        fstrt = np.sum((triplet_relation_same & relevant).astype(np.float32)) / (
            np.sum(relevant.astype(np.float32)) + 1e-12
        )

        return {"fraction_same_triplet_relation": fstr, "fraction_same_triplet_relation_thresholded": fstrt}

    def aggregate(self, all_metric_values):
        aggregated_metrics = super().aggregate(all_metric_values)
        aggregated_metrics = OrderedDict(
            [
                (metric_name, aggregated_metrics[metric_name])
                for metric_name in ["fraction_same_triplet_relation", "fraction_same_triplet_relation_thresholded"]
            ]
        )

        return aggregated_metrics
