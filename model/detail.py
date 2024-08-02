import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from collections import Counter


EPSILON, YPSILON = 1e-12, 1e12


def hinge(x, m):
    return F.relu(m - x)


def normalize(x, dim=1):
    x = x / (torch.linalg.norm(x, dim=dim, keepdim=True) + EPSILON)

    return x


def safe_division(x, positive_y):
    return x / (positive_y + EPSILON)


def safe_sqrt(quasi_positive_x):
    return torch.sqrt(torch.clamp(quasi_positive_x, min=EPSILON))


def safe_log(x):
    return torch.log(torch.clamp(x, min=EPSILON, max=YPSILON))


def safe_acos(x):
    return torch.acos(torch.clamp(x, min=-1.0 + EPSILON, max=1.0 - EPSILON))


def calculate_groups_pt(classes):
    classes = torch.unsqueeze(classes, 1)
    same_group = (classes == classes.t()).float()
    different_group = 1.0 - same_group
    same_group.fill_diagonal_(0.0)

    return same_group, different_group


def calculate_triplets(same_group, different_group):
    return torch.unsqueeze(same_group, 2) * torch.unsqueeze(different_group, 1)


def calculate_similarities_pt(embeddings):
    dot_products = torch.mm(embeddings, embeddings.t())
    squared_norms = dot_products.diag().unsqueeze(1).expand_as(dot_products)
    squared_distances = (squared_norms + squared_norms.t() - 2.0 * dot_products).clamp(min=0.0)
    distances = safe_sqrt(squared_distances)

    return distances, squared_distances, dot_products


def calculate_similarities_groups_pt(embeddings, group_size, directed):
    embeddings_by_group_offset = [embeddings[i : embeddings.shape[0] : group_size, :] for i in range(group_size)]

    dot_products = []
    squared_distances = []
    distances = []
    if directed:
        ijs = [(0, j) for j in range(1, group_size)]
    else:
        ijs = [(i, j) for i in range(group_size) for j in range(i + 1, group_size)]
    for i, j in ijs:
        dot_products_ij = torch.sum(embeddings_by_group_offset[i] * embeddings_by_group_offset[j], dim=1)
        squared_norms_i = torch.sum(embeddings_by_group_offset[i] ** 2, dim=1)
        squared_norms_j = torch.sum(embeddings_by_group_offset[j] ** 2, dim=1)
        squared_distances_ij = (squared_norms_i + squared_norms_j - 2.0 * dot_products_ij).clamp(min=0.0)
        distances_ij = safe_sqrt(squared_distances_ij)

        dot_products.append(dot_products_ij)
        squared_distances.append(squared_distances_ij)
        distances.append(distances_ij)

    return distances, squared_distances, dot_products


def calculate_triplet_angles(embeddings):
    embedding_differences = torch.unsqueeze(embeddings, 1) - torch.unsqueeze(embeddings, 0)
    normalized_embedding_differences = normalize(embedding_differences, dim=2)
    angles = torch.sum(
        torch.unsqueeze(normalized_embedding_differences, dim=2)
        * torch.unsqueeze(normalized_embedding_differences, dim=1),
        dim=3,
    )

    return angles


def calculate_triplet_angles_one_hot(embeddings):
    classes = torch.argmax(embeddings, 1)
    same_class = (torch.unsqueeze(classes, 0) == torch.unsqueeze(classes, 1)).to(torch.float)
    angles = 0.5 * (
        1.0 + torch.unsqueeze(same_class, 0) - torch.unsqueeze(same_class, 2) - torch.unsqueeze(same_class, 1)
    )

    return angles


def calculate_source_similarities(source_embeddings):
    source_distances, source_dot_products = [], []

    for i in range(source_embeddings.shape[1]):
        d, _, dp = calculate_similarities_pt(source_embeddings[:, i, :])
        source_distances.append(d)
        source_dot_products.append(dp)

    source_avg_distances = torch.mean(torch.stack(source_distances), dim=0)
    source_avg_dot_products = torch.mean(torch.stack(source_dot_products), dim=0)

    return source_distances, source_avg_distances, source_dot_products, source_avg_dot_products


def calculate_weights_from_distances(distances, tau, margin=0.0, pair_type=1, triplet_type=1):
    if pair_type == 1:
        pair_weights = torch.exp(-distances / tau)
    elif pair_type == 2:
        pair_weights = torch.exp(-(distances**2.0) / tau)
    else:
        pair_weights = None
    if pair_weights is not None:
        pair_weights_negated = 1.0 - pair_weights
        pair_weights.fill_diagonal_(0.0)
        pair_weights_negated.fill_diagonal_(0.0)
    else:
        pair_weights_negated = None

    if triplet_type == 1:
        triplet_weights = torch.unsqueeze(pair_weights, 2) * torch.unsqueeze(pair_weights_negated, 1)
    elif triplet_type == 2:
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_weights = torch.sigmoid((triplet_distance_differences - margin) / tau)
    elif triplet_type == 3:
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_weights = torch.exp(-triplet_distance_differences / tau)
    elif triplet_type == 4:
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_weights = torch.exp(-(triplet_distance_differences**2.0) / tau)
    else:
        triplet_weights = None

    return pair_weights, pair_weights_negated, triplet_weights


def compute_multi_teacher_triplet_weights(source_embeddings, margin):
    # Get the number of embeddings.
    num_source_embeddings = source_embeddings.shape[1]

    # Calculate distances according to the source embeddings.
    source_distances, _, _, _ = calculate_source_similarities(source_embeddings)

    # Calculate the triplet distance differences according to the source embeddings.
    source_triplet_distance_differences = []
    for d in source_distances:
        tdd = torch.unsqueeze(d, 1) - torch.unsqueeze(d, 2)
        source_triplet_distance_differences.append(tdd)

    source_triplet_distance_differences = torch.stack(source_triplet_distance_differences)

    # Assign weights to triplets according to the source embeddings.
    source_triplet_votes = torch.sum(source_triplet_distance_differences > margin, dim=0)
    source_triplet_weights = source_triplet_votes > (num_source_embeddings / 2.0)

    return source_triplet_weights.type(torch.float32)


def calculate_groups_np(classes):
    classes = np.expand_dims(classes, 1)
    same_group = classes == classes.T
    different_group = ~same_group
    np.fill_diagonal(same_group, False)

    return same_group, different_group


def calculate_similarities_np(embeddings):
    distances = sklearn.metrics.pairwise_distances(embeddings)
    squared_distances = distances**2
    dot_products = np.matmul(embeddings, embeddings.T)

    return distances, squared_distances, dot_products


def calculate_efficient_ranking_metrics(classes, embeddings, k=10):
    n = len(classes)

    embeddings = np.array(embeddings)
    if len(embeddings.shape) == 2:
        embeddings_transposed = np.ascontiguousarray(np.transpose(embeddings))
    elif len(embeddings.shape) == 3:
        embeddings_transposed = np.ascontiguousarray(np.moveaxis(embeddings, [0, 1, 2], [2, 0, 1]))
    else:
        embeddings_transposed = None

    def _get_indexes_top_k(a, k):
        indexes1 = np.argpartition(-a, k)[..., :k]

        if a.ndim == 1:
            indexes2 = np.argsort(-a[indexes1])
            return indexes1[indexes2]

        elif a.ndim == 2:
            rows = np.repeat(n.expand_dims(np.arange(a.shape[0]), 1), k, axis=1)
            indexes2 = np.argsort(-a[rows, indexes1])
            return indexes1[rows, indexes2]

    class_occurrences = Counter(classes)

    valid, top_corrects = np.zeros((n,)), np.zeros((n,))
    for i, (c, e) in enumerate(zip(classes, embeddings)):
        if class_occurrences[c] <= 1:
            continue
        valid[i] = 1

        # Compute the similarity of the current instance with all the other instances.
        if len(embeddings.shape) == 2:
            s = np.dot(e, embeddings_transposed)
        elif len(embeddings.shape) == 3:
            ss = []
            for j in range(embeddings.shape[1]):
                s = np.dot(e[j], embeddings_transposed[j])
                ss.append(s)
            ss = np.stack(ss)

            s = np.max(ss, axis=0)
        else:
            s = None

        # Get the indexes of the nearest neighbors of the current instance.
        nearest_neighbor_indexes = _get_indexes_top_k(s, k + 1)

        # Get the binarized labels of the nearest neighbors (1-same label as the current instance, 0-different label).
        binarized_labels = classes[nearest_neighbor_indexes] == c

        # Remove the binary label of the current instance, which is first in the list of nearest neighbors.
        binarized_labels = binarized_labels[1:]

        # closest_neighbor_index = nearest_neighbor_indexes[1]
        closest_neighbor_index = (
            nearest_neighbor_indexes[1] if nearest_neighbor_indexes[0] == i else nearest_neighbor_indexes[0]
        )
        top_corrects[i] = classes[i] == classes[closest_neighbor_index]

    return valid, top_corrects


def calculate_similarities_groups_np(embeddings, group_size, directed):
    embeddings_by_group_offset = [embeddings[i : embeddings.shape[0] : group_size, :] for i in range(group_size)]

    dot_products = []
    squared_distances = []
    distances = []
    if directed:
        ijs = [(0, j) for j in range(1, group_size)]
    else:
        ijs = [(i, j) for i in range(group_size) for j in range(i + 1, group_size)]
    for i, j in ijs:
        dot_products_ij = np.sum(embeddings_by_group_offset[i] * embeddings_by_group_offset[j], axis=1)
        squared_norms_i = np.sum(embeddings_by_group_offset[i] ** 2, axis=1)
        squared_norms_j = np.sum(embeddings_by_group_offset[j] ** 2, axis=1)
        squared_distances_ij = np.clip(squared_norms_i + squared_norms_j - 2.0 * dot_products_ij, 0.0, None)
        distances_ij = np.sqrt(squared_distances_ij)

        dot_products.append(dot_products_ij)
        squared_distances.append(squared_distances_ij)
        distances.append(distances_ij)

    return distances, squared_distances, dot_products


def calculate_fractions_same_triplet_relation(source_triplet_distance_differences, triplet_distance_differences):
    # FSTR
    triplet_relation_same = ((source_triplet_distance_differences > 0.0) & (triplet_distance_differences > 0.0)) | (
        (source_triplet_distance_differences < 0.0) & (triplet_distance_differences < 0.0)
    )
    fstr = np.sum(triplet_relation_same.astype(np.float32)) / (
        np.sum((source_triplet_distance_differences != 0.0).astype(np.float32)) + 1e-9
    )

    # FSTRT
    relevant = (source_triplet_distance_differences > 0.2) | (source_triplet_distance_differences < -0.2)
    fstrt = np.sum((triplet_relation_same & relevant).astype(np.float32)) / (np.sum(relevant.astype(np.float32)) + 1e-9)

    return fstr, fstrt
