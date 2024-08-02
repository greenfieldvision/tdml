import faiss
import numpy as np


def compute_nearest_neighbors_self(embeddings, num_neighbors):
    faiss_index = faiss.IndexFlatL2(embeddings.shape[-1])
    gpu_resources = faiss.StandardGpuResources()
    if gpu_resources is not None:
        faiss.index_cpu_to_gpu(gpu_resources, 0, faiss_index)
    faiss_index.add(embeddings)
    _, neighbors = faiss_index.search(embeddings, num_neighbors)

    return neighbors


def compute_metrics_classes(labels, embeddings):
    assert len(labels) == len(embeddings)

    # Compute k nearest neighbors for each instance, for small k.
    nearest_neighbors = compute_nearest_neighbors_self(np.array(embeddings), 10)

    # Calculate metrics for each instance.
    precisions_at_1 = []
    for i, (label, nearest_neighbor_indexes) in enumerate(zip(labels, nearest_neighbors)):
        # Check that at least one other instance has the same label as the current one.
        if not label in labels:
            print("invalid instance #{}: {} {}".format(i, labels, label))
            break

        # Get the binarized labels of the nearest neighbors (1-same label as the current instance, 0-different label).
        binarized_labels = [labels[j] == label for j in nearest_neighbor_indexes]

        # Remove the binary label of the current instance, which is first in the list of nearest neighbors.
        binarized_labels = binarized_labels[1:]

        # Calculate P@1 for the current instance.
        precision_at_1 = binarized_labels[0]

        # Append to the lists of metrics.
        precisions_at_1.append(precision_at_1)

    return np.mean(precisions_at_1)


def compute_metrics_triplets(binary_labels, embeddings, directed):
    assert len(binary_labels) == len(embeddings)

    positive_indexes, negative_indexes = [], []
    for i, l in enumerate(binary_labels):
        if l:
            positive_indexes.append(i)
        else:
            negative_indexes.append(i)

    num_correct_triplets, num_triplets = 0, 0

    if directed:
        i = positive_indexes[0]
        for j in positive_indexes[1:]:
            for k in negative_indexes:
                d_ij = np.linalg.norm(embeddings[i] - embeddings[j])
                d_ik = np.linalg.norm(embeddings[i] - embeddings[k])
                if d_ij < d_ik:
                    num_correct_triplets += 1
                num_triplets += 1

    else:
        for i_index, i in enumerate(positive_indexes):
            for j in positive_indexes[i_index + 1 :]:
                for k in negative_indexes:
                    d_ij = np.linalg.norm(embeddings[i] - embeddings[j])
                    d_ik = np.linalg.norm(embeddings[i] - embeddings[k])
                    d_jk = np.linalg.norm(embeddings[j] - embeddings[k])
                    if (d_ij < d_ik) and (d_ij < d_jk):
                        num_correct_triplets += 1
                    num_triplets += 1

    fraction_correct_triplets = num_correct_triplets / (num_triplets + 1e-12)
    return fraction_correct_triplets


def calculate_average_precision(sorted_binary_labels):
    num_instances = sorted_binary_labels.shape[0]

    cum_true_positives = np.cumsum(sorted_binary_labels, dtype=np.float)
    cum_positives = np.arange(1, num_instances + 1, dtype=np.float)
    precisions = cum_true_positives / (cum_positives + 1e-12)

    labeled_positives = np.sum(sorted_binary_labels)
    recalls = cum_true_positives / (labeled_positives + 1e-12)

    return precisions[0] * recalls[0] + np.sum((precisions[:-1] + precisions[1:]) / 2.0 * (recalls[1:] - recalls[:-1]))
