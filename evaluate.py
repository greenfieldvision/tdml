from omegaconf import OmegaConf

from tdml.evaluation.inference import Inference
from tdml.evaluation.metrics import compute_metrics_classes, compute_metrics_triplets
from tdml.factory import make_dataset_builder, make_preprocessing


def parse_args():
    args1 = OmegaConf.from_cli()
    args2 = OmegaConf.load(args1["configuration_file_name"])
    return OmegaConf.merge(args2, args1)


def make_inference_objects(args):
    dataset_builder = make_dataset_builder(args.dataset)
    preprocessing = make_preprocessing(dataset_builder, args.preprocessing)

    return dataset_builder, preprocessing


def evaluate(dataset_builder, preprocessing, args):
    inference = Inference(preprocessing, args.model_uri, dataset_builder, args.dataset.batch_size)
    instances = dataset_builder.get_instances_for_inference(args.subset_name)

    # Extract the labels and compute the embeddings. Then, free up the GPU/CPU memory claimed so far in order to reuse
    # it for pairwise distance computation.
    print("computing embeddings...")
    embeddings, labels, embeddings_by_instance = [], [], {}
    for e, l, i in inference.infer(instances):
        embeddings.append(e)
        labels.append(l)
        embeddings_by_instance[i] = e
    inference.deallocate()
    print("done")

    if args.evaluation_type == "classes":
        # Compute the class based metric learning metrics.
        mP_1 = compute_metrics_classes(labels, embeddings)

        # Pretty print the class based metric learning metrics.
        print("mP@1: {:.4f}".format(mP_1))

    elif args.evaluation_type == "triplets":
        # Compute the triplet based metric learning metrics.

        # Get the triplet directedness.
        directed = dataset_builder.get_triplet_directedness()

        # Go through all triplets and determine if the prediction is correct or not.
        fractions_correct_triplets = []
        for t in dataset_builder.get_triplet_indexes_for_evaluation(args.subset_name):
            f = compute_metrics_triplets([1, 1, 0], [embeddings_by_instance[i] for i in t], directed)
            fractions_correct_triplets.append(f)

        # Pretty print the triplet based metric learning metrics.
        print("FCT: {:.4f}".format(sum(fractions_correct_triplets) / len(fractions_correct_triplets)))

    else:
        raise ValueError("Unknown evaluation type.")


if __name__ == "__main__":
    args = parse_args()

    objects = make_inference_objects(args)
    evaluate(*objects, args)
