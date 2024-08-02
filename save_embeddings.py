import os
import re

import numpy as np

from omegaconf import OmegaConf

from tdml.factory import make_dataset_builder, make_preprocessing
from tdml.evaluation.inference import Inference


def parse_args():
    args1 = OmegaConf.from_cli()
    args2 = OmegaConf.load(args1["configuration_file_name"])
    return OmegaConf.merge(args2, args1)


def make_inference_objects(args):
    dataset_builder = make_dataset_builder(args.dataset)
    preprocessing = make_preprocessing(dataset_builder, args.preprocessing)
    model_uris = _expand_run_ids(args.model_uris)

    return dataset_builder, preprocessing, model_uris


def _expand_run_ids(model_uri):
    two_run_ids = "runs/(run-([0-9]{4})..([0-9]{4}))"
    m = re.search(two_run_ids, model_uri)
    if m is None:
        return model_uri

    new_model_uris = []
    first_run_id, last_run_id = int(m.group(2)), int(m.group(3))
    for run_id in range(first_run_id, last_run_id + 1):
        new_model_uri = model_uri.replace(m.group(0), "runs/run-{:04d}".format(run_id))
        new_model_uris.append(new_model_uri)

    return new_model_uris


def save_embeddings(dataset_builder, preprocessing, model_uris, args):
    # Compute the embeddings and store them.
    embeddings = None
    for i, model_uri in enumerate(model_uris):
        # Run inference on embedding model.
        inference = Inference(preprocessing, model_uri, dataset_builder)
        instances = dataset_builder.get_instances_for_inference("all")
        es = np.stack([yr[0] for yr in inference.infer(instances)])

        # Allocate NumPy array if needed and then accumulate current embeddings.
        if embeddings is None:
            embeddings = np.zeros((es.shape[0], len(model_uris), es.shape[1]), dtype=np.float32)
        embeddings[:, i, :] = es

    # Make output directory if it doesn't exist.
    os.makedirs(args.output_directory_name, exist_ok=True)

    # Save embeddings.
    file_name = os.path.join(args.output_directory_name, "embeddings_{}.npz".format(args.new_source_embeddings_id))
    print("writing embeddings to {}...".format(file_name))
    np.savez(file_name, embeddings=embeddings)
    print("done")


if __name__ == "__main__":
    args = parse_args()

    objects = make_inference_objects(args)
    save_embeddings(*objects, args)
