import copy
import os
import time

import numpy as np
import torch

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tdml.factory import make_dataset_builder, make_loss_and_metric, make_model, make_optimizer, make_preprocessing


RUN_DIRECTORY_NAME_TEMPLATE = "run-{run_id:04d}"
CHECKPOINT_FILE_NAME_TEMPLATE = "checkpoint-{epoch:04d}.pth"
BEST_CHECKPOINT_FILE_NAME = "checkpoint-best.pth"


def parse_args():
    args1 = OmegaConf.from_cli()
    args2 = OmegaConf.load(args1["configuration_file_name"])
    return OmegaConf.merge(args2, args1)


def make_training_objects(args):
    dataset_builder = make_dataset_builder(args.dataset)
    preprocessing = make_preprocessing(dataset_builder, args.preprocessing)
    data_loaders_by_subset = dataset_builder.make_pt_loaders(preprocessing)

    model = make_model(dataset_builder, args.model)
    loss_computer, metric_computer = make_loss_and_metric(dataset_builder, args.model)

    optimizer = make_optimizer(model, args.optimization)

    return data_loaders_by_subset, model, loss_computer, metric_computer, optimizer


def train(data_loaders_by_subset, model, loss_computer, metric_computer, optimizer, args):
    def _do_training_steps():
        # Set model to training mode.
        model.train()

        # Make dataset iterator that shows the step number and the loss.
        description = "training steps (loss {loss:.6f})"
        data_iterator = tqdm(data_loaders_by_subset["training"], desc=description.format(loss=0.0))

        # Do a gradient descent step for each batch.
        for instances, labels, indexes in data_iterator:
            instances, labels = instances.to(device), labels.to(device)
            model_outputs = model(x=instances)

            loss = loss_computer(labels, model_outputs) + model.get_custom_regularization_loss()
            training_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_description(description.format(loss=loss))

    def _do_validation_steps():
        nonlocal last_metric_value

        all_labels, all_model_outputs, all_metric_values = [], [], []

        # Set model to inference mode.
        model.eval()

        with torch.no_grad():
            # Make dataset iterator that shows the step number.
            data_iterator = tqdm(data_loaders_by_subset["validation"], desc="validation steps")

            # Do inference for each batch.
            for instances, labels, indexes in data_iterator:
                instances, labels = instances.to(device), labels.to(device)
                model_outputs = model(x=instances)

                loss = loss_computer(labels, model_outputs) + model.get_custom_regularization_loss()
                validation_losses.append(loss.item())

                labels, model_outputs = labels.detach().cpu().numpy(), model_outputs.detach().cpu().numpy()
                if args.optimization.evaluate_globally:
                    all_labels.append(labels)
                    all_model_outputs.append(model_outputs)
                else:
                    metric_value = metric_computer.compute(labels, model_outputs)
                    all_metric_values.append(metric_value)

        # Compute global metrics or average over batch metrics.
        if args.optimization.evaluate_globally:
            all_labels = np.concatenate(all_labels, axis=0)
            all_model_outputs = np.concatenate(all_model_outputs, axis=0)
            last_metric_value = metric_computer.compute_global(all_labels, all_model_outputs)
        else:
            last_metric_value = metric_computer.aggregate(all_metric_values)
        print(
            "{} metrics:  {} {:.4f}".format(
                "global" if args.optimization.evaluate_globally else "average",
                metric_name,
                last_metric_value[metric_name],
            )
        )

    def _save_model_on_improvement():
        nonlocal best_metric_value

        # Compare the metric values.
        comparison_score = 0
        if best_metric_value is None:
            comparison_score = 1
        else:
            if last_metric_value is None:
                comparison_score = -1
            else:
                comparison_score = metric_sign * (last_metric_value[metric_name] - best_metric_value[metric_name])

        # Save model checkpoint if the metric values improved.
        if comparison_score > 0:
            best_metric_value = last_metric_value
            best_checkpoint_file_name = os.path.join(model_directory_name, BEST_CHECKPOINT_FILE_NAME)
            torch.save(model.state_dict(), best_checkpoint_file_name)

    def _save_debug_info(epoch):
        training_writer.add_scalar("INFO/1_loss", np.mean(training_losses), epoch)
        training_writer.flush()

        validation_writer.add_scalar("INFO/1_loss", np.mean(validation_losses), epoch)
        validation_writer.add_scalar("INFO/2_{}".format(metric_name), last_metric_value[metric_name], epoch)
        validation_writer.flush()

    # Make the output directory if not present.
    model_directory_name = os.path.join(
        args.run.output_directory_name, RUN_DIRECTORY_NAME_TEMPLATE.format(run_id=args.run.id)
    )
    os.makedirs(model_directory_name, exist_ok=True)

    # Move the model to GPU if specified.
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = model.to(device)

    # Initialize the last and best metric values.
    last_metric_value, best_metric_value = None, None
    metric_name, metric_mode = args.model.metric_mode.split("/")
    metric_sign = 1 if metric_mode == "max" else -1 if metric_mode == "min" else None

    # Make summary writers for debugging.
    training_writer = SummaryWriter(os.path.join(model_directory_name, "training"))
    validation_writer = SummaryWriter(os.path.join(model_directory_name, "validation"))

    # Train for the specified number of epochs.
    for epoch in range(args.optimization.num_training_epochs):
        # Print progress information and start timer.
        print("===\nepoch {}".format(epoch + 1))
        start_time = time.time()

        # Do gradient descent steps on the training subset and inference+evaluation steps on the validation subset.
        training_losses, validation_losses = [], []
        _do_training_steps()
        _do_validation_steps()

        # Save current checkpoint if better than previous.
        _save_model_on_improvement()

        # Save loss and metric values for training and validation.
        _save_debug_info(epoch + 1)

        # Print training and evaluation time.
        print("time taken: {:.2f}s".format(time.time() - start_time))


def do_training_runs(args):
    # Run training as many times as specified.
    num_repetitions = args.get("num_repetitions", 1)
    for i in range(num_repetitions):
        # Log the repetition/split index.
        if args.get("different_splits", False):
            print("\nsplit {}\n".format(i))
        else:
            print("\nrepetition {}\n".format(i))

        # Set the repetition specific run id.
        a = copy.deepcopy(args)
        a.run.id = args.run.id + args.get("offset_for_repetition", 1) * i

        # Skip repetition if multiple repetitions requested and it seems to have run.
        if (num_repetitions > 1) and os.path.exists(
            os.path.join(a.run.output_directory_name, "run-{:04d}".format(a.run.id))
        ):
            print("skipping {}".format(a.run.id))
            continue

        # If different split, set the dataset version and source embedding id.
        if args.dataset.get("different_splits", False):
            a.dataset.version = "{}~{}".format(args.dataset.version, i)
            if args.dataset.get("source_embeddings_id") is not None:
                a.dataset.source_embeddings_id = "{}~{}".format(args.dataset.source_embeddings_id, i)

        # Run the training for the current repetition.
        objects = make_training_objects(a)
        train(*objects, a)


if __name__ == "__main__":
    # Parse arguments from command line and configuration file if specified.
    args = parse_args()

    # Do training runs as specified.
    do_training_runs(args)
