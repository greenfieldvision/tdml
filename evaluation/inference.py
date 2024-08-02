import more_itertools
import torch

from omegaconf import OmegaConf

from tdml.factory import make_model


def _parse_model_args_file_name(model_uri):
    name_and_parameters, file_name = model_uri.split("/", 1)

    tokens = name_and_parameters.split(",")
    name = tokens[0]
    parameters = {t.split(":")[0]: t.split(":")[1] for t in tokens[1:]}

    # Attempt conversion to number.
    def _maybe_convert_to_number(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    parameters = {k: _maybe_convert_to_number(v) for k, v in parameters.items()}

    # Convert name and parameters to OmegaConf arguments.
    name_and_parameters = {"name": name}
    name_and_parameters.update(parameters)
    args = OmegaConf.create(name_and_parameters)

    return args, file_name


class Inference:
    def __init__(self, preprocessing, model_uri, dataset_builder, batch_size=1):
        self.preprocessing = preprocessing

        model_args, model_file_name = _parse_model_args_file_name(model_uri)

        self.model = make_model(dataset_builder, model_args)
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.batch_size = batch_size

    def infer(self, input_records):
        output = []
        for r_batch in more_itertools.chunked(input_records, self.batch_size):
            x_batch = torch.stack([self.preprocessing(r[0]) for r in r_batch])
            x_batch = x_batch.to(self.device)

            with torch.no_grad():
                y_batch = self.model(x_batch)
                y_batch = y_batch.cpu().numpy()

            for y, r in zip(y_batch, r_batch):
                output.append((y,) + tuple(r[1:]))

        return output

    def deallocate(self):
        """Delete class variables that take up memory. Actual garbage collection is up to the user."""

        self.model = None

        torch.cuda.empty_cache()
