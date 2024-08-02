import os
import time

from abc import ABC, abstractmethod

from PIL import Image


class DatasetBuilder(ABC):
    def __init__(self, args):
        self.raw_data_directory_name = args.raw_data_directory_name
        self.data_directory_name = args.data_directory_name
        if (self.data_directory_name is not None) and (not os.path.exists(self.data_directory_name)):
            os.makedirs(self.data_directory_name)

        self.records_by_subset = None

    def make_pt_loaders(self, preprocessing):
        """Make PyTorch DataLoader objects for the training, validation and test subsets."""

        raise NotImplementedError()

    def get_instances_for_inference(self, subset_name):
        """Generate records for the instances from one of the subsets, to be used during inference."""

        # Read the raw record data if not already done so.
        self._read_raw_records_once()

        # Generate records one by one from the specified subset.
        for r in self.records_by_subset[subset_name]:
            yield self._convert_record_for_inference(r)

    def get_triplet_indexes_for_evaluation(self, subset_name):
        """Generate triplet indexes for one of the subsets, to be used during evaluation."""

        raise NotImplementedError()

    def get_triplet_directedness(self):
        return None

    @abstractmethod
    def _read_raw_records(self):
        """Make records for each subset from raw data on disk."""

        raise NotImplementedError()

    def _read_raw_records_once(self):
        if self.records_by_subset is None:
            print("reading raw records...")
            start_time = time.time()
            self._read_raw_records()
            time_delta = time.time() - start_time
            summary = " ".join(
                [
                    "{} {}".format(subset_name, len(self.records_by_subset[subset_name]))
                    for subset_name in ["training", "validation", "test", "all"]
                    if subset_name in self.records_by_subset.keys()
                ]
            )
            print("done ({:.2f} s): {}".format(time_delta, summary))

    def _convert_record_for_inference(self, record):
        """Convert record in internal format to format used for inference (eg load image based on file name)."""

        return record

    @staticmethod
    def _make_loader_dict(training_loader, validation_loader, test_loader):
        return {
            "training": training_loader,
            "validation": validation_loader,
            "test": test_loader,
        }


class ImageDatasetBuilder(DatasetBuilder):
    def _convert_record_for_inference(self, image_record):
        pil_image = Image.open(image_record[0]).convert("RGB")
        return (pil_image,) + image_record[1:]


class IndexDatasetBuilder(DatasetBuilder):
    def _convert_record_for_inference(self, image_record):
        return (image_record[2], image_record[1])
