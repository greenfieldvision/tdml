import os
import random

import numpy as np

from torch.utils.data import DataLoader

from tdml.data.dataset_builder import DatasetBuilder, ImageDatasetBuilder, IndexDatasetBuilder
from tdml.data.datasets import ImageClassDataset, ImageEmbeddingDataset, IndexDataset
from tdml.data.readers import CubDataReader, CarsDataReader, SopDataReader, IHSJCReader, ThingsReader, YummlyReader
from tdml.data.samplers import TripletSampler


NUM_WORKERS = 8


class TripletDatasetBuilder(DatasetBuilder):
    SEED = 0

    def __init__(self, args):
        super().__init__(args)

        self.batch_size = args.batch_size

        # Initialize the random number generator.
        random.seed(self.SEED)

    def make_pt_loaders(self, preprocessing):
        # Read the raw record data if not already done so.
        self._read_raw_records_once()

        # Make PyTorch datasets for training, validation and test.
        all_records = self.records_by_subset["all"]
        training_subset = self._make_dataset(all_records, preprocessing, True)
        validation_subset = self._make_dataset(all_records, preprocessing, False)
        test_subset = self._make_dataset(all_records, preprocessing, False)

        # Make PyTorch samplers that produce batches by randomly sampling labeled triplets from the training, validation
        # and test subsets.
        training_sampler = TripletSampler(self.triplets_by_subset["training"], self.batch_size, True)
        validation_sampler = TripletSampler(self.triplets_by_subset["validation"], self.batch_size, False)
        test_sampler = TripletSampler(self.triplets_by_subset["test"], self.batch_size, False)

        # Make PyTorch data loaders.
        training_loader = DataLoader(training_subset, batch_sampler=training_sampler, num_workers=NUM_WORKERS)
        validation_loader = DataLoader(validation_subset, batch_sampler=validation_sampler, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_subset, batch_sampler=test_sampler, num_workers=NUM_WORKERS)

        return self._make_loader_dict(training_loader, validation_loader, test_loader)

    def get_triplet_indexes_for_evaluation(self, subset_name):
        # Read the raw record data if not already done so.
        self._read_raw_records_once()

        yield from self.triplets_by_subset[subset_name]

    def _make_dataset(self, records, preprocessing, is_training):
        raise NotImplementedError()


class TripletOfImagesDatasetBuilder(TripletDatasetBuilder, ImageDatasetBuilder):
    def _make_dataset(self, records, preprocessing, is_training):
        return ImageClassDataset(records, preprocessing, is_training)


class TripletOfIndexesDatasetBuilder(TripletDatasetBuilder, IndexDatasetBuilder):
    def _make_dataset(self, records, preprocessing, is_training):
        return IndexDataset(records, preprocessing)


class SupervisionEmbeddingDatasetBuilder(ImageDatasetBuilder):
    SEED = 0

    def __init__(self, args):
        super().__init__(args)

        # Copy batch size.
        self.batch_size = args.batch_size

        # Initialize the embeddings according to whether the embedding file exists.
        file_name = os.path.join(args.data_directory_name, "embeddings_{}.npz".format(args.source_embeddings_id))
        if os.path.exists(file_name):
            print("loading embeddings from {}".format(file_name))

            # Load NumPy array with embeddings.
            npz_file = np.load(file_name)
            self.all_supervision_embeddings = npz_file["embeddings"].astype(np.float32)

        else:
            # Set embeddings to invalid.
            self.all_supervision_embeddings = None

        # Initialize the random number generator.
        random.seed(self.SEED)

    def make_pt_loaders(self, preprocessing):
        # Read the raw record data if not already done so.
        self._read_raw_records_once()

        # Make PyTorch datasets for training, validation and test.
        training_subset = self._make_dataset(
            self.records_by_subset["all"], self.instances_by_subset["training"], preprocessing, True
        )
        validation_subset = self._make_dataset(
            self.records_by_subset["all"], self.instances_by_subset["validation"], preprocessing, False
        )
        test_subset = self._make_dataset(
            self.records_by_subset["all"], self.instances_by_subset["test"], preprocessing, False
        )

        # Make PyTorch data loaders.
        training_loader = DataLoader(training_subset, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=True)
        validation_loader = DataLoader(validation_subset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

        return self._make_loader_dict(training_loader, validation_loader, test_loader)

    def _make_dataset(self, all_records, indexes, preprocessing, is_training):
        records = [all_records[i] for i in indexes]
        supervision_embeddings = [self.all_supervision_embeddings[i] for i in indexes]
        return ImageEmbeddingDataset(records, supervision_embeddings, preprocessing, is_training)


class SupervisionEmbeddingClassesDatasetBuilder(ImageDatasetBuilder):
    SEED = 0

    def __init__(self, args):
        super().__init__(args)

        # Copy batch size.
        self.batch_size = args.batch_size

        # Initialize additional supervision embeddings to invalid.
        self.all_supervision_embeddings = None

        if args.source_embeddings_id is not None:
            # Initialize the embeddings according to whether the embedding file exists.
            file_name = os.path.join(args.data_directory_name, "embeddings_{}.npz".format(args.source_embeddings_id))
            if os.path.exists(file_name):
                print("loading embeddings from {}".format(file_name))

                # Load NumPy array with embeddings.
                npz_file = np.load(file_name)
                self.all_supervision_embeddings = npz_file["embeddings"].astype(np.float32)

        # Initialize the random number generator.
        random.seed(self.SEED)

    def make_pt_loaders(self, preprocessing):
        # Read the raw record data if not already done so.
        self._read_raw_records_once()

        # Make PyTorch datasets for training, validation and test.
        training_subset = self._make_dataset(self.records_by_subset["training"], preprocessing, True)
        validation_subset = self._make_dataset(self.records_by_subset["test"], preprocessing, False)
        test_subset = self._make_dataset(self.records_by_subset["test"], preprocessing, False)

        # Make data loaders.
        training_loader = DataLoader(training_subset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        validation_loader = DataLoader(validation_subset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

        return self._make_loader_dict(training_loader, validation_loader, test_loader)

    def _make_dataset(self, records, preprocessing, is_training):
        if is_training and (self.all_supervision_embeddings is not None):
            # Note that there is no fixed mapping between training classes and supervision embeddings. The assumption
            # is that what matters is that the embeddings are spaced far apart.
            supervision_embeddings = [self.all_supervision_embeddings[r[1] - 1] for r in records]
        else:
            I = np.eye(self.num_classes, dtype=np.float32)
            supervision_embeddings = [np.expand_dims(I[r[1] - 1], 0) for r in records]
        return ImageEmbeddingDataset(records, supervision_embeddings, preprocessing, is_training)


class CUBSEC(SupervisionEmbeddingClassesDatasetBuilder, CubDataReader):
    pass


class CARSSEC(SupervisionEmbeddingClassesDatasetBuilder, CarsDataReader):
    pass


class SOPSEC(SupervisionEmbeddingClassesDatasetBuilder, SopDataReader):
    pass


class ThingsTImg(TripletOfImagesDatasetBuilder, ThingsReader):
    pass


class ThingsTInd(TripletOfIndexesDatasetBuilder, ThingsReader):
    @property
    def num_instances(self):
        self._read_raw_records_once()

        return len(self.records_by_subset["all"])


class ThingsSE(SupervisionEmbeddingDatasetBuilder, ThingsReader):
    pass


class IHSJCTImg(TripletOfImagesDatasetBuilder, IHSJCReader):
    pass


class IHSJCTInd(TripletOfIndexesDatasetBuilder, IHSJCReader):
    @property
    def num_instances(self):
        self._read_raw_records_once()

        return len(self.records_by_subset["all"])


class IHSJCSE(SupervisionEmbeddingDatasetBuilder, IHSJCReader):
    pass


class YummlyTImg(TripletOfImagesDatasetBuilder, YummlyReader):
    pass


class YummlyTInd(TripletOfIndexesDatasetBuilder, YummlyReader):
    @property
    def num_instances(self):
        self._read_raw_records_once()

        return len(self.records_by_subset["all"])


class YummlySE(SupervisionEmbeddingDatasetBuilder, YummlyReader):
    pass
