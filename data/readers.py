import os

import scipy.io

from ditdml.data_interfaces.ihsj_data_interface import IHSJDataInterface
from ditdml.data_interfaces.ihsjc_data_interface import IHSJCDataInterface
from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.data_interfaces.yummly_data_interface import YummlyDataInterface

from tdml.data.dataset_builder import DatasetBuilder


class CubDataReader(DatasetBuilder):
    def _read_raw_records(self):
        image_records = []

        image_directory_name = os.path.join(self.raw_data_directory_name, "images")
        for subdirectory_name in os.listdir(image_directory_name):
            full_subdirectory_name = os.path.join(image_directory_name, subdirectory_name)

            if (
                os.path.isdir(full_subdirectory_name)
                and (full_subdirectory_name != ".")
                and (full_subdirectory_name != "..")
            ):
                label_tokens = subdirectory_name.split(".")
                class_index = int(label_tokens[0])

                for file_name in os.listdir(full_subdirectory_name):
                    full_file_name = os.path.join(full_subdirectory_name, file_name)

                    if os.path.isfile(full_file_name):
                        image_records.append((full_file_name, class_index, class_index))

        image_records = sorted(image_records, key=lambda image_record: image_record[0])

        self.records_by_subset = {
            "training": [r for r in image_records if r[1] <= 100],
            "test": [r for r in image_records if r[1] > 100],
        }

    @property
    def num_classes(self):
        return 200


class CarsDataReader(DatasetBuilder):
    def _read_raw_records(self):
        image_records = []

        annotation_file_name = os.path.join(self.raw_data_directory_name, "cars_annos.mat")
        annotation_data = scipy.io.loadmat(annotation_file_name)
        annotations = annotation_data["annotations"]

        for l, fn in zip(annotations["class"][0], annotations["relative_im_path"][0]):
            class_index = int(l[0, 0])
            full_file_name = os.path.join(self.raw_data_directory_name, fn[0])
            image_records.append((full_file_name, class_index, class_index))

        self.records_by_subset = {
            "training": [r for r in image_records if r[1] <= 98],
            "test": [r for r in image_records if r[1] > 98],
        }

    @property
    def num_classes(self):
        return 196


class SopDataReader(DatasetBuilder):
    def _read_raw_records(self):
        self.records_by_subset = {
            "training": self._read_image_records_from_file("Ebay_train.txt"),
            "test": self._read_image_records_from_file("Ebay_test.txt"),
        }

    def _read_image_records_from_file(self, annotation_file_name):
        image_records = []

        full_annotation_file_name = os.path.join(self.raw_data_directory_name, annotation_file_name)
        with open(full_annotation_file_name, "rt") as f:
            for i, line in enumerate(f):
                if i > 0:
                    tokens = line.rstrip("\n").split(" ")
                    full_file_name = os.path.join(self.raw_data_directory_name, tokens[3])
                    class_index = int(tokens[1])
                    superclass_index = int(tokens[2])

                    image_records.append((full_file_name, class_index, superclass_index))

        return image_records

    @property
    def num_classes(self):
        return 22634


class ThingsReader(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)

        self.split_type, self.class_triplet_conversion_type = {
            "direct": ("by_class_same_training_validation", "all_instances"),
            "psycho": ("by_class_same_training_validation", "prototypes"),
            "source_embedding": ("by_class", "all_instances"),
        }.get(args.version, None)

        self.split_seed = 0

    def _read_raw_records(self):
        data_interface = ThingsDataInterface(
            self.raw_data_directory_name,
            self.split_type,
            self.split_seed,
            class_triplet_conversion_type=self.class_triplet_conversion_type,
        )
        records_with_index = [r + (i,) for i, r in enumerate(data_interface.reader.image_records)]

        self.triplets_by_subset = data_interface.triplets_by_subset
        self.instances_by_subset = data_interface.instances_by_subset
        self.prototype_instance_per_instance = [data_interface.prototypes_per_class[r[1]] for r in records_with_index]

        def records_in_subset(subset_name):
            return [records_with_index[i] for i in data_interface.instances_by_subset[subset_name]]

        self.records_by_subset = {
            "all": records_with_index,
            "training": records_in_subset("training"),
            "validation": records_in_subset("validation"),
            "test": records_in_subset("test"),
        }

    def get_triplet_directedness(self):
        return False

    def get_instance_index_remapping(self):
        self._read_raw_records_once()

        return self.prototype_instance_per_instance


class IHSJReader(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)

        self.split_type = {
            "direct": "by_class_same_training_validation",
            "psycho": "by_instance_same_training_validation",
            "source_embedding": "by_instance",
        }.get(args.version, None)

        self.split_seed = 0

    def _read_raw_records(self):
        data_interface = IHSJDataInterface(self.raw_data_directory_name, self.split_type, self.split_seed)
        records_with_index = [r + (i,) for i, r in enumerate(data_interface.reader.image_records)]

        self.triplets_by_subset = data_interface.triplets_by_subset
        self.ninelets_by_subset = data_interface.ninelets_by_subset
        self.instances_by_subset = data_interface.instances_by_subset

        def records_in_subset(subset_name):
            return [records_with_index[i] for i in self.instances_by_subset[subset_name]]

        self.records_by_subset = {
            "all": records_with_index,
            "training": records_in_subset("training"),
            "validation": records_in_subset("validation"),
            "test": records_in_subset("test"),
        }

    def get_triplet_directedness(self):
        return True

    def get_instance_index_remapping(self):
        self._read_raw_records_once()

        return list(range(len(self.records_by_subset["all"])))


class IHSJCReader(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)

        self.split_type, self.class_triplet_conversion_type = {
            "direct": ("by_class_same_training_validation", "all_instances"),
            "psycho": ("by_class_same_training_validation", "prototypes"),
            "source_embedding": ("by_class", "all_instances"),
        }.get(args.version, None)

        self.split_seed = 0

    def _read_raw_records(self):
        data_interface = IHSJCDataInterface(
            self.raw_data_directory_name,
            self.split_type,
            self.split_seed,
            class_triplet_conversion_type=self.class_triplet_conversion_type,
        )
        records_with_index = [r + (i,) for i, r in enumerate(data_interface.reader.image_records)]

        self.triplets_by_subset = data_interface.triplets_by_subset
        self.instances_by_subset = data_interface.instances_by_subset
        self.prototype_instance_per_instance = [data_interface.prototypes_per_class[r[1]] for r in records_with_index]

        def records_in_subset(subset_name):
            return [records_with_index[i] for i in self.instances_by_subset[subset_name]]

        self.records_by_subset = {
            "all": records_with_index,
            "training": records_in_subset("training"),
            "validation": records_in_subset("validation"),
            "test": records_in_subset("test"),
        }

    def get_triplet_directedness(self):
        return True

    def get_instance_index_remapping(self):
        self._read_raw_records_once()

        return self.prototype_instance_per_instance


class YummlyReader(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)

        parsed_dataset_version = args.version.split("~")
        self.split_type = {
            "direct": "by_instance_same_training_validation",
            "psycho": "by_instance_same_training_validation",
            "source_embedding": "by_instance",
        }.get(parsed_dataset_version[0], None)
        self.split_seed = 0 if len(parsed_dataset_version) == 1 else int(parsed_dataset_version[1])

    def _read_raw_records(self):
        data_interface = YummlyDataInterface(self.raw_data_directory_name, self.split_type, self.split_seed)
        records_with_index = [r + (i,) for i, r in enumerate(data_interface.reader.image_records)]

        self.triplets_by_subset = data_interface.triplets_by_subset
        self.instances_by_subset = data_interface.instances_by_subset

        def records_in_subset(subset_name):
            return [records_with_index[i] for i in self.instances_by_subset[subset_name]]

        self.records_by_subset = {
            "all": records_with_index,
            "training": records_in_subset("training"),
            "validation": records_in_subset("validation"),
            "test": records_in_subset("test"),
        }

    def get_triplet_directedness(self):
        return True

    def get_instance_index_remapping(self):
        self._read_raw_records_once()

        return list(range(len(self.records_by_subset["all"])))
