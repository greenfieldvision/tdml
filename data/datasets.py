from PIL import Image
from torch.utils.data import Dataset


class ImageClassDataset(Dataset):
    def __init__(self, image_records, preprocessing, is_training, class_index_offset=0):
        self.image_records = image_records
        self.preprocessing = preprocessing

        self.is_training = is_training

        self.class_index_offset = class_index_offset

    def __getitem__(self, idx):
        image_record = self.image_records[idx]

        image = Image.open(image_record[0])
        if len(image.size) == 2:
            image = image.convert("RGB")
        preprocessed_image = self.preprocessing(image, decode_bytes=False, augment=self.is_training)

        class_index = image_record[1] + self.class_index_offset

        return preprocessed_image, class_index, idx

    def __len__(self):
        return len(self.image_records)


class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_records, supervision_embeddings, preprocessing, is_training):
        self.image_records = image_records
        self.preprocessing = preprocessing

        self.supervision_embeddings = supervision_embeddings

        self.is_training = is_training

    def __getitem__(self, idx):
        image_record = self.image_records[idx]

        image = Image.open(image_record[0])
        if len(image.size) == 2:
            image = image.convert("RGB")
        preprocessed_image = self.preprocessing(image, decode_bytes=False, augment=self.is_training)

        if self.supervision_embeddings is None:
            supervision_embedding = None
        else:
            supervision_embedding = self.supervision_embeddings[idx]

        return preprocessed_image, supervision_embedding, idx

    def __len__(self):
        return len(self.image_records)


class IndexDataset(Dataset):
    def __init__(self, image_records, preprocessing):
        # Note that `image_records` contains _all_ the records in the dataset (training, validation and test).
        self.image_records = image_records
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        index = self.preprocessing(idx)
        return index, 0, idx

    def __len__(self):
        return len(self.image_records)
