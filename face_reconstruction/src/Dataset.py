import os
import cv2
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import sys
from recreate_icip2022_face_reconstruction.src.GenerateLFWView2 import GenerateLFWView2

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_all_filenames_from_dir(datasets_dir: str, data_dir: str):
    """
    Import filenames from a directory and its subdirectories into a list

    Args:
        datasets_dir (): Parent directory
        data_dir (): Directory name

    Returns:

    """
    all_files_list = []
    for subdir, dirs, files in os.walk(os.path.join(datasets_dir, data_dir)):
        for element in files:
            file = os.path.join(subdir, element)
            all_files_list.append(file)
    return all_files_list


class ParentDataset(Dataset):
    def __init__(self, dataset_dir='../data',
                 image_dir="",
                 embedding_dir="",
                 train=True,
                 device='cpu',
                 mix_id_train_test=True,
                 train_test_split=0.9,
                 random_seed=2021,
                 return_filename=False,
                 ):
        self.random_seed = random_seed
        self.train_test_split = train_test_split
        self.mixID_TrainTest = mix_id_train_test
        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.device = device
        self.train = train
        self.dir_all_images = []
        self.dir_all_embeddings = []
        self.return_filename = return_filename
        self.length_of_embedding_dir = len(self.dataset_dir) + len(self.embedding_dir) + 2

    def __len__(self):
        return len(self.dir_all_images)

    def __getitem__(self, idx):
        image = f"{self.dir_all_images[idx]}"
        image = self.transform_image(image)

        embedding = f"{self.dir_all_embeddings[idx]}"
        filename = embedding[self.length_of_embedding_dir:-4]  # Get filename of embedding w/o extension
        embedding = np.load(embedding)  # shape (512,1,1)
        embedding = self.transform_embedding(embedding)

        if self.return_filename:
            return embedding, image, filename
        else:
            return embedding, image

    def transform_image(self, image):
        # ======= RESIZE ========
        # image = image/255.
        # image     = np.load(image) # range (0,1), shape (3, 112, 112)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        image = torch.Tensor(image).to(self.device)
        return image

    def transform_embedding(self, embedding):
        embedding = np.reshape(embedding, [embedding.shape[-1], 1, 1])
        embedding = torch.Tensor(embedding).to(self.device)
        return embedding


class MyDataset(ParentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Iterate through all subdirectories in the source directory
        self.dir_all_images = get_all_filenames_from_dir(self.dataset_dir, self.image_dir)
        self.dir_all_embeddings = get_all_filenames_from_dir(self.dataset_dir, self.embedding_dir)

        # for folder in range(70):
        #     for npyfile in range(1000):
        #         self.dir_all_images.append(f"{folder:02d}/{folder:02d}{npyfile:03d}.npy")

        if self.mixID_TrainTest:
            random.seed(self.random_seed)
            shuffled_list = list(zip(self.dir_all_images, self.dir_all_embeddings))
            random.shuffle(shuffled_list)
            self.dir_all_images, self.dir_all_embeddings = zip(*shuffled_list)

        if self.train:
            self.dir_all_images = self.dir_all_images[:int(self.train_test_split * len(self.dir_all_images))]
            self.dir_all_embeddings = self.dir_all_embeddings[
                                      :int(self.train_test_split * len(self.dir_all_embeddings))]
        else:
            self.dir_all_images = self.dir_all_images[int(self.train_test_split * len(self.dir_all_images)):]
            self.dir_all_embeddings = self.dir_all_embeddings[
                                      int(self.train_test_split * len(self.dir_all_embeddings)):]


class LeanDataset(ParentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dir_all_images = get_all_filenames_from_dir(self.dataset_dir, self.image_dir)
        self.dir_all_embeddings = get_all_filenames_from_dir(self.dataset_dir, self.embedding_dir)


class LFWView2SpecificationDataset(ParentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        view2_gen = GenerateLFWView2()
        pairs = view2_gen.get_lfwview2_filenames(image_dir=self.image_dir, embedding_dir=self.embedding_dir)

        # Unpack pairs into separate lists
        target_1_image = [i[0] for i in pairs]
        target_1_embedding = [i[1] for i in pairs]
        target_2_image = [i[2] for i in pairs]
        target_2_embedding = [i[3] for i in pairs]

        # Append lists to self and remove duplicates without changing the order of the list
        [self.dir_all_images.append(x) for x in target_1_image if x not in self.dir_all_images]
        [self.dir_all_images.append(x) for x in target_2_image if x not in self.dir_all_images]
        [self.dir_all_embeddings.append(x) for x in target_1_embedding if x not in self.dir_all_embeddings]
        [self.dir_all_embeddings.append(x) for x in target_2_embedding if x not in self.dir_all_embeddings]


class LeanDatasetEmbeddings(ParentDataset):
    def __init__(self, iterations, **kwargs):
        super().__init__(**kwargs)
        self.iterations = iterations

        self.dir_all_embeddings = get_all_filenames_from_dir(self.dataset_dir, self.embedding_dir)

    def __len__(self):
        return len(self.dir_all_embeddings)

    def __getitem__(self, idx):
        embedding = f"{self.dir_all_embeddings[idx]}"
        filename = embedding.split('/')[-1].split('.')[0]  # Get filename of embedding w/o path and extension
        # filename = '_'.join(filename.split('_')[:-1])  # Get rid of _000x extension for pemiu16 experiment todo
        file_parent_dir = embedding.split('/')[-2]  # Get parent dir of embedding
        embedding = np.load(embedding)  # shape (512,1,1)
        embedding = self.transform_embedding(embedding)

        # Get corresponding original image by modifying the embedding filename
        image = f"{self.image_dir}/{'_'.join(filename.split('_')[:-1])}/{filename}.png"
        image = self.transform_image(image)

        # Return embedding and image
        if self.return_filename:
            return embedding, image, filename, file_parent_dir
        else:
            return embedding, image
