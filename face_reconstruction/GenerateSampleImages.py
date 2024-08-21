# Sample images
# Based on: icip2022_face_reconstruction

import os
import sys
import torch
import random
import numpy as np
import cv2
from scipy.spatial import distance
from .src.Dataset import LFWView2SpecificationDataset, LeanDatasetEmbeddings
from torch.utils.data import DataLoader
from .src.Network import Generator
from .src.loss.SSIMLoss import SSIM_Loss
from .src.loss.FaceIDLoss import ID_Loss, get_FaceRecognition_transformer
from .src.Dataset import get_all_filenames_from_dir


class GenerateSampleImages:
    def __init__(self,
                 dataset_dir="",
                 image_dir="",
                 embedding_dir="",
                 save_path="sample_images",
                 save_path_log="",
                 file_appendix="reconstructed",
                 generator_checkpoint_dir="training_files",
                 write_original_img=True,
                 create_subdirs=False,
                 batch_size=32,
                 epoch=90,
                 iterations=1):
        self.sample_dataloader = None
        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.save_path = save_path
        self.save_path_log = save_path_log
        self.file_appendix = file_appendix
        self.write_original_img = write_original_img
        self.batch_size = batch_size
        self.create_subdirs = create_subdirs
        self.generator_checkpoint_dir = generator_checkpoint_dir
        self.epoch = epoch
        self.iterations = iterations

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("The torch device is:", self.device)

    def generate(self):
        sys.path.append(os.getcwd())
        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        length_of_embedding = 512

        # =================== import Network =====================
        model_Generator = Generator(length_of_embedding=length_of_embedding)
        generator_checkpoint = f"{self.generator_checkpoint_dir}/models/Generator_{self.epoch}.pth"
        model_Generator.load_state_dict(
            torch.load(generator_checkpoint, map_location=self.device, )
        )
        model_Generator.to(self.device)
        # ========================================================

        # =================== import Loss ========================
        # ***** SSIM_Loss
        ssim_loss = SSIM_Loss()
        ssim_loss.to(self.device)

        # ***** ID_loss
        # ID_loss = ID_Loss(device=device)
        FaceRecognition_transformer = get_FaceRecognition_transformer(device=self.device)

        # ***** Other losses
        # MAE_loss = torch.nn.L1Loss()
        # MSE_loss = torch.nn.MSELoss()
        # BCE_loss = torch.nn.BCELoss()
        # ========================================================

        # Save models and logs
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(f'{self.save_path_log}/logs', exist_ok=True)
        with open(f'{self.save_path_log}/logs/sample_images_log_{self.file_appendix}.csv', 'w') as f:
            f.write(f"filename,"
                    f"MAE,"
                    f"Cosine Distance Bona Fide Embedding vs Input Embedding,"
                    f"Cosine Distance Bona Fide Embedding vs Synthesised Image Embedding\n")

        # Generate samples
        print(f"Generating around {len(self.sample_dataloader) * self.batch_size} samples")

        for embedding, real_image, filename, file_parent_dir in self.sample_dataloader:

            # new_embedding = FaceRecognition_transformer.transform(real_image.to(device) * 255.0)
            # new_embedding = new_embedding.cpu().numpy()
            # new_embedding = np.reshape(new_embedding, [new_embedding.shape[0], new_embedding.shape[-1], 1, 1])
            # new_embedding = torch.tensor(new_embedding).to(device)

            generated_image = model_Generator(embedding).detach().cpu()

            embedding_real_images = FaceRecognition_transformer.transform(
                real_image.to(self.device) * 255.0)  # Note: input img should be in (0,1)
            embedding_generated_images = FaceRecognition_transformer.transform(generated_image.to(self.device) * 255.0)

            for i in range(generated_image.size(0)):

                # With the LFW view2 comparison, each sample image should be saved in its own directory
                # to avoid conflicts with duplicate filenames.
                # Therefore, we create a new save_path_final variable, that stores either the same value as self,
                # or, for cases where samples should be stored in a dedicated parent directory, the path including
                # the parent directory.
                save_path_final = self.save_path
                if self.create_subdirs:
                    save_path_final = os.path.join(self.save_path, file_parent_dir[i])
                    os.makedirs(f'{save_path_final}', exist_ok=True)

                generated_img = generated_image[i].squeeze()
                real_img = real_image[i].squeeze()

                if self.write_original_img:
                    im = (real_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(int)
                    cv2.imwrite(f'{save_path_final}/{filename[i]}_original.png',
                                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))

                im = (generated_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(int)
                cv2.imwrite(f'{save_path_final}/{filename[i]}_{self.file_appendix}_{str(i).zfill(4)}.png',
                            np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))

                embedding_real = embedding_real_images[i].squeeze().cpu().numpy()
                embedding_gen = embedding_generated_images[i].squeeze().cpu().numpy()
                embedding_input = embedding[i].squeeze().cpu().numpy()

                with open(f'{self.save_path_log}/logs/sample_images_log_{self.file_appendix}.csv', 'a') as f:
                    f.write(
                        f"{filename[i]}_{self.file_appendix}_{str(i).zfill(4)}.png,"
                        f"{np.mean(np.abs(real_img.cpu().numpy() - generated_img.cpu().numpy()))},"
                        f"{round(1 - distance.cosine(embedding_real, embedding_input), 3)},"
                        f"{round(1 - distance.cosine(embedding_real, embedding_gen), 3)}\n")

        print("Done\n")


class GenerateSampleImagesDefault(GenerateSampleImages):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # import Dataset
        sample_dataset = LFWView2SpecificationDataset(train=False,
                                                      device=self.device,
                                                      image_dir=self.image_dir,
                                                      embedding_dir=self.embedding_dir,
                                                      # embedding_dir="lfw_embeddings_pemiu",
                                                      # embedding_dir="lfw_embeddings_pemiu_reconstructed",
                                                      return_filename=True)

        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.batch_size, shuffle=False)


class GenerateSampleImagesFromEmbeddings(GenerateSampleImages):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        sample_dataset = LeanDatasetEmbeddings(train=False,
                                               device=self.device,
                                               dataset_dir=self.dataset_dir,
                                               embedding_dir=self.embedding_dir,
                                               image_dir=self.image_dir,
                                               return_filename=True,
                                               iterations=self.iterations)

        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.batch_size, shuffle=False)
