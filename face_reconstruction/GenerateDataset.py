import os
import cv2
import torch
import numpy as np
from src.loss.FaceIDLoss import get_FaceRecognition_transformer
from src.Dataset import get_all_filenames_from_dir
from src.bob.bio.face.preprocessor.FaceCrop import FaceCrop


# Crop function from
# https://gitlab.idiap.ch/bob/bob.paper.icip2022_face_reconstruction/-/blob/master/experiments/ArcFace/DataGen/GenDataset.py
def crop(img):
    """
    
    Args:
        img ():

    Returns:

    Input:
        - img: RGB or BGR image in 0-1 or 0-255 scale
    Output:
        - new_img: RGB or BGR image in 0-1 or 0-255 scale
    """

    FFHQ_REYE_POS = (480 / 2, 380 / 2)
    FFHQ_LEYE_POS = (480 / 2, 650 / 2)

    CROPPED_IMAGE_SIZE = (112, 112)
    fixed_positions = {'reye': FFHQ_REYE_POS, 'leye': FFHQ_LEYE_POS}

    # annotation_type='eyes-center'
    # cropped_positions = dnn_default_cropping(CROPPED_IMAGE_SIZE, annotation_type)
    cropped_positions = {
        "leye": (51.5014, 73.5318),
        "reye": (51.6963, 38.2946)
    }

    cropper = FaceCrop(cropped_image_size=CROPPED_IMAGE_SIZE,
                       cropped_positions=cropped_positions,
                       color_channel='rgb',
                       fixed_positions=fixed_positions,
                       annotator="mtcnn",
                       )

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)  # (1, 3, 1024, 1024)
    new_image = cropper.transform(img)
    new_image = new_image[0]  # (3, 112, 112)
    new_image = new_image.transpose(1, 2, 0)  # (112, 112, 3) as OpenCV

    return new_image


def process_image(image, crop_image=True):
    """
    Read, normalizes resize, crop image. Expand dimension of output image by 1.

    Args:
        image ():
        crop_image (bool): enable the crop function (Warning: Currently works only with FFHQ image size)

    Returns: Shape (1, 3, 112, 112)

    """
    # Process image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    if crop_image:
        image = crop(image)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)  # (1, 3, 112, 112)
    return image


class GenerateDataset:
    def __init__(self,
                 device: str = 'cpu',
                 dataset_dir="../data",
                 image_dir="lfw_align",
                 save_dir="output",
                 ):
        """
        Crop and resize an image dataset and generate corresponding facial embeddings.

        Args:
            device (str):
            dataset_dir ():
            image_dir ():
            save_dir ():
        """
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.image_dir = image_dir
        self.device = device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_recognition_transformer = get_FaceRecognition_transformer(device=device)
        self.image_data = self.load_image_data()
        self.batch_size = 0
        self.batch_cnt = 0

        print("************ NOTE: The torch device is:", device)

    def load_image_data(self):
        return get_all_filenames_from_dir(self.dataset_dir, self.image_dir)

    def generate_embedding(self, image):
        new_embedding = self.face_recognition_transformer.transform(image.to(self.device))
        new_embedding = new_embedding.cpu().numpy()
        new_embedding = np.reshape(new_embedding, [new_embedding.shape[0], new_embedding.shape[-1], 1, 1])
        new_embedding = np.squeeze(new_embedding)
        return new_embedding

    def generate_embeddings_for_reconstructed_images(self):
        os.makedirs(f'{os.path.join(self.dataset_dir, self.save_dir)}', exist_ok=True)

        images = get_all_filenames_from_dir(self.dataset_dir, self.image_dir)
        only_reconstructed_images = []

        for image in images:
            if image.split("_")[-1] == "reconstructed.png":
                only_reconstructed_images.append(image)

        for x in only_reconstructed_images:
            image_processed = process_image(x, crop_image=False)

            # Generate embedding
            image_tensor = torch.Tensor((image_processed * 255.).astype('uint8')).type(torch.FloatTensor)
            embedding = self.generate_embedding(image_tensor)

            # Save embedding
            np.save(f"{os.path.join(self.dataset_dir, self.save_dir)}/{x.split('/')[-1].split('.')[0]}.npy", embedding)

    def generate_dataset(self):
        os.makedirs(f'{os.path.join(self.dataset_dir, self.save_dir)}/images', exist_ok=True)
        os.makedirs(f'{os.path.join(self.dataset_dir, self.save_dir)}/embeddings', exist_ok=True)

        file_id_cnt = 0
        for image in self.image_data:
            image = process_image(image)

            # Generate embedding
            image_tensor = torch.Tensor((image * 255.).astype('uint8')).type(torch.FloatTensor)
            embedding = self.generate_embedding(image_tensor)

            # Generate final image
            image = image[0]  # (3, 112, 112)
            image = image.transpose(1, 2, 0)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Generate filename
            file_id = str(self.batch_cnt * self.batch_size + file_id_cnt).zfill(5)

            # Generate and save image and embedding
            cv2.imwrite(f"{os.path.join(self.dataset_dir, self.save_dir)}/images/image_{file_id}.jpg",
                        np.array([image[:, :, 2], image[:, :, 1], image[:, :, 0]]).transpose(1, 2, 0))
            np.save(f"{os.path.join(self.dataset_dir, self.save_dir)}/embeddings/embedding_{file_id}", embedding)

            # Increase filename counter
            file_id_cnt = file_id_cnt + 1
