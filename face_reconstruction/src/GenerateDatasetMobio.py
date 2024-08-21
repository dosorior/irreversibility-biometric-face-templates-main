import os
import cv2
import torch
import numpy as np
from .loss.FaceIDLoss import get_FaceRecognition_transformer
from .Dataset import get_all_filenames_from_dir
from .bob.bio.face.preprocessor.FaceCrop import FaceCrop


# Crop function from
# https://gitlab.idiap.ch/bob/bob.paper.icip2022_face_reconstruction/-/blob/master/experiments/ArcFace/DataGen/GenDataset.py
def crop(img, eye_positions):
    """

    Args:
        img ():
        eye_positions (list): Eye position parameters: l x, l y, r x, r y

    Returns:

    """
    cropped_image_size = (112, 112)
    left_eye_x, left_eye_y, right_eye_x, right_eye_y = eye_positions

    # From the description of FaceCrop.py:
    # 'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)
    fixed_positions = {'reye': (right_eye_y, right_eye_x), 'leye': (left_eye_y, left_eye_x)}

    # annotation_type='eyes-center'
    # cropped_positions = dnn_default_cropping(CROPPED_IMAGE_SIZE, annotation_type)
    cropped_positions = {
        "leye": (51.6963, 38.2946),
        "reye": (51.5014, 73.5318)
    }

    cropper = FaceCrop(cropped_image_size=cropped_image_size,
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


def process_image(image, eye_positions, crop_image=True):
    """
    Read, normalizes resize, crop image. Expand dimension of output image by 1.

    Args:
        image ():
        eye_positions (list): Eye position parameters: l x, l y, r x, r y
        crop_image (bool): enable the crop function

    Returns: Shape (1, 3, 112, 112)

    """
    # Process image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    if crop_image:
        image = crop(image, eye_positions)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)  # (1, 3, 112, 112)
    return image


class GenerateDatasetMobio:
    def __init__(self,
                 device: str = 'cpu',
                 dataset_dir="../data",
                 image_dir="lfw_align",
                 save_dir="output",
                 eye_positions_txt_path="Mobio/lists/eye_positions.txt"
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
        self.eye_positions_txt_path = os.path.join(dataset_dir, eye_positions_txt_path)
        self.device = device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_recognition_transformer = get_FaceRecognition_transformer(device=device)
        self.image_data = self.load_image_data()
        self.eye_positions = self.load_eye_pos_file()

        print("************ NOTE: The torch device is:", device)

    def load_image_data(self):
        return get_all_filenames_from_dir(self.dataset_dir, self.image_dir)

    def load_eye_pos_file(self):
        file = []
        with open(self.eye_positions_txt_path, 'r') as f:
            for line in f.readlines():
                file.append(line.strip().split())
        return np.array(file, dtype=object)

    def get_eye_position(self, image_filename):
        index = np.where(self.eye_positions == image_filename)  # Find index of eye positions in positions file
        positions = self.eye_positions[index[0]][0][1:]  # Convert file entry to list of four positions
        positions = [int(x) for x in positions]  # Convert positions from str to int
        return positions

    def generate_embedding(self, image):
        new_embedding = self.face_recognition_transformer.transform(image.to(self.device))
        new_embedding = new_embedding.cpu().numpy()
        new_embedding = np.reshape(new_embedding, [new_embedding.shape[0], new_embedding.shape[-1], 1, 1])
        new_embedding = np.squeeze(new_embedding)
        return new_embedding

    def generate_dataset(self):
        os.makedirs(f'{os.path.join(self.dataset_dir, self.save_dir)}/images', exist_ok=True)
        os.makedirs(f'{os.path.join(self.dataset_dir, self.save_dir)}/embeddings', exist_ok=True)

        for image in self.image_data:
            # Get filename with extension
            filename = image.split('/')[-1]
            # Get filename with three parent directories and without extension as string
            filename_incl_parent_dir = image.split('/')[-4:]
            filename_incl_parent_dir[-1] = filename_incl_parent_dir[-1].split('.')[0]
            filename_incl_parent_dir = '/'.join(filename_incl_parent_dir)

            # Get eye positions and process image
            eye_pos = self.get_eye_position(filename_incl_parent_dir)
            image = process_image(image, eye_pos)

            # Generate embedding
            image_tensor = torch.Tensor((image * 255.).astype('uint8')).type(torch.FloatTensor)
            embedding = self.generate_embedding(image_tensor)

            # Generate final image
            image = image[0]  # (3, 112, 112)
            image = image.transpose(1, 2, 0)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Generate filename
            # file_id = str(self.batch_cnt * self.batch_size + file_id_cnt).zfill(5)

            # Generate and save image and embedding
            cv2.imwrite(os.path.join(self.dataset_dir, self.save_dir, 'images', filename),
                        np.array([image[:, :, 2], image[:, :, 1], image[:, :, 0]]).transpose(1, 2, 0))
            np.save(f"{os.path.join(self.dataset_dir, self.save_dir, 'embeddings', filename.split('.')[0])}.npy", embedding)
