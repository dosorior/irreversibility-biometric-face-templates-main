import torch
import numpy as np
import random
import cv2
from ..pemiu.privacy_enhancing_miu import PrivacyEnhancingMIU
from scipy.spatial import distance
from ..loss.FaceIDLoss import get_FaceRecognition_transformer


def process_image(image):
    """

    Args:
        image ():

    Returns: Shape (1, 3, 112, 112)

    """
    # Process image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)  # (1, 3, 112, 112)
    image = torch.Tensor((image * 255.).astype('uint8')).type(torch.FloatTensor)
    return image


class CosineSimilarity:
    def __init__(self):
        # Initialize face transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("The torch device is:", self.device)

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        length_of_embedding = 512

        self.face_recognition_transformer = get_FaceRecognition_transformer(device=self.device)

        # Initialize PEMIU
        # Block size doesn't matter for the cos_sim function
        self.pemiu = PrivacyEnhancingMIU(block_size=16)

    def calc_cos_sim_pemiu_method(self, embedding_a, embedding_b):
        return self.pemiu.cos_sim(embedding_a, embedding_b)

    def get_from_img(self, image_a, image_b):
        """
        Get cosine similarity between two images by generating their respective embeddings

        Args:
            image_a ():
            image_b ():

        Returns: Cosine similarity value using method from PEMIU implementation

        """
        # Process images
        image_a = process_image(image_a)
        image_b = process_image(image_b)

        # Create embeddings from images
        # Note: input img should be in (0,1)
        embedding_a = self.face_recognition_transformer.transform(image_a.to(self.device))
        embedding_b = self.face_recognition_transformer.transform(image_b.to(self.device))
        embedding_a = embedding_a.squeeze().cpu().numpy()
        embedding_b = embedding_b.squeeze().cpu().numpy()

        # Calculate cosine similarity
        cos_sim = self.calc_cos_sim_pemiu_method(embedding_a, embedding_b)

        return cos_sim
