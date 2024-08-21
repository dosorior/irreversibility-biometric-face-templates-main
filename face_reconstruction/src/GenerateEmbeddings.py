import torch
import numpy as np
import random
import cv2
from .pemiu.privacy_enhancing_miu import PrivacyEnhancingMIU
from scipy.spatial import distance
from .loss.FaceIDLoss import get_FaceRecognition_transformer
from .CosineSimilarity.CosineSimilarity import process_image


class GenerateEmbeddings:
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

    def embedding_from_img(self, image):
        """
        Generate embedding from image

        Args:
            image ():

        Returns: numpy array

        """
        # Process images
        image = process_image(image)

        # Create embeddings from images
        # Note: input img should be in (0,1)
        embedding = self.face_recognition_transformer.transform(image.to(self.device))
        embedding = embedding.squeeze().cpu().numpy()

        return embedding
