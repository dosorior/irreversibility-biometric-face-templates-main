import argparse
import os
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class AttributePredictionInsightFace:
    def __init__(self,
                 data_dir="../data",
                 lfw_gender_dir="lfw_gender",
                 file_female="female_names.txt",
                 file_male="male_names.txt",
                 ):
        self.lfw_gender_m, self.lfw_gender_f = [], []

        # Add filenames of lfw files to gender lists seperated by gender
        # Remove file extension
        filepath_male = os.path.join(data_dir, lfw_gender_dir, file_male)
        filepath_female = os.path.join(data_dir, lfw_gender_dir, file_female)
        with open(filepath_male, 'r') as f_m, open(filepath_female, 'r') as f_f:
            [self.lfw_gender_m.append(line.strip().split('.')[0]) for line in f_m.readlines()]
            [self.lfw_gender_f.append(line.strip().split('.')[0]) for line in f_f.readlines()]

        # Initialise InsightFace detection
        parser = argparse.ArgumentParser(description='insightface gender-age test')
        # general
        parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
        args = parser.parse_args()

        self.app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
        self.app.prepare(ctx_id=args.ctx, det_size=(128, 128))

    def get_gender_to_lfw_specification(self, image_filenames):
        """

        Get gender of a list of lfw face image files according to the official lfw gender attribution.

        Args:
            image_filenames ():

        Returns: List of strings "M", "F" or "X" in the same order as the input list

        """
        # Remove file extension

        gender_list = []
        for file in image_filenames:
            if file.split('.')[0] in self.lfw_gender_m:
                gender_list.append("M")
            elif file.split('.')[0] in self.lfw_gender_f:
                gender_list.append("F")
            else:
                gender_list.append("X")

        return gender_list

    def get_gender_with_insightface_attribute_model(self, image_filepaths):
        # Load image filename and image into a list of lists
        print(f"Reading {len(image_filepaths)} images")
        all_images = []
        for image in image_filepaths:
            all_images.append([image.split('/')[-1], cv2.imread(image)])

        # Get faces
        print(f"Getting faces for {len(all_images)} images")
        for index, image in enumerate(all_images):
            all_images[index].append(self.app.get(image[1]))

        # Append string representation of gender
        print(f"Evaluate genders for faces")
        genders = []
        for index, image in enumerate(all_images):
            if len(image[2]) > 1:
                print(f"Found {len(image[2])} faces in {image[0]}."
                      f"\n\tDet_score face 1: {image[2][0].det_score}, "
                      f"Det_score face 2: {image[2][1].det_score}, "
                      f"Attributed gender is: {image[2][0].sex}")
            genders.append(image[2][0].sex)

        return genders
