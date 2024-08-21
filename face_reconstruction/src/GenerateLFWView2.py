import os
import numpy as np
from sklearn.model_selection import KFold


def get_filename(folder_name, index, extension):
    return folder_name + "_" + index.zfill(4) + "." + extension


class GenerateLFWView2:
    def __init__(self,
                 data_dir="../data",
                 pairs_txt_dir="lfw_view2",
                 filename_pairs="pairs.txt"):
        self.data_dir = data_dir
        self.pairs_txt_dir = pairs_txt_dir

        self.filename_pairs = filename_pairs
        self.pairs_txt_file = os.path.join(data_dir, pairs_txt_dir, filename_pairs)

    def load_pairs_textfile(self):
        """
        Import pairs.txt file

        Returns: numpy representation of comparison pairs in text format

        """
        pairs = []
        with open(self.pairs_txt_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        assert (len(pairs) == 6000)
        return np.array(pairs, dtype=object)

    def get_lfwview2_filenames(self,
                               image_dir="lfw",
                               embedding_dir="features_cleaner",
                               ):
        """

        Args:
            image_dir ():
            embedding_dir ():

        Returns:

        """
        # Initialize filepaths
        image_filepath = os.path.join(self.data_dir, image_dir)
        embedding_filepath = os.path.join(self.data_dir, embedding_dir)

        bona_fide_comparison = bool
        target_2_image, target_2_embedding = str, str
        pairs = self.load_pairs_textfile()
        comparison_images_and_embeddings = []

        for pair in pairs:
            # The first target in pairs.txt is always in the same format
            target_1_image = os.path.join(image_filepath, pair[0],
                                          get_filename(pair[0], pair[1], "png"))
            target_1_embedding = os.path.join(embedding_filepath,
                                              get_filename(pair[0], pair[1], "npy"))

            # On genuine pairs, the file name of target_1 and target_2 is identical
            if len(pair) == 3:
                target_2_image = os.path.join(image_filepath, pair[0],
                                              get_filename(pair[0], pair[2], "png"))
                target_2_embedding = os.path.join(embedding_filepath,
                                                  get_filename(pair[0], pair[2], "npy"))
                bona_fide_comparison = True

            # On impostor pairs, the file name of target_1 and target_2 is different
            if len(pair) == 4:
                target_2_image = os.path.join(image_filepath, pair[2],
                                              get_filename(pair[2], pair[3], "png"))
                target_2_embedding = os.path.join(embedding_filepath,
                                                  get_filename(pair[2], pair[3], "npy"))
                bona_fide_comparison = False

            # Assert files exist
            assert (os.path.isfile(target_1_image))
            assert (os.path.isfile(target_1_embedding))
            assert (os.path.isfile(target_2_image))
            assert (os.path.isfile(target_2_embedding))

            comparison_pair = [target_1_image, target_1_embedding,
                               target_2_image, target_2_embedding,
                               bona_fide_comparison]
            comparison_images_and_embeddings.append(comparison_pair)

        # Assert the View2 specification has the correct amount of image pairs
        assert (len(comparison_images_and_embeddings) == 6000)

        return comparison_images_and_embeddings
