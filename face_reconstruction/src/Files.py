import os
import numpy as np
import pandas as pd
from .GenerateLFWView2 import GenerateLFWView2
from .Dataset import get_all_filenames_from_dir
from sklearn.preprocessing import MinMaxScaler


class Files:
    def __init__(self,
                 data_dir="../data",
                 lfw_image_dir="lfw_align",
                 lfw_embedding_dir="features_cleaner",
                 lfw_embedding_pemiu_dir="features_cleaner_pemiu16",
                 ffhq_image_dir="ffhq_align/images",
                 ffhq_embedding_dir="ffhq_align/embeddings",
                 sample_image_dir=""):
        self.data_dir = data_dir
        self.lfw_image_dir = lfw_image_dir
        self.lfw_embedding_dir = lfw_embedding_dir
        self.lfw_embedding_pemiu_dir = lfw_embedding_pemiu_dir
        self.ffhq_image_dir = ffhq_image_dir
        self.ffhq_embedding_dir = ffhq_embedding_dir

    def get_filenames_lfw_images(self):
        files = get_all_filenames_from_dir(self.data_dir, self.lfw_image_dir)
        assert [os.path.isfile(file) for file in files]
        return files

    def get_filenames_lfw_embeddings(self, with_path=True):
        """
        Get filenames of all lfw embeddings (13,233 items). By default, the method returns filenames including a path.
        The method asserts that the files are present before returning.

        Args:
            with_path (): bool. If True, the returned list includes the relative filepath.
                                If False, the returned list includes only the filename, with file extension

        Returns: List of filenames

        """
        files = get_all_filenames_from_dir(self.data_dir, self.lfw_embedding_dir)
        assert [os.path.isfile(file) for file in files]
        if with_path:
            return files
        else:
            return [file.split('/')[-1] for file in files]

    def get_filenames_ffhq_embeddings(self, with_path=True):
        """
        Get filenames of all ffhq embeddings (52,001 items). By default, the method returns filenames including a path.
        The method asserts that the files are present before returning.

        Args:
            with_path (): bool. If True, the returned list includes the relative filepath.
                                If False, the returned list includes only the filename, with file extension

        Returns: List of filenames

        """
        files = get_all_filenames_from_dir(self.data_dir, self.ffhq_embedding_dir)
        assert [os.path.isfile(file) for file in files]
        if with_path:
            return files
        else:
            return [file.split('/')[-1] for file in files]

    def get_lfw_view2_as_dataframe(self):
        view2_gen = GenerateLFWView2()
        df = pd.DataFrame(
            view2_gen.get_lfwview2_filenames(
                image_dir=self.lfw_image_dir,
                embedding_dir=self.lfw_embedding_dir
            ))
        df.columns = ['a_img', 'a_embedding', 'b_img', 'b_embedding', 'genuine']
        df['genuine'] = df['genuine'].astype(bool)

        # Add paths of pemiu enhanced images and embeddings by modifying the filenames
        df['a_img_reconstructed'] = df['a_img'].apply(
            lambda x: f"sample_images/{x.split('/')[-1].split('.')[0]}_reconstructed.png")
        df['a_embedding_reconstructed'] = df['a_img'].apply(
            lambda x: f"sample_images_embeddings/{x.split('/')[-1].rsplit('.')[0]}_reconstructed.npy")

        df['b_img_reconstructed'] = df['b_img'].apply(
            lambda x: f"sample_images/{x.split('/')[-1].split('.')[0]}_reconstructed.png")
        df['b_embedding_reconstructed'] = df['b_img'].apply(
            lambda x: f"sample_images_embeddings/{x.split('/')[-1].rsplit('.')[0]}_reconstructed.npy")

        df['a_img_pemiu'] = df['a_img'].apply(
            lambda x: f"sample_images_lfw_pemiu/{x.split('/')[-1].split('.')[0]}_pemiu.png")
        df['a_embedding_pemiu'] = df['a_embedding'].apply(
            lambda x: f"../data/lfw_embeddings_pemiu/{x.split('/')[-1]}")

        df['b_img_pemiu'] = df['b_img'].apply(
            lambda x: f"sample_images_lfw_pemiu/{x.split('/')[-1].split('.')[0]}_pemiu.png")
        df['b_embedding_pemiu'] = df['b_embedding'].apply(
            lambda x: f"../data/lfw_embeddings_pemiu/{x.split('/')[-1]}")

        # todo: this assert doesn't work
        assert [os.path.isfile(x) for x in df['a_img_reconstructed']]
        assert [os.path.isfile(x) for x in df['a_embedding_reconstructed']]
        assert [os.path.isfile(x) for x in df['b_img_reconstructed']]
        assert [os.path.isfile(x) for x in df['b_embedding_reconstructed']]
        assert [os.path.isfile(x) for x in df['a_img_pemiu']]
        assert [os.path.isfile(x) for x in df['a_embedding_pemiu']]
        assert [os.path.isfile(x) for x in df['b_img_pemiu']]
        assert [os.path.isfile(x) for x in df['b_embedding_pemiu']]

        return df

    def get_lfw_view2_genuine_impostor_as_separate_dataframes(self):
        df = self.get_lfw_view2_as_dataframe()
        df_genuine = df.loc[df['genuine'] == True]
        df_impostor = df.loc[df['genuine'] == False]

        # Add pemiu enhanced reconstructed embeddings
        df_genuine['a_embedding_pemiu_reconstructed'] = get_all_filenames_from_dir(
            self.data_dir, "lfw_embeddings_pemiu_genuine_reconstructed")
        df_impostor['a_embedding_pemiu_reconstructed'] = get_all_filenames_from_dir(
            self.data_dir, "lfw_embeddings_pemiu_impostor_reconstructed")

        # Assert that embeddings are present as files
        assert [os.path.isfile(x) for x in df_genuine['a_embedding_pemiu_reconstructed']]
        assert [os.path.isfile(x) for x in df_impostor['a_embedding_pemiu_reconstructed']]

        # Assert that filename of target_a embedding is the same as the pemiu reconstructed embedding
        assert [x.split('/')[-1] == y.split('/')[-1] for x, y in
                zip(df_genuine['a_embedding'], df_genuine['a_embedding_pemiu_reconstructed'])]
        assert [x.split('/')[-1] == y.split('/')[-1] for x, y in
                zip(df_impostor['a_embedding'], df_impostor['a_embedding_pemiu_reconstructed'])]

        return df_genuine, df_impostor

    def get_np_arrays_lfw_embeddings(self):
        files = self.get_filenames_lfw_embeddings()
        files_loaded = [np.load(file) for file in files]
        return files_loaded

    def get_np_arrays_ffhq_embeddings(self):
        files = self.get_filenames_ffhq_embeddings()
        files_loaded = [np.load(file) for file in files]
        return files_loaded

    def get_all_lfw_filenames_as_dataframe(self):
        """

        Gather all 13,233 lfw filenames in different processing steps:
         - unmodified
         - privacy enhanced using PEMIU

        Assert that files are present.

        Returns: Pandas dataframe with the columns:
                'lfw_image',             'lfw_embedding',
                'lfw_image_pemiu',       'lfw_embedding_pemiu'
                'lfw_filename_raw'

        """
        files = list(zip(
            get_all_filenames_from_dir(self.data_dir, self.lfw_image_dir),  # lfw_images
            get_all_filenames_from_dir(self.data_dir, self.lfw_embedding_dir),  # lfw_embeddings
            get_all_filenames_from_dir(".", "sample_images_lfw_pemiu"),  # lfw_images_pemiu
            get_all_filenames_from_dir(self.data_dir, self.lfw_embedding_pemiu_dir)  # lfw_embeddings_pemiu
        ))

        assert [[os.path.isfile(file) for file in file_lists] for file_lists in files[0]]

        files = pd.DataFrame(files, columns=[
            'lfw_image',
            'lfw_embedding',
            'lfw_image_pemiu',
            'lfw_embedding_pemiu'
        ])
        files['lfw_filename_raw'] = files['lfw_image'].apply(lambda x: x.split('/')[-1])

        return files

    def _get_original_image(self, filename):
        return f"../data/lfw_align/{'_'.join(filename.split('_')[:-1])}/{filename}.png"

    def _get_reconstructed_image(self, filename):
        path = f"../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/sample_images_unprotected"
        return f"{path}/{filename}_reconstructed.png"

    def _get_pemiu_image(self, filename, blocksize):
        path = f"../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/sample_images_pemiu{blocksize}"
        return f"{path}/{filename}_pemiu{blocksize}.png"

    def get_dataframe_with_all_lfw_images_incl_pemiu(self,
                                                     embeddings=False,
                                                     gender_labels=False,
                                                     drop_missing_gender_labels=False,
                                                     cos_sim=False
                                                     ):

        # Prepare dataframe 'genders' with LFW image filenames and original genders
        block_sizes = ['16', '32', '64', '128']
        df = pd.read_csv("../evaluation/lfw_complete_genders_with_quality.csv", sep=",")
        dataframe = pd.DataFrame()

        # Add filename and remove file extension
        dataframe['filename'] = df['lfw_filename_raw'].apply(lambda x: x.split('.')[0])

        # Add original image path
        dataframe['img_original'] = dataframe['filename'].apply(lambda x: self._get_original_image(x))

        # Add reconstructed image path
        dataframe['img_reconstructed'] = dataframe['filename'].apply(lambda x: self._get_reconstructed_image(x))

        # Add PEMIU image path
        for blocksize in block_sizes:
            dataframe[f'img_pemiu_{blocksize}'] = dataframe['filename'].apply(
                lambda x: self._get_pemiu_image(x, blocksize))

        # Add image quality measurement
        # dataframe['quality_img_original'] = df['img_original_quality']

        if embeddings:
            # Add filepaths of templates from sample images created from PEMIU protected templates at different block sizes
            for blocksize in block_sizes:
                dataframe[f'pemiu{blocksize}_fromimg'] = dataframe['filename'].apply(
                    lambda
                        x: f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_pemiu{blocksize}_embeddings/{x}_pemiu{blocksize}.npy")

            # Add filepaths of templates from images created from unprotected templates
            dataframe['pemiu0_fromimg'] = dataframe['filename'].apply(
                lambda
                    x: f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_unprotected_embeddings/{x}_reconstructed.npy")

            # Add filepaths of pemiu enhanced templates
            for blocksize in block_sizes:
                dataframe[f'pemiu{blocksize}'] = dataframe['filename'].apply(
                    lambda x: f"../data/features_cleaner_pemiu{blocksize}/{x}.npy")

            # Add filepaths of original templates
            dataframe['pemiu0'] = dataframe['filename'].apply(
                lambda x: f"../data/features_cleaner/{x}.npy")

        if gender_labels:
            # Add official gender labels
            dataframe['gender_official'] = df['gender_official']

        if cos_sim:
            # Get cosine similarity data
            cosine_similarity_pemiu = "../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples_pemiu.csv"
            cosine_similarity_unprotected = "../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples_unprotected.csv"
            df_cos_sim_pemiu = pd.read_csv(cosine_similarity_pemiu)
            df_cos_sim_unprotected = pd.read_csv(cosine_similarity_unprotected)
            df_cos_sim = pd.DataFrame()

            # Add cos_sim calculations with varying block sizes to dataframe
            for index, blocksize in enumerate(block_sizes):
                df_cos_sim[f'cos_sim_bonafide_synthesized_pemiu{blocksize}'] = df_cos_sim_pemiu[
                    f'cos_sim_pemiu{blocksize}_vs_real']

            # Add cos_sim of unprotected synthesized images
            df_cos_sim[f'cos_sim_bonafide_synthesized_unprotected'] = df_cos_sim_unprotected[
                'cos_sim_reconstructed_vs_real']

            # Normalize cos_sim data
            # Set normalization range to include values from genuine and impostor comparisons
            min_max_values = [[0.2430551121288237],
                              [-0.0050609096468108294],
                              [0.9729986512423155],
                              [0.9729986512423154],
                              [0.1912166066597454],
                              [-0.23413056292959689],
                              [0.3636036840601125],
                              [0.30872918671248395]]

            # Simplify gathering of columns by using a mask
            mask = df_cos_sim.columns.str.contains('cos_sim_*')

            # Step 1: Set limits of normalization scale to all columns within the mask by finding min, max values
            min_max_values_new = [array.agg(['min', 'max']) for array in [df_cos_sim.loc[:, mask]]]
            min_max_values_new = np.array(min_max_values_new).reshape(-1, 1).tolist()
            min_max_values = min_max_values + min_max_values_new

            # Step 2: Set normalization scale
            scaler = MinMaxScaler()
            scaler.fit(min_max_values)

            # Step 3: Apply normalization to all columns within the mask
            # Append extension "_normalized" to the new column name
            for column in df_cos_sim.loc[:, mask]:
                df_cos_sim[f'{column}_normalized'] = scaler.transform(df_cos_sim[column].values.reshape(-1, 1))

            # Get column names ending in '_normalized', add them to self.genders
            mask = df_cos_sim.columns.str.endswith('_normalized')
            for column in df_cos_sim.loc[:, mask]:
                dataframe[column] = df_cos_sim[column]

        if drop_missing_gender_labels:
            # Four samples have missing genders. Drop them.
            dataframe = dataframe.drop(
                dataframe[dataframe['gender_official'] == "X"].index)

            # Reset index after dropping to fix a bug when iterating over the dataset further down in the program
            dataframe.reset_index(drop=True, inplace=True)

        return dataframe
