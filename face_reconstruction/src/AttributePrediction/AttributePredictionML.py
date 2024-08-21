import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def one_sample_per_subject(dataframe, evaluation_criteria: str):
    """
    Return a dataframe of LFW image set with only one sample per subject.
    If a subject has more than one sample, it is compared to the other choices by the provided evaluation criterion.

    Args:
        dataframe ():
        evaluation_criteria (): Dataframe column name. Valid choices, e.g.:
                                - img_original_quality
                                - cos_sim_bonafide_synthesized_pemiu{blocksize}

    Returns: pd.DataFrame

    """
    # Dataset should only include one sample per subject
    # Create a dict with name of subject as key and an empty list as value
    subjects = dataframe['filename'].apply(lambda x: x.split('_')[:-1]).tolist()
    subjects = ['_'.join(x) for x in subjects]
    indices = [[] for _ in range(len(subjects))]
    subjects_dict = dict(zip(subjects, indices))
    # Store row indices for each subject as values
    for index, row in dataframe.iterrows():
        if '_'.join(row['filename'].split('_')[:-1]) in subjects_dict:
            subjects_dict['_'.join(row['filename'].split('_')[:-1])].append(index)
    # Iterate over dict, remove all values except one for each key
    for key, value in subjects_dict.items():
        if len(value) > 1:
            # We keep the sample with the greatest value considering the evaluation criteria
            # Get all rows for indices of this subject as series.
            data = dataframe.iloc[subjects_dict[key]]
            # Get the index of the sample with the greatest value considering the criteria
            # Reduce the values to the chosen sample.
            subjects_dict[key] = [data[evaluation_criteria].idxmax()]
    # Iterate over dataframe.
    # Remove all rows which are not in the dictionary containing only unique subjects and samples
    for index, row in dataframe.iterrows():
        if [index] not in subjects_dict.values():
            dataframe.drop(index, inplace=True)

    return dataframe


class AttributePredictionML:
    def __init__(self,
                 train_on_one_sample_per_subject=False,
                 balance_genders=False,
                 alternate_genders=False,
                 evaluate_on_samples_that_pass_threshold=False,
                 threshold=0.34):
        """
        ML methods to predict the gender of a subject using embeddings generated from sample images.

        Args:
            train_on_one_sample_per_subject (bool): By default, the LFW dataset contains multiple samples for certain
                                    subjects. When set to True, the dataset is reduced to only consist of one sample
                                    per subject.
            balance_genders (bool): The LFW dataset is skewed to contain around 77% male samples.
                                    When set to True, the amount of male samples is adjusted to match the amount of
                                    female samples. The excess of male samples is discarded in the order of where
                                    they occur in the dataset.
            alternate_genders (bool): To alternately sort male and female samples
            evaluate_on_samples_that_pass_threshold:
            threshold:
        """
        self.evaluate_on_samples_that_pass_threshold = evaluate_on_samples_that_pass_threshold
        self.threshold = threshold

        # Prepare dataframe 'genders' with LFW image filenames and original genders
        block_sizes = ['16', '32', '64', '128']
        df = pd.read_csv("../evaluation/lfw_complete_genders_with_quality.csv", sep=",")
        self.genders = pd.DataFrame()

        # Add filename and remove file extension
        self.genders['filename'] = df['lfw_filename_raw'].apply(lambda x: x.split('.')[0])

        # Add image quality measurement
        self.genders['img_original_quality'] = df['img_original_quality']

        # Add filepaths of templates from sample images created from PEMIU protected templates at different block sizes
        for blocksize in block_sizes:
            self.genders[f'pemiu{blocksize}_fromimg'] = self.genders['filename'].apply(
                lambda
                    x: f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_pemiu{blocksize}_embeddings/{x}_pemiu{blocksize}.npy")

        # Add filepaths of templates from images created from unprotected templates
        self.genders['pemiu0_fromimg'] = self.genders['filename'].apply(
            lambda
                x: f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_unprotected_embeddings/{x}_reconstructed.npy")

        # Add filepaths of pemiu enhanced templates
        for blocksize in block_sizes:
            self.genders[f'pemiu{blocksize}'] = self.genders['filename'].apply(
                lambda x: f"../data/features_cleaner_pemiu{blocksize}/{x}.npy")

        # Add filepaths of original templates
        self.genders['pemiu0'] = self.genders['filename'].apply(
            lambda x: f"../data/features_cleaner/{x}.npy")

        # Add official gender labels
        self.genders['gender_official'] = df['gender_official']

        # Get cosine similarity data
        cosine_similarity_pemiu = "../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples_pemiu.csv"
        cosine_similarity_unprotected = "../recreate_icip2022_face_reconstruction/experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples_unprotected.csv"
        df_cos_sim_pemiu = pd.read_csv(cosine_similarity_pemiu)
        df_cos_sim_unprotected = pd.read_csv(cosine_similarity_unprotected)
        df_cos_sim = pd.DataFrame()

        # Add cos_sim calculations with varying block sizes to dataframe
        for index, blocksize in enumerate(block_sizes):
            df_cos_sim[f'cos_sim_bonafide_synthesized_pemiu{blocksize}'] = df_cos_sim_pemiu[f'cos_sim_pemiu{blocksize}_vs_real']

        # Add cos_sim of unprotected synthesized images
        df_cos_sim[f'cos_sim_bonafide_synthesized_unprotected'] = df_cos_sim_unprotected['cos_sim_reconstructed_vs_real']

        # Four samples have missing genders. Drop them.
        self.genders = self.genders.drop(
            self.genders[self.genders['gender_official'] == "X"].index)

        # Reset index after dropping to fix a bug when iterating over the dataset further down in the program
        self.genders.reset_index(drop=True, inplace=True)

        # Copy the dataframe for evaluation - it now contains the whole dataset
        self.evaluation_dataset = self.genders.copy(deep=True)

        # We only select one sample per identity. It's the one with the highest image quality.
        if train_on_one_sample_per_subject:
            self.genders = one_sample_per_subject(self.genders, "img_original_quality")

        if balance_genders:
            # LFW dataset is skewed with 77% male samples
            # Balance dataset to include a 50/50 amount of male and female samples
            num_female = len(self.genders[self.genders["gender_official"] == "F"])
            # Get indices of rows with male gender
            male_indices = self.genders[self.genders['gender_official'] == "M"].index
            # Reduce amount of male indices so that the total matches the number of female samples
            male_indices = male_indices[num_female:]
            # Drop superfluous indices
            self.genders = self.genders.drop(male_indices)

        if alternate_genders:
            # To balance the training and testing, we reorder the dataframe so that male and female samples alternate.
            # We first gather the indices of male and female samples. Then, we create a dictionary which has the
            # index of the sample, and create a new order key, that defines the new position of the sample.
            # The new position will then alternate between male and female.
            male_indices = self.genders[self.genders['gender_official'] == "M"].index
            female_indices = self.genders[self.genders['gender_official'] == "F"].index
            sortMap = {}
            counter_m, counter_f = 0, 0
            for i in range(len(self.genders)):
                if i % 2 == 0:
                    sortMap[male_indices[counter_m]] = i
                    counter_m = counter_m + 1
                else:
                    sortMap[female_indices[counter_f]] = i
                    counter_f = counter_f + 1
            # We create a new 'order' column and sort the dataframe accordingly to achieve male and female sample
            # alternation
            self.genders['order'] = self.genders.index.map(sortMap)
            self.genders.sort_values('order', inplace=True)

        # self.genders Dataframe columns:
        # filename
        # pemiu16_fromimg
        # pemiu32_fromimg
        # pemiu64_fromimg
        # pemiu128_fromimg
        # pemiu0_fromimg    <- templates from reconstructed images, without privacy enhancement
        # pemiu16
        # pemiu32
        # pemiu64
        # pemiu128
        # pemiu0            <- original templates, without privacy enhancement
        # gender_official

    def _get_fromimg_string(self, embedding_from_image: bool):
        if embedding_from_image:
            return "_fromimg"
        else:
            return ""

    def _get_train_test_split(self, blocksize, fromimg="", scaler=False):
        # Load numpy arrays
        X = [np.asarray(np.load(x)) for x in self.genders[f'pemiu{blocksize}{fromimg}']]

        if scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Transform gender labels to int: 0 for M, 1 for F
        y = [0 if x == 'M' else 1 for x in self.genders['gender_official'].tolist()]

        # Split the dataset
        return train_test_split(X, y, test_size=0.2, random_state=0)

    def _instantiate_model(self, _model: str):
        # Instantiate model
        if _model == "knn":
            return KNeighborsClassifier(n_neighbors=3)
        elif _model == "svm_poly":
            return svm.SVC(random_state=42, kernel="poly")
        elif _model == "svm_rbf":
            return svm.SVC(random_state=42, kernel="rbf")
        elif _model == "svm_sigmoid":
            return svm.SVC(random_state=42, kernel="sigmoid")
        else:
            raise ValueError('Model name not implemented')

    def attribute_prediction_knn(self, blocksize, k=3, fromimg=""):
        """

        Args:
            blocksize (): '0', '16', '32', '64', '128'.
                           Blocksize of 0 means: No PEMIU enhancement (unprotected template)
            k (): k-nearest neighbor, default is 3
            fromimg (): either empty string (then it calculates from embedding)
                        or "_fromimg" to calculate from embedding created from reconstructed image

        Returns:

        """
        X_train, X_test, y_train, y_test = self._get_train_test_split(blocksize, fromimg, scaler=True)

        # Instantiate the model
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the model to the training set
        knn.fit(X_train, y_train)

        # Output blocksize
        print("-----------------")
        print(f"Block size: {blocksize}, k = {k}, from reconstructed image = {'false' if fromimg == '' else 'true'}")

        # Predict results on test split
        y_pred = knn.predict(X_test)
        print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

        return '{0:0.4f}'.format(accuracy_score(y_test, y_pred))

        # Predict results on train split to analyze overfitting
        # y_pred_train = knn.predict(X_train)
        # print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

        # print the scores on training and test set
        # print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
        # print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))

    def attribute_prediction_svm(self, blocksize, kernel="sigmoid", fromimg=""):
        """

        Args:
            blocksize ():
            kernel (): poly, rbf, sigmoid
            fromimg ():

        Returns:

        """
        # Split the dataset
        X_train, X_test, y_train, y_test = self._get_train_test_split(blocksize, fromimg, scaler=False)

        # Instantiate the model
        model = svm.SVC(random_state=42, kernel=kernel)

        model.fit(X_train, y_train)

        # Output blocksize
        print("-----------------")
        print(
            f"Block size: {blocksize}, kernel = {kernel}, from reconstructed image = {'false' if fromimg == '' else 'true'}")

        # Predict results on test split
        y_pred = model.predict(X_test)
        print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

        return '{0:0.4f}'.format(accuracy_score(y_test, y_pred))

    def attribute_prediction_cross_validation(self,
                                              blocksize,
                                              _model: str,
                                              embedding_from_image: bool,
                                              k_fold_splits=10):
        # Load numpy arrays
        X = [np.asarray(np.load(x)) for x in
             self.genders[f'pemiu{blocksize}{self._get_fromimg_string(embedding_from_image)}']]

        # Transform gender labels to int: 0 for M, 1 for F
        y = [0 if x == 'M' else 1 for x in self.genders['gender_official'].tolist()]

        # Define k-fold splits
        kfold = KFold(n_splits=k_fold_splits)

        # Instantiate model
        if _model == "knn":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            model = KNeighborsClassifier(n_neighbors=3)
        elif _model == "svm_poly":
            model = svm.SVC(random_state=42, kernel="poly")
        elif _model == "svm_rbf":
            model = svm.SVC(random_state=42, kernel="rbf")
        elif _model == "svm_sigmoid":
            model = svm.SVC(random_state=42, kernel="sigmoid")
        else:
            raise ValueError('Model name not implemented')

        results = cross_val_score(model,
                                  X, y,
                                  cv=kfold)
        # Output blocksize
        print("-----------------")
        print(f"Block size: {blocksize}, model: {_model}, from reconstructed image = {embedding_from_image}")
        print(results)
        print(f"Mean: {results.mean()}")
        print(f"Standard deviation: {results.std()}")

        return round(results.mean(), 4), round(results.std(), 4)

    def attack_training_on_unprotected_prediction_on_pemiu(self,
                                                           _model: str,
                                                           embedding_from_image: bool):
        # Load data for training
        # Features
        X = [np.asarray(np.load(x)) for x in self.genders[f'pemiu0{self._get_fromimg_string(embedding_from_image)}']]

        # Labels
        # Transform gender labels to int: 0 for M, 1 for F
        y = [0 if x == 'M' else 1 for x in self.genders['gender_official'].tolist()]

        # Instantiate model
        model = self._instantiate_model(_model)

        # We perform the scaler if the model is knn, to be consistent with the other implementations
        if _model == "knn":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Fit model
        model.fit(X, y)

        # Prediction
        block_sizes = ['16', '32', '64', '128']
        results = {}
        for blocksize in block_sizes:
            df_reduced = pd.DataFrame()

            # Load data for prediction
            if self.evaluate_on_samples_that_pass_threshold:
                # Only include samples where the cos_sim of the reconstructed image passes the system threshold.
                # First, get the best sample per identity in terms of cos_sim
                df_reduced = one_sample_per_subject(
                    self.evaluation_dataset, f"cos_sim_bonafide_synthesized_pemiu{blocksize}")
                # Reduce dataset to only include samples that pass the threshold
                df_reduced = df_reduced[
                    df_reduced[f'cos_sim_bonafide_synthesized_pemiu{blocksize}'] > self.threshold]
                # Load features
                X_pemiu = [np.asarray(np.load(x)) for x in
                           df_reduced[f'pemiu{blocksize}{self._get_fromimg_string(embedding_from_image)}']]
                # Set correct labels for the newly reduced dataframe
                y = [0 if x == 'M' else 1 for x in df_reduced['gender_official'].tolist()]
            else:
                X_pemiu = [np.asarray(np.load(x)) for x in
                           self.genders[f'pemiu{blocksize}{self._get_fromimg_string(embedding_from_image)}']]

            # Predict results on test split
            y_pred = model.predict(X_pemiu)

            # Print results
            print(f"Block size: {blocksize}, "
                  f"model: {_model}, "
                  f"from reconstructed image: {embedding_from_image}")
            print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y, y_pred)))

            # Store results: Prediction score
            results[blocksize] = accuracy_score(y, y_pred)

            # Append predicted labels to dataframe
            if self.evaluate_on_samples_that_pass_threshold:
                # When we only predict samples that match a certain threshold, we only add the predicted
                # ones to the self.genders dataframe
                df_reduced = df_reduced[['filename']]
                df_reduced[f"pred_"
                             f"{'fromimg' if embedding_from_image else 'fromembedding'}_"
                             f"{blocksize}_"
                             f"{_model}"] = ['M' if x == 0 else 'F' for x in y_pred]
                self.evaluation_dataset = self.evaluation_dataset.merge(df_reduced, on='filename', how='outer')
            else:
                self.genders[f"pred_"
                             f"{'fromimg' if embedding_from_image else 'fromembedding'}_"
                             f"{blocksize}_"
                             f"{_model}"] = ['M' if x == 0 else 'F' for x in y_pred]  # Change labels to M and F

        if self.evaluate_on_samples_that_pass_threshold:
            return results, self.evaluation_dataset
        else:
            return results, self.genders
