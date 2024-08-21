import os
from GenerateDataset import GenerateDataset
from src.GenerateDatasetMobio import GenerateDatasetMobio
from recreate_icip2022_face_reconstruction.src.GenerateLFWView2 import GenerateLFWView2
from src.Dataset import get_all_filenames_from_dir
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import csv
from src.pemiu.privacy_enhancing_miu import PrivacyEnhancingMIU
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
from GenerateSampleImages import GenerateSampleImagesFromEmbeddings
from recreate_icip2022_face_reconstruction.src.AttributePrediction.AttributePredictionInsightFace import \
    AttributePredictionInsightFace
from src.Files import Files
from src.CosineSimilarity import CosineSimilarity
from src.AttributePrediction.AttributePredictionML import AttributePredictionML
from src.GenerateEmbeddings import GenerateEmbeddings
from FaceImageQuality.face_image_quality import SER_FIQ
import cv2


def printout(function_name):
    print(f"\n#### {function_name} ####\n")


def display_menu(menu):
    """
    Display a menu where the key identifies the name of a function.
    :param menu: dictionary, key identifies a value which is a function name
    :return:
    """
    for k, function in menu.items():
        print(k, function.__name__)


def train():
    pass


def evaluation():
    pass

#####For settings in Testing#####
def generate_sample_images():
    # samples_1 = GenerateSampleImagesDefault(
    #     image_dir="lfw_align",
    #     embedding_dir="features_cleaner",
    #     file_appendix="reconstructed")
    # samples_1.generate()

    # samples_2 = GenerateSampleImagesDefault(
    #     image_dir="lfw_align",
    #     embedding_dir="lfw_embeddings_pemiu",
    #     file_appendix="pemiu")
    # samples_2.generate()

    # samples_3 = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir="lfw_embeddings_pemiu_genuine_reconstructed",
    #     file_appendix="pemiu_reconstructed",
    #     save_path="sample_images_pemiu_genuine_reconstructed",
    #     write_original_img=False,
    #     create_subdirs=True
    # )
    # samples_3.generate()

    # samples_4 = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir="lfw_embeddings_pemiu_impostor_reconstructed",
    #     file_appendix="pemiu_reconstructed",
    #     save_path="sample_images_pemiu_impostor_reconstructed",
    #     write_original_img=False,
    #     create_subdirs=True
    # )
    # samples_4.generate()

    # samples_5 = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir="features_cleaner_pemiu",
    #     file_appendix="pemiu",
    #     save_path="sample_images_lfw_pemiu",
    #     write_original_img=False,
    #     create_subdirs=False
    # )
    # samples_5.generate()

    # Experiment 6.1
    # Model is trained on FFHQ image set with protected embeddings (pemiu block size 16)
    # Sample sources are the pemiu protected embeddings from LFW view 2 dataset
    # samples_6_pemiu16 = GenerateSampleImagesDefault(
    #     dataset_dir="../data",
    #     image_dir="lfw_align",
    #     embedding_dir="features_cleaner_pemiu",
    #     file_appendix="_pemiu_reconstructed",
    #     save_path="experiment_pemiu16/sample_images",
    #     generator_checkpoint_dir="training_files_pemiu_16",
    #     write_original_img=True,
    #     create_subdirs=False
    # )
    # samples_6_pemiu16.generate()

    # Experiment 6.2 - cont.
    # Generate samples from genuine pemiu enhanced reconstructed templates
    # samples_6_2_pemiu16 = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir="lfw_embeddings_pemiu_genuine_reconstructed",
    #     file_appendix="pemiu_reconstructed",
    #     save_path="experiment_pemiu16/sample_images_pemiu_genuine_reconstructed",
    #     generator_checkpoint_dir="training_files_pemiu_16",
    #     write_original_img=False,
    #     create_subdirs=True
    # )
    # samples_6_2_pemiu16.generate()

    # Experiment 7
    # Generate samples from protected embeddings using pemiu and block sizes 16, 32, 64 and 128
    # Used Model: Model 1, trained on FFHQ and unprotected embeddings
    # for block_size in ["16", "32", "64", "128"]:
    #     samples_7_pemiu_block_sizes = GenerateSampleImagesFromEmbeddings(
    #         dataset_dir="../data",
    #         embedding_dir=f"features_cleaner_pemiu{block_size}",
    #         file_appendix=f"pemiu{block_size}",
    #         save_path=f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_pemiu{block_size}",
    #         save_path_log="experiments/experiment_7_model1_pemiu_block_sizes",
    #         generator_checkpoint_dir="training_files",
    #         write_original_img=False,
    #         create_subdirs=False
    #     )
    #     samples_7_pemiu_block_sizes.generate()

    # Mated Reconstructed
    # unprotected_reconstructed_sample = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir=f"features_cleaner",
    #     file_appendix=f"reconstructed",
    #     save_path=f"experiments/experiment_7_model1_pemiu_block_sizes/sample_images_unprotected",
    #     save_path_log="experiments/experiment_7_model1_pemiu_block_sizes",
    #     generator_checkpoint_dir="training_files",
    #     write_original_img=False,
    #     create_subdirs=False
    # )
    # unprotected_reconstructed_sample.generate()

    # Experiment 8
    # Generate samples from new model 3 trained with modified shuffle method to adapt to PEMIU block size 16
    # samples_8_model3_pemiu16 = GenerateSampleImagesFromEmbeddings(
    #     dataset_dir="../data",
    #     embedding_dir=f"features_cleaner_pemiu16",
    #     file_appendix=f"model3_pemiu16",
    #     save_path=f"experiments/experiment_8_model3_pemiu_adapted/sample_images_pemiu16",
    #     save_path_log="experiments/experiment_8_model3_pemiu_adapted",
    #     generator_checkpoint_dir="model3_training_files_pemiu_16",
    #     epoch=84,
    #     write_original_img=False,
    #     create_subdirs=False
    # )
    # samples_8_model3_pemiu16.generate()

    # Experiment 9
    # Generate multiple pemiu 16 from same subject/sample
    experiment_path = "experiment_9_pemiu_randomization"
    samples_9_model1_pemiu16 = GenerateSampleImagesFromEmbeddings(
        dataset_dir=f"./experiments/{experiment_path}",
        embedding_dir=f"sample_embeddings_pemiu16",
        image_dir=f"experiments/{experiment_path}/sample_images",
        file_appendix=f"model1_pemiu16",
        save_path=f"experiments/{experiment_path}/sample_images_pemiu16",
        save_path_log=f"experiments/{experiment_path}",
        generator_checkpoint_dir="training_files",
        epoch=90,
        write_original_img=False,
        create_subdirs=False,
        iterations=32,
        batch_size=32
    )
    samples_9_model1_pemiu16.generate()


def generate_dataset():
    data_gen = GenerateDataset(device="cuda",
                               image_dir="ffhq",
                               save_dir="ffhq_align")
    data_gen.generate_dataset()


def generate_dataset_mobio():
    data_gen_mobio = GenerateDatasetMobio(device="cuda",
                                          image_dir="Mobio/files/image",
                                          save_dir="mobio_align")
    data_gen_mobio.generate_dataset()


def generate_embeddings_for_reconstructed_images():
    data_gen = GenerateDataset(device="cuda",
                               dataset_dir="",
                               image_dir="sample_images",
                               save_dir="sample_images_embeddings")
    data_gen.generate_embeddings_for_reconstructed_images()


def generate_lfw_view2():
    view2_gen = GenerateLFWView2(pairs_txt_dir="lfw_view2")
    pairs = view2_gen.get_lfwview2_filenames(
        image_dir="lfw_align",
        embedding_dir="features_cleaner"
    )
    np.savetxt("../data/lfw_view2/pairs_filenames.csv",
               pairs,
               delimiter=",",
               header="a_img,a_embedding,"
                      "b_img,b_embedding,"
                      "genuine",
               fmt='% s',
               comments='')

    # ,a_embedding_pemiu,a_embedding_pemiu_reconstructed,"
    # ,b_embedding_pemiu,b_embedding_pemiu_reconstructed,"


def calc_cosine_distance():
    """
    Load csv file containing filenames of embeddings using the LFW view2 genuine and impostor split.

    Returns: two csv files with cosine similarity score, divided by genuine and impostor comparisons
    """
    # Load pandas dataframe from csv containing filenames of images and embeddings from LFW dataset
    # Column names:
    # - a_img                               - b_img
    # - a_embedding                         - b_embedding
    # - a_embedding_pemiu                   - b_embedding_pemiu
    # - a_embedding_pemiu_reconstructed     - b_embedding_pemiu_reconstructed
    # - genuine
    df = pd.read_csv("../data/lfw_view2/pairs_filenames.csv",
                     sep=',')

    # Get data in lists for processing
    target_a = df['a_embedding'].tolist()
    target_a_pemiu = df['a_embedding_pemiu'].tolist()
    target_a_pemiu_reconstructed = df['a_embedding_pemiu_reconstructed'].tolist()
    target_b = df['b_embedding'].tolist()
    target_b_pemiu = df['b_embedding_pemiu'].tolist()
    target_b_pemiu_reconstructed = df['b_embedding_pemiu_reconstructed'].tolist()

    pemiu = PrivacyEnhancingMIU(block_size=16)

    # Calculate cosine distance
    # Real embeddings according to LFW view2 genuine/impostor comparison
    df['cos_dist_real'] = [pemiu.cos_sim(np.load(a.strip()), np.load(b.strip()))
                           for a, b in zip(target_a, target_b)]

    # target_a real vs. pemiu
    df['cos_dist_target_a_real_vs_target_a_pemiu'] = [
        pemiu.cos_sim(np.load(a.strip()), np.load(b.strip()))
        for a, b in zip(target_a, target_a_pemiu)]

    # target_a pemiu vs. target_b real
    df['cos_dist_target_a_pemiu_vs_target_b_real'] = [
        pemiu.cos_sim(np.load(a.strip()), np.load(b.strip()))
        for a, b in zip(target_a_pemiu, target_b)]

    # target_a vs. target_a_pemiu_reconstructed
    df['cos_dist_target_a_real_vs_target_a_pemiu_reconstructed'] = [
        pemiu.cos_sim(np.load(a.strip()), np.load(b.strip()))
        for a, b in zip(target_a, target_a_pemiu_reconstructed)]

    # Normalize cosine distance to range (0, 1)
    df['cos_dist_real_normalized'] = \
        preprocessing.minmax_scale(df['cos_dist_real'])
    df['cos_dist_target_a_real_vs_target_a_pemiu_normalized'] = \
        preprocessing.minmax_scale(df['cos_dist_target_a_real_vs_target_a_pemiu'])
    df['cos_dist_target_a_pemiu_vs_target_b_real_normalized'] = \
        preprocessing.minmax_scale(df['cos_dist_target_a_pemiu_vs_target_b_real'])
    df['cos_dist_target_a_real_vs_target_a_pemiu_reconstructed_normalized'] = \
        preprocessing.minmax_scale(df['cos_dist_target_a_real_vs_target_a_pemiu_reconstructed'])

    # Export
    os.makedirs('../evaluation/', exist_ok=True)
    path = "../evaluation/lfw_view2_pairs_cosdist_"
    with open(path + "genuine.csv", 'w') as csv_genuine, open(path + "impostor.csv", 'w') as csv_impostor:
        writer_genuine = csv.writer(csv_genuine, delimiter=",")
        writer_impostor = csv.writer(csv_impostor, delimiter=",")
        # Add header
        writer_genuine.writerow(df.columns.astype(str))
        writer_impostor.writerow(df.columns.astype(str))
        for index, row in df.iterrows():
            if row[8]:  # Genuine comparison
                writer_genuine.writerow(row)
            else:  # Impostor comparison
                writer_impostor.writerow(row)


def generate_pyeer_eer_report():
    save_path = "../evaluation/pyeer_ffhq_unprotected_lfwview2/"
    path = "../evaluation/"

    df_genuine = pd.read_csv(path + "lfwview2_genuine_cos_sim.csv",
                             sep=',')
    df_impostor = pd.read_csv(path + "lfwview2_impostor_cos_sim.csv",
                              sep=',')

    # ds_scores: True for Hamming, euclidian distance or False for cosine
    stats_unprotected = get_eer_stats(
        df_genuine['cos_sim_a_b_normalized'],
        df_impostor['cos_sim_a_b_normalized'],
        ds_scores=False)
    stats_inversion_attack_sisfe = get_eer_stats(
        df_genuine['cos_sim_a_a_reconstructed_normalized'],
        df_impostor['cos_sim_a_b_normalized'],
        ds_scores=False)
    stats_inversion_attack_disfe = get_eer_stats(
        df_genuine['cos_sim_a_b_reconstructed_normalized'],
        df_impostor['cos_sim_a_b_normalized'],
        ds_scores=False)
    stats_genuine_pemiu_reconstructed = get_eer_stats(
        df_genuine['cos_sim_a_pemiu_a_pemiu_reconstructed_normalized'],
        df_impostor['cos_sim_a_b_normalized'],
        ds_scores=False)

    generate_eer_report([stats_unprotected,
                         stats_inversion_attack_sisfe,
                         stats_inversion_attack_disfe,
                         stats_genuine_pemiu_reconstructed],
                        ['Unprotected',
                         'SISFE',
                         'DISFE',
                         'Genuine PEMIU Reconstructed'],
                        save_path + "eer_report.csv")

    export_error_rates(stats_unprotected.fmr, stats_unprotected.fnmr, save_path + 'A_DET.csv')
    plot_eer_stats([stats_unprotected], ['A'], save_path=save_path)


def pemiu_test():
    # Load embeddings, divided by genuine and impostor pairs
    with open("../data/lfw_view2/pairs_filenames.csv") as file:
        reader = csv.reader(file)
        data = list(reader)
        embeddings_genuine, embeddings_impostor = [], []
        for element in data[1:]:
            if element[4].strip() == "True":
                embeddings_genuine.append([element[1].strip(), element[3].strip()])
            else:
                embeddings_impostor.append([element[1].strip(), element[3].strip()])

    print(embeddings_genuine[0][0], embeddings_genuine[0][1])
    target_a = [np.load(embeddings_genuine[0][0])]
    target_b = [np.load(embeddings_genuine[0][1])]

    print(embeddings_impostor[0][1])
    target_b_impostor = [np.load(embeddings_impostor[0][1])]

    # PrivacyEnhancing with MIU
    pemiu = PrivacyEnhancingMIU(block_size=16)
    # embeddings_genuine_shuffled = pemiu.shuffle(embeddings_genuine)

    # Shuffle
    target_a_shuffled = pemiu.shuffle(target_a)
    target_b_shuffled = pemiu.shuffle(target_b)
    target_b_impostor_shuffled = pemiu.shuffle(target_b_impostor)

    # reconstruct a genuine to reference vector alt[0]
    rec_gen = pemiu.reconstruct(target_a_shuffled[0], target_b_shuffled[0])
    rec_imp = pemiu.reconstruct(target_a_shuffled[0], target_b_impostor_shuffled[0])

    # compare genuine
    print("Genuine Comparison - original", pemiu.cos_sim(target_a[0], target_b[0]))
    print("Genuine Comparison - not reconstructed",
          pemiu.cos_sim(target_a_shuffled[0], target_b_shuffled[0]))
    print("Genuine Comparison - reconstructed:", pemiu.cos_sim(target_a_shuffled[0], rec_gen))
    print("Genuine Comparison - vs real target a:", pemiu.cos_sim(target_a[0], rec_gen))
    print("Genuine Comparison - vs real target b:", pemiu.cos_sim(target_b[0], rec_gen))

    # compare imposter
    print("Imposter Comparison - original", pemiu.cos_sim(target_a[0], target_b_impostor[0]))
    print("Imposter Comparison - not reconstructed",
          pemiu.cos_sim(target_a_shuffled[0], target_b_impostor_shuffled[0]))
    print("Imposter Comparison - reconstructed:", pemiu.cos_sim(target_a_shuffled[0], rec_imp))


def create_pemiu_templates():
    path = "../data/"
    folder_name = "features_cleaner"
    filename_path_length = len(path + folder_name) + 1

    # Load embeddings, divided by genuine and impostor pairs
    with open("../data/lfw_view2/pairs_filenames.csv") as file:
        reader = csv.reader(file)
        data = list(reader)
        genuine_target_a, genuine_target_b = [], []
        genuine_target_a_filename, genuine_target_b_filename = [], []
        impostor_target_a, impostor_target_b = [], []
        impostor_target_a_filename, impostor_target_b_filename = [], []
        for element in data[1:]:
            if element[4].strip() == "True":
                genuine_target_a.append(np.load(element[1].strip()))
                genuine_target_b.append(np.load(element[3].strip()))
                genuine_target_a_filename.append(element[1][filename_path_length:])
                genuine_target_b_filename.append(element[3][filename_path_length:])
            else:
                impostor_target_a.append(np.load(element[1].strip()))
                impostor_target_b.append(np.load(element[3].strip()))
                impostor_target_a_filename.append(element[1][filename_path_length:])
                impostor_target_b_filename.append(element[3][filename_path_length:])

    # lfw_embeddings = dataset.dir_all_embeddings
    # lfw_embeddings_numpy = [np.load(element) for element in lfw_embeddings]
    pemiu = PrivacyEnhancingMIU(block_size=16)
    genuine_pemiu_target_a = pemiu.shuffle(genuine_target_a)
    genuine_pemiu_target_b = pemiu.shuffle(genuine_target_b)
    impostor_pemiu_target_a = pemiu.shuffle(impostor_target_a)
    impostor_pemiu_target_b = pemiu.shuffle(impostor_target_b)

    genuine_pemiu_reconstructed = [pemiu.reconstruct(genuine_pemiu_target_a[i], genuine_pemiu_target_b[i])
                                   for i in range(len(genuine_pemiu_target_a))]
    impostor_pemiu_reconstructed = [pemiu.reconstruct(impostor_pemiu_target_a[i], impostor_pemiu_target_b[i])
                                    for i in range(len(impostor_pemiu_target_a))]

    os.makedirs(f"{path}lfw_embeddings_pemiu/", exist_ok=True)
    os.makedirs(f"{path}lfw_embeddings_pemiu_genuine_reconstructed/", exist_ok=True)
    os.makedirs(f"{path}lfw_embeddings_pemiu_impostor_reconstructed/", exist_ok=True)

    # Save pemiu enhanced templates
    save = False
    if save:
        [np.save(f"{path}lfw_embeddings_pemiu/{filename}", element)
         for element, filename in zip(genuine_pemiu_target_a, genuine_target_a_filename)]
        [np.save(f"{path}lfw_embeddings_pemiu/{filename}", element)
         for element, filename in zip(genuine_pemiu_target_b, genuine_target_b_filename)]
        [np.save(f"{path}lfw_embeddings_pemiu/{filename}", element)
         for element, filename in zip(impostor_pemiu_target_a, impostor_target_a_filename)]
        [np.save(f"{path}lfw_embeddings_pemiu/{filename}", element)
         for element, filename in zip(impostor_pemiu_target_b, impostor_target_b_filename)]

    for i in range(len(genuine_target_a_filename)):
        os.makedirs(f"{path}lfw_embeddings_pemiu_genuine_reconstructed/{str(i).zfill(4)}", exist_ok=True)
        np.save(f"{path}lfw_embeddings_pemiu_genuine_reconstructed/{str(i).zfill(4)}/{genuine_target_a_filename[i]}",
                genuine_pemiu_reconstructed[i])
        os.makedirs(f"{path}lfw_embeddings_pemiu_impostor_reconstructed/{str(i).zfill(4)}", exist_ok=True)
        np.save(f"{path}lfw_embeddings_pemiu_impostor_reconstructed/{str(i).zfill(4)}/{impostor_target_a_filename[i]}",
                impostor_pemiu_reconstructed[i])


def create_pemiu_templates_from_all_lfw():
    block_size = 128
    save_path = f"../data/features_cleaner_pemiu{block_size}/"
    files = Files()
    lfw_image_filenames = files.get_filenames_lfw_embeddings(with_path=False)
    lfw_image_arrays = files.get_np_arrays_lfw_embeddings()
    pemiu = PrivacyEnhancingMIU(block_size=block_size)
    lfw_images_shuffled = pemiu.shuffle(lfw_image_arrays)
    os.makedirs(save_path, exist_ok=True)
    [np.save(f"{save_path}/{filename}", element) for element, filename in zip(lfw_images_shuffled, lfw_image_filenames)]
    print(f"Done saving {len(lfw_images_shuffled)} embeddings in {save_path} with block_size {block_size}")


def create_pemiu_reconstructed():
    block_size = 128

    # Get dataframe with LFW View 2 comparison pairs and PEMIU shuffled embeddings
    view2_gen = GenerateLFWView2()
    df = pd.DataFrame(
        view2_gen.get_lfwview2_filenames(
            image_dir="lfw_align",
            embedding_dir=f"features_cleaner_pemiu{block_size}"
        ))
    df.columns = ['a_img', f'a_embedding_pemiu{block_size}', 'b_img', f'b_embedding_pemiu{block_size}', 'genuine']
    df['genuine'] = df['genuine'].astype(bool)

    # PEMIU reconstruction
    save_path = "../data"
    pemiu = PrivacyEnhancingMIU(block_size=block_size)
    cos_sim, path_list = [], []

    for index, row in df.iterrows():
        target_a = np.load(row[f'a_embedding_pemiu{block_size}'])
        target_b = np.load(row[f'b_embedding_pemiu{block_size}'])
        reconstructed = pemiu.reconstruct(target_a, target_b)
        cos_sim.append(pemiu.cos_sim(target_a, reconstructed))
        label = "genuine" if row['genuine'] == True else "impostor"
        path = os.path.join(save_path, f"features_cleaner_pemiu{block_size}_reconstructed_{label}", str(index).zfill(4))
        filename = f"{str(row['a_img']).split('/')[-1].split('.')[0]}.npy"
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), reconstructed)
        path_list.append(os.path.join(path, filename))

    df[f'a_embedding_reconstructed{block_size}'] = path_list
    df[f'cos_sim_a_pemiu_a_pemiu_reconstructed{block_size}'] = cos_sim

    # Split dataframe into two by separating genuine and impostor
    df_genuine = df.loc[df['genuine'] == True]
    df_impostor = df.loc[df['genuine'] == False]

    df_genuine.to_csv(f"../dataframes/lfw_view2_genuine_pemiu{block_size}.csv")
    df_impostor.to_csv(f"../dataframes/lfw_view2_impostor_pemiu{block_size}.csv")


def create_pemiu_templates_from_all_ffhq():
    save_path = "../data/ffhq_align/embeddings_pemiu_16"
    files = Files()
    filenames = files.get_filenames_ffhq_embeddings(with_path=False)
    embedding_arrays = files.get_np_arrays_ffhq_embeddings()
    pemiu = PrivacyEnhancingMIU(block_size=16)
    embeddings_shuffled = pemiu.shuffle(embedding_arrays)
    os.makedirs(save_path, exist_ok=True)
    [np.save(f"{save_path}/{filename}", element) for element, filename in zip(embeddings_shuffled, filenames)]
    print(f"Done saving {len(embeddings_shuffled)} embeddings in {save_path}")


def calc_cosine_distance_pemiu():
    """

    Returns:

    """
    # Load pandas dataframe from csv containing filenames of images and embeddings from LFW dataset
    df = pd.read_csv("../data/lfw_view2/pairs_filenames.csv",
                     sep=',')

    df_genuine = df.loc[df['genuine'] == True]
    df_impostor = df.loc[df['genuine'] == False]

    # Get filenames of embeddings
    a_genuine_pemiu = [f"../data/lfw_embeddings_pemiu/{x[25:-4]}.npy" for x in df_genuine['a_embedding'].tolist()]
    a_genuine_pemiu_reconstructed = [
        f"../data/lfw_embeddings_pemiu_genuine_reconstructed/{str(index).zfill(4)}/{x[25:-4]}.npy" for index, x in
        enumerate(df_genuine['a_embedding'].tolist())]
    a_impostor_pemiu = [f"../data/lfw_embeddings_pemiu/{x[25:-4]}.npy" for x in df_impostor['a_embedding'].tolist()]
    a_impostor_pemiu_reconstructed = [
        f"../data/lfw_embeddings_pemiu_impostor_reconstructed/{str(index).zfill(4)}/{x[25:-4]}.npy" for index, x in
        enumerate(df_impostor['a_embedding'].tolist())]

    # Check integrity of files
    assert ([os.path.isfile(a_genuine_pemiu[x]) for x in range(len(a_genuine_pemiu))])
    assert ([os.path.isfile(a_genuine_pemiu_reconstructed[x]) for x in range(len(a_genuine_pemiu_reconstructed))])
    assert ([os.path.isfile(a_impostor_pemiu[x]) for x in range(len(a_impostor_pemiu))])
    assert ([os.path.isfile(a_impostor_pemiu_reconstructed[x]) for x in range(len(a_impostor_pemiu_reconstructed))])

    # Load embeddings as numpy arrays
    a_genuine_pemiu = [np.load(x) for x in a_genuine_pemiu]
    a_genuine_pemiu_reconstructed = [np.load(x) for x in a_genuine_pemiu_reconstructed]
    a_impostor_pemiu = [np.load(x) for x in a_impostor_pemiu]
    a_impostor_pemiu_reconstructed = [np.load(x) for x in a_impostor_pemiu_reconstructed]

    # Calculate cosine similarity
    pemiu = PrivacyEnhancingMIU(block_size=16)
    df_results = pd.DataFrame()

    df_results['cos_sim_pemiu_reconstructed_genuine'] = [pemiu.cos_sim(x, y) for x, y in
                                                         zip(a_genuine_pemiu, a_genuine_pemiu_reconstructed)]
    df_results['cos_sim_pemiu_reconstructed_impostor'] = [pemiu.cos_sim(x, y) for x, y in
                                                          zip(a_impostor_pemiu, a_impostor_pemiu_reconstructed)]

    # Normalize cosine distance to range (0, 1)
    # Fit scaler to column of genuine results
    scaler = MinMaxScaler()
    scaler.fit(df_results['cos_sim_pemiu_reconstructed_genuine'].values.reshape(-1, 1))

    df_results['cos_sim_pemiu_reconstructed_genuine_normalized'] = scaler.transform(
        df_results['cos_sim_pemiu_reconstructed_genuine'].values.reshape(-1, 1))
    df_results['cos_sim_pemiu_reconstructed_impostor_normalized'] = scaler.transform(
        df_results['cos_sim_pemiu_reconstructed_impostor'].values.reshape(-1, 1))
    print(df_results)

    # Export
    os.makedirs('../evaluation/', exist_ok=True)
    path = "../evaluation/cos_sim_pemiu_reconstructed_lfwview2.csv"
    df_results.to_csv(path)


def calc_cosine_distance_view2():
    files = Files()
    df_genuine, df_impostor = files.get_lfw_view2_genuine_impostor_as_separate_dataframes()

    # Load embeddings
    genuine_a = df_genuine['a_embedding'].apply(lambda x: np.load(x)).tolist()
    genuine_b = df_genuine['b_embedding'].apply(lambda x: np.load(x)).tolist()
    genuine_a_reconstructed = df_genuine['a_embedding_reconstructed'].apply(lambda x: np.load(x)).tolist()
    genuine_b_reconstructed = df_genuine['b_embedding_reconstructed'].apply(lambda x: np.load(x)).tolist()
    pemiu_a = df_genuine['a_embedding_pemiu'].apply(lambda x: np.load(x)).tolist()
    pemiu_b = df_genuine['b_embedding_pemiu'].apply(lambda x: np.load(x)).tolist()
    pemiu_reconstructed_a = df_genuine['a_embedding_pemiu_reconstructed'].apply(lambda x: np.load(x)).tolist()

    impostor_a = df_impostor['a_embedding'].apply(lambda x: np.load(x)).tolist()
    impostor_b = df_impostor['b_embedding'].apply(lambda x: np.load(x)).tolist()
    impostor_a_reconstructed = df_impostor['a_embedding_reconstructed'].apply(lambda x: np.load(x)).tolist()
    impostor_b_reconstructed = df_impostor['b_embedding_reconstructed'].apply(lambda x: np.load(x)).tolist()
    impostor_pemiu_a = df_impostor['a_embedding_pemiu'].apply(lambda x: np.load(x)).tolist()
    impostor_pemiu_reconstructed_a = df_impostor['a_embedding_pemiu_reconstructed'].apply(lambda x: np.load(x)).tolist()

    # Calculate cosine similarity using the method provided in the PEMIU implementation
    pemiu = PrivacyEnhancingMIU(block_size=16)

    df_genuine['cos_sim_a_b'] = [pemiu.cos_sim(x, y) for x, y in zip(genuine_a, genuine_b)]
    df_genuine['cos_sim_a_a_pemiu'] = [pemiu.cos_sim(x, y) for x, y in zip(genuine_a, pemiu_a)]
    df_genuine['cos_sim_a_b_pemiu'] = [pemiu.cos_sim(x, y) for x, y in zip(genuine_a, pemiu_b)]
    df_genuine['cos_sim_a_pemiu_b_pemiu'] = [pemiu.cos_sim(x, y) for x, y in zip(pemiu_a, pemiu_b)]
    df_genuine['cos_sim_a_pemiu_a_pemiu_reconstructed'] = [pemiu.cos_sim(x, y) for x, y in
                                                           zip(pemiu_a, pemiu_reconstructed_a)]
    df_genuine['cos_sim_a_a_pemiu_reconstructed'] = [pemiu.cos_sim(x, y) for x, y in
                                                     zip(genuine_a, pemiu_reconstructed_a)]

    df_genuine['cos_sim_a_a_reconstructed'] = [pemiu.cos_sim(x, y) for x, y in zip(genuine_a, genuine_a_reconstructed)]
    df_genuine['cos_sim_a_b_reconstructed'] = [pemiu.cos_sim(x, y) for x, y in zip(genuine_a, genuine_b_reconstructed)]

    df_impostor['cos_sim_a_b'] = [pemiu.cos_sim(x, y) for x, y in zip(impostor_a, impostor_b)]
    df_impostor['cos_sim_a_pemiu_a_pemiu_reconstructed'] = [pemiu.cos_sim(x, y) for x, y in
                                                            zip(impostor_pemiu_a, impostor_pemiu_reconstructed_a)]

    # Calculate cosine similarity using scipy spacial distance
    # (the method used in Hatef's paper about image reconstruction)
    df_genuine['cos_sim_scipy_a_b'] = [round(1 - distance.cosine(x, y), 3) for x, y in zip(genuine_a, genuine_b)]
    df_genuine['cos_sim_scipy_a_a_pemiu'] = [round(1 - distance.cosine(x, y), 3) for x, y in zip(genuine_a, pemiu_a)]

    df_impostor['cos_sim_scipy_a_b'] = [round(1 - distance.cosine(x, y), 3) for x, y in zip(impostor_a, impostor_b)]

    # Normalize cosine distance to range (0, 1)
    # Simplify gathering of columns by using a mask
    mask_genuine = df_genuine.columns.str.contains('cos_sim_*')
    mask_impostor = df_impostor.columns.str.contains('cos_sim_*')

    # Step 1: Set limits of normalization scale to all columns within the mask by finding min, max values
    min_max_values_genuine = [array.agg(['min', 'max']) for array in [df_genuine.loc[:, mask_genuine]]]
    min_max_values_impostor = [array.agg(['min', 'max']) for array in [df_impostor.loc[:, mask_impostor]]]
    min_max_values_genuine = np.array(min_max_values_genuine).reshape(-1, 1).tolist()
    min_max_values_impostor = np.array(min_max_values_impostor).reshape(-1, 1).tolist()
    min_max_values = min_max_values_genuine + min_max_values_impostor

    # Step 2: Set normalization scale
    scaler = MinMaxScaler()
    scaler.fit(min_max_values)

    # Step 3: Apply normalization to all columns within the mask
    # Append extension "_normalized" to the new column name
    for column in df_genuine.loc[:, mask_genuine]:
        df_genuine[f'{column}_normalized'] = scaler.transform(df_genuine[column].values.reshape(-1, 1))

    for column in df_impostor.loc[:, mask_impostor]:
        df_impostor[f'{column}_normalized'] = scaler.transform(df_impostor[column].values.reshape(-1, 1))

    # Export
    path = "../evaluation/"
    os.makedirs(path, exist_ok=True)
    df_genuine.to_csv(f"{path}lfwview2_genuine_cos_sim.csv")
    df_impostor.to_csv(f"{path}lfwview2_impostor_cos_sim.csv")


def get_gender_attribute():
    df = pd.read_csv("../data/lfw_view2/pairs_filenames.csv",
                     sep=',')

    filenames = []
    [filenames.append(element.split('/')[-1]) for element in df['a_img']]

    attr_gender = AttributePredictionInsightFace()
    df['a_gender_lfw'] = attr_gender.get_gender_to_lfw_specification(image_filenames=filenames)

    df['a_gender_insightface'] = attr_gender.get_gender_with_insightface_attribute_model(df['a_img'])

    # Get pemiu sample images filepath
    df['a_pemiu_img'] = ["sample_images/" + file.split('.')[0] + "_pemiu.png" for file in filenames]

    df['a_pemiu_gender_insightface'] = attr_gender.get_gender_with_insightface_attribute_model(df['a_pemiu_img'])

    df.to_csv("../evaluation/lfw_pairs_genders.csv")


def get_gender_attribute_for_all_lfw_files():
    files = Files()
    df = files.get_all_lfw_filenames_as_dataframe()

    # Add official gender attribution
    attr_gender = AttributePredictionInsightFace()
    df['gender_official'] = attr_gender.get_gender_to_lfw_specification(df['lfw_filename_raw'])

    # Add detected gender using insightface for original image and pemiu enhanced image
    df['gender_insightface'] = attr_gender.get_gender_with_insightface_attribute_model(df['lfw_image'])
    df['gender_insightface_pemiu'] = attr_gender.get_gender_with_insightface_attribute_model(df['lfw_image_pemiu'])

    # Save
    df.to_csv("../evaluation/lfw_complete_genders.csv")


def test_cosine_similarity_calculate_package():
    cos_sim_calc = CosineSimilarity()

    path_original = "../data/lfw_align/"
    path_pemiu = "experiments/experiment_7_model1_pemiu_block_sizes/sample_images_pemiu16/"
    image_a = f"{path_original}Aaron_Eckhart/Aaron_Eckhart_0001.png"
    image_b = f"{path_pemiu}Aaron_Eckhart_0001_pemiu16.png"

    print(cos_sim_calc.get_from_img(image_a, image_b))


def calculate_cos_sim_for_experiment_7():
    images = {}
    block_sizes = ["16", "32", "64", "128"]

    # Get image files
    images['unprotected'] = get_all_filenames_from_dir("../data", "lfw_align")
    for block_size in block_sizes:
        images[block_size] = get_all_filenames_from_dir("experiments/experiment_7_model1_pemiu_block_sizes/",
                                                        f"sample_images_pemiu{block_size}")

    # Setup dataframe
    df = pd.DataFrame()
    df['filename'] = [x.split("/")[-1].split(".")[0] for x in images['unprotected']]

    # Calculate cosine similarity
    cos_sim_calc = CosineSimilarity()
    for block_size in block_sizes:
        df[f'cos_sim_pemiu{block_size}_vs_real'] = [cos_sim_calc.get_from_img(a, b) for a, b in
                                                    zip(images['unprotected'], images[f'{block_size}'])]
        print(f"done with {block_size}")

    # Save dataframe
    df.to_csv("experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples.csv")


def calculate_cos_sim_for_mated_attack():
    # Get image files
    images = {'unprotected': get_all_filenames_from_dir("../data", "lfw_align"),
              'reconstructed': get_all_filenames_from_dir("experiments/experiment_7_model1_pemiu_block_sizes",
                                                          "sample_images_unprotected")}

    # Setup dataframe
    df = pd.DataFrame()
    df['filename'] = [x.split("/")[-1].split(".")[0] for x in images['unprotected']]

    # Calculate cosine similarity
    cos_sim_calc = CosineSimilarity()
    df[f'cos_sim_reconstructed_vs_real'] = [cos_sim_calc.get_from_img(a, b) for a, b in
                                            zip(images['unprotected'], images['reconstructed'])]

    # Save dataframe
    df.to_csv("experiments/experiment_7_model1_pemiu_block_sizes/cosine_similarity/cos_sim_samples_unprotected.csv")


def calculate_cos_sim_for_experiment_8():
    images = {}
    block_sizes = ["16"]
    path = "experiments/experiment_8_model3_pemiu_adapted/"

    # Get image files
    images['unprotected'] = get_all_filenames_from_dir("../data", "lfw_align")
    for block_size in block_sizes:
        images[block_size] = get_all_filenames_from_dir(path,
                                                        f"sample_images_pemiu{block_size}")

    # Setup dataframe
    df = pd.DataFrame()
    df['filename'] = [x.split("/")[-1].split(".")[0] for x in images['unprotected']]

    # Calculate cosine similarity
    cos_sim_calc = CosineSimilarity()
    for block_size in block_sizes:
        df[f'cos_sim_pemiu{block_size}_vs_real'] = [cos_sim_calc.get_from_img(a, b) for a, b in
                                                    zip(images['unprotected'], images[f'{block_size}'])]
        print(f"done with {block_size}")

    # Save dataframe
    df.to_csv(f"{path}cosine_similarity/cos_sim_samples_pemiu_model3.csv")


def attribute_prediction_knn():
    attr_ml = AttributePredictionML(balance_genders=True,
                                    train_on_one_sample_per_subject=True,
                                    alternate_genders=True)
    for blocksize in ['0', '16', '32', '64', '128']:
        attr_ml.attribute_prediction_knn(blocksize, fromimg="")
        attr_ml.attribute_prediction_knn(blocksize, fromimg="_fromimg")


def attribute_prediction_svm():
    attr_ml = AttributePredictionML(balance_genders=True)
    for blocksize in ['0', '16', '32', '64', '128']:
        for kernel in ['poly', 'rbf', 'sigmoid']:
            for fromimg in ['', '_fromimg']:
                attr_ml.attribute_prediction_svm(blocksize, fromimg=fromimg, kernel=kernel)


def attribute_prediction_cross_validation():
    attr_ml = AttributePredictionML(balance_genders=True,
                                    train_on_one_sample_per_subject=True,
                                    alternate_genders=True)

    # Dataframes for storing results
    df_from_img = pd.DataFrame(columns=['blocksize',
                                        'knn_mean',
                                        'knn_std',
                                        'svm_poly_mean',
                                        'svm_poly_std',
                                        'svm_rbf_mean',
                                        'svm_rbf_std',
                                        'svm_sigmoid_mean',
                                        'svm_sigmoid_std'])
    df_from_embedding = df_from_img.copy()

    # Path for saving
    path = "../evaluation/attribute_prediction_ml_with_cross_validation"
    os.makedirs(f'{path}', exist_ok=True)

    # Run evaluation
    for embedding_from_image in [True, False]:
        for blocksize in ['0', '16', '32', '64', '128']:
            results = [blocksize]
            results.extend(attr_ml.attribute_prediction_cross_validation(blocksize,
                                                                         'knn',
                                                                         embedding_from_image))
            for model in ['svm_poly', 'svm_rbf', 'svm_sigmoid']:
                results.extend(attr_ml.attribute_prediction_cross_validation(blocksize,
                                                                             model,
                                                                             embedding_from_image))
            if embedding_from_image:
                df_from_img.loc[len(df_from_img.index)] = results
            else:
                df_from_embedding.loc[len(df_from_embedding.index)] = results

    # Save results
    df_from_img.to_csv(f"{path}/attribute_from_img.csv")
    df_from_embedding.to_csv(f"{path}/attribute_from_embedding.csv")


def attribute_prediction_save_results():
    attr_ml = AttributePredictionML(balance_genders=True)
    df_from_img = pd.DataFrame(columns=['blocksize',
                                        'knn',
                                        'svm_poly',
                                        'svm_rbf',
                                        'svm_sigmoid'])
    df_from_embedding = df_from_img.copy()

    for fromimg in ['', '_fromimg']:
        for blocksize in ['0', '16', '32', '64', '128']:
            results = [blocksize, attr_ml.attribute_prediction_knn(blocksize, fromimg=fromimg)]
            for kernel in ['poly', 'rbf', 'sigmoid']:
                results.append(attr_ml.attribute_prediction_svm(blocksize, fromimg=fromimg, kernel=kernel))
            if fromimg == '_fromimg':
                df_from_img.loc[len(df_from_img.index)] = results
            else:
                df_from_embedding.loc[len(df_from_embedding.index)] = results

    path = "../evaluation/attribute_prediction_ml"
    os.makedirs(f'{path}', exist_ok=True)
    df_from_img.to_csv(f"{path}/attribute_from_img.csv")
    df_from_embedding.to_csv(f"{path}/attribute_from_embedding.csv")


def attribute_prediction_attack():
    attr_ml = AttributePredictionML(balance_genders=True,
                                    train_on_one_sample_per_subject=True,
                                    alternate_genders=True)
    df_from_img = pd.DataFrame(columns=['blocksize',
                                        'knn',
                                        'svm_poly',
                                        'svm_rbf',
                                        'svm_sigmoid'])
    df_from_embedding = df_from_img.copy()

    # Path for saving
    path = "../evaluation/attribute_prediction_ml_attack_training_on_unprotected_prediction_on_pemiu"
    os.makedirs(f'{path}', exist_ok=True)

    # Run evaluation
    results_from_img, results_from_embedding = {}, {}
    predicted_labels = pd.DataFrame()
    for embedding_from_image in [True, False]:
        for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']:
            scores, df = attr_ml.attack_training_on_unprotected_prediction_on_pemiu(model, embedding_from_image)
            predicted_labels = pd.concat((predicted_labels, df), axis=1)
            if embedding_from_image:
                results_from_img[f'{model}'] = scores
            else:
                results_from_embedding[f'{model}'] = scores

    for blocksize in ['16', '32', '64', '128']:
        row = [blocksize]
        row.extend([results_from_img[model][blocksize] for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']])
        df_from_img.loc[len(df_from_img.index)] = row

        row = [blocksize]
        row.extend(
            [results_from_embedding[model][blocksize] for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']])
        df_from_embedding.loc[len(df_from_embedding.index)] = row

    # Save results
    df_from_img.to_csv(f"{path}/attribute_from_img.csv")
    df_from_embedding.to_csv(f"{path}/attribute_from_embedding.csv")
    predicted_labels.T.drop_duplicates().T.to_csv(f"{path}/predicted_labels.csv")


def attribute_prediction_attack_on_templates_that_pass_threshold():
    # FMR100_TH = 0.34, FMR1000_TH = 0.4
    threshold_name = "100"
    threshold_number = 0.34

    attr_ml = AttributePredictionML(balance_genders=True,
                                    train_on_one_sample_per_subject=True,
                                    alternate_genders=True,
                                    evaluate_on_samples_that_pass_threshold=True,
                                    threshold=threshold_number)
    df_from_img = pd.DataFrame(columns=['blocksize',
                                        'knn',
                                        'svm_poly',
                                        'svm_rbf',
                                        'svm_sigmoid'])
    df_from_embedding = df_from_img.copy()

    # Path for saving
    path = f"../evaluation/attribute_prediction_ml_attack_training_on_" \
           f"unprotected_prediction_on_pemiu_samples_that_pass_threshold{threshold_name}"
    os.makedirs(f'{path}', exist_ok=True)

    # Run evaluation
    results_from_img, results_from_embedding = {}, {}
    predicted_labels = pd.DataFrame()
    for embedding_from_image in [True, False]:
        for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']:
            scores, df = attr_ml.attack_training_on_unprotected_prediction_on_pemiu(model, embedding_from_image)
            predicted_labels = pd.concat((predicted_labels, df), axis=1)
            if embedding_from_image:
                results_from_img[f'{model}'] = scores
            else:
                results_from_embedding[f'{model}'] = scores

    for blocksize in ['16', '32', '64', '128']:
        row = [blocksize]
        row.extend([results_from_img[model][blocksize] for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']])
        df_from_img.loc[len(df_from_img.index)] = row

        row = [blocksize]
        row.extend(
            [results_from_embedding[model][blocksize] for model in ['knn', 'svm_poly', 'svm_rbf', 'svm_sigmoid']])
        df_from_embedding.loc[len(df_from_embedding.index)] = row

    # Save results
    df_from_img.to_csv(f"{path}/attribute_from_img.csv")
    df_from_embedding.to_csv(f"{path}/attribute_from_embedding.csv")
    predicted_labels.T.drop_duplicates().T.to_csv(f"{path}/predicted_labels.csv")


def generate_embeddings():
    dataset_dir = "experiments/experiment_9_pemiu_randomization"
    block_sizes = ['16', '32', '64', '128']

    gen_emb = GenerateEmbeddings()

    # # PEMIU protected
    # for blocksize in block_sizes:
    #     samples = get_all_filenames_from_dir(dataset_dir, f"sample_images_pemiu{blocksize}")
    #
    #     os.makedirs(f'{os.path.join(dataset_dir, f"sample_images_pemiu{blocksize}_embeddings")}', exist_ok=True)
    #
    #     for image in samples:
    #         embedding = gen_emb.embedding_from_img(image)
    #         np.save(
    #         f"{os.path.join(dataset_dir, f'sample_images_pemiu{blocksize}_embeddings')}/{image.split('/')[-1].split('.')[0]}.npy",
    #         embedding)

    # Unprotected
    block_size = 16
    amount = 16
    samples = get_all_filenames_from_dir(dataset_dir, "sample_embeddings")
    os.makedirs(f'{os.path.join(dataset_dir, f"sample_embeddings_pemiu16")}', exist_ok=True)
    pemiu = PrivacyEnhancingMIU(block_size=block_size)

    for element in samples:
        for index in range(amount):
            # embedding = gen_emb.embedding_from_img(element)
            embedding = [np.load(element)]
            embedding = pemiu.shuffle(embedding)
            np.save(
                f"{os.path.join(dataset_dir, f'sample_embeddings_pemiu16')}/"
                f"{element.split('/')[-1].split('.')[0]}_{str(index).zfill(4)}.npy",
                embedding)


def experiment9():
    # Experiment 9
    # Generate multiple pemiu 16 from same subject/sample

    # Dirs
    dataset_dir = "experiments/experiment_9_pemiu_randomization"
    embedding_dir = "sample_embeddings"
    image_dir = "sample_images"

    # Settings
    block_size = 16
    amount = 128

    # Unused variables
    block_sizes = ['16', '32', '64', '128']

    # todo: Generate embeddings from sample images
    gen_emb = GenerateEmbeddings()

    # Generate PEMIU shuffled embeddings from embeddings
    samples = get_all_filenames_from_dir(dataset_dir, embedding_dir)
    os.makedirs(f'{os.path.join(dataset_dir, f"sample_embeddings_pemiu16")}', exist_ok=True)
    pemiu = PrivacyEnhancingMIU(block_size=block_size)

    for element in samples:
        for index in range(amount):
            # embedding = gen_emb.embedding_from_img(element)
            embedding = [np.load(element)]
            embedding = pemiu.shuffle(embedding)
            np.save(
                f"{os.path.join(dataset_dir, f'sample_embeddings_pemiu16')}/"
                f"{element.split('/')[-1].split('.')[0]}_{str(index).zfill(4)}.npy",  # File extension _0000.npy
                embedding)

    # Generate sample images
    samples_9_model1_pemiu16 = GenerateSampleImagesFromEmbeddings(
        dataset_dir=f"{dataset_dir}",
        embedding_dir=f"sample_embeddings_pemiu16",
        image_dir=f"{os.path.join(dataset_dir, image_dir)}",
        file_appendix=f"model1_pemiu16",
        save_path=f"{dataset_dir}/sample_images_pemiu16",
        save_path_log=f"{dataset_dir}",
        generator_checkpoint_dir="training_files",
        epoch=90,
        write_original_img=False,
        create_subdirs=False,
        iterations=amount,
        batch_size=amount
    )
    samples_9_model1_pemiu16.generate()

    # Calc cos_sim
    images = {}
    block_sizes = ["16"]
    path = "experiments/experiment_9_pemiu_randomization/"

    # Get image files
    all_files_list = []
    for subdir, dirs, files in os.walk(os.path.join(path, "sample_images_pemiu16")):
        for element in files:
            file = os.path.join(subdir, element)
            all_files_list.append(file)

    # Setup dataframe
    df = pd.DataFrame()
    filename = all_files_list[0].split("/")[-1].split(".")[0]
    df['filename'] = [x for x in all_files_list]
    df['unprotected'] = [f"{os.path.join(path, 'sample_images', '_'.join(filename.split('_')[:-3]))}.png" for x in
                         range(len(all_files_list))]

    # Calculate cosine similarity
    cos_sim_calc = CosineSimilarity()
    for block_size in block_sizes:
        df[f'cos_sim_pemiu{block_size}_vs_real'] = [cos_sim_calc.get_from_img(a, b) for a, b in
                                                    zip(df['unprotected'], df['filename'])]
        print(f"done with {block_size}")

    # Save dataframe
    os.makedirs(f"{path}cosine_similarity/", exist_ok=True)
    df.to_csv(f"{path}cosine_similarity/cos_sim_samples_pemiu_model3.csv")


def ser_fiq_example():
    # Sample code of calculating the score of an image

    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    ser_fiq = SER_FIQ(gpu=0)

    # Load the test image
    test_img = cv2.imread("../FaceImageQuality/data/test_img.jpeg")

    # Align the image
    aligned_img = ser_fiq.apply_mtcnn(test_img)

    # Calculate the quality score of the image
    # T=100 (default) is a good choice
    # Alpha and r parameters can be used to scale your
    # score distribution.
    score = ser_fiq.get_score(aligned_img, T=100)

    print("SER-FIQ quality score of image 1 is", score)

    # Do the same thing for the second image as well
    test_img2 = cv2.imread("../FaceImageQuality/data/test_img2.jpeg")

    aligned_img2 = ser_fiq.apply_mtcnn(test_img2)

    score2 = ser_fiq.get_score(aligned_img2, T=100)

    print("SER-FIQ quality score of image 2 is", score2)


def ser_fiq_image_quality_assessment():
    # Create the SER-FIQ Model
    ser_fiq = SER_FIQ(gpu=0)

    # Load data
    df = pd.read_csv("../evaluation/lfw_complete_genders.csv")

    # Load images
    images = [cv2.imread(img_path) for img_path in df['lfw_image']]

    # Align images
    aligned_images = [ser_fiq.apply_mtcnn(image) for image in images]

    # Calculate the quality score of the image
    # T=100 (default) is a good choice
    # Alpha and r parameters can be used to scale your
    # score distribution.
    score = [ser_fiq.get_score(image, T=100) for image in aligned_images]

    df['img_original_quality'] = score

    df.to_csv("../evaluation/lfw_complete_genders_with_quality.csv")


def ser_fiq_image_quality_for_all_files():
    save_path = "../evaluation/image_quality"
    os.makedirs(save_path, exist_ok=True)

    files = Files()
    dataframe = files.get_dataframe_with_all_lfw_images_incl_pemiu()

    # Create the SER-FIQ Model
    ser_fiq = SER_FIQ(gpu=0)

    # Get columns containing image paths by using a mask
    mask = dataframe.columns.str.contains('img_*')

    # Iterate over img_ columns
    for column in dataframe.loc[:, mask]:
        print(f"Load {len(dataframe)} {column} images ...")
        # Load images
        images = [cv2.imread(img_path) for img_path in dataframe[column]]

        print(f"Align {len(images)} images ...")
        # Align images
        aligned_images = [ser_fiq.apply_mtcnn(image) for image in images]

        print(f"Score {len(aligned_images)} images ...")
        # Calculate the quality score of the image
        score = [ser_fiq.get_score(image, T=100) for image in aligned_images]

        print(f"Saving ...\n")
        # Save score
        dataframe[f"quality_{column}"] = score

    dataframe.to_csv(f"{save_path}/image_quality_reconstructed_pemiu.csv")


def insightface_attribute_prediction_for_all_lfw_pemiu():
    save_path = "../evaluation/insightface_gender_prediction"
    os.makedirs(save_path, exist_ok=True)

    files = Files()
    df = files.get_dataframe_with_all_lfw_images_incl_pemiu(gender_labels=True,
                                                            drop_missing_gender_labels=True)

    # Initialize algorithm
    attr_gender_if = AttributePredictionInsightFace()

    # Get columns containing image paths by using a mask
    mask = df.columns.str.contains('img_*')

    # Iterate over img_ columns
    for column in df.loc[:, mask]:
        # Detect gender using insightface
        df[f'gender_insightface_{column}'] = attr_gender_if.get_gender_with_insightface_attribute_model(
            df[column])

    # Save
    df.to_csv(f"{save_path}/insightface_gender_prediction.csv")


def insightface_attribute_prediction_for_img_that_pass_threshold():
    # Settings
    threshold_name = "1000"
    threshold_number = 0.4   # FMR100_TH = 0.34, FMR1000_TH = 0.4

    # Variables
    block_sizes = ['16', '32', '64', '128']
    save_path = f"../evaluation/insightface_gender_prediction_img_that_pass_threshold{threshold_name}"
    os.makedirs(save_path, exist_ok=True)

    # Get reduced dataset only containing one sample per identity
    df = pd.read_csv(f"../evaluation/attribute_prediction_ml_attack_training_on_unprotected_prediction_on_pemiu_samples_that_pass_threshold{threshold_name}/predicted_labels.csv")
    # Merge with dataframe containing filepaths for images
    files = Files()
    df_all = files.get_dataframe_with_all_lfw_images_incl_pemiu()
    df = df.merge(df_all, on=['filename'], how='left')

    # Initialize algorithm
    attr_gender_if = AttributePredictionInsightFace()

    # Get only rows where cos_sim of bonafide vs. pemiu@blocksize is greater than threshold
    for blocksize in block_sizes:
        # Filter dataframe
        df_prediction = df[df[f'cos_sim_bonafide_synthesized_pemiu{blocksize}_normalized'] > threshold_number].copy(deep=True)
        # Predict
        df_prediction[f'gender_insightface_{blocksize}'] = attr_gender_if.get_gender_with_insightface_attribute_model(
            df_prediction[f"img_pemiu_{blocksize}"])
        # Drop columns we don't need for merge
        df_prediction = df_prediction[['filename', f'gender_insightface_{blocksize}']]
        # Merge predictions with larger dataframe
        df = df.merge(df_prediction, on=['filename'], how='left')

    # Save
    df.to_csv(f"{save_path}/insightface_gender_prediction_that_pass_th{threshold_name}.csv")


def main():
    function_names = [train,
                      evaluation,
                      generate_sample_images,
                      generate_dataset,
                      generate_lfw_view2,
                      calc_cosine_distance,
                      generate_pyeer_eer_report,
                      pemiu_test,
                      create_pemiu_templates,
                      calc_cosine_distance_pemiu,
                      get_gender_attribute,
                      create_pemiu_templates_from_all_lfw,
                      create_pemiu_templates_from_all_ffhq,
                      get_gender_attribute_for_all_lfw_files,
                      calc_cosine_distance_view2,
                      generate_embeddings_for_reconstructed_images,
                      generate_dataset_mobio,
                      create_pemiu_reconstructed,
                      test_cosine_similarity_calculate_package,
                      calculate_cos_sim_for_experiment_7,
                      calculate_cos_sim_for_mated_attack,
                      calculate_cos_sim_for_experiment_8,
                      attribute_prediction_knn,
                      attribute_prediction_svm,
                      attribute_prediction_save_results,
                      attribute_prediction_cross_validation,
                      attribute_prediction_attack,
                      attribute_prediction_attack_on_templates_that_pass_threshold,
                      generate_embeddings,
                      ser_fiq_example,
                      ser_fiq_image_quality_assessment,
                      ser_fiq_image_quality_for_all_files,
                      experiment9,
                      insightface_attribute_prediction_for_all_lfw_pemiu,
                      insightface_attribute_prediction_for_img_that_pass_threshold]
    menu_items = dict(enumerate(function_names, start=1))

    while True:
        display_menu(menu_items)
        selection = int(
            input("Please enter your selection number: "))  # Get function key
        selected_value = menu_items[selection]  # Gets the function name
        printout(str(menu_items[selection]).split(" ")[1])
        selected_value()  # add parentheses to call the function


if __name__ == '__main__':
    main()
