import time
import math
import numpy as np
import random
from multiprocessing import Pool
from multiprocessing import freeze_support
from src.pemiu.privacy_enhancing_miu import PrivacyEnhancingMIU
from threading import Lock

# Global vars
vector = '../data/features_cleaner/Aaron_Eckhart_0001.npy'
block_size = 32
amount = 10 ** 6
n_processors = 20

# Init
pemiu = PrivacyEnhancingMIU(block_size=block_size)
vector = [np.load(vector)]
seed_dict = {}
max_range = int(512 / block_size + 1)
permutation_degree = [x * block_size for x in range(0, max_range)]


# Define function to run multiple processors and pool the results together
def run_multiprocessing(func, i, processors):
    with Pool(processes=processors) as pool:
        return pool.map(func, i)


# Define task function
def calc_permutation_complexity(seed):
    random.seed(seed)
    target_a_shuffled = pemiu.shuffle(vector)
    positions = [vector[i] == target_a_shuffled[i] for i in range(len(vector))]
    if sum(positions[0]) in permutation_degree:
        seed_dict[seed] = sum(positions[0])
        permutation_degree.remove(sum(positions[0]))


def main():
    # Setup
    x_ls = list(range(amount))

    # Benchmark setup
    start = time.perf_counter()

    # pass the task function, followed by the parameters to processors
    out = run_multiprocessing(calc_permutation_complexity, x_ls, n_processors)

    print("Input length: {}".format(len(x_ls)))
    print("Output length: {}".format(len(out)))
    print("Multiprocessing time: {} mins\n".format((time.perf_counter() - start) / 60))

    print("Permutation degrees that couldn't be found:", permutation_degree, "\n")

    # Sort dict by permutation degree
    # seed_dict = dict(sorted(seed_dict.items(), key=lambda item: item[1], reverse=True))

    # Seeds and their permutation degree
    print("---------\nSeed, permutation degree")
    for key, value in seed_dict.items():
        print(key, "\t", value)

    print("\nFound", len(seed_dict), "seeds of possible", max_range - 1)

    # Save results to disk
    path = "../evaluation/seed_discovery_32.txt"
    with open(path, 'w') as file:
        for key, value in seed_dict.items():
            file.write(f"{key}\t{value}\n")


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
