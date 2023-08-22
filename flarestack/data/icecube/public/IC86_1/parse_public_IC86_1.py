# type: ignore

# exclude this file from `mypy` checking since the attribute IC86_1_dict can no longer be imported
# this also implies that this file is currently untested, otherwise such a test would fail
# if in the future its functionality is restored, remove the `type: ignore` comment


"""Script to convert public data files provided by IceCube for the first year
of IC86_1 into a format useable for science with flarestack. The data files
themselves are duplicates of those provided at:

    https://icecube.wisc.edu/science/data/PS-IC86-2011

"""

import os
import numpy as np
import csv
import pickle

from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.icecube_utils.dataset_loader import data_loader
from flarestack.shared import input_dir

ref_data = data_loader(IC86_1_dict["exp_path"])

src_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

data_dir = src_dir + "raw_data/"
output_path = input_dir + "public_datasets/"


data_dtype = np.dtype(
    [
        ("ra", float),
        ("dec", float),
        ("logE", float),
        ("sigma", float),
        ("time", float),
        ("sinDec", float),
    ]
)


def parse_numpy_dataset():
    """Function to parse the .txt file  of events into a numpy format
    readable by flarestack, which is the saved in the products/ subdirectory.
    """

    data = []

    for dataset in ["upgoing_events.txt", "downgoing_events.txt"]:
        path = data_dir + dataset

        with open(path, "r") as f:
            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):
                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    entry = (
                        np.deg2rad(row[3]),
                        np.deg2rad(row[4]),
                        row[1],
                        np.deg2rad(row[2]),
                        row[0],
                        np.sin(np.deg2rad(row[4])),
                    )

                    data.append(entry)

    data = np.array(data, dtype=data_dtype)

    exp_path = output_path + "public_IC86_1.npy"

    with open(exp_path, "wb") as f:
        print("Saving converted numpy array to", exp_path)
        pickle.dump(data, f)


parse_numpy_dataset()


def parse_effective_areas():
    """Function to parse effective areas .txt into a format that flarestack
    can use to build Signal/Background splines.
    """
    file_name = data_dir + "TabulatedAeff.txt"

    data = []

    with open(file_name, "r") as f:
        csv_reader = csv.reader(f, delimiter=" ")

        for i, row in enumerate(csv_reader):
            if i > 0:
                row = [float(x) for x in row if x != ""]

                entry = tuple(row)

                data.append(entry)

    data = np.array(data)

    print(sorted(set(data[:, 0] + data[:, 1])))


parse_effective_areas()
