import os
import numpy as np
import socket
import pickle
import json
import zlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# Directory substructure creation
# ==============================================================================

# fs_dir is the path of the

fs_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

try:
    fs_scratch_dir = os.environ["FLARESTACK_SCRATCH_DIR"]
except KeyError:
    fs_scratch_dir = str(Path.home())
    logger.warning(
        "No scratch directory has been set. Using home directory as default."
    )

fs_scratch_dir = os.path.join(fs_scratch_dir, "flarestack__data/")

logger.info("Scratch Directory is: {0}".format(fs_scratch_dir))

input_dir = fs_scratch_dir + "input/"
storage_dir = fs_scratch_dir + "storage/"
output_dir = fs_scratch_dir + "output/"
cluster_dir = fs_scratch_dir + "cluster/"
log_dir = cluster_dir + "logs/"

cache_dir = storage_dir + "cache/"
cat_cache_dir = cache_dir + "catalogue_cache/"

public_dataset_dir = input_dir + "public_datasets/"
sim_dataset_dir = input_dir + "sim_datasets/"

catalogue_dir = input_dir + "catalogues/"
transients_dir = catalogue_dir + "transients/"
analysis_dir = input_dir + "analysis/"

pickle_dir = storage_dir + "pickles/"
inj_param_dir = pickle_dir + "injection_values/"

plots_dir = output_dir + "plots/"
unbliding_dir = output_dir + "unblinding_results/"
limits_dir = output_dir + "limits/"
dataset_plot_dir = output_dir + "dataset_plots/"
eff_a_plot_dir = dataset_plot_dir + "effective_area_plots/"
energy_proxy_plot_dir = dataset_plot_dir + "energy_proxy_map/"
ang_res_plot_dir = dataset_plot_dir + "angular_resolution_plots/"

illustration_dir = plots_dir + "illustrations/"

acc_f_dir = input_dir + "acceptance_functions/"
SoB_spline_dir = input_dir + "SoB_splines/"
energy_spline_dir = input_dir + "energy_pdf_splines/"
bkg_spline_dir = input_dir + "bkg_splines/"
energy_proxy_dir = input_dir + "energy_proxy_weighting/"
med_ang_res_dir = input_dir + "median_angular_resolution/"
pc_dir = input_dir + "pull_corrections/"
pull_dir = pc_dir + "pulls/"
floor_dir = pc_dir + "floors/"

all_dirs = [
    fs_scratch_dir,
    input_dir,
    storage_dir,
    output_dir,
    cluster_dir,
    pc_dir,
    log_dir,
    catalogue_dir,
    acc_f_dir,
    energy_spline_dir,
    pickle_dir,
    plots_dir,
    SoB_spline_dir,
    analysis_dir,
    illustration_dir,
    transients_dir,
    bkg_spline_dir,
    dataset_plot_dir,
    unbliding_dir,
    limits_dir,
    pull_dir,
    floor_dir,
    cache_dir,
    cat_cache_dir,
    public_dataset_dir,
    energy_proxy_dir,
    eff_a_plot_dir,
    med_ang_res_dir,
    ang_res_plot_dir,
    energy_proxy_plot_dir,
    sim_dataset_dir,
]

for dirname in all_dirs:
    if not os.path.isdir(dirname):
        logger.info("Making Directory: {0}".format(dirname))
        os.makedirs(dirname)
    else:
        logger.info("Found Directory: {0}".format(dirname))

# ==============================================================================
# Check host and specify path to dataset storage
# ==============================================================================

host = socket.getfqdn()

host_server: Optional[str] = None

if np.logical_or("ifh.de" in host, "zeuthen.desy.de" in host):
    host_server = "DESY"
elif "icecube.wisc.edu" in host:
    host_server = "WIPAC"

# gamma_range = [1., 4.]
# gamma_precision = .025
flarestack_gamma_precision = 0.025

default_gamma_precision = {"flarestack": flarestack_gamma_precision, "skylab": 0.1}

default_smoothing_order = {"flarestack": 2, "skylab": 1}


def deterministic_hash(hash_dict):
    """Generic function to convert a given dictionary into a 32bit hash
    value. This process is deterministic, so can be used as a general uid to
    save/load values.

    :param hash_dict: Dictionary contaiing relevant information
    :return: A 32bit number representing the data
    """
    return zlib.adler32((json.dumps(hash_dict, sort_keys=True)).encode())


# Sets the minimum angular error

min_angular_err = np.deg2rad(0.2)

# Sets an angular error floor based on the 25th quantile

base_floor_quantile = 0.25


def floor_pickle(floor_dict):
    hash_dict = dict(floor_dict)
    season = hash_dict["season"]
    hash_dict["season"] = season.sample_name + "/" + season.season_name
    try:
        del hash_dict["pull_name"]
    except KeyError:
        pass
    unique_key = deterministic_hash(hash_dict)
    return floor_dir + str(unique_key) + ".pkl"


def pull_pickle(pull_dict):
    hash_dict = dict(pull_dict)
    season = hash_dict["season"]
    hash_dict["season"] = season.sample_name + "/" + season.season_name
    unique_key = deterministic_hash(hash_dict)
    return pull_dir + str(unique_key) + ".pkl"


def llh_energy_hash_pickles(llh_dict, season):
    hash_dict = dict(llh_dict["llh_energy_pdf"])
    hash_dict["llh_name"] = llh_dict["llh_name"]

    precision = llh_dict.get("gamma_precision", "flarestack")
    smoothing_order = llh_dict.get("smoothing_order", "flarestack")

    key = deterministic_hash(hash_dict)
    season_path = (
        str(key)
        + "/"
        + season.sample_name
        + "/"
        + season.season_name
        + smoothing_precision_string(smoothing_order, precision)
        + ".pkl"
    )
    SoB_path = SoB_spline_dir + season_path
    acc_path = acc_f_dir + season_path
    return SoB_path, acc_path


band_mask_chunk_size = 100


def band_mask_hash_dir(catalogue):
    return cat_cache_dir + str(zlib.adler32(catalogue)) + "/"


def band_mask_cache_name(season, catalogue, injection_bandwidth):
    n_chunks = int((len(catalogue) + band_mask_chunk_size - 1) / band_mask_chunk_size)
    logger.info(
        "Breaking catalogue into {0} chunks of {1}".format(
            n_chunks, band_mask_chunk_size
        )
    )

    cats = []
    mask_indices = []
    source_indices = []

    for i in range(n_chunks):
        cat = catalogue[(i * band_mask_chunk_size) : ((i + 1) * band_mask_chunk_size)]
        cats.append(cat)
        mask_indices += [i for _ in range(band_mask_chunk_size)]
        source_indices += [x for x in range(band_mask_chunk_size)]

    paths = [
        band_mask_hash_dir(cat)
        + season.sample_name
        + f"_{injection_bandwidth:.8f}/"
        + season.season_name
        + ".npz"
        for cat in cats
    ]

    mask_indices = mask_indices[: len(catalogue)]
    source_indices = source_indices[: len(catalogue)]

    return cats, paths, mask_indices, source_indices


def name_pickle_output_dir(name):
    return os.path.join(pickle_dir, name)


def inj_dir_name(name):
    return os.path.join(inj_param_dir, name)


def plot_output_dir(name):
    return os.path.join(plots_dir, name)


def unblinding_output_path(name):
    path = os.path.join(unbliding_dir, name + "unblinding_results.pkl")
    return path


def limit_output_path(name):
    path = os.path.join(limits_dir, name + "limit.pkl")
    return path


def sim_dataset_dir_path(sample_name, season_name, bkg_flux_model):
    return (
        f"{sim_dataset_dir}{sample_name}/"
        f"{season_name}/{bkg_flux_model.unique_name()}/"
    )


def get_base_sob_plot_dir(season):
    return (
        dataset_plot_dir
        + "Signal_over_background/"
        + season.sample_name
        + "/"
        + season.season_name
        + "/"
    )


def smoothing_precision_string(smoothing_order="flarestack", gamma_precision="skylab"):
    if isinstance(smoothing_order, str):
        if smoothing_order in default_smoothing_order.keys():
            smoothing_order = default_smoothing_order[smoothing_order]
        else:
            raise ValueError(f"Smoothing order for {smoothing_order} not known!")

    if isinstance(gamma_precision, str):
        if gamma_precision in default_gamma_precision.keys():
            gamma_precision = default_gamma_precision[gamma_precision]
        else:
            raise ValueError(f"Smoothing order for {smoothing_order} not known!")

    logger.debug(
        f"smoothing order is {smoothing_order}, gamma precision is {gamma_precision}"
    )
    s = ""
    if smoothing_order != default_smoothing_order["flarestack"]:
        s += f"_smoothing{smoothing_order}"
    if gamma_precision != default_gamma_precision["flarestack"]:
        s += f"_precision{gamma_precision}"
    return s


def acceptance_path(season):
    return acc_f_dir + season.sample_name + "/" + season.season_name + ".pkl"


def SoB_spline_path(season, *args, **kwargs):
    return (
        SoB_spline_dir
        + season.sample_name
        + "/"
        + season.season_name
        + smoothing_precision_string(*args, **kwargs)
        + ".pkl"
    )


def bkg_spline_path(season):
    return bkg_spline_dir + season.sample_name + "/" + season.season_name + ".pkl"


def energy_proxy_path(season):
    return energy_proxy_dir + season.sample_name + "/" + season.season_name + ".pkl"


def med_ang_res_path(season):
    return med_ang_res_dir + season.sample_name + "/" + season.season_name + ".pkl"


def ang_res_plot_path(season):
    return ang_res_plot_dir + season.sample_name + "/" + season.season_name + ".pdf"


def energy_proxy_plot_path(season):
    return (
        energy_proxy_plot_dir + season.sample_name + "/" + season.season_name + ".pdf"
    )


def effective_area_plot_path(season):
    return eff_a_plot_dir + season.sample_name + "/" + season.season_name + ".pdf"


k_flux_factor = 10**-9


def k_to_flux(k):
    """k is a flux scale, with k=1 equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2.
    The k values can be converted into values for the flux.

    :param k: Flux scale
    :return: Flux value
    """
    return k * k_flux_factor


def flux_to_k(flux):
    """k is a flux scale, with k=1 equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2.
    The flux values can be converted into values for k.

    :param flux: Flux value
    :return: Flux scale (k)
    """
    return flux / k_flux_factor


def scale_shortener(scale):
    """Function to trim number of significant figures for flux scales when
    required for dictionary keys or saving pickle files.

    :param scale: Flux Scale
    :return: Flux Scale to 4.s.f
    """
    return "{0:.4G}".format(float(scale))


def analysis_pickle_path(mh_dict=None, name=None):
    """Converts a Minimisation Handler dictionary to a corresponding analysis
    config pickle. This pickle can be used to run a Minimisation Handler.

    :param mh_dict: Minimisation Handler dictionary
    :param name: unique Minimisation Handler name
    :return: Path to analysis pickle
    """

    dict_name = "dict.pkl"

    if mh_dict is not None:
        try:
            if np.logical_and(name is not None, name != mh_dict["name"]):
                raise Exception(
                    f"Both name and mh_dict arguments provided. "
                    f"There is a conflict between: \n"
                    f"'name' parameter: {name} \n"
                    f"'mh_dict'-derived name: {mh_dict['name']} \n"
                    f"Please resolve this conflict, by specifying the correct name."
                )

            name = mh_dict["name"]
        except KeyError:
            raise Exception(
                "No field 'name' was specified in mh_dict object. "
                "Cannot save results without a unique directory"
                " name being specified."
            )

        if "fixed_scale" in mh_dict.keys():
            dict_name = f"fixed_scale_{mh_dict['fixed_scale']}_dict.pkl"

    analysis_path = os.path.join(analysis_dir, name)

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = os.path.join(analysis_path, dict_name)
    return pkl_file


def make_analysis_pickle(mh_dict):
    """Takes a Minimisation Handler Dictionary, finds the corresponding
    analysis pickle path, and saves the dictionary to this path

    :param mh_dict: Minimisation Handler dictionary
    """

    for season in mh_dict["dataset"].values():
        season.clean_season_cache()

    pkl_file = analysis_pickle_path(mh_dict)

    with open(pkl_file, "wb") as f:
        pickle.dump(mh_dict, f)

    return pkl_file


def weighted_quantile(values, quantiles, weight):
    """Calculated quantiles accounting for weights.

    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weight: array-like of the same length as `array`
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
