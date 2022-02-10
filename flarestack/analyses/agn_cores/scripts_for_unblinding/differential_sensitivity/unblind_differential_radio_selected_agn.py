"""Script to unblind the Radio-selected AGN sample (9749 sources) in terms of differential significance/upper limits.
Draws the background TS values generated by differential_sensitivit/diff_sens_radio_selected_agn_analysis.py,
in order to quantify the significance of the result. Produces relevant post-unblinding plots.
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import diffuse_8_year
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.analyses.agn_cores.shared_agncores import (
    agn_subset_catalogue,
    complete_cats_north,
    complete_cats_north,
    agn_catalogue_name,
    agn_subset_catalogue_north,
)
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.unblinding import create_unblinder


import logging
import os
import psutil, resource  # to get memory usage info


logging.getLogger().setLevel("INFO")


analyses = dict()

# Initialise Injectors/LLHs

llh_time = {"time_pdf_name": "Steady"}

llh_energy = {"energy_pdf_name": "PowerLaw"}

llh_dict = {
    "llh_name": "standard_matrix",
    "llh_sig_time_pdf": llh_time,
    "llh_energy_pdf": llh_energy,
}


def bkg_ts_base_name(cat_key, gamma):
    return (
        "analyses/agn_cores/stacking_analysis_8yrNTsample_diff_sens_pre_unblinding/{0}/"
        "{1}/".format(cat_key, gamma)
    )


def bkg_ts_generate_name(cat_key, n_sources, gamma):
    return bkg_ts_base_name(cat_key, gamma) + "NrSrcs={0}/".format(n_sources)


def base_name(cat_key, gamma):
    return (
        "analyses/agn_cores/stacking_analysis_8yrNTsample_diff_sens_unblinding/{0}/"
        "{1}/".format(cat_key, gamma)
    )


def generate_name(cat_key, n_sources, gamma):
    return base_name(cat_key, gamma) + "NrSrcs={0}/".format(n_sources)


gammas = [2.0, 2.5]

nr_brightest_sources = [9749]

all_res = dict()

for (cat_type, method) in complete_cats_north[:1]:

    unique_key = cat_type + "_" + method

    gamma_dict = dict()

    for gamma_index in gammas:
        res = dict()
        nr_srcs = int(nr_brightest_sources[0])
        cat_path = agn_subset_catalogue(cat_type, method, nr_srcs)
        catalogue = load_catalogue(cat_path)
        cat = np.load(cat_path)

        name = generate_name(unique_key, nr_srcs, gamma_index)
        bkg_ts = bkg_ts_generate_name(unique_key, nr_srcs, gamma_index)

        injection_time = llh_time
        injection_energy = dict(llh_energy)
        injection_energy["gamma"] = gamma_index

        inj_kwargs = {
            "injection_energy_pdf": injection_energy,
            "injection_sig_time_pdf": injection_time,
        }

        unblind_dict = {
            "name": name,
            "mh_name": "large_catalogue",
            "dataset": diffuse_8_year.get_seasons(),
            "catalogue": cat_path,
            "llh_dict": llh_dict,
            "background_ts": bkg_ts,
        }
        ub = create_unblinder(unblind_dict, mock_unblind=True, full_plots=True)
