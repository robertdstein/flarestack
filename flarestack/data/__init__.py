"""Generic classes for a Dataset, and for a Season.
"""
import os
import numpy as np
import copy
import logging
from numpy.lib.recfunctions import append_fields, drop_fields
from flarestack.core.injector import MCInjector, EffectiveAreaInjector
from flarestack.utils.make_SoB_splines import make_background_spline
from flarestack.utils.create_acceptance_functions import make_acceptance_season
from flarestack.core.time_pdf import (
    TimePDF,
    DetectorOnOffList,
    FixedEndBox,
    FixedRefBox,
)
import flarestack


logger = logging.getLogger(__name__)


class DatasetHolder:
    def __init__(self, sample_name):
        self.sample_name = sample_name

        self.datasets = dict()
        self.current = None

    def set_current(self, version):
        if version in self.datasets.keys():
            self.current = version
        else:
            raise Exception(
                "Unrecognised version key: {0} \n "
                "Stored dataset keys are: {1}".format(version, self.datasets.keys())
            )

    def get_current(self):
        if self.current is not None:
            return self.datasets[self.current]
        else:
            logger.warning("Warning: no file listed as current.")
            key = sorted(list(self.datasets.keys()))
            logger.warning("Using key {0} out of {1}".format(key, self.datasets.keys()))
            return self.datasets[key]


class Dataset:
    def __init__(self, **kwargs):
        self.seasons = dict()
        self.subseasons = dict()
        self.name = kwargs.get("name", None)

    def add_season(self, season, **kwargs):
        self.seasons[season.season_name] = copy.copy(season)
        self.seasons[season.season_name].setup(**kwargs)

    def add_subseason(self, season, **kwargs):
        self.subseasons[season.season_name] = copy.copy(season)
        self.subseasons[season.season_name].setup(**kwargs)

    def get_seasons(self, *args, **kwargs):
        season_names = list(args)

        if len(season_names) == 0:
            return self.make_copy()
        else:
            cd = self.make_copy()
            cd.seasons = dict()
            cd.subseasons = dict()
            for name in season_names:
                if name in self.seasons:
                    cd.add_season(self.seasons[name], **kwargs)
                elif name in self.subseasons:
                    cd.add_season(self.subseasons[name], **kwargs)
                else:
                    raise Exception(
                        "Unrecognised season name: {0} not found. \n"
                        "Available seasons are: {1} \n"
                        "Available subseasons are: {2}".format(
                            name, self.seasons.keys(), self.subseasons.keys()
                        )
                    )

            return cd

    def get_single_season(self, name, **kwargs):
        return self.get_seasons(name, **kwargs)[name]

    def __iter__(self):
        return self.seasons.__iter__()

    def items(self):
        return self.seasons.items()

    def values(self):
        return self.seasons.values()

    def keys(self):
        return self.seasons.keys()

    def __getitem__(self, item):
        return self.seasons[item]

    def get(self, item):
        return self.seasons[item]

    def __len__(self):
        return self.seasons.__len__()

    def make_copy(self):
        return copy.copy(self)


class Season:
    def __init__(self, season_name, sample_name, exp_path, **kwargs):
        self.season_name = season_name
        self.sample_name = sample_name
        self.exp_path = exp_path
        self.loaded_background_model = None
        self.loaded_trial_model = None
        self.pseudo_mc_path = None
        self.background_dtype = None
        self._time_pdf = None
        self.all_paths = [self.exp_path]
        self._subselection_fraction = None
        self.get_trial_model = self.get_background_model

        # If any keywords left over, they're unrecognized, raise an error
        if kwargs:
            # Arbitrarily select alphabetically first unknown keyword arg
            raise TypeError(f"Season got unexpected keyword argument {min(kwargs)}")

    def setup(self, **kwargs):
        trial_with_data = kwargs.pop("trial_with_data", None)
        subselection_fraction = kwargs.pop("subselection_fraction", None)

        # If any keywords left over, they're unrecognized, raise an error
        if kwargs:
            # Arbitrarily select alphabetically first unknown keyword arg
            raise TypeError(f"Season got unexpected keyword argument {min(kwargs)}")

        # Subselection fraction

        if subselection_fraction is not None:
            if float(subselection_fraction) > 1.0:
                raise ValueError(
                    "Subselection {0} is greater than 1."
                    "Please specify a different subselection value"
                )

            self.set_subselection_fraction(subselection_fraction)

        if trial_with_data is True:
            self.use_data_for_trials()

    def load_background_model(self):
        """Generic function to load background data to memory. It is useful
        for Injector, but does not always need to be used."""
        self.loaded_background_model = self.get_background_model()

    def load_trial_model(self):
        """Generic function to load background data to memory. It is useful
        for Injector, but does not always need to be used."""
        self.loaded_trial_model = self.get_trial_model()

    def set_subselection_fraction(self, subselection_fraction):
        if float(subselection_fraction) > 1.0:
            raise ValueError(
                "Subselection {0} is greater than 1."
                "Please specify a subselection value <=1. "
            )

        self._subselection_fraction = subselection_fraction

    def get_background_dtype(self):
        if self.background_dtype is None:
            exp = self.get_exp_data()
            self.background_dtype = exp.dtype
            del exp
        return self.background_dtype

    def data_background_model(self, **kwargs):
        """Function to return data as a background model."""
        exp = self.get_exp_data(**kwargs)
        weight = np.ones(len(exp))
        exp = append_fields(exp, "weight", weight, usemask=False, dtypes=[float]).copy()
        return exp

    def get_background_model(self) -> dict:
        """Generic Function to return background model. This could be
        the experimental data (if the signal contamination is small),
        or a weighted MC dataset. By default, uses data."""
        return self.data_background_model()

    def generate_trial_dataset(self):
        if self.loaded_trial_model is None:
            self.load_trial_model()
        return np.copy(self.loaded_trial_model)

    def use_data_for_trials(self):
        if self.__class__.get_background_model == Season.get_background_model:
            logger.warning(
                "This season is already set to generate trials using scrambled data. "
                "No need to set it again!"
            )
        else:
            self.get_trial_model = self.get_exp_data
            logger.info("Set trial model to use scrambled data.")

    def pseudo_background(self):
        """Scrambles the raw dataset to "blind" the data. Assigns a flat Right
        Ascension distribution, and randomly redistributes the arrival times
        in the dataset. Returns a shuffled dataset, which can be used for
        blinded analysis.
        :return: data: The scrambled dataset
        """
        data = self.generate_trial_dataset()
        # Assigns a flat random distribution for Right Ascension
        data["ra"] = np.random.uniform(0, 2 * np.pi, size=len(data))
        # Randomly reorders the times
        np.random.shuffle(data["time"])
        return np.array(data[list(self.get_background_dtype().names)].copy())[:,]

    def simulate_background(self):
        data = self.pseudo_background()
        if self._subselection_fraction is not None:
            data = np.random.choice(data, int(len(data) * self._subselection_fraction))
        return data

    def get_exp_data(self, **kwargs):
        return np.array(self.load_data(self.exp_path, **kwargs))

    def build_time_pdf_dict(self):
        """Function to build a pdf for the livetime of the season. By
        default, this is assumed to be uniform, spanning from the first to
        the last event found in the data.

        :return: Time pdf dictionary
        """
        exp = self.load_data(self.exp_path)
        t0 = min(exp["time"])
        t1 = max(exp["time"])

        t_pdf_dict = {
            "time_pdf_name": "fixed_end_box",
            "start_time_mjd": t0,
            "end_time_mjd": t1,
        }

        return t_pdf_dict

    def build_time_pdf(self):
        t_pdf_dict = self.build_time_pdf_dict()
        time_pdf = TimePDF.create(t_pdf_dict)

        compatible_time_pdfs = [FixedEndBox, FixedRefBox, DetectorOnOffList]
        if np.sum([isinstance(time_pdf, x) for x in compatible_time_pdfs]) == 0:
            raise ValueError(
                "Attempting to use a time PDF that is not an "
                "allowed time PDF class. Only {0} are allowed, "
                " as these PDFs have well-defined start and "
                "end points. Please prove one of these as a "
                "background_time_pdf for the simulation.".format(compatible_time_pdfs)
            )
        return time_pdf

    def get_time_pdf(self):
        if self._time_pdf is None:
            self._time_pdf = self.build_time_pdf()
        return self._time_pdf

    def clean_season_cache(self):
        self._time_pdf = None

    def load_data(self, path, **kwargs):
        return np.load(path)

    def make_injector(self, sources, **inj_kwargs):
        pass

    def return_name(self):
        return self.sample_name + "/" + self.season_name

    def make_background_spatial(self):
        make_background_spline(self)

    def get_pseudo_mc(self, **kwargs):
        return self.load_data(self.pseudo_mc_path, **kwargs)

    def check_files_exist(self):
        all_paths = []

        for x in self.all_paths:
            if isinstance(x, list):
                all_paths += x
            else:
                all_paths.append(x)

        for x in all_paths:
            if not os.path.isfile(x):
                raise Exception("File Not Found: {0}".format(x))
            else:
                print("Found:", x)

    def check_data_quality(self):
        pass

    # def make_acceptance_function(self, acc_path):
    #     make_acceptance_season(self, acc_path)


class SeasonWithMC(Season):
    def __init__(self, season_name, sample_name, exp_path, mc_path, **kwargs):
        Season.__init__(self, season_name, sample_name, exp_path, **kwargs)
        self.mc_path = mc_path
        self.all_paths.append(self.mc_path)
        self.pseudo_mc_path = mc_path

    def make_injector(self, sources, **inj_kwargs):
        return MCInjector.create(self, sources, **inj_kwargs)

    def get_mc(self, **kwargs):
        return self.load_data(self.mc_path, **kwargs)


class SeasonWithoutMC(Season):
    def __init__(self, season_name, sample_name, exp_path, pseudo_mc_path, **kwargs):
        Season.__init__(self, season_name, sample_name, exp_path, **kwargs)
        self.pseudo_mc_path = pseudo_mc_path
        self.all_paths.append(self.pseudo_mc_path)

    def make_injector(self, sources, **inj_kwargs):
        return EffectiveAreaInjector.create(self, sources, **inj_kwargs)

    def load_effective_area(self):
        return

    def load_angular_resolution(self):
        return

    def load_energy_proxy_mapping(self):
        """Function to construct the mapping between energy proxy and true
        energy. By default, this is a simple 1:1 mapping.

        :return: function mapping true energy to energy proxy
        """
        return lambda x: np.log10(x)
