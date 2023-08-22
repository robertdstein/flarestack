import logging
import numexpr
import os
import numpy as np
import scipy.interpolate
from scipy import sparse
import pickle
from flarestack.shared import (
    acceptance_path,
    llh_energy_hash_pickles,
    SoB_spline_path,
    default_gamma_precision,
    default_smoothing_order,
)
from flarestack.core.time_pdf import TimePDF, read_t_pdf_dict
from flarestack.utils.make_SoB_splines import load_spline
from flarestack.core.energy_pdf import EnergyPDF, read_e_pdf_dict
from flarestack.core.spatial_pdf import SpatialPDF
from flarestack.utils.create_acceptance_functions import (
    dec_range,
    make_acceptance_season,
)
from flarestack.utils.make_SoB_splines import (
    create_2d_ratio_hist,
    make_2d_spline_from_hist,
    make_individual_spline_set,
)

logger = logging.getLogger(__name__)

spatial_mask_threshold = 1e-21
default_spacial_box_width = 5.0  # box width around source positions in degrees


def read_llh_dict(llh_dict):
    """Ensures that llh dictionaries remain backwards-compatible

    :param llh_dict: LLH Dictionary
    :return: LLH Dictionary compatible with new format
    """

    maps = [
        ("name", "llh_name"),
        ("LLH Time PDF", "llh_sig_time_pdf"),
        ("LLH Energy PDF", "llh_energy_pdf"),
        ("llh_sig_energy_pdf", "llh_energy_pdf"),
        ("Fit Negative n_s?", "negative_ns_bool"),
        ("llh_time_pdf", "llh_sig_time_pdf"),
    ]

    for old_key, new_key in maps:
        if old_key in list(llh_dict.keys()):
            logger.warning(
                "Deprecated llh_dict key'{0}' was used. Please use '{1}' in future.".format(
                    old_key, new_key
                )
            )
            llh_dict[new_key] = llh_dict[old_key]

    pairs = [("llh_energy_pdf", read_e_pdf_dict), ("llh_sig_time_pdf", read_t_pdf_dict)]

    for key, f in pairs:
        if key in list(llh_dict.keys()):
            llh_dict[key] = f(llh_dict[key])
        else:
            llh_dict[key] = {}

    if "llh_spatial_pdf" not in llh_dict.keys():
        llh_dict["llh_spatial_pdf"] = {}

    if "llh_bkg_time_pdf" not in llh_dict.keys():
        logger.warning(
            "No 'llh_bkg_time_pdf' was specified. A 'steady' pdf will be assumed."
        )
        llh_dict["llh_bkg_time_pdf"] = {"time_pdf_name": "steady"}

    return llh_dict


class LLH(object):
    """Base class LLH."""

    subclasses: dict[str, object] = {}

    def __init__(self, season, sources, llh_dict):
        self.season = season
        self.sources = sources
        self.llh_dict = llh_dict
        self.spatial_pdf = SpatialPDF(llh_dict["llh_spatial_pdf"], season)
        self.spatial_box_width = llh_dict.get(
            "spatial_box_width", default_spacial_box_width
        )

        try:
            time_dict = llh_dict["llh_sig_time_pdf"]
            self.sig_time_pdf = TimePDF.create(time_dict, season.get_time_pdf())
        except KeyError:
            raise KeyError(
                "No Signal Time PDF specified. Please add an "
                "'llh_sig_time_pdf' entry to the llh_dict, and try "
                "again. If you do not want time dependence in your "
                "likelihood, please specify a 'steady' Time PDF."
            )

        try:
            time_dict = llh_dict["llh_bkg_time_pdf"]
            self.bkg_time_pdf = TimePDF.create(time_dict, season.get_time_pdf())
        except KeyError:
            raise KeyError(
                "No Background Time PDF specified. Please add an "
                "'llh_bkg_time_pdf' entry to the llh_dict, and try "
                "again. If you do not want time dependence in your "
                "likelihood, please specify a 'steady' Time PDF."
            )

        self.acceptance, self.energy_weight_f = self.create_energy_functions()

    @classmethod
    def register_subclass(cls, llh_name):
        """Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """

        def decorator(subclass):
            cls.subclasses[llh_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, season, sources, llh_dict):
        llh_dict = read_llh_dict(llh_dict)
        llh_name = llh_dict["llh_name"]

        if llh_name not in cls.subclasses:
            raise ValueError("Bad LLH name {}".format(llh_name))

        return cls.subclasses[llh_name](season, sources, llh_dict)

    @classmethod
    def get_parameters(cls, llh_dict):
        llh_name = llh_dict["llh_name"]

        if llh_name not in cls.subclasses:
            raise ValueError("Bad LLH name {}".format(llh_name))

        return cls.subclasses[llh_name].return_llh_parameters(llh_dict)

    @classmethod
    def get_injected_parameters(cls, mh_dict):
        llh_name = mh_dict["llh_dict"]["llh_name"]

        if llh_name not in cls.subclasses:
            raise ValueError("Bad LLH name {}".format(llh_name))

        return cls.subclasses[llh_name].return_injected_parameters(mh_dict)

    # ==========================================================================
    # Signal PDF
    # ==========================================================================

    def signal_pdf(self, source, cut_data):
        """Calculates the value of the signal spatial PDF for a given source
        for each event in the coincident data subsample. If there is a Time PDF
        given, also calculates the value of the signal Time PDF for each event.
        Returns either the signal spatial PDF values, or the product of the
        signal spatial and time PDFs.

        :param source: Source to be considered
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Signal Spacetime PDF values
        """
        space_term = self.spatial_pdf.signal_spatial(source, cut_data)

        if hasattr(self, "sig_time_pdf"):
            time_term = self.sig_time_pdf.f(cut_data["time"], source)

            sig_pdf = space_term * time_term

        else:
            sig_pdf = space_term

        return sig_pdf

    # @staticmethod
    # def signal(source, cut_data):
    #     """Calculates the angular distance between the source and the
    #     coincident dataset. Uses a Gaussian PDF function, centered on the
    #     source. Returns the value of the Gaussian at the given distances.
    #
    #     :param source: Single Source
    #     :param cut_data: Subset of Dataset with coincident events
    #     :return: Array of Spatial PDF values
    #     """
    #     distance = flarestack.core.astro.angular_distance(
    #         cut_data['ra'], cut_data['dec'],
    #         source['ra_rad'], source['dec_rad']
    #     )
    #     space_term = (1. / (2. * np.pi * cut_data['sigma'] ** 2.) *
    #                   np.exp(-0.5 * (distance / cut_data['sigma']) ** 2.))
    #
    #     return space_term

    # ==========================================================================
    # Background PDF
    # ==========================================================================

    def background_pdf(self, source, cut_data):
        """Calculates the value of the background spatial PDF for a given
        source for each event in the coincident data subsample. Thus is done
        by calling the self.bkg_spline spline function, which was fitted to
        the Sin(Declination) distribution of the data.

        If there is a signal Time PDF given, then the background time PDF
        is also calculated for each event. This is assumed to be a normalised
        uniform distribution for the season.

        Returns either the background spatial PDF values, or the product of the
        background spatial and time PDFs.

        :param source: Source to be considered
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Background Spacetime PDF values
        """
        space_term = self.spatial_pdf.background_spatial(cut_data)

        if hasattr(self, "sig_time_pdf"):
            time_term = self.bkg_time_pdf.f(cut_data["time"], source)
            sig_pdf = space_term * time_term
        else:
            sig_pdf = space_term

        return sig_pdf

    # def acceptance(self, source, params=None):
    #     """Calculates the detector acceptance for a given source, using the
    #     1D interpolation of the acceptance as a function of declination based
    #     on the IC data rate. This is a crude estimation.
    #
    #     :param source: Source to be considered
    #     :param params: Parameter array
    #     :return: Value for the acceptance of the detector, in the given
    #     season, for the source
    #     """
    #     return self.acceptance_f(source, params)

    def create_energy_functions(self):
        """Creates the acceptance function, which parameterises signal
        acceptance as a function of declination, and the energy weighting
        function, which gives the energy signal-over-background ratio

        :return: Acceptance function, energy_weighting_function
        """

        def acc_f(source, params=None):
            return 1.0

        def energy_weight_f(event, params=None):
            return 1.0

        return acc_f, energy_weight_f

    def select_spatially_coincident_data(self, data, sources):
        """Checks each source, and only identifies events in data which are
        both spatially and time-coincident with the source. Spatial
        coincidence is defined as a +/- 5 degree box centered on the  given
        source. Time coincidence is determined by the parameters of the LLH
        Time PDF. Produces a mask for the dataset, which removes all events
        which are not coincident with at least one source.

        :param data: Dataset to be tested
        :param sources: Sources to be tested
        :return: Mask to remove
        """
        veto = np.ones_like(data["ra"], dtype=bool)

        for source in sources:
            # Sets half width of spatial box
            width = np.deg2rad(self.spatial_box_width)

            # Sets a declination band 5 degrees above and below the source
            min_dec = max(-np.pi / 2.0, source["dec_rad"] - width)
            max_dec = min(np.pi / 2.0, source["dec_rad"] + width)

            # Accepts events lying within a 5 degree band of the source
            dec_mask = np.logical_and(
                np.greater(data["dec"], min_dec), np.less(data["dec"], max_dec)
            )

            # Sets the minimum value of cos(dec)
            cos_factor = np.amin(np.cos([min_dec, max_dec]))

            # Scales the width of the box in ra, to give a roughly constant
            # area. However, if the width would have to be greater that +/- pi,
            # then sets the area to be exactly 2 pi.
            dPhi = np.amin([2.0 * np.pi, 2.0 * width / cos_factor])

            # Accounts for wrapping effects at ra=0, calculates the distance
            # of each event to the source.
            ra_dist = np.fabs(
                (data["ra"] - source["ra_rad"] + np.pi) % (2.0 * np.pi) - np.pi
            )
            ra_mask = ra_dist < dPhi / 2.0

            spatial_mask = dec_mask & ra_mask

            veto = veto & ~spatial_mask

        return ~veto

    @staticmethod
    def assume_background(n_s, n_coincident, n_all):
        """To save time with likelihood calculation, it can be assumed that
        all events defined as "non-coincident", because of distance in space
        and time to the source, are in fact background events. This is
        equivalent to setting S=0 for all non-coincident events. IN this
        case, the likelihood can be calculated as the product of the number
        of non-coincident events, and the likelihood of an event which has S=0.

        :param n_s: Array of expected number of events
        :param n_coincident: Number of events that were not assumed to have S=0
        :param n_all: The total number of events
        :return: Log Likelihood value for the given
        """
        return (n_all - n_coincident) * np.log1p(-n_s / n_all)

    def create_kwargs(self, data, pull_corrector, weight_f=None):
        kwargs = dict()
        return kwargs

    def create_llh_function(self, data, pull_corrector, weight_f=None):
        """Creates a likelihood function to minimise, based on the dataset.

        :param data: Dataset
        :return: LLH function that can be minimised
        """

        kwargs = self.create_kwargs(data, pull_corrector, weight_f)

        def test_statistic(params, weights):
            return self.calculate_test_statistic(params, weights, **kwargs)

        return test_statistic

    def calculate_test_statistic(self, params, weights, **kwargs):
        pass

    @staticmethod
    def return_llh_parameters(llh_dict):
        seeds = []
        bounds = []
        names = []

        return seeds, names, bounds

    @staticmethod
    def return_injected_parameters(mh_dict):
        return {}


@LLH.register_subclass("spatial")
class SpatialLLH(LLH):
    """Most basic LLH, in which only spatial, and optionally also temporal,
    information is included. No Energy PDF is used, and no energy weighting
    is applied.

    """

    fit_energy = False

    def __init__(self, season, sources, llh_dict):
        LLH.__init__(self, season, sources, llh_dict)

        if "energy_pdf_name" in list(llh_dict["llh_energy_pdf"]):
            raise Exception(
                "Found 'llh_energy_pdf' entry in llh_dict, "
                "but SpatialLLH does not use Energy PDFs. \n"
                "Please remove this entry, and try again."
            )

    def create_energy_function(self):
        """In the most simple case of spatial-only weighting, you would
        neglect the energy weighting of events. Then, you can simply assume
        that the detector acceptance is roughly proportional to the data rate,
        i.e assuming that the incident background atmospheric neutrino flux
        is uniform. Thus the acceptance of the detector is simply the
        background spatial PDF (which is a spline fitted to data as a
        function of declination). This method does, admittedly neglect the
        fact that background in the southern hemisphere is mainly composed of
        muon bundles, rather than atmospheric neutrinos. Still, it's slighty
        better than assuming a uniform detector acceptance

        :return: 1D linear interpolation
        """
        exp = self.season.get_background_model()
        data_rate = float(len(exp))
        del exp

        # return lambda x: data_rate
        return lambda x: np.exp(self.bkg_spatial(np.sin(x))) * data_rate

    def create_llh_function(self, data, pull_corrector, weight_f=None):
        """Creates a likelihood function to minimise, based on the dataset.

        :param data: Dataset
        :param pull_corrector: pull_corrector
        :return: LLH function that can be minimised
        """
        n_all = float(len(data))
        SoB_spacetime = []

        assumed_bkg_mask = np.ones(len(data), dtype=bool)

        for i, source in enumerate(self.sources):
            s_mask = self.select_spatially_coincident_data(data, [source])

            assumed_bkg_mask *= ~s_mask
            coincident_data = data[s_mask]

            if len(coincident_data) > 0:
                sig = self.signal_pdf(source, coincident_data)

                bkg = np.array(self.background_pdf(source, coincident_data))

                SoB_spacetime.append(sig / bkg)
                del sig
                del bkg

        n_coincident = np.sum(~assumed_bkg_mask)

        SoB_spacetime = np.array(SoB_spacetime)

        def test_statistic(params, weights):
            return self.calculate_test_statistic(
                params,
                weights,
                n_all=n_all,
                n_coincident=n_coincident,
                SoB_spacetime=SoB_spacetime,
            )

        return test_statistic

    def calculate_test_statistic(self, params, weights, **kwargs):
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from Minimisation
        :param weights: Normalised fraction of n_s allocated to each source
        :return: 2 * llh value (Equal to Test Statistic)
        """

        n_s = np.array(params)

        # Calculates the expected number of signal events for each source in
        # the season
        all_n_j = n_s * weights.T[0]

        x = []

        for i, n_j in enumerate(all_n_j):
            x.append(1 + (n_j / kwargs["n_all"]) * (kwargs["SoB_spacetime"][i] - 1.0))

        if np.sum([np.sum(x_row <= 0.0) for x_row in x]) > 0:
            llh_value = -50.0 + all_n_j

        else:
            llh_value = np.array([np.sum(np.log(y)) for y in x])

            llh_value += self.assume_background(
                all_n_j, kwargs["n_coincident"], kwargs["n_all"]
            )

            if np.logical_and(
                np.sum(all_n_j) < 0, np.sum(llh_value) < np.sum(-50.0 + all_n_j)
            ):
                llh_value = -50.0 + all_n_j

        # Definition of test statistic
        return 2.0 * np.sum(llh_value)


@LLH.register_subclass("fixed_energy")
class FixedEnergyLLH(LLH):
    fit_energy = False

    def __init__(self, season, sources, llh_dict):
        try:
            e_pdf_dict = llh_dict["llh_energy_pdf"]
            self.energy_pdf = EnergyPDF.create(e_pdf_dict)
        except KeyError:
            raise KeyError(
                "LLH with energy term selected, but no energy PDF "
                "has been provided. Please add an 'llh_energy_pdf' "
                "dictionary to the LLH dictionary, and try "
                "again."
            )

        # defines the order of the splines used in the building of the energy PDF
        self.smoothing_order = None
        smoothing_order = llh_dict.get("smoothing_order", "flarestack")
        if isinstance(smoothing_order, str):
            self.smoothing_order = default_smoothing_order[smoothing_order]
        else:
            self.smoothing_order = smoothing_order

        # used to construct the support points in gamma when the energy PDF is built
        self.precision = None
        precision = llh_dict.get("gamma_precision", "flarestack")
        if isinstance(precision, str):
            self.precision = default_gamma_precision[precision]
        else:
            self.precision = precision

        LLH.__init__(self, season, sources, llh_dict)

    def create_energy_functions(self):
        """Creates the acceptance function, which parameterises signal
        acceptance as a function of declination, and the energy weighting
        function, which gives the energy signal-over-background ratio

        :return: Acceptance function, energy_weighting_function
        """

        SoB_path, acc_path = llh_energy_hash_pickles(self.llh_dict, self.season)

        # Set up acceptance function, creating values if they have not been
        # created before

        if not os.path.isfile(acc_path):
            self.create_acceptance_function(acc_path)

        logger.debug("Loading from {0}".format(acc_path))

        with open(acc_path, "rb") as f:
            [dec_vals, acc] = pickle.load(f)

        acc_spline = scipy.interpolate.interp1d(dec_vals, acc, kind="linear")

        def acc_f(source, params=None):
            return acc_spline(source["dec_rad"])

        # Sets up energy weighting function, creating values if they have not
        # been created before

        if not os.path.isfile(SoB_path):
            self.create_energy_weighting_function(SoB_path)

        logger.debug("Loading from {0}".format(SoB_path))

        with open(SoB_path, "rb") as f:
            [dec_vals, ratio_hist] = pickle.load(f)

        spline = make_2d_spline_from_hist(
            np.array(ratio_hist), dec_vals, self.season.log_e_bins, self.smoothing_order
        )

        def energy_weight_f(event, params=None):
            return np.exp(spline(event["logE"], event["sinDec"], grid=False))

        return acc_f, energy_weight_f

    def create_acceptance_function(self, acc_path):
        logger.info(
            "Building acceptance functions in sin(dec) bins "
            "(with fixed energy weighting)"
        )

        mc = self.season.get_pseudo_mc()

        acc = np.ones_like(dec_range, dtype=float)

        for i, dec in enumerate(dec_range):
            # Sets half width of band
            dec_width = np.deg2rad(5.0)

            # Sets a declination band 5 degrees above and below the source
            min_dec = max(-np.pi / 2.0, dec - dec_width)
            max_dec = min(np.pi / 2.0, dec + dec_width)
            # Gives the solid angle coverage of the sky for the band
            omega = 2.0 * np.pi * (np.sin(max_dec) - np.sin(min_dec))

            band_mask = np.logical_and(
                np.greater(mc["trueDec"], min_dec), np.less(mc["trueDec"], max_dec)
            )

            cut_mc = mc[band_mask]
            weights = self.energy_pdf.weight_mc(cut_mc)
            acc[i] = np.sum(weights / omega)

        del mc

        try:
            os.makedirs(os.path.dirname(acc_path))
        except OSError:
            pass

        logger.info("Saving to {0}".format(acc_path))

        with open(acc_path, "wb") as f:
            pickle.dump([dec_range, acc], f)

        return f

    def create_energy_weighting_function(self, SoB_path):
        logger.info(
            "Building energy-weighting functions in sin(dec) vs log E bins "
            "(with fixed energy weighting)"
        )

        # dec_range = self.season["sinDec bins"]

        ratio_hist = create_2d_ratio_hist(
            exp=self.season.get_background_model(),
            mc=self.season.get_pseudo_mc(),
            sin_dec_bins=dec_range,
            log_e_bins=self.season.log_e_bins,
            weight_function=self.energy_pdf.weight_mc,
        )

        try:
            os.makedirs(os.path.dirname(SoB_path))
        except OSError:
            pass

        logger.info("Saving to {0}".format(SoB_path))

        with open(SoB_path, "wb") as f:
            pickle.dump([dec_range, ratio_hist], f)

    def create_kwargs(self, data, pull_corrector, weight_f=None):
        """Creates a likelihood function to minimise, based on the dataset.

        :param data: Dataset
        :return: LLH function that can be minimised
        """
        kwargs = dict()
        kwargs["n_all"] = float(len(data))
        SoB = []

        assumed_bkg_mask = np.ones(len(data), dtype=bool)

        for i, source in enumerate(self.sources):
            s_mask = self.select_spatially_coincident_data(data, [source])

            assumed_bkg_mask *= ~s_mask
            coincident_data = data[s_mask]

            if len(coincident_data) > 0:
                sig = self.signal_pdf(source, coincident_data)
                bkg = np.array(self.background_pdf(source, coincident_data))

                SoB_energy_ratio = self.energy_weight_f(coincident_data)

                SoB.append(SoB_energy_ratio * sig / bkg)

        kwargs["n_coincident"] = np.sum(~assumed_bkg_mask)

        kwargs["SoB"] = np.array(SoB)
        return kwargs

    def calculate_test_statistic(self, params, weights, **kwargs):
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from Minimisation
        :param weights: Normalised fraction of n_s allocated to each source
        :return: 2 * llh value (Equal to Test Statistic)
        """
        n_s = np.array(params)

        # Calculates the expected number of signal events for each source in
        # the season
        all_n_j = n_s * weights.T[0]

        x = []

        for i, n_j in enumerate(all_n_j):
            x.append(1 + (n_j / kwargs["n_all"]) * (kwargs["SoB"][i] - 1.0))

        if np.sum([np.sum(x_row <= 0.0) for x_row in x]) > 0:
            llh_value = -50.0 + all_n_j

        else:
            llh_value = np.array([np.sum(np.log(y)) for y in x])

            llh_value += self.assume_background(
                all_n_j, kwargs["n_coincident"], kwargs["n_all"]
            )

            if np.logical_and(
                np.sum(all_n_j) < 0, np.sum(llh_value) < np.sum(-50.0 + all_n_j)
            ):
                llh_value = -50.0 + all_n_j

        # Definition of test statistic
        return 2.0 * np.sum(llh_value)


@LLH.register_subclass("standard")
class StandardLLH(FixedEnergyLLH):
    fit_energy = True

    def __init__(self, season, sources, llh_dict):
        FixedEnergyLLH.__init__(self, season, sources, llh_dict)

        # Bins for energy Log(E/GeV)
        self.energy_bins = np.linspace(1.0, 10.0, 40 + 1)

        # Sets precision for energy SoB
        # self.precision = .1

        self.SoB_spline_2Ds = load_spline(
            self.season,
            smoothing_order=self.smoothing_order,
            gamma_precision=self.precision,
        )

        if self.SoB_spline_2Ds:
            logger.debug("Loaded {0} splines.".format(len(self.SoB_spline_2Ds)))
        else:
            logger.warning("Didn't load SoB splines!!!")

        logger.debug("Loaded {0} splines.".format(len(self.SoB_spline_2Ds)))

        self.acceptance_f = self.create_acceptance_function()
        self.acceptance = self.new_acceptance

        # self.SoB_energy_cache = []

    def _around(self, value):
        """Produces an array in which the precision of the value
        is rounded to the nearest integer. This is then multiplied
        by the precision, and the new value is returned.

        :param value: value to be processed
        :return: value after processed
        """
        return np.around(float(value) / self.precision) * self.precision

    def create_energy_functions(self):
        """Creates the acceptance function, which parameterises signal
        acceptance as a function of declination, and the energy weighting
        function, which gives the energy signal-over-background ratio

        :return: Acceptance function, energy_weighting_function
        """

        SoB_path = SoB_spline_path(
            self.season,
            smoothing_order=self.smoothing_order,
            gamma_precision=self.precision,
        )
        acc_path = acceptance_path(self.season)

        # Set up acceptance function, creating values if they have not been
        # created before

        if not os.path.isfile(acc_path):
            make_acceptance_season(self.season, acc_path)

        logger.debug("Loading from {0}".format(acc_path))

        with open(acc_path, "rb") as f:
            [dec_bins, gamma_bins, acc] = pickle.load(f)

        # acc_spline = scipy.interpolate.interp2d(
        #     dec_bins, gamma_bins, np.array(acc).T, kind='linear')
        #
        # def acc_f(source, params):
        #     return acc_spline(source["dec_rad"], params[-1])

        acc_f = None

        # Checks if energy weighting functions have been created

        if not os.path.isfile(SoB_path):
            make_individual_spline_set(
                self.season,
                SoB_path,
                smoothing_order=self.smoothing_order,
                gamma_precision=self.precision,
            )

        return acc_f, None

    # def create_acceptance_function(self, acc_path):
    #     """Creates a 2D linear interpolation of the acceptance of the detector
    #     for the given season, as a function of declination and gamma. Returns
    #     this interpolation function.
    #
    #     :return: 2D linear interpolation
    #     """
    #
    #     # acc_path = acceptance_path(self.season)
    #
    #     with open(acc_path) as f:
    #         acc_dict = pickle.load(f)
    #
    #     dec_bins = acc_dict["dec"]
    #     gamma_bins = acc_dict["gamma"]
    #     values = acc_dict["acceptance"]
    #     f = scipy.interpolate.interp2d(
    #         dec_bins, gamma_bins, values.T, kind='linear')
    #     return f

    def create_acceptance_function(self):
        """Creates a 2D linear interpolation of the acceptance of the detector
        for the given season, as a function of declination and gamma. Returns
        this interpolation function.

        :return: 2D linear interpolation
        """

        acc_path = acceptance_path(self.season)

        with open(acc_path, "rb") as f:
            [dec_bins, gamma_bins, acc] = pickle.load(f)

        f = scipy.interpolate.interp2d(dec_bins, gamma_bins, acc.T, kind="linear")
        return f

    def new_acceptance(self, source, params=None):
        """Calculates the detector acceptance for a given source, using the
        2D interpolation of the acceptance as a function of declination and
        gamma. If gamma IS NOT being fit, uses the default value of gamma for
        weighting (determined in __init__). If gamma IS being fit, it will be
        the last entry in the parameter array, and is the acceptance uses
        this value.

        :param source: Source to be considered
        :param params: Parameter array
        :return: Value for the acceptance of the detector, in the given
        season, for the source
        """
        dec = source["dec_rad"]
        gamma = params[-1]

        return self.acceptance_f(dec, gamma)

    def create_kwargs(self, data, pull_corrector, weight_f=None):
        kwargs = dict()

        kwargs["n_all"] = float(len(data))
        SoB_spacetime = []
        SoB_energy_cache = []

        assumed_background_mask = np.ones(len(data), dtype=bool)

        for i, source in enumerate(self.sources):
            s_mask = self.select_spatially_coincident_data(data, [source])

            coincident_data = data[s_mask].copy()

            if len(coincident_data) > 0:
                # Only bother accepting neutrinos where the spatial
                # likelihood is greater than 1e-21. This prevents 0s
                # appearing in dynamic pull corrections, but also speeds
                # things up (those neutrinos won't contribute anything to the
                # likelihood!)

                sig = self.signal_pdf(source, coincident_data)
                nonzero_mask = sig > spatial_mask_threshold

                s_mask[s_mask] *= nonzero_mask

                assumed_background_mask *= ~s_mask
                coincident_data = data[s_mask]

                if len(coincident_data) > 0:
                    SoB_pdf = lambda x: self.signal_pdf(
                        source, x
                    ) / self.background_pdf(source, x)

                    spatial_cache = pull_corrector.create_spatial_cache(
                        coincident_data, SoB_pdf
                    )

                    SoB_spacetime.append(spatial_cache)

                    energy_cache = self.create_SoB_energy_cache(coincident_data)

                    SoB_energy_cache.append(energy_cache)

                else:
                    SoB_spacetime.append([])
                    SoB_energy_cache.append([])

            else:
                SoB_spacetime.append([])
                SoB_energy_cache.append([])

        kwargs["n_coincident"] = np.sum(~assumed_background_mask)

        # SoB_spacetime contains one list per source.
        # The list contains one value for each associated neutrino.
        kwargs["SoB_spacetime_cache"] = SoB_spacetime
        kwargs["SoB_energy_cache"] = SoB_energy_cache
        kwargs["pull_corrector"] = pull_corrector

        return kwargs

    def calculate_test_statistic(self, params, weights, **kwargs) -> float:
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from Minimisation
        :param weights: Normalised fraction of n_s allocated to each source
        :return: 2 * llh value (Equal to Test Statistic)
        """
        n_s = np.array(params[:-1])
        gamma = params[-1]

        # Calculates the expected number of signal events for each source in
        # the season
        all_n_j = n_s * weights.T[0]

        x = []

        # If n_s if negative, then removes the energy term from the likelihood

        for i, n_j in enumerate(all_n_j):
            SoB_spacetime_data: list = kwargs["SoB_spacetime_cache"][i]

            # Switches off Energy term for negative n_s, which should in theory
            # be a continuous change that does not alter the likelihood for
            # n_s > 0 (as it is not included for n_s=0).

            if len(SoB_spacetime_data) == 0:
                x.append(np.array([1.0]))

            else:
                SoB_spacetime = kwargs["pull_corrector"].estimate_spatial(
                    gamma, SoB_spacetime_data
                )

                if n_j < 0:
                    x.append(1.0 + (n_j / kwargs["n_all"]) * (SoB_spacetime - 1.0))

                else:
                    SoB_energy = self.estimate_energy_weights(
                        gamma, kwargs["SoB_energy_cache"][i]
                    )

                    x.append(
                        1.0
                        + (
                            (n_j / kwargs["n_all"])
                            * ((SoB_energy * SoB_spacetime) - 1.0)
                        )
                    )

        if np.sum([np.sum(x_row <= 0.0) for x_row in x]) > 0:
            llh_value = -50.0 + all_n_j

        else:
            llh_value = np.sum([np.sum(np.log(y)) for y in x])

            llh_value += np.sum(
                self.assume_background(
                    np.sum(all_n_j), kwargs["n_coincident"], kwargs["n_all"]
                )
            )

            if np.logical_and(
                np.sum(all_n_j) < 0, np.sum(llh_value) < np.sum(-50.0 + all_n_j)
            ):
                llh_value = -50.0 + all_n_j

        # Definition of test statistic
        return 2.0 * np.sum(llh_value)

    # ==============================================================================
    # Energy Log(Signal/Background) Ratio
    # ==============================================================================

    def create_SoB_energy_cache(self, cut_data):
        """Evaluates the Log(Signal/Background) values for all coincident
        data. For each value of gamma in self.gamma_support_points, calculates
        the Log(Signal/Background) values for the coincident data. Then saves
        each weight array to a dictionary.

        :param cut_data: Subset of the data containing only coincident events
        :return: Dictionary containing SoB values for each event for each
        gamma value.
        """

        energy_SoB_cache = dict()

        for gamma in list(self.SoB_spline_2Ds.keys()):
            try:
                energy_SoB_cache[gamma] = self.SoB_spline_2Ds[gamma].ev(
                    cut_data["logE"], cut_data["sinDec"]
                )
            except:  # this is in case the splines were produced using the RegularGridInterpolator
                energy_SoB_cache[gamma] = self.SoB_spline_2Ds[gamma](
                    (cut_data["logE"], cut_data["sinDec"])
                )

        return energy_SoB_cache

    def estimate_energy_weights(self, gamma, energy_SoB_cache):
        """Quickly estimates the value of Signal/Background for Gamma.
        Uses pre-calculated values for first and second derivatives.
        Uses a Taylor series to estimate S(gamma), unless SoB has already
        been calculated for a given gamma.

        :param gamma: Spectral Index
        :param energy_SoB_cache: Weight cache
        :return: Estimated value for S(gamma)
        """
        if gamma in list(energy_SoB_cache.keys()):
            val = np.exp(energy_SoB_cache[gamma])
        else:
            g1 = self._around(gamma)
            dg = self.precision

            g0 = self._around(g1 - dg)
            g2 = self._around(g1 + dg)

            # Uses Numexpr to quickly estimate S(gamma)

            S0 = energy_SoB_cache[g0]
            S1 = energy_SoB_cache[g1]
            S2 = energy_SoB_cache[g2]

            val = numexpr.evaluate(
                "exp((S0 - 2.*S1 + S2) / (2. * dg**2) * (gamma - g1)**2"
                + " + (S2 -S0) / (2. * dg) * (gamma - g1) + S1)"
            )

        return val

    @staticmethod
    def return_llh_parameters(llh_dict):
        e_pdf = EnergyPDF.create(llh_dict["llh_energy_pdf"])
        return e_pdf.return_energy_parameters()

    @staticmethod
    def return_injected_parameters(mh_dict):
        try:
            inj = mh_dict["inj_dict"]["injection_energy_pdf"]
            llh = mh_dict["llh_dict"]["llh_energy_pdf"]

            if inj["energy_pdf_name"] == llh["energy_pdf_name"]:
                e_pdf = EnergyPDF.create(inj)
                return e_pdf.return_injected_parameters()

        except KeyError:
            pass

        seeds, bounds, names = LLH.get_parameters(mh_dict["llh_dict"])

        res_dict = {}
        for key in names:
            res_dict[key] = np.nan

        return res_dict


@LLH.register_subclass("standard_overlapping")
class StandardOverlappingLLH(StandardLLH):
    def create_kwargs(self, data, pull_corrector, weight_f=None):
        if weight_f is None:
            raise Exception(
                "Weight function not passed, but is required for "
                "standard_overlapping LLH functions."
            )

        season_weight = lambda x: weight_f([1.0, x], self.season)

        kwargs = dict()

        kwargs["n_all"] = float(len(data))

        assumed_background_mask = np.ones(len(data), dtype=bool)

        for i, source in enumerate(self.sources):
            s_mask = self.select_spatially_coincident_data(data, [source])

            coincident_data = data[s_mask]

            if len(coincident_data) > 0:
                # Only bother accepting neutrinos where the spacial
                # likelihood is greater than 1e-21. This prevents 0s
                # appearing in dynamic pull corrections, but also speeds
                # things up (those neutrinos won't contribute anything to the
                # likelihood!)

                sig = self.signal_pdf(source, coincident_data)
                nonzero_mask = sig > spatial_mask_threshold
                s_mask[s_mask] *= nonzero_mask

                assumed_background_mask *= ~s_mask

        coincident_data = data[~assumed_background_mask]

        SoB_energy_cache = self.create_SoB_energy_cache(coincident_data)

        def joint_SoB(dataset, gamma):
            return np.sum(
                [
                    np.array(
                        season_weight(gamma)[i]
                        * self.signal_pdf(source, dataset)
                        / self.background_pdf(source, dataset)
                    )
                    for i, source in enumerate(self.sources)
                ],
                axis=0,
            ) / np.sum(season_weight(gamma))

        SoB_spacetime = pull_corrector.create_spatial_cache(coincident_data, joint_SoB)

        kwargs["n_coincident"] = np.sum(~assumed_background_mask)
        kwargs["SoB_spacetime_cache"] = SoB_spacetime
        kwargs["SoB_energy_cache"] = SoB_energy_cache
        kwargs["pull_corrector"] = pull_corrector

        return kwargs

    def calculate_test_statistic(self, params, weights, **kwargs):
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from Minimisation
        :param weights: Normalised fraction of n_s allocated to each source
        :return: 2 * llh value (Equal to Test Statistic)
        """
        n_s = np.array(params[:-1])
        gamma = params[-1]

        SoB_spacetime = kwargs["pull_corrector"].estimate_spatial(
            gamma, kwargs["SoB_spacetime_cache"]
        )

        SoB_energy = self.estimate_energy_weights(gamma, kwargs["SoB_energy_cache"])

        # Calculates the expected number of signal events for each source in
        # the season
        n_j = n_s * np.sum(weights)

        # If n_s if negative, then removes the energy term from the likelihood

        # Switches off Energy term for negative n_s, which should in theory
        # be a continuous change that does not alter the likelihood for
        # n_s > 0 (as it is not included for n_s=0).
        if n_j < 0.0:
            x = 1.0 + ((n_j / kwargs["n_all"]) * (SoB_spacetime - 1.0))
        else:
            x = 1.0 + ((n_j / kwargs["n_all"]) * ((SoB_energy * SoB_spacetime) - 1.0))

        llh_value = np.sum(np.log(x))

        llh_value += self.assume_background(
            n_j, kwargs["n_coincident"], kwargs["n_all"]
        )

        # Definition of test statistic
        return 2.0 * np.sum(llh_value)


@LLH.register_subclass("standard_matrix")
class StandardMatrixLLH(StandardOverlappingLLH):
    def create_kwargs(self, data, pull_corrector, weight_f=None):
        if weight_f is None:
            raise Exception(
                "Weight function not passed, but is required for "
                "standard_overlapping LLH functions."
            )

        coincidence_matrix = sparse.lil_matrix(
            (len(self.sources), len(data)), dtype=bool
        )

        kwargs = dict()

        kwargs["n_all"] = float(len(data))

        sources = self.sources

        for i, source in enumerate(sources):
            s_mask = self.select_spatially_coincident_data(data, [source])

            coincident_data = data[s_mask]

            if len(coincident_data) > 0:
                # Only bother accepting neutrinos where the spacial
                # likelihood is greater than 1e-21. This prevents 0s
                # appearing in dynamic pull corrections, but also speeds
                # things up (those neutrinos won't contribute anything to the
                # likelihood!)

                sig = self.signal_pdf(source, coincident_data)
                nonzero_mask = sig > spatial_mask_threshold
                s_mask[s_mask] *= nonzero_mask

                coincidence_matrix[i] = s_mask

        # Using Sparse matrixes
        coincident_nu_mask = np.sum(coincidence_matrix, axis=0) > 0
        coincident_nu_mask = np.array(coincident_nu_mask).ravel()
        coincident_source_mask = np.sum(coincidence_matrix, axis=1) > 0
        coincident_source_mask = np.array(coincident_source_mask).ravel()

        coincidence_matrix = (
            coincidence_matrix[coincident_source_mask].T[coincident_nu_mask].T
        )
        coincidence_matrix.tocsr()

        coincident_data = data[coincident_nu_mask]
        coincident_sources = sources[coincident_source_mask]

        season_weight = lambda x: weight_f([1.0, x], self.season)[
            coincident_source_mask
        ]

        SoB_energy_cache = self.create_SoB_energy_cache(coincident_data)

        def joint_SoB(dataset, gamma):
            weight = np.array(season_weight(gamma))
            weight /= np.sum(weight)

            # create an empty lil_matrix (good for matrix creation) with shape
            # of coincidence_matrix and type float
            SoB_matrix_sparse = sparse.lil_matrix(coincidence_matrix.shape, dtype=float)

            for i, src in enumerate(coincident_sources):
                mask = (coincidence_matrix.getrow(i)).toarray()[0]
                SoB_matrix_sparse[i, mask] = (
                    weight[i]
                    * self.signal_pdf(src, dataset[mask])
                    / self.background_pdf(src, dataset[mask])
                )

            SoB_sum = SoB_matrix_sparse.sum(axis=0)
            return_value = np.array(SoB_sum).ravel()

            return return_value

        SoB_spacetime = pull_corrector.create_spatial_cache(coincident_data, joint_SoB)

        kwargs["n_coincident"] = np.sum(coincident_nu_mask)
        kwargs["SoB_spacetime_cache"] = SoB_spacetime
        kwargs["SoB_energy_cache"] = SoB_energy_cache
        kwargs["pull_corrector"] = pull_corrector

        return kwargs


def generate_dynamic_flare_class(season, sources, llh_dict):
    try:
        mh_name = llh_dict["llh_name"]
    except KeyError:
        raise KeyError("No LLH specified.")

    # Set up dynamic inheritance

    try:
        ParentLLH = LLH.subclasses[mh_name]
    except KeyError:
        raise KeyError("Parent class {} not found.".format(mh_name))

    # Defines custom Flare class

    class FlareLLH(ParentLLH):
        def create_flare_llh_function(
            self, data, flare_veto, n_all, src, n_season, pull_corrector
        ):
            coincident_data = data[~flare_veto]
            kwargs = self.create_kwargs(coincident_data, pull_corrector)
            kwargs["n_all"] = n_all
            weights = np.array([1.0])

            def test_statistic(params):
                return self.calculate_test_statistic(params, weights, **kwargs)

            # Super ugly-looking code that magically takes the old llh
            # object, sets the assume_background contribution to zero,
            # and then adds on a new assume_season_background where mutiple
            # datasets are treated as one season of data.

            def combined_test_statistic(params):
                return test_statistic(params) + (
                    2
                    * self.assume_season_background(
                        params[0], np.sum(~flare_veto), n_season, n_all
                    )
                )

            return combined_test_statistic

        @staticmethod
        def assume_background(n_s, n_coincident, n_all):
            """In the standard create_llh_function method that the FlareClass
            inherits, this method will be called. To maintain modularity, we
            simply set it to 0 here. The Flare class treats all neutrino events
            collectively, rather than splitting them by season. As a result,
            the assume_season_background method is called seperately to handle
            the non-coincident neutrinos.

            :param n_s: Array of expected number of events
            :param n_coincident: Number of events that were not assumed to have S=0
            :param n_all: The total number of events
            :return: 0.
            """
            return 0.0

        def signal_pdf(self, source, cut_data):
            """Calculates the value of the signal spatial PDF for a given source
            for each event in the coincident data subsample. If there is a Time PDF
            given, also calculates the value of the signal Time PDF for each event.
            Returns either the signal spatial PDF values, or the product of the
            signal spatial and time PDFs.

            :param source: Source to be considered
            :param cut_data: Subset of Dataset with coincident events
            :return: Array of Signal Spacetime PDF values
            """
            space_term = self.spatial_pdf.signal_spatial(source, cut_data)

            return space_term

        # ==========================================================================
        # Background PDF
        # ==========================================================================

        def background_pdf(self, source, cut_data):
            """For the flare search, generating repeated box time PDFs would
            be required to recalculate the
            """
            space_term = self.spatial_pdf.background_spatial(cut_data)

            return space_term

        @staticmethod
        def assume_season_background(n_s, n_mask, n_season, n_all):
            """To save time with likelihood calculation, it can be assumed that
            all events defined as "non-coincident", because of distance in space
            and time to the source, are in fact background events. This is
            equivalent to setting S=0 for all non-coincident events. IN this
            case, the likelihood can be calculated as the product of the number
            of non-coincident events, and the likelihood of an event which has S=0.

            :param n_s: Array of expected number of events
            :param n_mask: Number of events that were not assumed to have S=0
            :param n_all: The total number of events
            :return: Log Likelihood value for the given
            """
            return (n_season - n_mask) * np.log1p(-n_s / n_all)

        def estimate_significance(self, coincident_data, source):
            """Finds events in the coincident dataset (spatially and temporally
            overlapping sources), which are significant. This is defined as having a
            Signal/Background Ratio that is greater than 1. The S/B ratio is
            calculating using spatial and energy PDFs.

            :param coincident_data: Data overlapping the source spatially/temporally
            :param source: Source to be considered
            :return: SoB of events in coincident dataset
            """
            sig = self.spatial_pdf.signal_spatial(source, coincident_data)
            bkg = self.spatial_pdf.background_spatial(coincident_data)
            SoB_space = sig / bkg

            SoB_energy_cache = self.create_SoB_energy_cache(coincident_data)

            # ChangeME?

            SoB_energy = self.estimate_energy_weights(
                gamma=3.0, energy_SoB_cache=SoB_energy_cache
            )

            SoB = SoB_space * SoB_energy
            return SoB

        def find_significant_events(self, coincident_data, source):
            """Finds events in the coincident dataset (spatially and temporally
            overlapping sources), which are significant. This is defined as having a
            Signal/Background Ratio that is greater than 1. The S/B ratio is
            calculating using spatial and energy PDFs.

            :param coincident_data: Data overlapping the source spatially/temporally
            :param source: Source to be considered
            :return: Significant events in coincident dataset
            """

            SoB = self.estimate_significance(coincident_data, source)

            mask = SoB > 1.0

            return coincident_data[mask]

    return FlareLLH(season, sources, llh_dict)


"""
# NOTE: `mypy` allowed to find some deprecations in the following __main__ block that were never noticed before, because the block is not part of any testing routine. It should be checked whether equivalent code is covered by an unit test, before removing the (now commented) block altogether.

if __name__ == "__main__":


    from flarestack.shared import fs_scratch_dir
    from scipy.interpolate import InterpolatedUnivariateSpline

    g = EnergyPDF.create({"energy_pdf_name": "PowerLaw", "gamma": 2.2})

    e_range = np.logspace(start=0, stop=7, num=1000)

    f = InterpolatedUnivariateSpline(e_range, np.log(g.f(e_range)))

    path = fs_scratch_dir + "tester_spline.npy"

    print(path)

    with open(path, "wb") as h:
        pickle.dump(f, h)

    e_pdf = {
        "energy_pdf_name": "Spline",
        "spline_path": path,
    }

    from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
    from flarestack.utils.prepare_catalogue import ps_catalogue_name
    from flarestack.core.injector import Injector

    llh_dict = {
        "llh_name": "FixedEnergy",
        "llh_time_pdf": {"time_pdf_name": "Steady"},
        "llh_energy_pdf": {"energy_pdf_name": "PowerLaw", "gamma": 2.2}
        # "LLH Energy PDF": e_pdf_dict
    }
    source = np.load(ps_catalogue_name(0.0))

    llh = LLH.create(IC86_1_dict, source, llh_dict)

    # inj = MockUnblindedInjector(IC86_1_dict, source)
    inj = Injector(IC86_1_dict, source)

    data = inj.create_dataset(0.0)
    f = llh.create_llh_function(data)

    weights = np.array([1.0])

    for i in np.linspace(0.0, 10.0, 21):
        print(i, f([i], weights))
"""
