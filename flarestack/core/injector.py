import logging
import os
import numpy as np
import random
import zipfile
import zlib
from flarestack.shared import band_mask_cache_name
from flarestack.core.energy_pdf import EnergyPDF, read_e_pdf_dict
from flarestack.core.time_pdf import TimePDF, read_t_pdf_dict
from flarestack.core.spatial_pdf import SpatialPDF
from flarestack.utils.catalogue_loader import calculate_source_weight
from scipy import sparse, interpolate
from flarestack.shared import k_to_flux

logger = logging.getLogger(__name__)


def read_injector_dict(inj_dict):
    """Ensures that injection dictionaries remain backwards-compatible

    :param inj_dict: Injection Dictionary
    :return: Injection Dictionary compatible with new format
    """

    if inj_dict != {}:
        maps = [
            ("Injection Time PDF", "injection_sig_time_pdf"),
            ("Injection Energy PDF", "injection_energy_pdf"),
            ("injection_sig_energy_pdf", "injection_energy_pdf"),
            ("inj_energy_pdf", "injection_energy_pdf"),
            ("Poisson Smear?", "poisson_smear_bool"),
            ("injection_time_pdf", "injection_sig_time_pdf"),
            ("inj_sig_time_pdf", "injection_sig_time_pdf"),
        ]

        for old_key, new_key in maps:
            if old_key in list(inj_dict.keys()):
                logger.warning(
                    "Deprecated inj_dict key '{0}' was used. Please use '{1}' in future.".format(
                        old_key, new_key
                    )
                )
                inj_dict[new_key] = inj_dict[old_key]

        pairs = [
            ("injection_energy_pdf", read_e_pdf_dict),
            ("injection_sig_time_pdf", read_t_pdf_dict),
        ]

        for key, f in pairs:
            if key in list(inj_dict.keys()):
                inj_dict[key] = f(inj_dict[key])

        if "injection_spatial_pdf" not in inj_dict.keys():
            inj_dict["injection_spatial_pdf"] = {}

        # if "injection_bkg_time_pdf" not in inj_dict.keys():
        #     logger.warning("No 'injection_bkg_time_pdf' was specified. A 'steady' pdf will be assumed.")
        #     inj_dict["injection_bkg_time_pdf"] = {"time_pdf_name": "steady"}

    return inj_dict


class BaseInjector:
    """Base Injector Class"""

    subclasses: dict[str, object] = {}

    def __init__(self, season, sources, **kwargs):
        kwargs = read_injector_dict(kwargs)
        self.inj_kwargs = kwargs

        logger.info("Initialising Injector for {0}".format(season.season_name))
        self.injection_band_mask = dict()
        self.season = season
        self.season.load_background_model()

        self.sources = sources

        if len(sources) > 0:
            self.weight_scale = calculate_source_weight(self.sources)

        try:
            self.sig_time_pdf = TimePDF.create(
                kwargs["injection_sig_time_pdf"], season.get_time_pdf()
            )
            # self.bkg_time_pdf = TimePDF.create(kwargs["injection_bkg_time_pdf"],
            #                                    season.get_time_pdf())
            self.energy_pdf = EnergyPDF.create(kwargs["injection_energy_pdf"])
            self.spatial_pdf = SpatialPDF(kwargs["injection_spatial_pdf"], season)
        except KeyError:
            raise Exception(
                "Injection Arguments missing. \n "
                "'injection_energy_pdf', 'injection_time_pdf',"
                "and 'injection_spatial_pdf' are required. \n"
                "Found: \n {0}".format(kwargs)
            )

        if "poisson_smear_bool" in list(kwargs.keys()):
            self.poisson_smear = kwargs["poisson_smear_bool"]
        else:
            self.poisson_smear = True

        self.n_exp = np.nan

        try:
            self.fixed_n = kwargs["fixed_n"]
        except KeyError:
            self.fixed_n = np.nan

    def calculate_n_exp(self):
        all_n_exp = np.empty(
            (len(self.sources), 1),
            dtype=np.dtype([("source_name", "a30"), ("n_exp", float)]),
        )

        for i, source in enumerate(self.sources):
            all_n_exp[i]["source_name"] = source["source_name"]
            all_n_exp[i]["n_exp"] = self.calculate_n_exp_single(source)
        return all_n_exp

    def calculate_n_exp_single(self, source):
        raise NotImplementedError

    def get_n_exp_single(self, source):
        if not isinstance(source["source_name"], bytes):
            name = bytes(source["source_name"], encoding="utf8")
        else:
            name = source["source_name"]

        return np.copy(self.n_exp[self.n_exp["source_name"] == name])

    def get_expectation(self, source, scale):
        return float(self.get_n_exp_single(source)["n_exp"]) * scale

    def update_sources(self, sources):
        """Reuses an injector with new sources

        :param sources: Sources to be added
        """
        self.sources = sources
        self.weight_scale = np.sum(
            self.sources["base_weight"] * self.sources["distance_mpc"] ** -2
        )
        self.n_exp = self.calculate_n_exp()

    def create_dataset(self, scale, angular_error_modifier=None):
        """Create a dataset based on scrambled data for background, and Monte
        Carlo simulation for signal. Returns the composite dataset. The source
        flux can be scaled by the scale parameter.

        :param scale: Ratio of Injected Flux to source flux
        :param angular_error_modifier: AngularErrorModifier to change angular errors
        :return: Simulated dataset
        """
        bkg_events = self.season.simulate_background()

        if scale > 0.0:
            sig_events = self.inject_signal(scale)
        else:
            sig_events = []

        if len(sig_events) > 0:
            simulated_data = np.concatenate((bkg_events, sig_events))
        else:
            simulated_data = bkg_events

        if angular_error_modifier is not None:
            simulated_data = angular_error_modifier.pull_correct_static(simulated_data)

        return simulated_data

    def inject_signal(self, scale):
        return

    @classmethod
    def register_subclass(cls, inj_name):
        """Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """

        def decorator(subclass):
            BaseInjector.subclasses[inj_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, season, sources, **kwargs):
        inj_dict = read_injector_dict(kwargs)

        if "injector_name" not in inj_dict.keys():
            return cls(season, sources, **inj_dict)

        inj_name = inj_dict["injector_name"]

        if inj_name not in BaseInjector.subclasses:
            raise ValueError(
                f"Bad Injector name {inj_name}. "
                f"Available options are {BaseInjector.subclasses.keys()}"
            )
        else:
            return BaseInjector.subclasses[inj_name](season, sources, **inj_dict)

    @staticmethod
    def get_dec_and_omega(source, bandwidth):
        # Sets half width of band
        dec_width = np.sin(np.deg2rad(bandwidth))

        sinDec = np.sin(source["dec_rad"])

        # Sets a declination band above and below the source
        min_dec = max(-1, sinDec - dec_width)
        max_dec = min(1.0, sinDec + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2.0 * np.pi * (max_dec - min_dec)
        return np.arcsin(dec_width), np.arcsin(min_dec), np.arcsin(max_dec), omega


@BaseInjector.register_subclass("mc_injector")
class MCInjector(BaseInjector):
    """Core Injector Class, returns a dataset on which calculations can be
    performed. This base class is tailored for injection of MC into mock
    background. This can be either MC background, or scrambled real data.
    """

    subclasses: dict[str, object] = {}

    def __init__(self, season, sources, **kwargs):
        kwargs = read_injector_dict(kwargs)
        self._mc = season.get_mc()
        BaseInjector.__init__(self, season, sources, **kwargs)

        self.injection_declination_bandwidth = self.inj_kwargs.pop(
            "injection_declination_bandwidth", 1.5
        )

        try:
            self.mc_weights = self.energy_pdf.weight_mc(self._mc)
            self.n_exp = self.calculate_n_exp()

        except KeyError:
            logger.warning("No Injection Arguments. Are you unblinding?")
            pass

    def select_mc_band(self, source):
        """For a given source, selects MC events within a declination band of
        width +/- 5 degrees that contains the source. Then returns the MC data
        subset containing only those MC events.

        :param source: Source to be simulated
        :return: mc (cut): Simulated events which lie within the band
        :return: omega: Solid Angle of the chosen band
        :return: band_mask: The mask which removes events outside band
        """

        dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(
            source, self.injection_declination_bandwidth
        )

        band_mask = self.get_band_mask(source, min_dec, max_dec)

        return np.copy(self._mc[band_mask]), omega, band_mask

    def get_band_mask(self, source, min_dec, max_dec):
        # Checks if the mask has already been evaluated for the source
        # If not, creates the mask for this source, and saves it
        if source["source_name"] in list(self.injection_band_mask.keys()):
            band_mask = self.injection_band_mask[source["source_name"]]
        else:
            band_mask = np.logical_and(
                np.greater(self._mc["trueDec"], min_dec),
                np.less(self._mc["trueDec"], max_dec),
            )
            self.injection_band_mask[source["source_name"]] = band_mask

        return band_mask

    def calculate_single_source(self, source, scale):
        """Calculate the weighted MC for a single source, given a flux scale
        and a distance scale.

        :param source:
        :param scale:
        :return:
        """
        # Selects MC events lying in a +/- 5 degree declination band
        source_mc, omega, band_mask = self.select_mc_band(source)

        source_mc = self.calculate_fluence(source, scale, source_mc, band_mask, omega)

        return source_mc

    def calculate_n_exp_single(self, source):
        return np.sum(self.calculate_single_source(source, 1.0)["ow"])

    def calculate_fluence(self, source, scale, source_mc, band_mask, omega):
        """Function to calculate the fluence for a given source, and multiply
        the oneweights by this. After this step, the oneweight sum is equal
        to the expected neutrino number.

        :param source: Source to be calculated
        :param scale: Flux scale
        :param source_mc: MC that is close to source
        :param band_mask: Closeness mask for MC
        :param omega: Solid angle covered by MC mask
        :return: Modified source MC
        """
        # Calculate the effective injection time for simulation. Equal to
        # the overlap between the season and the injection time PDF for
        # the source, scaled if the injection PDF is not uniform in time.
        eff_inj_time = self.sig_time_pdf.effective_injection_time(source)

        # All injection fluxes are given in terms of k, equal to 1e-9
        inj_flux = k_to_flux(source["injection_weight_modifier"] * scale)

        # Fraction of total flux allocated to given source, assuming
        # standard candles with flux proportional to 1/d^2 multiplied by the
        # sources weight

        weight = calculate_source_weight(source) / self.weight_scale

        # Calculate the fluence, using the effective injection time.
        fluence = inj_flux * eff_inj_time * weight

        # Recalculates the oneweights to account for the declination
        # band, and the relative distance of the sources.
        # Multiplies by the fluence, to enable calculations of n_inj,
        # the expected number of injected events

        source_mc["ow"] = fluence * self.mc_weights[band_mask] / omega

        return source_mc

    def inject_signal(self, scale):
        """Randomly select simulated events from the Monte Carlo dataset to
        simulate a signal for each source. The source flux can be scaled by
        the scale parameter.

        :param scale: Ratio of Injected Flux to source flux.
        :return: Set of signal events for the given IC Season.
        """
        # Creates empty signal event array
        sig_events = np.empty((0,), dtype=self.season.get_background_dtype())

        n_tot_exp = 0

        # Loop over each source to be simulated
        for i, source in enumerate(self.sources):
            # If a number of neutrinos to inject is specified, use that.
            # Otherwise, inject based on the flux scale as normal.

            if not np.isnan(self.fixed_n):
                n_inj = int(self.fixed_n)
            else:
                n_inj = self.get_expectation(source, scale)

            n_tot_exp += n_inj

            # Simulates poisson noise around the expectation value n_inj.
            if self.poisson_smear:
                n_s = int(np.random.poisson(n_inj))
            # If there is no poisson noise, rounds n_s down to nearest integer
            else:
                n_s = int(n_inj)

            try:
                f_n_inj = float(n_inj[0])
            except (TypeError, IndexError):
                f_n_inj = float(n_inj)

            logger.debug(
                "Injected {0} events with an expectation of {1:.2f} events for {2}".format(
                    n_s, f_n_inj, source["source_name"]
                )
            )

            #  If n_s = 0, skips simulation step.
            if n_s < 1:
                continue

            source_mc = self.calculate_single_source(source, scale)

            # Creates a normalised array of OneWeights
            p_select = source_mc["ow"] / np.sum(source_mc["ow"])

            # Creates an array with n_signal entries.
            # Each entry is a random integer between 0 and no. of sources.
            # The probability for each integer is equal to the OneWeight of
            # the corresponding source_path.
            ind = np.random.choice(len(source_mc["ow"]), size=n_s, p=p_select)

            # Selects the sources corresponding to the random integer array
            sim_ev = source_mc[ind]

            # Rotates the Monte Carlo events onto the source_path
            sim_ev = self.spatial_pdf.rotate_to_position(
                sim_ev, source["ra_rad"], source["dec_rad"]
            )

            # Generates times for each simulated event, drawing from the
            # Injector time PDF.
            sim_ev["time"] = self.sig_time_pdf.simulate_times(source, n_s)

            # Joins the new events to the signal events
            sig_events = np.concatenate(
                (sig_events, sim_ev[list(self.season.get_background_dtype().names)])
            )

        return sig_events


@MCInjector.register_subclass("low_memory_injector")
class LowMemoryInjector(MCInjector):
    """For large numbers of sources O(~100), saving MC masks becomes
    increasingly burdensome. As a solution, the LowMemoryInjector should be
    used instead. It will be somewhat slower, but will have much more
    reasonable memory consumption.
    """

    def __init__(self, season, sources, **kwargs):
        self.split_cats = None
        self.injection_band_paths = None
        self.band_mask_cache = None
        self.band_mask_index = None

        MCInjector.__init__(self, season, sources, **kwargs)

    def calculate_n_exp(self):
        cats, paths, m_index, s_index = band_mask_cache_name(
            self.season, self.sources, self.injection_declination_bandwidth
        )
        self.split_cats = cats
        self.injection_band_paths = paths

        if np.sum([not os.path.isfile(x) for x in self.injection_band_paths]) > 0.0:
            logger.info("No saved band masks found. These will have to be made first.")
            self.make_injection_band_mask()

        self.n_exp = np.zeros(
            (len(self.sources), 1),
            dtype=np.dtype(
                [
                    ("source_name", "a30"),
                    ("n_exp", float),
                    ("mask_index", int),
                    ("source_index", int),
                ]
            ),
        )

        self.n_exp["mask_index"] = np.array(m_index).reshape(len(m_index), 1)
        self.n_exp["source_index"] = np.array(s_index).reshape(len(s_index), 1)

        for i, source in enumerate(self.sources):
            self.n_exp[i]["source_name"] = source["source_name"]
            self.n_exp[i]["n_exp"] = self.calculate_n_exp_single(source)

        return self.n_exp

    def make_injection_band_mask(self):
        for j, cat in enumerate(self.split_cats):
            path = self.injection_band_paths[j]

            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass

            # Make mask
            injection_band_mask = sparse.lil_matrix(
                (len(cat), len(self._mc)), dtype=bool
            )
            for i, source in enumerate(cat):
                dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(
                    source, self.injection_declination_bandwidth
                )
                band_mask = np.logical_and(
                    np.greater(self._mc["trueDec"], min_dec),
                    np.less(self._mc["trueDec"], max_dec),
                )
                injection_band_mask[i, :] = band_mask
            injection_band_mask = injection_band_mask.tocsr()
            sparse.save_npz(path, injection_band_mask)

            del injection_band_mask

            logger.info(f"Saving to {path}")

    def load_band_mask(self, index):
        path = self.injection_band_paths[index]
        # logger.debug(f'type(band_mask_cache) = {type(self.band_mask_cache)}')
        del self.band_mask_cache
        logger.debug(f"loading bandmask from {path}")
        self.band_mask_cache = sparse.load_npz(path)
        self.band_mask_index = index
        # return sparse.load_npz(path)

    def get_band_mask(self, source, min_dec, max_dec):
        entry = self.get_n_exp_single(source)
        if len(entry) != 1:
            raise ValueError(
                f"Length of found entries for {source['source_name']} "
                f"is {len(entry)} but should be 1!"
            )
        mask_index = entry["mask_index"]

        if not np.logical_and(
            not isinstance(self.band_mask_cache, type(None)),
            self.band_mask_index == mask_index,
        ):
            try:
                self.load_band_mask(mask_index[0])
            except (zlib.error, zipfile.BadZipFile):
                self.make_injection_band_mask()
                self.load_band_mask(mask_index[0])

            # self.load_band_mask(mask_index[0])

        # band_mask = self.load_band_mask(mask_index[0])

        return self.band_mask_cache.getrow(entry["source_index"][0]).toarray()[0]


@MCInjector.register_subclass("effective_area_injector")
class EffectiveAreaInjector(BaseInjector):
    """Class for injecting signal events by relying on effective areas rather
    than pre-existing Monte Carlo simulation. This Injector should be used
    for analysing public data, as no MC is provided.
    """

    def __init__(self, season, sources, **kwargs):
        self.effective_area_f = season.load_effective_area()
        self.energy_proxy_mapping = season.load_energy_proxy_mapping()
        self.angular_res_f = season.load_angular_resolution()
        BaseInjector.__init__(self, season, sources, **kwargs)
        self.n_exp = self.calculate_n_exp()
        self.conversion_cache = dict()

    def inject_signal(self, scale):
        # Creates empty signal event array
        sig_events = np.empty((0,), dtype=self.season.get_background_dtype())

        n_tot_exp = 0

        # Loop over each source to be simulated
        for i, source in enumerate(self.sources):
            # If a number of neutrinos to inject is specified, use that.
            # Otherwise, inject based on the flux scale as normal.

            if not np.isnan(self.fixed_n):
                n_inj = int(self.fixed_n)
            else:
                n_inj = self.get_expectation(source, scale)

            n_tot_exp += n_inj

            # Simulates poisson noise around the expectation value n_inj.
            if self.poisson_smear:
                n_s = int(np.random.poisson(n_inj))
            # If there is no poisson noise, rounds n_s to nearest integer
            else:
                n_s = int(n_inj)

            logger.debug(
                "Injected {0} events with an expectation of {1:.2f} events for {2}".format(
                    n_s,
                    n_inj if isinstance(n_inj, float) else float(n_inj[0]),
                    source["source_name"],
                )
            )

            #  If n_s = 0, skips simulation step.
            if n_s < 1:
                continue

            sim_ev = np.empty((n_s,), dtype=self.season.get_background_dtype())

            # Fills the energy proxy conversion cache

            if source["source_name"] not in self.conversion_cache.keys():
                self.calculate_energy_proxy(source)

            # Produces random seeds, converts this using a convolution
            # of energy pdf and approximated energy proxy mapping to
            # produce final energy proxy values

            convert_f = self.conversion_cache[source["source_name"]]

            random_fraction = [random.random() for _ in range(n_s)]

            sim_ev["logE"] = convert_f(random_fraction)

            # Simulates times according to Time PDF

            sim_ev["time"] = self.sig_time_pdf.simulate_times(source, n_s)
            sim_ev["sigma"] = self.angular_res_f(sim_ev["logE"]).copy()
            sim_ev["raw_sigma"] = sim_ev["sigma"].copy()

            sim_ev = self.spatial_pdf.simulate_distribution(source, sim_ev)

            sim_ev = sim_ev[list(self.season.get_background_dtype().names)].copy()
            #

            # Joins the new events to the signal events
            sig_events = np.concatenate((sig_events, sim_ev))

        sig_events = np.array(sig_events)

        return sig_events

    def calculate_single_source(self, source, scale):
        # Calculate the effective injection time for simulation. Equal to
        # the overlap between the season and the injection time PDF for
        # the source, scaled if the injection PDF is not uniform in time.
        eff_inj_time = self.sig_time_pdf.effective_injection_time(source)

        # All injection fluxes are given in terms of k, equal to 1e-9
        inj_flux = k_to_flux(source["injection_weight_modifier"] * scale)

        # Fraction of total flux allocated to given source, assuming
        # standard candles with flux proportional to 1/d^2 multiplied by the
        # sources weight

        weight = calculate_source_weight(source) / self.weight_scale

        # Calculate the fluence, using the effective injection time.
        fluence = inj_flux * eff_inj_time * weight

        def source_eff_area(e):
            return self.effective_area_f(
                np.log10(e), np.sin(source["dec_rad"])
            ) * self.energy_pdf.f(e)

        int_eff_a = self.energy_pdf.integrate_over_E(source_eff_area)

        # Effective areas are given in m2, but flux is in per cm2

        int_eff_a *= 10**4

        n_inj = fluence * int_eff_a

        return n_inj

    def calculate_n_exp_single(self, source):
        return self.calculate_single_source(source, scale=1.0)

    def calculate_energy_proxy(self, source):
        # Simulates energy proxy values

        def source_eff_area(log_e):
            return (
                self.effective_area_f(log_e, np.sin(source["dec_rad"]))
                * self.energy_pdf.f(log_e)
                * self.energy_proxy_mapping(log_e)
            )

        start_x = np.log10(self.energy_pdf.integral_e_min)

        x_vals = np.linspace(
            start_x + 1e-7, np.log10(self.energy_pdf.integral_e_max), 100
        )[1:]

        y_vals = np.array(
            [
                self.energy_pdf.integrate_over_E(source_eff_area, upper=np.exp(x))
                for x in x_vals
            ]
        )
        y_vals /= max(y_vals)

        f = interpolate.interp1d([0.0] + list(y_vals), [start_x] + list(x_vals))
        self.conversion_cache[source["source_name"]] = f


class MockUnblindedInjector:
    """If the data is not really to be unblinded, then MockUnblindedInjector
    should be called. In this case, the create_dataset function simply returns
    one background scramble.
    """

    def __init__(self, season, sources=np.nan, **kwargs):
        self.season = season
        self._raw_data = season.get_exp_data()

    def create_dataset(self, scale, angular_error_modifier=None):
        """Returns a background scramble

        :return: Scrambled data
        """
        seed = int(123456)
        np.random.seed(seed)

        simulated_data = self.season.simulate_background()
        if angular_error_modifier is not None:
            simulated_data = angular_error_modifier.pull_correct_static(simulated_data)

        return simulated_data


class TrueUnblindedInjector:
    """If the data is unblinded, then UnblindedInjector should be called. In
    this case, the create_dataset function simply returns the unblinded dataset.
    """

    def __init__(self, season, sources, **kwargs):
        self.season = season

    def create_dataset(self, scale, angular_error_modifier=None):
        exp_data = self.season.get_exp_data()

        if angular_error_modifier is not None:
            exp_data = angular_error_modifier.pull_correct_static(exp_data)

        return exp_data


# if __name__ == "__main__":
#     from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
#     data = ps_v002_p01[0]
#     from flarestack.analyses.agn_cores.shared_agncores import agncores_cat_dir
#     cat = np.load(
#         agncores_cat_dir +
#         "radioloud_2rxs_noBL_2000brightest_srcs_weight1.npy"
#     )
#
#     LowMemoryInjector(data, cat)
