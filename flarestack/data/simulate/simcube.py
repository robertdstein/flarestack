import numpy as np
import random
from scipy.interpolate import interp1d
import logging
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.data.simulate import SimSeason, SimDataset

logger = logging.getLogger(__name__)


class IceCubeBackgroundFluxModel:
    def __init__(self):
        self.atmo = self.atmo_flux()
        self.muon_bundle = self.muon_bundle_flux()

    @staticmethod
    def atmo_flux():
        e_pdf_dict = {
            "energy_pdf_name": "power_law",
            "gamma": 3.7,
        }
        return EnergyPDF.create(e_pdf_dict)

    @staticmethod
    def muon_bundle_flux():
        e_pdf_dict = {
            "energy_pdf_name": "power_law",
            "gamma": 3.0,
        }
        return EnergyPDF.create(e_pdf_dict)

    @staticmethod
    def path_through_earth(sindec):
        ic_angle = np.arccos(sindec)

        dec = np.arccos(sindec)

        earth_radius = 6400.0
        ic_depth = 2.0

        dist_to_center = earth_radius - ic_depth

        center_adjacent = earth_radius * np.cos(dec)
        center_opposite = earth_radius * np.sin(dec)

        adjacent_full = dist_to_center + center_adjacent
        full_dist = np.sqrt(adjacent_full**2 + center_opposite**2)

        return full_dist

    def muon_bundle_extent(self, sindec):
        length = self.path_through_earth(sindec)

        # This is the fudgiest factor known to man

        muon_decay_length = 3.0

        return np.exp(-length / muon_decay_length)

    def flux_model_f(self, e, sindec):
        muon_contrib = self.muon_bundle.f(e) * self.muon_bundle_extent(sindec)

        return self.atmo.f(e) + muon_contrib


# ibfm = IceCubeBackgroundFluxModel()
#
# for i in [-1., -0.5, -0.1, 0.0, 0.1, 0.5, 1.]:
#     print(ibfm.path_through_earth(i))
# input("?")

bkg_e_pdf_dict = {"energy_pdf_name": "power_law", "gamma": 3.7}


class SimCubeSeason(SimSeason):
    def __init__(
        self,
        season_name,
        sample_name,
        pseudo_mc_path,
        event_dtype,
        load_effective_area,
        load_angular_resolution,
        bkg_time_pdf_dict,
        bkg_flux_norm,
        bkg_e_pdf_dict,
        energy_proxy_map,
        sin_dec_bins,
        log_e_bins,
        **kwargs
    ):
        self.log_e_bins = log_e_bins
        self.sin_dec_bins = sin_dec_bins

        SimSeason.__init__(
            self,
            season_name,
            sample_name,
            pseudo_mc_path,
            event_dtype,
            load_effective_area,
            load_angular_resolution,
            bkg_time_pdf_dict,
            bkg_flux_norm,
            bkg_e_pdf_dict,
            energy_proxy_map,
            **kwargs
        )

    def generate_sim_data(self, fluence):
        logger.info("Simulating events:")
        sim_events = np.empty((0,), dtype=self.event_dtype)

        for i, lower_sin_dec in enumerate(self.sin_dec_bins[:-1]):
            upper_sin_dec = self.sin_dec_bins[i + 1]
            new_events = self.simulate_dec_range(fluence, lower_sin_dec, upper_sin_dec)

            logger.info(
                "Simulated {0} events between sin(dec)={1} and "
                "sin(dec)={2}".format(len(new_events), lower_sin_dec, upper_sin_dec)
            )

            # Joins the new events to the signal events
            sim_events = np.concatenate((sim_events, new_events))

        sim_events = np.array(sim_events)

        logger.info("Simulated {0} events in total".format(len(sim_events)))

        return sim_events

    def simulate_dec_range(self, fluence, lower_sin_dec, upper_sin_dec):
        mean_sin_dec = 0.5 * (lower_sin_dec + upper_sin_dec)
        solid_angle = 2 * np.pi * (upper_sin_dec - lower_sin_dec)
        sim_fluence = fluence * solid_angle  # GeV^-1 cm^-2

        def source_eff_area(e):
            return self.effective_area_f(
                np.log10(e), mean_sin_dec
            ) * self.bkg_energy_pdf.f(e)

        int_eff_a = self.bkg_energy_pdf.integrate_over_E(source_eff_area)

        # Effective areas are given in m2, but flux is in per cm2

        int_eff_a *= 10**4

        n_exp = sim_fluence * int_eff_a
        n_sim = np.random.poisson(n_exp)

        new_events = np.empty((n_sim,), dtype=self.event_dtype)

        new_events["ra"] = [random.random() * 2 * np.pi for _ in range(n_sim)]

        new_events["sinDec"] = [
            lower_sin_dec + (random.random() * (upper_sin_dec - lower_sin_dec))
            for _ in range(n_sim)
        ]
        new_events["dec"] = np.arcsin(new_events["sinDec"])
        new_events["time"] = self.get_time_pdf().simulate_times([], n_sim)

        fluence_ints, log_e_range = self.bkg_energy_pdf.piecewise_integrate_over_energy(
            source_eff_area
        )
        fluence_ints = fluence_ints

        fluence_ints = np.array(fluence_ints)
        fluence_ints /= np.sum(fluence_ints)

        fluence_cumulative = [
            np.sum(fluence_ints[:i]) for i, _ in enumerate(fluence_ints)
        ]

        fluence_cumulative = [0.0] + fluence_cumulative + [1.0]

        log_e_range = list(log_e_range) + [log_e_range[-1]]

        sim_true_e = interp1d(fluence_cumulative, log_e_range)

        true_e_vals = np.array(
            [10 ** sim_true_e(random.random()) for _ in range(n_sim)]
        )

        new_events["logE"] = self.energy_proxy_map(true_e_vals)

        new_events["sigma"] = self.angular_res_f(new_events["logE"]).copy()
        new_events["raw_sigma"] = new_events["sigma"].copy()

        return new_events


simcube_dataset = SimDataset()

for name, season in icecube_ps_3_year.get_seasons().items():

    def ideal_energy_proxy(e):
        return np.log10(e)

    def wrapper_f(
        bkg_time_pdf_dict,
        bkg_flux_model,
        energy_proxy_map=None,
        sim_name=None,
        **kwargs
    ):
        if np.logical_and(energy_proxy_map is None, sim_name is None):
            energy_proxy_map = ideal_energy_proxy
            sim_name = "default"

        if np.logical_and(energy_proxy_map != ideal_energy_proxy, sim_name is None):
            raise ValueError(
                "Non-default energy proxy mapping was used, "
                "but no unique sim_name was provided. Please "
                "provide a unique 'sim_name' to describe this "
                "simulation."
            )

        sin_dec_bins = season.sin_dec_bins[season.sin_dec_bins > 0.0]

        sim_season = SimCubeSeason(
            season_name=name,
            sample_name="SimCube_{0}".format(sim_name),
            pseudo_mc_path=season.pseudo_mc_path,
            load_effective_area=season.load_effective_area(),
            event_dtype=season.get_background_dtype(),
            load_angular_resolution=season.load_angular_resolution,
            bkg_time_pdf_dict=bkg_time_pdf_dict,
            bkg_flux_model=bkg_flux_model,
            energy_proxy_map=energy_proxy_map,
            sin_dec_bins=sin_dec_bins,
            log_e_bins=season.log_e_bins,
            **kwargs
        )
        return sim_season

    simcube_dataset.add_sim_season(name, wrapper_f)

# after mypy: it seems the following block of code is not compatible with the current syntax of `simcube_dataset.set_sim_params()`
# - error: Missing positional argument "bkg_flux_model" in call to "set_sim_params"  [call-arg]
# this implies this module is likely untested, otherwise such a test would fail
# the following is commented out, for the time being

# bkg_time_pdf_dict = {
#     "time_pdf_name": "fixed_ref_box",
#     "fixed_ref_time_mjd": 50000,
#     "pre_window": 0.0,
#     "post_window": 500.0,
# }
#
# simcube_season = simcube_dataset.set_sim_params(
#     name="IC86-2012",
#     bkg_time_pdf_dict=bkg_time_pdf_dict,
#     bkg_flux_norm=1e8,
#     bkg_e_pdf_dict=bkg_e_pdf_dict,
# )

# nicecube_10year = SimCubeSeason(0, 100, 1., e_pdf_dict)
