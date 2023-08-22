"""Script used to estimate discovery potential quickly. It uses some pretty
absurd approximations, but is REALLY REALLY fast (equivalent or faster than a
single trial, and is good to within ~30-50%). Use with some caution to
quickly guess appropriate flux scales, or understand trends, without full
calculations.
"""
import numpy as np
from flarestack.core.llh import LLH
from flarestack.core.astro import angular_distance
from flarestack.shared import k_to_flux
from flarestack.utils.catalogue_loader import load_catalogue, calculate_source_weight
from scipy.stats import norm
import logging


def estimate_discovery_potential(seasons, inj_dict, sources, llh_dict, raw_scale=1.0):
    """Function to estimate discovery potential given an injection model. It
    assumes an optimal LLH construction, i.e aligned time windows and correct
    energy weighting etc. Takes injectors and seasons.

    :param injectors: Injectors to be used
    :param sources: Sources to be evaluated
    :return: An estimate for the discovery potential
    """
    logging.info("Trying to guess scale using AsimovEstimator.")

    season_bkg = []
    season_sig = []

    def weight_f(n_s, n_bkg):
        metric = np.array(n_s)  # /np.sqrt(np.array(n_bkg))
        return 1.0  # metric #/ np.mean(metric)#/ max(metric)

    def ts_weight(n_s):
        return 1.0
        # return n_s / np.sum(n_s)

    # def weight_ts(ts, n_s)

    weight_scale = calculate_source_weight(sources)

    livetime = 0.0

    n_s_tot = 0.0
    n_tot = 0.0
    n_tot_coincident = 0.0

    all_ns = []
    all_nbkg = []

    all_ts = []
    all_bkg_ts = []

    final_ts = []

    new_n_s = 0.0
    new_n_bkg = 0.0

    for season in seasons.values():
        new_llh_dict = dict(llh_dict)
        new_llh_dict["llh_name"] = "fixed_energy"
        new_llh_dict["llh_energy_pdf"] = inj_dict["injection_energy_pdf"]
        llh = LLH.create(season, sources, new_llh_dict)

        data = season.get_background_model()
        n_tot += np.sum(data["weight"])
        livetime += llh.bkg_time_pdf.livetime * 60 * 60 * 24

        def signalness(sig_over_background):
            """Converts a signal over background ratio into a signal
            probability. This is ratio/(1 + ratio)

            :param sig_over_background: Ratio of signal to background
            probability
            :return: Percentage probability of signal
            """

            return sig_over_background / (1.0 + sig_over_background)

        n_sigs = []
        n_bkgs = []

        ts_vals = []
        bkg_vals = []
        n_s_season = 0.0

        # n_exp = np.sum(inj.n_exp["n_exp"]) * raw_scale

        sig_times = np.array(
            [llh.sig_time_pdf.effective_injection_time(x) for x in sources]
        )
        source_weights = np.array([calculate_source_weight(x) for x in sources])
        mean_time = np.sum(sig_times * source_weights) / weight_scale

        # print(source_weights)

        fluences = (
            np.array([x * sig_times[i] for i, x in enumerate(source_weights)])
            / weight_scale
        )
        # print(sources.dtype.names)
        # print(sources["dec_rad"], np.sin(sources["dec_rad"]))
        # print(fluences)
        # input("?")
        res = np.histogram(
            np.sin(sources["dec_rad"]), bins=season.sin_dec_bins, weights=fluences
        )

        dummy_sources = []
        bounds = []
        n_eff_sources = []

        for i, w in enumerate(res[0]):
            if w > 0:
                lower = res[1][i]
                upper = res[1][i + 1]
                mid = np.mean([upper, lower])

                mask = np.logical_and(
                    np.sin(sources["dec_rad"]) > lower,
                    np.sin(sources["dec_rad"]) < upper,
                )

                n_eff_sources.append(
                    (np.sum(fluences[mask]) ** 2.0 / np.sum(fluences[mask] ** 2))
                )

                # print(n_eff_sources)
                # print(fluences[mask])
                #
                # tester = np.array([1.5, 1.5, 1.5])
                #
                #
                # print(np.sum(tester**2)/(np.mean(tester)**2.))
                # input("?")

                dummy_sources.append(
                    (np.arcsin(mid), res[0][i], 1.0, 1.0, "dummy_{0}".format(mid))
                )
                bounds.append((lower, upper))

        dummy_sources = np.array(
            dummy_sources,
            dtype=np.dtype(
                [
                    ("dec_rad", float),
                    ("base_weight", float),
                    ("distance_mpc", float),
                    ("injection_weight_modifier", float),
                    ("source_name", np.str),
                ]
            ),
        )
        inj = season.make_injector(dummy_sources, **inj_dict)

        for j, dummy_source in enumerate(dummy_sources):
            lower, upper = bounds[j]

            n_eff = n_eff_sources[j]

            source_mc = inj.calculate_single_source(dummy_source, scale=raw_scale)

            if len(source_mc) == 0:
                logging.warning(
                    f"Warning, no MC found for dummy source at declination {np.arcsin(lower)} {np.arcsin(upper)}"
                )
                ts_vals.append(0.0)
                n_sigs.append(0.0)
                n_bkgs.append(0.0)
            else:
                # Gives the solid angle coverage of the sky for the band
                omega = 2.0 * np.pi * (upper - lower)

                data_mask = np.logical_and(
                    np.greater(data["dec"], np.arcsin(lower)),
                    np.less(data["dec"], np.arcsin(upper)),
                )
                local_data = data[data_mask]

                data_weights = (
                    signalness(llh.energy_weight_f(local_data)) * local_data["weight"]
                )

                # print("source_mc", source_mc)

                mc_weights = signalness(llh.energy_weight_f(source_mc))

                true_errors = angular_distance(
                    source_mc["ra"],
                    source_mc["dec"],
                    source_mc["trueRa"],
                    source_mc["trueDec"],
                )

                # median_sigma = weighted_quantile(
                #             true_errors, 0.5, source_mc["ow"] * mc_weights)

                median_sigma = np.mean(local_data["sigma"])

                area = (
                    np.pi * (2.0 * median_sigma) ** 2 / np.cos(dummy_source["dec_rad"])
                )

                local_rate = np.sum(data_weights)

                # n_bkg = local_rate * area  # * source_weight
                n_bkg = np.sum(local_data["weight"])

                n_tot_coincident += n_bkg

                ratio_time = livetime / mean_time

                sig_spatial = signalness(
                    (
                        1.0
                        / (2.0 * np.pi * source_mc["sigma"] ** 2.0)
                        * np.exp(-0.5 * ((true_errors / source_mc["sigma"]) ** 2.0))
                    )
                    / llh.spatial_pdf.background_spatial(source_mc)
                )

                ra_steps = np.linspace(-np.pi, np.pi, 100)
                dec_steps = np.linspace(lower, upper, 10)

                mean_dec = np.mean(
                    signalness(
                        norm.pdf(
                            dec_steps,
                            scale=median_sigma / np.cos(dummy_source["dec_rad"]),
                            loc=np.mean([lower, upper]),
                        )
                        * (upper - lower)
                    )
                )

                mean_ra = np.mean(
                    signalness(
                        norm.pdf(ra_steps, scale=median_sigma, loc=0.0) * 2.0 * np.pi
                    )
                )

                bkg_spatial = mean_dec * mean_ra  # * n_eff

                n_s_tot += np.sum(source_mc["ow"])
                n_s_season += np.sum(source_mc["ow"])

                med_sig = (
                    np.mean(sig_spatial * mc_weights)
                    * signalness(ratio_time)
                    * np.sum(source_mc["ow"])
                )
                med_bkg = (
                    np.mean(bkg_spatial * data_weights)
                    * (1.0 - signalness(ratio_time))
                    * n_bkg
                )

                new_n_s += med_sig
                new_n_bkg += med_bkg

    scaler_ratio = new_n_s / n_s_tot

    scaler_ratio = new_n_bkg / n_tot_coincident

    print("Scaler Ratio", scaler_ratio)

    disc_count = norm.ppf(
        norm.cdf(5.0), loc=0.0, scale=np.sqrt(new_n_bkg)
    )  # * scaler_ratio

    simple = 5.0 * np.sqrt(new_n_bkg)  # * scaler_ratio
    #
    # disc_count = simple

    # print(disc_count, simple, simple/disc_count, n_s_tot)
    #
    # print("testerer", new_n_s, new_n_bkg)
    #
    print("Disc count", disc_count, disc_count / scaler_ratio)
    scale = disc_count / new_n_s
    print(scale)

    # Convert from scale factor to flux units

    scale = k_to_flux(scale) * raw_scale

    logging.info(
        "Estimated Discovery Potential is: {:.3g} GeV sr^-1 s^-1 cm^-2".format(scale)
    )

    return scale


class AsimovEstimator(object):
    def __init__(self, seasons, inj_dict, llh_dict):
        self.seasons = seasons
        self.injectors = dict()
        self.llh_dict = llh_dict

        for season in self.seasons.values():
            self.injectors[season.season_name] = season.make_injector([], **inj_dict)

    def guess_discovery_potential(self, source_path):
        sources = load_catalogue(source_path)

        for inj in self.injectors.values():
            inj.update_sources(sources)

        return estimate_discovery_potential(self.injectors, sources, self.llh_dict)
