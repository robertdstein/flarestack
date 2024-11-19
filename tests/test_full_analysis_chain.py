"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack import create_unblinder, MinimisationHandler
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack import analyse, ResultsHandler, OverfluctuationError

logging.getLogger().setLevel("INFO")

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):
    def setUp(self):
        pass

    def test_full_chain(self):
        logging.info("Testing MinimisationHandler analysis chain")

        base_name = "tests/test_analysis_chain"

        try:
            for j, gamma in enumerate([2.0, 2.5]):
                # Initialise Injectors/LLHs

                inj_dict = {
                    "injection_sig_time_pdf": {"time_pdf_name": "steady"},
                    "injection_energy_pdf": {
                        "energy_pdf_name": "power_law",
                        "gamma": gamma,
                    },
                }

                llh_dict = {
                    "llh_name": "standard_matrix",
                    "llh_sig_time_pdf": {"time_pdf_name": "steady"},
                    "llh_bkg_time_pdf": {
                        "time_pdf_name": "steady",
                    },
                    "llh_energy_pdf": {"energy_pdf_name": "power_law"},
                }

                # Test three declinations

                mh_dict = {
                    "name": f"{base_name}/{gamma}/",
                    "mh_name": "fixed_weights",
                    "dataset": icecube_ps_3_year.get_seasons("IC86-2011"),
                    "catalogue": catalogue,
                    "inj_dict": inj_dict,
                    "llh_dict": llh_dict,
                    "n_steps": 5,
                    "n_trials": 10,
                    "scale": [3.0, 500.0][j],
                }

                analyse(mh_dict, n_cpu=24, cluster=False)

                rh = ResultsHandler(mh_dict)

                # Deliberately test a second time, to see performance once results have been combined

                rh = ResultsHandler(mh_dict)

            ub_dict = dict(mh_dict)

            # Test without background TS

            ub = create_unblinder(ub_dict, full_plots=True)

            # Test with background TS

            ub_dict["background_ts"] = base_name

            ub = create_unblinder(ub_dict, full_plots=True, scan_2d=True)

            mh = MinimisationHandler.create(mh_dict)
            mh.iterate_run(scale=1.0, n_steps=3, n_trials=1)

        except OverfluctuationError:
            pass


if __name__ == "__main__":
    unittest.main()
