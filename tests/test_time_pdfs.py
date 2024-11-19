"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack import MinimisationHandler

# Initialise Injectors/LLHs

time_pdfs = [
    {"time_pdf_name": "steady"},
    {"time_pdf_name": "box", "pre_window": 0.0, "post_window": 100.0},
    {
        "time_pdf_name": "custom_source_box",
    },
    {
        "time_pdf_name": "fixed_ref_box",
        "pre_window": 0.0,
        "post_window": 100.0,
        "fixed_ref_time_mjd": 56000,
    },
]

true_parameters = [
    [1.877671588900102, 3.4651997149577394],
    [0.0, 2.111438613892292],
    [0.0, 2.110474052128495],
    [0.0, 2.0993342075261676],
]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):
    def setUp(self):
        pass

    def test_declination_sensitivity(self):
        logging.info("Testing 'fixed_weight' MinimisationHandler class")

        for i, t_pdf_dict in enumerate(time_pdfs):
            llh_dict = {
                "llh_name": "standard_matrix",
                "llh_sig_time_pdf": t_pdf_dict,
                "llh_bkg_time_pdf": {
                    "time_pdf_name": "steady",
                },
                "llh_energy_pdf": {"energy_pdf_name": "power_law"},
            }

            # Test three declinations

            unblind_dict = {
                "mh_name": "fixed_weights",
                "name": "tests/test_time_pdfs/",
                "dataset": icecube_ps_3_year.get_seasons("IC79-2010", "IC86-2011"),
                "catalogue": catalogue,
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]

            logging.info("Best fit values {0}".format(list(res["x"])))
            logging.info("Reference best fit {0}".format(true_parameters[i]))

            for j, x in enumerate(res["x"]):
                self.assertAlmostEqual(x, true_parameters[i][j], delta=0.1)

            # inj_dict = {
            #     "injection_sig_time_pdf": t_pdf_dict,
            #     "injection_bkg_time_pdf": {
            #         "time_pdf_name": "steady",
            #     },
            #     "injection_energy_pdf": {
            #         "energy_pdf_name": "power_law",
            #         "gamma": 2.0
            #     }
            # }
            #
            # mh_dict = dict(unblind_dict)
            # mh_dict["inj_dict"] = inj_dict
            #
            # mh = MinimisationHandler.create(mh_dict)
            # res = mh.simulate_and_run(5.)
            # print(res)


if __name__ == "__main__":
    unittest.main()
