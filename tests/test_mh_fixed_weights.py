"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack import create_unblinder, MinimisationHandler

# Initialise Injectors/LLHs

llh_dict = {
    "llh_name": "standard_matrix",
    "llh_sig_time_pdf": {"time_pdf_name": "steady"},
    "llh_bkg_time_pdf": {
        "time_pdf_name": "steady",
    },
    "llh_energy_pdf": {"energy_pdf_name": "power_law"},
}

true_parameters = [3.70369960756338, 4.0]

catalogue = tde_catalogue_name("jetted")


logging.getLogger().setLevel("INFO")


class TestTimeIntegrated(unittest.TestCase):
    def setUp(self):
        pass

    def test_declination_sensitivity(self):
        logging.info("Testing 'fixed_weight' MinimisationHandler class")

        # Test three declinations

        unblind_dict = {
            "name": "test/test_fixed_weights/",
            "mh_name": "fixed_weights",
            "dataset": icecube_ps_3_year.get_seasons("IC86-2011"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]

        logging.info("Best fit values {0}".format(list(res["x"])))
        logging.info("Reference best fit {0}".format(true_parameters))

        for i, x in enumerate(res["x"]):
            self.assertAlmostEqual(x, true_parameters[i], delta=0.1)


if __name__ == "__main__":
    unittest.main()
