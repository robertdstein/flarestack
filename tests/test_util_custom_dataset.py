"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.utils import custom_dataset, load_catalogue

true_parameters = [0.0, 2.07606811]
true_keys = ["IC79-2010", "IC86-2011"]

catalogue = tde_catalogue_name("jetted")


class TestUtilCustomDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_custom_dataset(self):
        logging.info("Testing custom_dataset util function.")

        llh_dict = {
            "llh_name": "standard_matrix",
            "llh_sig_time_pdf": {
                "time_pdf_name": "box",
                "pre_window": 0.0,
                "post_window": 100.0,
            },
            "llh_bkg_time_pdf": {
                "time_pdf_name": "steady",
            },
            "llh_energy_pdf": {"energy_pdf_name": "power_law"},
        }

        dataset = custom_dataset(
            icecube_ps_3_year, load_catalogue(catalogue), llh_dict["llh_sig_time_pdf"]
        )

        keys = sorted(list(dataset.keys()))

        self.assertEqual(keys, true_keys)

        # Test three declinations

        unblind_dict = {
            "name": "test_custom_dataset",
            "mh_name": "fixed_weights",
            "dataset": custom_dataset(
                icecube_ps_3_year,
                load_catalogue(catalogue),
                llh_dict["llh_sig_time_pdf"],
            ),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        for j, x in enumerate(res["x"]):
            self.assertAlmostEqual(x, true_parameters[j], delta=0.1)

        logging.info("Best fit values {0}".format(list(res["x"])))
        logging.info("Reference best fit {0}".format(true_parameters))


if __name__ == "__main__":
    unittest.main()
