"""This script contains the EnergyPDF classes, that are used for weighting
events based on a given energy PDF.

"""
import numexpr
import numpy as np
import pickle as Pickle
import logging

logger = logging.getLogger(__name__)


gamma_range = [1.0, 4.0]

default_emin = 100
default_emax = 10**7


def read_e_pdf_dict(e_pdf_dict):
    """Ensures backwards compatibility of e_pdf_dict objects.

    :param e_pdf_dict: Energy PDF dictionary
    :return: Updated Energy PDF dictionary compatible with new format
    """

    if e_pdf_dict != {}:
        maps = [
            ("E Min", "e_min_gev"),
            ("E Max", "e_max_gev"),
            ("Name", "energy_pdf_name"),
            ("Gamma", "gamma"),
            ("Spline Path", "spline_path"),
            ("Source Energy (erg)", "source_energy_erg"),
        ]

        for old_key, new_key in maps:
            if old_key in list(e_pdf_dict.keys()):
                logger.warning(
                    "Deprecated e_pdf_key '{0}' was used. "
                    "Please use '{1}' in future.".format(old_key, new_key)
                )
                e_pdf_dict[new_key] = e_pdf_dict[old_key]

        name_maps = [
            ("Power Law", "power_law"),
            ("PowerLaw", "power_law"),
            ("Spline", "spline"),
        ]

        for old_key, new_key in name_maps:
            if e_pdf_dict["energy_pdf_name"] == old_key:
                logger.warning(
                    "Deprecated energy_pdf_name '{0}' was used. "
                    "Please use '{1}' in future.".format(old_key, new_key)
                )
                e_pdf_dict["energy_pdf_name"] = new_key

    return e_pdf_dict


class EnergyPDF(object):
    """
    A base class for Energy PDFs.
    """

    subclasses: dict[str, object] = {}

    def __init__(self, e_pdf_dict):
        """
        constructor
        :param e_pdf_dict:
        """

        # Set up minimum/maximum energy

        if "e_min_gev" in list(e_pdf_dict.keys()):
            self.e_min = e_pdf_dict["e_min_gev"]
            logger.info(f"Minimum Energy is {self.e_min:.2g} GeV.")
            self.integral_e_min = self.e_min
        else:
            self.integral_e_min = default_emin

        if "e_max_gev" in list(e_pdf_dict.keys()):
            self.e_max = e_pdf_dict["e_max_gev"]
            logger.info(f"Maximum Energy is {self.e_max:.2g} GeV.")
            self.integral_e_max = self.e_max
        else:
            self.integral_e_max = default_emax

    @classmethod
    def register_subclass(cls, energy_pdf_name):
        """
        Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """

        def decorator(subclass):
            cls.subclasses[energy_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, e_pdf_dict):
        e_pdf_dict = read_e_pdf_dict(e_pdf_dict)

        e_pdf_name = e_pdf_dict["energy_pdf_name"]

        if e_pdf_name not in cls.subclasses:
            raise ValueError("Bad energy PDF name {}".format(e_pdf_name))

        return cls.subclasses[e_pdf_name](e_pdf_dict)

    @staticmethod
    def f(energy):
        pass

    def integrate_over_E(self, f, lower=None, upper=None):
        """Uses Newton's method to integrate function f over the energy
        range. By default, uses 100 GeV to 10 PeV, unless otherwise specified.
        Uses 1000 logarithmically-spaced bins to calculate integral.

        :param f: Function to be integrated
        :param lower: Lower bound for integration
        :param upper: Upper bound for integration
        :return: Integral of function
        """

        if lower is None:
            lower = self.integral_e_min

        if upper is None:
            upper = self.integral_e_max

        return self.integrate(f, lower, upper)

    @staticmethod
    def integrate(f, lower, upper):
        diff_sum, _ = EnergyPDF.piecewise_integrate(f, lower, upper)
        int_sum = np.sum(diff_sum)
        return int_sum

    def piecewise_integrate_over_energy(self, f, lower=None, upper=None):
        """Uses Newton's method to integrate function f over the energy
        range. By default, uses 100 GeV to 10 PeV, unless otherwise specified.
        Uses 1000 logarithmically-spaced bins to calculate integral.

        :param f: Function to be integrated
        :param lower: Lower bound for integration
        :param upper: Upper bound for integration
        :return: Integral of function bins
        """

        if lower is None:
            lower = self.integral_e_min

        if upper is None:
            upper = self.integral_e_max

        diff_sum, e_range = self.piecewise_integrate(f, lower, upper)

        return diff_sum, e_range

    @staticmethod
    def piecewise_integrate(f, lower, upper):
        nsteps = int(1.0e3)

        e_range = np.linspace(np.log10(lower), np.log10(upper), nsteps + 1)
        diff_sum = []

        for i, log_e in enumerate(e_range[:-1]):
            e0 = 10.0 ** (log_e)
            e1 = 10.0 ** (e_range[i + 1])
            diff_sum.append(0.5 * (e1 - e0) * (f(e0) + f(e1)))

        return diff_sum, e_range

    def flux_integral(self, lower=None, upper=None):
        """Integrates over energy PDF to give integrated flux (dN/dT)"""
        return self.integrate_over_E(self.f, lower, upper)

    def fluence_integral(self, lower=None, upper=None):
        """Performs an integral for fluence over a given energy range. This is
        gives the total energy per unit area per second that is radiated.
        """

        def g(energy):
            return energy * self.f(energy)

        return self.integrate_over_E(g, lower, upper)

    def return_energy_parameters(self):
        default = []
        bounds = []
        name = []
        return default, bounds, name

    def simulate_true_energies(self, n_s):
        raise NotImplementedError(
            "Simulate_true_energies not implemented "
            "for {0}".format(self.__class__.__name__)
        )


@EnergyPDF.register_subclass("power_law")
class PowerLaw(EnergyPDF):
    """A Power Law energy PDF. Takes an argument of gamma in the dictionary
    for the init function, where gamma is the spectral index of the Power Law.
    """

    def __init__(self, e_pdf_dict=None):
        """Creates a PowerLaw object, which is an energy PDF based on a power
        law. The power law is generated from e_pdf_dict, which can specify a
        spectral index (Gamma), as well as an optional minimum energy (E Min)
        and a maximum energy (E Max)

        :param e_pdf_dict: Dictionary containing parameters
        """

        if e_pdf_dict is None:
            e_pdf_dict = dict()

        EnergyPDF.__init__(self, e_pdf_dict)

        if "gamma" in list(e_pdf_dict.keys()):
            self.gamma = float(e_pdf_dict["gamma"])

    def weight_mc(self, mc, gamma=None):
        """Returns an array containing the weights for each MC event,
        given that the spectral index gamma has been chosen. Weights each
        event as (E/GeV)^-gamma, and multiplies this by the pre-existing MC
        oneweight value, to give the overall oneweight.

        :param mc: Monte Carlo
        :param gamma: Spectral Index (default is value in e_pdf_dict)
        :return: Weights Array
        """
        # Uses numexpr for faster processing
        ow = mc["ow"]
        trueE = mc["trueE"]
        if gamma is None:
            gamma = self.gamma
        weights = numexpr.evaluate("ow * trueE **(-gamma)")

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            mask = trueE < self.e_min
            weights[mask] = 0.0

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            mask = trueE > self.e_max
            weights[mask] = 0.0

        del ow, trueE

        return weights

    def f(self, energy):
        val = energy**-self.gamma

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            if energy < self.e_min:
                return 0.0

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            if energy > self.e_max:
                return 0.0

        return val

    def flux_integral(self, e_min=None, e_max=None):
        """Integrates over energy PDF to give integrated flux (dN/dT)"""

        if e_max is None:
            e_max = self.integral_e_max
        if e_min is None:
            e_min = self.integral_e_min

        # Integrate over flux to get dN/dt

        if self.gamma == 1.0:
            phi_integral = np.log(e_max / e_min)
        else:
            phi_power = 1 - self.gamma

            phi_integral = (1.0 / phi_power) * (
                (e_max**phi_power) - (e_min**phi_power)
            )

        return phi_integral

    def fluence_integral(self, e_min=None, e_max=None):
        """Performs an integral for fluence over a given energy range. This is
        gives the total energy per unit area per second that is radiated.
        """

        if e_max is None:
            e_max = self.integral_e_max
        if e_min is None:
            e_min = self.integral_e_min

        if self.gamma == 2:
            e_integral = np.log(e_max / e_min)
        else:
            power = 2 - self.gamma

            # Get around astropy power rounding error (does not give
            # EXACTLY 2)

            e_integral = (1.0 / power) * ((e_max**power) - (e_min**power))

        return e_integral

    # def simulate_true_energies(self, n_s, eff_a_f):
    #
    #     log_e_range = np.linspace(
    #         np.log10(self.integral_e_min), np.log10(self.integral_e_max), 1e3)
    #
    #     fluence_ints = [
    #         self.fluence_integral(10**emin, 10**log_e_range[i+1])
    #         for i, emin in enumerate(log_e_range[:-1])
    #     ]
    #
    #     fluence_ints = np.array(fluence_ints)
    #     fluence_ints /= np.sum(fluence_ints)
    #     # fluence_vals = [0] + fluence_vals
    #     # fluence_vals = np.array(fluence_vals)
    #     #
    #     # mean_fluences = 0.5 * (fluence_vals[:-1] + fluence_vals[1:])
    #     # widths = 10**log_e_range[1:] - 10**log_e_range[:-1]
    #     #
    #     # fluence_ints = widths * mean_fluences
    #
    #     print(fluence_ints)
    #     input("")

    def return_energy_parameters(self):
        default = [2.0]
        bounds = [(gamma_range[0], gamma_range[1])]
        name = ["gamma"]
        return default, bounds, name

    def return_injected_parameters(self):
        return {"gamma": self.gamma}


@EnergyPDF.register_subclass("spline")
class Spline(EnergyPDF):
    """A Power Law energy PDF. Takes an argument of gamma in the dictionary
    for the init function, where gamma is the spectral index of the Power Law.
    """

    def __init__(self, e_pdf_dict={}):
        """Creates a PowerLaw object, which is an energy PDF based on a power
        law. The power law is generated from e_pdf_dict, which can specify a
        spectral index (Gamma), as well as an optional minimum energy (E Min)
        and a maximum energy (E Max)

        :param e_pdf_dict: Dictionary containing parameters
        """
        EnergyPDF.__init__(self, e_pdf_dict)

        with open(e_pdf_dict["spline_path"], "rb") as g:
            f = Pickle.load(g)
            self.f = lambda x: np.exp(f(x))

    def weight_mc(self, mc):
        """Returns an array containing the weights for each MC event,
        given that the spectral index gamma has been chosen. Weights each
        event using the energy spline, and multiplies this by the
        pre-existing MC oneweight value, to give the overall oneweight.

        :param mc: Monte Carlo
        :return: Weights Array
        """
        # Uses numexpr for faster processing
        weights = mc["ow"] * self.f(mc["trueE"])

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            mask = mc["trueE"] < self.e_min
            weights[mask] = 0.0

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            mask = mc["trueE"] > self.e_max
            weights[mask] = 0.0

        return weights
