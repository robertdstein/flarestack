import logging
import numpy as np
import os
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.shared import (
    min_angular_err,
    base_floor_quantile,
    floor_pickle,
    pull_pickle,
    weighted_quantile,
)
from flarestack.utils.dynamic_pull_correction import (
    create_quantile_floor_0d,
    create_quantile_floor_0d_e,
    create_quantile_floor_1d,
    create_quantile_floor_1d_e,
    create_pull_0d_e,
    create_pull_1d,
    create_pull_1d_e,
    create_pull_2d,
    create_pull_2d_e,
)
import pickle as Pickle
from scipy.interpolate import interp1d, RectBivariateSpline
from flarestack.utils.make_SoB_splines import (
    get_gamma_support_points,
    get_gamma_precision,
    _around,
)
import numexpr
import inspect

logger = logging.getLogger(__name__)


class BaseFloorClass(object):
    subclasses: dict[str, object] = {}

    def __init__(self, floor_dict):
        self.floor_dict = floor_dict
        self.season = floor_dict["season"]
        self.pickle_name = floor_pickle(floor_dict)

    @classmethod
    def register_subclass(cls, floor_name):
        """Adds a new subclass of BaseFloorClass, with class name equal to
        "floor_name".
        """

        def decorator(subclass):
            cls.subclasses[floor_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, floor_dict):
        floor_name = floor_dict["floor_name"]

        if floor_name not in cls.subclasses:
            raise ValueError("Bad floor name {}".format(floor_name))

        return cls.subclasses[floor_name](floor_dict)

    def floor(self, data):
        return np.array([0.0 for _ in data])

    def apply_floor(self, data):
        mask = data["raw_sigma"] < self.floor(data)
        new_data = data.copy()
        new_data["sigma"][mask] = np.sqrt(
            self.floor(data[mask].copy()) ** 2.0 + data["raw_sigma"][mask].copy() ** 2.0
        )
        return new_data

    def apply_dynamic(self, data):
        return data

    def apply_static(self, data):
        return data


@BaseFloorClass.register_subclass("no_floor")
class NoFloor(BaseFloorClass):
    pass


class BaseStaticFloor(BaseFloorClass):
    """Class that enables the application of a static floor. Rewrites the
    apply_static method to update the "sigma" field AND update the "raw
    sigma" field. This means that the floor is applied once, and then is
    permanently in effect. It will run faster because it will not be
    reapplied in each iteration.
    """

    def apply_static(self, data):
        data = self.apply_floor(data)
        data["raw_sigma"] = data["sigma"]
        return data


class BaseDynamicFloorClass(BaseFloorClass):
    """Class that enables the application of a dynamic floor. Rewrites the
    apply_dynamic method to update only the "sigma" field". This means that
    the floor is applied for each iteration. It will run slower because it will
    be reapplied in each iteration.
    """

    def apply_dynamic(self, data):
        return self.floor(data)


@BaseFloorClass.register_subclass("static_floor")
class StaticFloor(BaseStaticFloor):
    def __init__(self, floor_dict):
        BaseFloorClass.__init__(self, floor_dict)

        try:
            self.min_error = np.deg2rad(floor_dict["min_error_deg"])
        except KeyError:
            self.min_error = min_angular_err

        logger.debug(
            "Applying an angular error floor of {0} degrees".format(
                np.degrees(self.min_error)
            )
        )

    def floor(self, data):
        return np.array([self.min_error for _ in data])


class BaseQuantileFloor(BaseFloorClass):
    def __init__(self, floor_dict):
        try:
            self.floor_quantile = floor_dict["floor_quantile"]
        except KeyError:
            self.floor_quantile = base_floor_quantile
            floor_dict["floor_quantile"] = self.floor_quantile

        BaseFloorClass.__init__(self, floor_dict)

        logger.debug(
            "Applying an angular error floor using quantile {0}".format(
                self.floor_quantile
            )
        )

        if not os.path.isfile(self.pickle_name):
            self.create_pickle()
        else:
            logger.debug(f"Loading from {self.pickle_name}")

        with open(self.pickle_name, "r") as f:
            pickled_data = Pickle.load(f)

        self.f = self.create_function(pickled_data)

    def create_pickle(self):
        pass

    def create_function(self, pickled_array):
        pass


@BaseFloorClass.register_subclass("quantile_floor_0d")
class QuantileFloor0D(BaseQuantileFloor, BaseStaticFloor):
    def create_pickle(self):
        create_quantile_floor_0d(self.floor_dict)

    def create_function(self, pickled_array):
        return lambda data, params: np.array([pickled_array for _ in data])


@BaseFloorClass.register_subclass("quantile_floor_0d_e")
class QuantileFloorEParam0D(BaseQuantileFloor, BaseDynamicFloorClass):
    def create_pickle(self):
        create_quantile_floor_0d_e(self.floor_dict)

    def create_function(self, pickled_array):
        func = interp1d(pickled_array[0], pickled_array[1])
        return lambda data, params: np.array([func(params) for _ in data])


"""
# This class has been commented out since its name is redefined below.
# However, it seems QuantileFloor1D is never explicitly called in the code and is hence likely untested.
@BaseFloorClass.register_subclass("quantile_floor_1d")
class QuantileFloor1D(BaseQuantileFloor, BaseStaticFloor):
    def create_pickle(self):
        create_quantile_floor_1d(self.floor_dict)

    def create_function(self, pickled_array):
        func = interp1d(pickled_array[0], pickled_array[1])
        return lambda data, params: func(data["logE"])
"""


@BaseFloorClass.register_subclass("quantile_floor_1d_e")
class QuantileFloor1D(BaseQuantileFloor, BaseDynamicFloorClass):
    def create_pickle(self):
        create_quantile_floor_1d_e(self.floor_dict)

    def create_function(self, pickled_array):
        func = RectBivariateSpline(
            pickled_array[0],
            pickled_array[1],
            np.log(pickled_array[2]),
            kx=1,
            ky=1,
            s=0,
        )
        return lambda data, params: np.array(
            [np.exp(func(x["logE"], params[0])[0]) for x in data]
        ).T


class BaseAngularErrorModifier(object):
    subclasses: dict[str, object] = {}

    def __init__(self, pull_dict):
        self.season = pull_dict["season"]
        self.floor = BaseFloorClass.create(pull_dict)
        self.pull_dict = pull_dict
        self.pull_name = pull_pickle(pull_dict)

        # precision in gamma
        self.precision = pull_dict.get("gamma_precision", "flarestack")

    @classmethod
    def register_subclass(cls, aem_name):
        """Adds a new subclass of BaseAngularErrorModifier,
        with class name equal to aem_name.

        :param aem_name: AngularErrorModifier name
        :return: AngularErrorModifier object
        """

        def decorator(subclass):
            cls.subclasses[aem_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(
        cls,
        season,
        e_pdf_dict,
        floor_name="static_floor",
        aem_name="no_modifier",
        **kwargs,
    ):
        pull_dict = dict()
        pull_dict["season"] = season
        pull_dict["e_pdf_dict"] = e_pdf_dict
        pull_dict["floor_name"] = floor_name
        pull_dict["aem_name"] = aem_name
        pull_dict.update(kwargs)

        if aem_name not in cls.subclasses:
            raise ValueError("Bad pull name {}".format(aem_name))

        return cls.subclasses[aem_name](pull_dict)

    def pull_correct(self, data, params):
        return data

    def pull_correct_static(self, data):
        data = self.floor.apply_static(data)
        return data

    def pull_correct_dynamic(self, data, params):
        data = self.floor.apply_dynamic(data, params)
        return data

    def create_spatial_cache(self, cut_data, SoB_pdf):
        if len(inspect.getfullargspec(SoB_pdf)[0]) == 2:
            SoB = dict()
            for gamma in get_gamma_support_points(precision=self.precision):
                SoB[gamma] = np.log(SoB_pdf(cut_data, gamma))
        else:
            SoB = SoB_pdf(cut_data)
        return SoB

    def estimate_spatial(self, gamma, spatial_cache):
        if isinstance(spatial_cache, dict):
            return self.estimate_spatial_dynamic(gamma, spatial_cache)
        else:
            return spatial_cache

    def estimate_spatial_dynamic(self, gamma, spatial_cache):
        """Quickly estimates the value of pull for Gamma.
        Uses pre-calculated values for first and second derivatives.
        Uses a Taylor series to estimate S(gamma), unless pull has already
        been calculated for a given gamma.

        :param gamma: Spectral Index
        :param spatial_cache: Median Pull cache
        :return: Estimated value for S(gamma)
        """
        if gamma in list(spatial_cache.keys()):
            val = np.exp(spatial_cache[gamma])
            # val = spatial_cache[gamma]
        else:
            g1 = _around(gamma, self.precision)
            dg = get_gamma_precision(self.precision)

            g0 = _around(g1 - dg, self.precision)
            g2 = _around(g1 + dg, self.precision)

            # Uses Numexpr to quickly estimate S(gamma)

            S0 = spatial_cache[g0]
            S1 = spatial_cache[g1]
            S2 = spatial_cache[g2]

            val = numexpr.evaluate(
                "exp((S0 - 2.*S1 + S2) / (2. * dg**2) * (gamma - g1)**2"
                + " + (S2 -S0) / (2. * dg) * (gamma - g1) + S1)"
            )
            # val = numexpr.evaluate(
            #     "((S0 - 2.*S1 + S2) / (2. * dg**2) * (gamma - g1)**2" + \
            #     " + (S2 -S0) / (2. * dg) * (gamma - g1) + S1)"
            # )

        return val


@BaseAngularErrorModifier.register_subclass("no_pull")
class NoPull(BaseAngularErrorModifier):
    pass


class BaseMedianAngularErrorModifier(BaseAngularErrorModifier):
    def __init__(self, pull_dict):
        BaseAngularErrorModifier.__init__(self, pull_dict)

        if not os.path.isfile(self.pull_name):
            self.create_pickle()
        else:
            logger.debug(f"Loading from {self.pull_name}")

        with open(self.pull_name, "r") as f:
            self.pickled_data = Pickle.load(f)

    def pull_correct(self, f, data):
        data["sigma"] = np.exp(f(data)) * data["raw_sigma"]
        return data

    def create_pickle(self):
        pass

    def create_static(self):
        return lambda data: np.array([1.0 for _ in data])

    def create_dynamic(self, pickled_array):
        return lambda data: np.array([1.0 for _ in data])


class StaticMedianPullCorrector(BaseMedianAngularErrorModifier):
    def __init__(self, pull_dict):
        BaseMedianAngularErrorModifier.__init__(self, pull_dict)

        self.static_f = self.create_static()

    def pull_correct_static(self, data):
        data = self.floor.apply_static(data)
        data = self.pull_correct(self.static_f, data)

        data["raw_sigma"] = data["sigma"]

        return data


class DynamicMedianPullCorrector(BaseMedianAngularErrorModifier):
    def __init__(self, pull_dict):
        BaseAngularErrorModifier.__init__(self, pull_dict)

    def estimate_spatial(self, gamma, spatial_cache):
        return self.estimate_spatial_dynamic(gamma, spatial_cache)

    def pull_correct_dynamic(self, data, param):
        data = self.floor.apply_dynamic(data)
        f = self.create_dynamic(self.pickled_data[param])
        data["sigma"] = np.exp(f(data)) * data["raw_sigma"]
        return data

    def create_spatial_cache(self, cut_data, SoB_pdf):
        """Evaluates the median pull values for all coincidentdata. For each
        value of gamma in self.gamma_support_points, calculates
        the Log(Signal/Background) values for the coincident data. Then saves
        each weight array to a dictionary.

        :param cut_data: Subset of the data containing only coincident events
        :return: Dictionary containing SoB values for each event for each
        gamma value.
        """

        spatial_cache = dict()

        for key in sorted(self.pickled_data.keys()):
            cut_data = self.pull_correct_dynamic(cut_data, key)

            # If gamma is needed to evaluate spatial PDF (say because you
            # have overlapping PDFs and you need to know the weights,
            # then you pass the key. Otherwise just evaluate as normal.

            if len(inspect.getargspec(SoB_pdf)[0]) == 2:
                SoB = SoB_pdf(cut_data, key)
            else:
                SoB = SoB_pdf(cut_data)

            spatial_cache[key] = np.log(SoB)

        return spatial_cache


@BaseAngularErrorModifier.register_subclass("median_0d_e")
class MedianPullEParam0D(DynamicMedianPullCorrector):
    def create_pickle(self):
        create_pull_0d_e(self.pull_dict)

    def create_dynamic(self, pickled_array):
        return lambda data: np.array([pickled_array for _ in data])


@BaseAngularErrorModifier.register_subclass("median_1d")
class MedianPull1D(StaticMedianPullCorrector):
    def create_pickle(self):
        create_pull_1d(self.pull_dict)

    def create_static(self):
        func = interp1d(self.pickled_data[0], self.pickled_data[1])
        return lambda data: func(data["logE"])


@BaseAngularErrorModifier.register_subclass("median_1d_e")
class MedianPullEParam1D(DynamicMedianPullCorrector):
    def create_pickle(self):
        create_pull_1d_e(self.pull_dict)

    def create_dynamic(self, pickled_array):
        func = interp1d(pickled_array[0], pickled_array[1])
        return lambda data: func(data["logE"])


@BaseAngularErrorModifier.register_subclass("median_2d")
class MedianPull2D(StaticMedianPullCorrector):
    def create_pickle(self):
        create_pull_2d(self.pull_dict)

    def create_static(self):
        func = RectBivariateSpline(
            self.pickled_data[0], self.pickled_data[1], self.pickled_data[2]
        )

        return lambda data: [func(x["logE"], x["sinDec"])[0][0] for x in data]
        # return lambda data:


@BaseAngularErrorModifier.register_subclass("median_2d_e")
class MedianPullEParam2D(DynamicMedianPullCorrector):
    def create_pickle(self):
        create_pull_2d_e(self.pull_dict)

    def create_dynamic(self, pickled_array):
        func = RectBivariateSpline(pickled_array[0], pickled_array[1], pickled_array[2])

        return lambda data: func(data["logE"], data["sinDec"])


"""
# NOTE: `mypy` allowed to find some deprecations in the following __main__ block that were never noticed before, because the block is not part of any testing routine. It should be checked whether equivalent code is covered by an unit test, before removing the (now commented) block altogether.

if __name__ == "__main__":
    from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
    from flarestack.analyses.angular_error_floor.plot_bias import (
        get_data,
        weighted_quantile,
    )
    from scipy.stats import norm

    print(norm.cdf(1.0))

    def symmetric_gauss(sigma):
        return 1 - 2 * norm.sf(sigma)

    def gauss_2d(sigma):
        # return symmetric_gauss(sigma) ** 2
        return symmetric_gauss(sigma)

    print(symmetric_gauss(1.0))
    print(gauss_2d(1.0))
    print(gauss_2d(1.177))

    e_pdf_dict = {"Name": "Power Law", "Gamma": 3.0}

    e_pdf = EnergyPDF.create(e_pdf_dict)

    pc = BaseAngularErrorModifier.create(
        IC86_1_dict, e_pdf_dict, "no_floor", "median_2d"
    )
    mc, x, y = get_data(IC86_1_dict)[:10]

    pulls = x / y

    weights = e_pdf.weight_mc(mc)

    median_pull = weighted_quantile(pulls, 0.5, weights)

    def med_pull(data):
        y = np.degrees(data["sigma"])
        pulls = x / y
        med = weighted_quantile(pulls, 0.5, weights)
        return med

    print(mc["sigma"][:5])

    mc = pc.pull_correct_static(mc)

    print(mc["sigma"][:5])

    print(med_pull(mc))

    print(median_pull)

# @BaseAngularErrorModifier.register_subclass('static_pull_corrector')
# class Static1DPullCorrector(BaseAngularErrorModifier):
#
#     def __init__(self, season, e_pdf_dict, **kwargs):
#         BaseAngularErrorModifier.__init__(self, season, e_pdf_dict)


# x = BaseFloorClass.create(IC86_1_dict, e_pdf_dict, "quantile_floor_1d_e",
#                           floor_quantile=0.1)
# for gamma in np.linspace(1.0, 4.0, 4):
#     print data_loader(IC86_1_dict["exp_path"])["logE"][:8]
#     print np.degrees(x.f(data_loader(IC86_1_dict["exp_path"])[:8], [gamma]))
"""
