import logging
import numpy as np
from scipy.interpolate import interp1d


logger = logging.getLogger(__name__)


def box_func(t, t0, t1):
    """Box function that is equal to 1 between t0 and t1, and 0 otherwise.
    Equal to 0.5 at to and t1.

    :param t: Time to be evaluated
    :param t0: Start time of box
    :param t1: End time of box
    :return: Value of Box function at t
    """
    val = 0.5 * ((np.sign(t - t0)) - (np.sign(t - t1)))
    return val


def decay_fct(t, t0, decay_time, truncation=np.inf):
    """
    Decay function that is equal to 0 before t0, equal to 1 at t0 and then decays with a decay time

    :param t: time to be evaluated
    :param t0: start time of the function
    :param decay_time: decay time
    :param truncation: truncation time, function will give zero for t < truncation
    :return: value at t
    """
    val = (
        np.heaviside(t - t0, 1)
        * (1 / (1 + (t - t0) / decay_time))
        * np.heaviside(t0 + truncation - t, 1)
    )
    return val


def decay_fct_integral(tstart, tend, t0, decay_time, truncation=np.inf):
    """
    The integral function of decay_function based on the analytical form

    :param tstart: float, integrating from
    :param tend: float, integrating to
    :param t0: float, parameter t0 in decay_function, start time of the decay function
    :param decay_time: float, decay time
    :param truncation: float, truncation time, decay function will be 0 for t < truncation
    :return: float
    """
    val = decay_time * np.log(
        (decay_time + np.maximum(0, np.minimum(tend - t0, truncation)))
        / (decay_time + np.maximum(tstart - t0, 0))
    )
    return val


def read_t_pdf_dict(t_pdf_dict):
    """Ensures backwards compatibility for t_pdf_dict objects"""

    maps = [
        ("Offset", "offset"),
        ("Pre-Window", "pre_window"),
        ("Post-Window", "post_window"),
        ("Fixed Ref Time (MJD)", "fixed_ref_time_mjd"),
        ("Name", "time_pdf_name"),
        ("Max Offset", "max_offset"),
        ("Min Offset", "min_offset"),
        ("Max Flare", "max_flare"),
    ]

    for old_key, new_key in maps:
        if old_key in list(t_pdf_dict.keys()):
            logger.warning(
                "Deprecated t_pdf_key '{0}' was used. "
                "Please use '{1}' in future.".format(old_key, new_key)
            )
            t_pdf_dict[new_key] = t_pdf_dict[old_key]

    name_maps = [
        ("Steady", "steady"),
        ("Box", "box"),
        ("FixedRefBox", "fixed_ref_box"),
        ("FixedEndBox", "custom_source_box"),  # Not a typo! Class renamed.
    ]

    for old_key, new_key in name_maps:
        if t_pdf_dict["time_pdf_name"] == old_key:
            logger.warning(
                "Deprecated time_pdf_name '{0}' was used. "
                "Please use '{1}' in future.".format(old_key, new_key)
            )
            t_pdf_dict["time_pdf_name"] = new_key

    return t_pdf_dict


class TimePDF:
    subclasses: dict[str, object] = {}

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        self.t_dict = t_pdf_dict

        if livetime_pdf is not None:
            self.livetime_f = lambda x: livetime_pdf.livetime_f(
                x
            )  # * livetime_pdf.livetime
            self.livetime_pdf = livetime_pdf
            self.t0 = livetime_pdf.sig_t0()
            self.t1 = livetime_pdf.sig_t1()
            (
                self.mjd_to_livetime,
                self.livetime_to_mjd,
            ) = self.livetime_pdf.get_mjd_conversion()

        else:
            self.livetime_pdf = None
            self.t0 = -np.inf
            self.t1 = np.inf
            self.livetime = None

    @classmethod
    def register_subclass(cls, time_pdf_name):
        def decorator(subclass):
            cls.subclasses[time_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, t_pdf_dict, livetime_pdf=None):
        t_pdf_dict = read_t_pdf_dict(t_pdf_dict)

        t_pdf_name = t_pdf_dict["time_pdf_name"]

        if t_pdf_name not in cls.subclasses:
            raise ValueError(
                f"Bad time PDF name {t_pdf_name}. "
                f"Available PDFs are {cls.subclasses.keys()}"
            )

        return cls.subclasses[t_pdf_name](t_pdf_dict, livetime_pdf)

    def product_integral(self, t, source):
        """Calculates the product of the given signal PDF with the season box
        function. Thus gives 0 everywhere outside the season, and otherwise
        the value of the normalised integral. The season function is offset
        by 1e-9, to ensure that f(t1) is equal to 1. (i.e the function is
        equal to 1 at the end of the box).

        :param t: Time
        :param source: Source to be considered
        :return: Product of signal integral and season
        """

        f = np.array(self.signal_integral(t, source))

        f[f < 0.0] = 0.0
        f[f > 1.0] = 1.0

        return f

    def inverse_cumulative(self, source):
        """Calculates the inverse cumulative of the time PDF.
        First, calculates the values for the integral of the signal PDF within
        the season. Then rescales these values, such that the start of the
        season yields 0, and then end of the season yields 1. Creates a
        function to interpolate between these values. Then, for a number
        between 0 and 1, the interpolated function will return the MJD time
        at which that fraction of the cumulative distribution was reached.

        :param source: Source to be considered
        :return: Interpolated function
        """
        max_int = self.product_integral(self.sig_t1(source), source)
        min_int = self.product_integral(self.sig_t0(source), source)

        logger.debug("Integral of the signal PDF:")
        logger.debug(f"F({self.sig_t0(source)}) = {min_int}")
        logger.debug(f"F({self.sig_t1(source)}) = {max_int}")

        fraction = max_int - min_int

        # number of points for the basis of interpolation
        n_points = int(1e4)

        t_range = np.linspace(
            float(self.sig_t0(source)), float(self.sig_t1(source)), n_points
        )
        cumu = (self.product_integral(t_range, source) - min_int) / fraction

        # Checks to ensure the cumulative fraction spans 0 to 1
        if max(cumu) > 1.0:
            raise Exception("Cumulative Distribution exceeds 1.")
        elif min(cumu) < 0.0:
            raise Exception("Cumulative Distribution extends below 0.")

        return interp1d(cumu, t_range, kind="linear")

    def simulate_times(self, source, n_s):
        """Randomly draws times for n_s events for a given source,
        all lying within the current season. The values are based on an
        interpolation of the integrated time PDF.

        :param source: Source being considered
        :param n_s: Number of event times to be simulated
        :return: Array of times in MJD for a given source
        """
        f = self.inverse_cumulative(source)

        sims = f(np.random.uniform(0.0, 1.0, n_s))

        return sims

    def f(self, t, source):
        raise NotImplementedError(
            "No 'f' has been implemented for {0}".format(self.__class__.__name__)
        )

    def sig_t0(self, source):
        """Calculates the starting time for the time pdf.

        :param source: Source to be considered
        :return: Time of PDF start
        """
        raise NotImplementedError(
            "sig_t0 function not implemented for " "{0}".format(self.__class__.__name__)
        )

    def sig_t1(self, source):
        """Calculates the ending time for the time pdf.

        :param source: Source to be considered
        :return: Time of PDF end
        """
        raise NotImplementedError(
            "sig_t1 function not implemented for " "{0}".format(self.__class__.__name__)
        )

    def integral_to_infinity(self, source):
        max_int = self.product_integral(self.sig_t1(source), source)
        min_int = self.product_integral(self.sig_t0(source), source)
        return max_int - min_int

    # def convert_livetime_mjd(self):
    #     t_range = np.linspace(
    #         self.livetime_pdf.sig_t0(), self.livetime_pdf.sig_t1(), int(1e3))
    #
    #     f_range = np.array([self.livetime_pdf.livetime_f(t) for t in t_range])
    #     f_range /= np.sum(f_range)
    #
    #     sum_range = [np.sum(f_range[:i]) for i, _ in enumerate(f_range)]
    #
    #     mjd_to_livetime = interp1d(t_range, sum_range)
    #     livetime_to_mjd = interp1d(sum_range, t_range)
    #     return mjd_to_livetime, livetime_to_mjd

    def effective_injection_time(self, source=None):
        raise NotImplementedError

    def get_livetime(self):
        if self.livetime is not None:
            return self.livetime
        raise NotImplementedError

    def get_mjd_conversion(self):
        return self.mjd_to_livetime, self.livetime_to_mjd


@TimePDF.register_subclass("steady")
class Steady(TimePDF):
    """The time-independent case for a Time PDF. Requires no additional
    arguments in the dictionary for __init__. Used for a steady source that
    is continuously emitting.
    """

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        TimePDF.__init__(self, t_pdf_dict, livetime_pdf)

        if self.livetime_pdf is None:
            raise ValueError(
                "No livetime pdf has been provided, but a Steady "
                "Time PDF has been chosen. Without a fixed start "
                "and end point, no PDF can be defined. Please "
                "provide a livetime_pdf, or use a different Time "
                "PDF class such as FixedEndBox."
            )
        else:
            self.livetime = livetime_pdf.get_livetime()

    def f(self, t, source):
        """In the case of a steady source, the signal PDF is a uniform PDF in
        time. It is thus simply equal to the season_f, normalised with the
        length of the season to give an integral of 1. It is thus equal to
        the background PDF.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        return 1.0 / self.livetime  # self.livetime_f(t)# * self.livetime

    def signal_integral(self, t, source):
        """In the case of a steady source, the signal PDF is a uniform PDF in
        time. Thus, the integral is simply a linear function increasing
        between t0 (box start) and t1 (box end). After t1, the integral is
        equal to 1, while it is equal to 0 for t < t0.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """

        return self.mjd_to_livetime(t) / self.livetime

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        entire season. Thus returns the start and end times for the season.

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = self.t0
        end = self.t1

        return start, end

    def effective_injection_time(self, source=None):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        season_length = self.integral_to_infinity(source) * self.livetime

        return season_length * (60 * 60 * 24)

    def raw_injection_time(self, source):
        """Calculates the 'raw injection time' which is the injection time
        assuming a detector with 100% uptime. Useful for calculating source
        emission times for source-frame energy estimation.

        :param source: Source to be considered
        :return: Time in seconds for 100% uptime
        """
        return (self.t1 - self.t0) * (60 * 60 * 24)

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """
        return self.t0

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return self.t1


@TimePDF.register_subclass("box")
class Box(TimePDF):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. Requires arguments of
    Pre-Window and Post_window, and gives a box from Pre-Window days before
    the reference time to Post-Window days after the reference time.
    """

    def __init__(self, t_pdf_dict, season):
        TimePDF.__init__(self, t_pdf_dict, season)
        self.pre_window = self.t_dict["pre_window"]
        self.post_window = self.t_dict["post_window"]

        if "offset" in list(t_pdf_dict.keys()):
            self.offset = self.t_dict["offset"]
            self.pre_window -= self.offset
            self.post_window += self.offset

        try:
            if self.t_dict["livetime"] is True:
                logger.debug("Using time PDF as a detector livetime PDF.")
                self.mjd_to_livetime = lambda x: x - self.sig_t0([])
                self.livetime_to_mjd = lambda x: x + self.sig_t0([])
                self.livetime = self.t_dict["post_window"] + self.t_dict["pre_window"]
                # print(self.t1, self.t0, self.livetime, self.livetime_to_mjd(7.), self.pre_window, self.sig_t0([]))
        except KeyError:
            pass

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """
        return max(self.t0, source["ref_time_mjd"] - self.pre_window)

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return min(source["ref_time_mjd"] + self.post_window, self.t1)

    def f(self, t, source=None):
        """In this case, the signal PDF is a uniform PDF for a fixed duration of
        time. It is normalised with the length of the box in LIVETIME rather
        than days, to give an integral of 1.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        length = self.mjd_to_livetime(t1) - self.mjd_to_livetime(t0)

        if length > 0.0:
            return box_func(t, t0, t1) / length

        else:
            return np.zeros_like(t)

    def signal_integral(self, t, source):
        """In this case, the signal PDF is a uniform PDF for a fixed duration of
        time. Thus, the integral is simply a linear function increasing
        between t0 (box start) and t1 (box end). After t1, the integral is
        equal to 1, while it is equal to 0 for t < t0.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """

        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        length = self.mjd_to_livetime(t1) - self.mjd_to_livetime(t0)

        return np.abs(
            (self.mjd_to_livetime(t) - self.mjd_to_livetime(t0))
            * box_func(t, t0, t1 + 1e-9)
            / length
        )

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        period of overlap of the flare and the box. Thus, for a given season,
        return the source and data

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = max(self.t0, self.sig_t0(source))
        end = min(self.t1, self.sig_t1(source))

        return start, end

    def effective_injection_time(self, source=None):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        t0 = self.mjd_to_livetime(self.sig_t0(source))
        t1 = self.mjd_to_livetime(self.sig_t1(source))
        time = (t1 - t0) * 60 * 60 * 24
        return max(time, 0.0)

    def raw_injection_time(self, source):
        """Calculates the 'raw injection time' which is the injection time
        assuming a detector with 100% uptime. Useful for calculating source
        emission times for source-frame energy estimation.

        :param source: Source to be considered
        :return: Time in seconds for 100% uptime
        """

        diff = max(self.sig_t1(source) - self.sig_t0(source), 0)
        return diff * (60 * 60 * 24)


@TimePDF.register_subclass("fixed_ref_box")
class FixedRefBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is unique for each source. The sources must have
    a field "Start Time (MJD)" and another "End Time (MJD)", specifying the
    period of the Time PDF.
    """

    def __init__(self, t_pdf_dict, season):
        Box.__init__(self, t_pdf_dict, season)
        self.fixed_ref = t_pdf_dict["fixed_ref_time_mjd"]

    def sig_t0(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        return max(self.t0, self.fixed_ref - self.pre_window)

    def sig_t1(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return min(self.fixed_ref + self.post_window, self.t1)


@TimePDF.register_subclass("fixed_end_box")
class FixedEndBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is the same for all sources.
    """

    def __init__(self, t_pdf_dict, season):
        self.start_time_mjd = t_pdf_dict["start_time_mjd"]
        self.end_time_mjd = t_pdf_dict["end_time_mjd"]
        if "offset" in t_pdf_dict:
            self.offset = self.t_dict["offset"]
        else:
            self.offset = 0
        TimePDF.__init__(self, t_pdf_dict, season)

    def sig_t0(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        t0 = self.start_time_mjd + self.offset

        return max(self.t0, t0)

    def sig_t1(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """

        t1 = self.end_time_mjd + self.offset

        return min(t1, self.t1)

    def get_livetime(self):
        if self.livetime_pdf is not None:
            raise Exception("Livetime PDF already provided.")
        else:
            return self.sig_t1() - self.sig_t0()

    def get_mjd_conversion(self):
        if self.livetime_pdf is not None:
            raise Exception("Livetime PDF already provided.")
        else:
            mjd_to_l = lambda x: x - self.sig_t0()
            l_to_mjd = lambda x: x + self.sig_t0()
            return mjd_to_l, l_to_mjd


@TimePDF.register_subclass("custom_source_box")
class CustomSourceBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is unique for each source. The sources must have
    a field "Start Time (MJD)" and another "End Time (MJD)", specifying the
    period of the Time PDF.
    """

    def __init__(self, t_pdf_dict, season):
        TimePDF.__init__(self, t_pdf_dict, season)
        if "offset" in t_pdf_dict:
            self.offset = self.t_dict["offset"]
        else:
            self.offset = 0

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        t0 = source["start_time_mjd"] + self.offset

        return max(self.t0, t0)

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """

        t1 = source["end_time_mjd"] + self.offset

        return min(t1, self.t1)


@TimePDF.register_subclass("detector_on_off_list")
class DetectorOnOffList(TimePDF):
    """TimePDF with predefined on/off periods. Can be used for a livetime
    function, in which observations are divided into runs with gaps. Can also
    be used for e.g pre-defined interesting period for a variable source.
    """

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        TimePDF.__init__(self, t_pdf_dict, livetime_pdf)

        try:
            self.on_off_list = t_pdf_dict["on_off_list"]
        except KeyError:
            raise KeyError(
                "No 'on_off_list' found in t_pdf_dict. The "
                "following fields were provided: {0}".format(t_pdf_dict.keys())
            )
        (
            self.t0,
            self.t1,
            self._livetime,
            self.season_f,
            self.mjd_to_livetime,
            self.livetime_to_mjd,
        ) = self.parse_list()

    def f(self, t, source=None):
        return self.season_f(t) / self.livetime

    def livetime_f(self, t, source=None):
        return self.f(t, source)

    def sig_t0(self, source=None):
        return self.t0

    def sig_t1(self, source=None):
        return self.t1

    def parse_list(self):
        raise NotImplementedError

    def get_livetime(self):
        return self._livetime

    def get_mjd_conversion(self):
        return self.mjd_to_livetime, self.livetime_to_mjd

    def signal_integral(self, t, source=None):
        return self.mjd_to_livetime(t) / self.get_livetime()

    def inverse_cumulative(self, source=None):
        return lambda t: self.livetime_to_mjd(t * self.get_livetime())


@TimePDF.register_subclass("decay")
class DecayPDF(TimePDF):
    r"""
    Describing a decaying emission as expected for interacting supernovae (type IIn) following

    Zirakashvili, V.N., und V.S. Ptuskin.
    „Type IIn supernovae as sources of high energy astrophysical neutrinos“.
    Astropart. Phys. 78 (2016): 28–34.
    https://doi.org/10.1016/j.astropartphys.2016.02.004.

    .. math::
        \mathcal{T}(t) \varpropto \left( 1 + \frac{t}{t_{\mathrm{decay}}} \right)^{-1}

    The elements in the `t_pdf_dict` are

        - `decay_time`: :math:`t_{\mathrm{decay}}` in the equation above
        - `decay_length`: A time after which :math:`\mathcal{T}(t)` is truncated such that

    .. math::
        \mathcal{T}(t) \varpropto \begin{cases}
            \left( 1 + \frac{t}{t_{\mathrm{decay}}} \right)^{-1},\, &\mathrm{for}\, t < t_{\mathrm{length}} \\
            0, &\mathrm{else}
        \end{cases}

    """

    def __init__(self, t_pdf_dict, season):
        TimePDF.__init__(self, t_pdf_dict, season)

        if not "decay_time" in self.t_dict:
            raise KeyError(
                "In order to use a Decay PDF, a decay time has to be included in the time pdf dictionary!"
            )

        self.decay_time = self.t_dict["decay_time"]
        self.decay_length = (
            self.t_dict["decay_length"] if "decay_length" in self.t_dict else np.inf
        )
        if not "decay_length" in self.t_dict:
            logger.warning("No decay length given! Assuming endless decay")

    def decay_function(self, t, t0):
        return decay_fct(t, t0, self.decay_time, self.decay_length)

    def decay_integral(self, a, b, t0):
        return decay_fct_integral(a, b, t0, self.decay_time, self.decay_length)

    def sig_t0(self, source):
        """
        Gives the start time of the signal from a source. If the start time lies within the season,
        the start time is the source's "ref_time_mjd". If not it"s the start of the season.
        :param source: source to be considered
        :return: time of signal start
        """
        return max(self.t0, source["ref_time_mjd"])

    def sig_t1(self, source):
        """
        Gives the end time of a signal.
        For an endless decay, that's just the end of the season.
        If the decay length is not infinite, the signal might end before the season ends.
        :param source: source to be considered
        :return: end time of signal
        """
        return min((self.t1, source["ref_time_mjd"] + self.decay_length))

    def signal_integral(self, t, source):
        """
        Gives the integrated signal using decay_fct_integral()
        :param t: float or array like
        :param source: the sources to be considered
        :return: float
        """

        integration_result = self.decay_integral(self.t0, t, source["ref_time_mjd"])
        normalization_factor = self.decay_integral(
            self.t0, self.t1, source["ref_time_mjd"]
        )
        return integration_result / normalization_factor

    def effective_injection_time(self, source=None):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        sig_t0 = self.sig_t0(source)
        sig_t1 = self.sig_t1(source)
        complete_integral = self.decay_integral(
            source["ref_time_mjd"],
            source["ref_time_mjd"] + self.decay_length,
            source["ref_time_mjd"],
        )
        portion = self.decay_integral(sig_t0, sig_t1, source["ref_time_mjd"])
        weight = portion / complete_integral

        t0 = self.mjd_to_livetime(sig_t0)
        t1 = self.mjd_to_livetime(sig_t1)
        livetime_weight = (t1 - t0) / (sig_t1 - sig_t0)

        return self.decay_length * weight * livetime_weight * 60 * 60 * 24

    def raw_injection_time(self, source):
        """Calculates the 'raw injection time' which is the injection time
        assuming a detector with 100% uptime. Useful for calculating source
        emission times for source-frame energy estimation.

        :param source: Source to be considered
        :return: Time in seconds for 100% uptime
        """

        sig_t0 = self.sig_t0(source)
        sig_t1 = self.sig_t1(source)
        complete_integral = self.decay_integral(
            source["ref_time_mjd"],
            source["ref_time_mjd"] + self.decay_length,
            source["ref_time_mjd"],
        )
        portion = self.decay_integral(sig_t0, sig_t1, source["ref_time_mjd"])
        weight = portion / complete_integral

        return self.decay_length * weight * 60 * 60 * 24

    def f(self, t, source=None):
        """
        In this case the PDF is the decay function, normalized to the integral over livetime
        :param t: float or array_like, Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        if t0 >= t1:
            # in this case the source emission ends before the start of the season so the pdf is zero everywhere
            return np.zeros_like(t)

        # to normalize the function, integrate over the whole livetime
        a, b = self.mjd_to_livetime(t0), self.mjd_to_livetime(t1)
        normalization_factor = self.decay_integral(
            a, b, source["ref_time_mjd"] - self.livetime_to_mjd(0)
        )

        if normalization_factor > 0.0:
            r = self.decay_function(t, source["ref_time_mjd"])
            return r / normalization_factor

        else:
            # the normalization factor should always be greater than zero,
            # so here something went wrong.
            logger.error(
                f"\nintegrating from {a:.2f} to {b:.2f}. \n"
                f"t = {t} \n"
                f"t0 = {t0:.2f} \n"
                f"t_pp={self.decay_time:.2f} \n"
                f"trunc={self.decay_length:.2f}"
            )
            raise ValueError("Normalization factor <= 0!")


# from data.icecube_pointsource_7_year import ps_v002_p01
# from shared import catalogue_dir
#
# cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
# catalogue = np.load(cat_path)
#
# for t in np.linspace(0, 500, 101):
#     time_dict = {
#         "Name": "FixedRefBox",
#         "Fixed Ref Time (MJD)": 55650,
#         "Pre-Window": 0,
#         "Post-Window": t
#     }
#
#     livetime = 0
#     tot = 0
#
#     for season in ps_v002_p01[-2:-1]:
#         tpdf = TimePDF.create(time_dict, season)
#         livetime += tpdf.effective_injection_time(
#             catalogue
#         )
#         tot += tpdf.livetime
#
#     print t, livetime/(60*60*24), tot
