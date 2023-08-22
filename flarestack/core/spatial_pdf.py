import numpy as np
import healpy as hp
import os
from scipy.stats import norm
from numpy.lib.recfunctions import append_fields
from flarestack.core.astro import angular_distance
from flarestack.shared import bkg_spline_path
from flarestack.utils.make_SoB_splines import load_bkg_spatial_spline


class SpatialPDF:
    """General SpatialPDF holder class. Has separate signal and background
    spatial PDF objects.
    """

    def __init__(self, spatial_pdf_dict, season):
        self.signal = SignalSpatialPDF.create(spatial_pdf_dict)
        self.background = BackgroundSpatialPDF.create(spatial_pdf_dict, season)

        self.simulate_distribution = self.signal.simulate_distribution
        self.signal_spatial = self.signal.signal_spatial
        self.rotate_to_position = self.signal.rotate_to_position

        self.background_spatial = self.background.background_spatial


# ==============================================================================
# Signal Spatial PDFs
# ==============================================================================


class SignalSpatialPDF:
    """Base Signal Spatial PDF class."""

    subclasses: dict[str, object] = {}

    def __init__(self, spatial_pdf_dict):
        pass

    @staticmethod
    def simulate_distribution(source, data):
        return data

    @staticmethod
    def signal_spatial(source, events):
        return

    @classmethod
    def register_subclass(cls, spatial_pdf_name):
        """Adds a new subclass of SpatialPDF, with class name equal to
        "spatial_pdf_name".
        """

        def decorator(subclass):
            cls.subclasses[spatial_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, s_pdf_dict):
        try:
            s_pdf_name = s_pdf_dict["spatial_pdf_name"]
        except KeyError:
            s_pdf_name = "circular_gaussian"

        if s_pdf_name not in cls.subclasses:
            raise ValueError(
                "Bad Signal Spatial PDF name {0} \n"
                "Available names are {1}".format(s_pdf_name, cls.subclasses)
            )

        return cls.subclasses[s_pdf_name](s_pdf_dict)

    @staticmethod
    def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
        """Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
        onto ra3 and dec3, respectively. All angles are treated as radians.
        Essentially rotates the events, so that they behave as if they were
        originally incident on the source.

        :param ra1: Event Right Ascension
        :param dec1: Event Declination
        :param ra2: True Event Right Ascension
        :param dec2: True Event Declination
        :param ra3: Source Right Ascension
        :param dec3: Source Declination
        :return: Returns new Right Ascensions and Declinations
        """
        # Turns Right Ascension/Declination into Azimuth/Zenith for healpy
        phi1 = ra1 - np.pi
        zen1 = np.pi / 2.0 - dec1
        phi2 = ra2 - np.pi
        zen2 = np.pi / 2.0 - dec2
        phi3 = ra3 - np.pi
        zen3 = np.pi / 2.0 - dec3

        # Rotate each ra1 and dec1 towards the pole?
        x = np.array(
            [
                hp.rotator.rotateDirection(
                    hp.rotator.get_rotation_matrix((dp, -dz, 0.0))[0], z, p
                )
                for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)
            ]
        )

        # Rotate **all** these vectors towards ra3, dec3 (source_path)
        zen, phi = hp.rotator.rotateDirection(
            np.dot(
                hp.rotator.get_rotation_matrix((-phi3, 0, 0))[0],
                hp.rotator.get_rotation_matrix((0, zen3, 0.0))[0],
            ),
            x[:, 0],
            x[:, 1],
        )

        dec = np.pi / 2.0 - zen

        ra = phi + np.pi
        return np.atleast_1d(ra), np.atleast_1d(dec)

    def rotate_to_position(self, ev, ra, dec):
        """Modifies the events by reassigning the Right Ascension and
        Declination of the events. Rotates the events, so that they are
        distributed as if they originated from the source. Removes the
        additional Monte Carlo information from sampled events, so that they
        appear like regular data.

        The fields removed are:
            True Right Ascension,
            True Declination,
            True Energy,
            OneWeight

        :param ev: Events
        :param ra: Source Right Ascension (radians)
        :param dec: Source Declination (radians)
        :return: Events (modified)
        """
        names = ev.dtype.names

        # Rotates the events to lie on the source
        ev["ra"], rot_dec = self.rotate(
            ev["ra"], np.arcsin(ev["sinDec"]), ev["trueRa"], ev["trueDec"], ra, dec
        )

        if "dec" in names:
            ev["dec"] = rot_dec
        ev["sinDec"] = np.sin(rot_dec)

        # Deletes the Monte Carlo information from sampled events
        non_mc = [
            name for name in names if name not in ["trueRa", "trueDec", "trueE", "ow"]
        ]
        ev = ev[non_mc].copy()

        return ev


@SignalSpatialPDF.register_subclass("circular_gaussian")
class CircularGaussian(SignalSpatialPDF):
    def simulate_distribution(self, source, data):
        data["ra"] = np.pi + norm.rvs(size=len(data)) * data["sigma"]
        data["dec"] = norm.rvs(size=len(data)) * data["sigma"]
        data["sinDec"] = np.sin(data["dec"])
        data = append_fields(
            data,
            ["trueRa", "trueDec"],
            [np.ones_like(data["dec"]) * np.pi, np.zeros_like(data["dec"])],
        ).copy()

        data = self.rotate_to_position(data, source["ra_rad"], source["dec_rad"]).copy()

        return data.copy()

    @staticmethod
    def signal_spatial(source, cut_data):
        """Calculates the angular distance between the source and the
        coincident dataset. Uses a Gaussian PDF function, centered on the
        source. Returns the value of the Gaussian at the given distances.

        :param source: Single Source
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Spatial PDF values
        """
        distance = angular_distance(
            cut_data["ra"], cut_data["dec"], source["ra_rad"], source["dec_rad"]
        )
        space_term = (
            1.0
            / (2.0 * np.pi * cut_data["sigma"] ** 2.0)
            * np.exp(-0.5 * (distance / cut_data["sigma"]) ** 2.0)
        )

        return space_term


# ==============================================================================
# Background Spatial PDFs
# ==============================================================================


class BackgroundSpatialPDF:
    subclasses: dict[str, object] = {}

    def __init__(self, spatial_pdf_dict, season):
        pass

    @classmethod
    def register_subclass(cls, bkg_spatial_name):
        """Adds a new subclass of BackgroundSpatialPDF, with class name equal to
        "spatial_pdf_name".
        """

        def decorator(subclass):
            cls.subclasses[bkg_spatial_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, s_pdf_dict, season):
        try:
            s_pdf_name = s_pdf_dict["bkg_spatial_pdf"]
        except KeyError:
            s_pdf_name = "zenith_spline"

        if s_pdf_name not in cls.subclasses:
            raise ValueError(
                "Bad Background Spatial PDF name {0} \n"
                "Available names are {1}".format(s_pdf_name, cls.subclasses)
            )

        return cls.subclasses[s_pdf_name](s_pdf_dict, season)

    def background_spatial(self, events):
        return np.ones(len(events))


@BackgroundSpatialPDF.register_subclass("uniform")
class UniformPDF(BackgroundSpatialPDF):
    """A highly-simplified spatial PDF in which events are distributed
    uniformly over the celestial sphere.
    """

    def background_spatial(self, events):
        space_term = (1.0 / (4.0 * np.pi)) * np.ones(len(events))
        return space_term


@BackgroundSpatialPDF.register_subclass("uniform_solid_angle")
class UniformSolidAngle(BackgroundSpatialPDF):
    """Generic class for a background PDF that is uniform over some fixed
    area, and 0 otherwise. Requires an argument to be passed in the
    'spatial_pdf_dict', with key 'background_solid_angle'. In the limit of a
    solid angle of 4 pi, this becomes identical to the UniformPDF class.
    """

    def __init__(self, spatial_pdf_dict, season):
        BackgroundSpatialPDF.__init__(self, spatial_pdf_dict, season)

        try:
            self.solid_angle = spatial_pdf_dict["background_solid_angle"]
        except KeyError:
            raise KeyError(
                "No solid angle passed to UniformSolidAngle class.\n"
                "Please include an entry 'background_solid_angle' "
                "in the 'spatial_pdf_dict'."
            )

        if self.solid_angle > 4 * np.pi:
            raise ValueError(
                "Solid angle {0} was provided, but this is "
                "larger than 4pi. Please provide a valid solid "
                "angle."
            )

    def background_spatial(self, events):
        space_term = (1.0 / self.solid_angle) * np.ones(len(events))
        return space_term


@BackgroundSpatialPDF.register_subclass("zenith_spline")
class ZenithSpline(BackgroundSpatialPDF):
    """A 1D background spatial PDF, in which the background is assumed to be
    uniform in azimuth, but varying as a function of zenith. A
    spline is used to parameterise this distribution.
    """

    def __init__(self, spatial_pdf_dict, season):
        BackgroundSpatialPDF.__init__(self, spatial_pdf_dict, season)
        self.bkg_f = self.create_background_function(season)

    @staticmethod
    def create_background_function(season):
        # Checks if background spatial spline has been created

        if not os.path.isfile(bkg_spline_path(season)):
            season.make_background_spatial()

        return load_bkg_spatial_spline(season)

    def background_spatial(self, events):
        space_term = (1.0 / (2.0 * np.pi)) * np.exp(self.bkg_f(events["sinDec"]))

        return space_term
