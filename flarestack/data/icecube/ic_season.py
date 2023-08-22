import numpy as np
import os
from flarestack.data import Dataset, SeasonWithMC
from flarestack.icecube_utils.dataset_loader import (
    data_loader,
    grl_loader,
    verify_grl_with_data,
)
from flarestack.shared import host_server
from flarestack.core.time_pdf import TimePDF, DetectorOnOffList
from scipy.interpolate import interp1d
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

flarestack_dataset_dir: Optional[str] = os.environ.get("FLARESTACK_DATASET_DIR")

"""
Source data on the WIPAC cluster.
- The 7yr sensitivity data are contained in a directory.
- The 10yr sensitivity data are in a single file. 
"""
WIPAC_dataset_dir = Path("/data/ana/analyses/")
WIPAC_7yr_dir = Path(
    "/data/ana/PointSource/PS/version-002-p01/results/time_integrated_fullsky"
)
WIPAC_10yr_dir = Path(
    "/data/user/tcarver/skylab_scripts/skylab_trunk/doc/analyses/combined_tracks"
)
ref_10yr_filename = (
    "TenYr_E2andE3_sensitivity_and_discpot.npy"  # expected identical at all locations
)

"""
When mirroring the data locally, the expected structure is the following:
- 7yr: ${FLARESTACK_DATASET_DIR}/mirror-7year-PS-sens
- 10yr: ${FLARESTACK_DATASET_DIR}/TenYr_E2andE3_sensitivity_and_discpot.npy

In the DESY mirror, the structure is a bit different:
- 7yr: ${FLARESTACK_DATASET_DIR}/ref_sensitivity/mirror-7year-PS-sens
- 10yr: ${FLARESTACK_DATASET_DIR}/ref_sensitivity/TenYr_E2andE3_sensitivity_and_discpot.npy
"""
mirror_7yr_dirname = "mirror-7year-PS-sens"  # expected identical at all mirrors

# NOTE: the following block is somehow convoluted and the logic should be restructured.
if flarestack_dataset_dir is not None:
    logger.info(f"Loading datasets from {flarestack_dataset_dir} (local)")

    icecube_dataset_dir = Path(flarestack_dataset_dir)

    ref_7yr_path: Path = icecube_dataset_dir / mirror_7yr_dirname
    if ref_7yr_path.is_dir():
        ref_dir_7yr: Optional[Path] = ref_7yr_path
    else:
        logger.warning(f"No 7yr sensitivity directory found at {ref_7yr_path}")
        ref_dir_7yr = None

    ref_10yr_path: Path = icecube_dataset_dir / ref_10yr_filename
    if ref_10yr_path.is_file():
        ref_10yr: Optional[Path] = ref_10yr_path
    else:
        logger.warning(f"No 10yr sensitivity found at {ref_10yr_path}")
        ref_10yr = None
else:
    logger.info(
        "Local dataset directory not found. Assuming we are running on an supported datacenter (WIPAC, DESY), I will try to fetch the data from central storage."
    )

DESY_data_path = Path("/lustre/fs22/group/icecube/data_mirror")
DESY_sens_path = DESY_data_path / "ref_sensitivity"

# Only load from central storage if $FLARESTACK_DATASET_DIR is not set.
if flarestack_dataset_dir is None:
    # NOTE: he following block has no failsafe against changes in the directory structure.
    if host_server == "DESY":
        icecube_dataset_dir = DESY_data_path
        ref_dir_7yr = DESY_sens_path / mirror_7yr_dirname
        ref_10yr = DESY_sens_path / ref_10yr_filename
        logger.info(f"Loading datasets from {icecube_dataset_dir} (DESY)")

    elif host_server == "WIPAC":
        icecube_dataset_dir = WIPAC_dataset_dir
        ref_dir_7yr = WIPAC_7yr_dir
        ref_10yr = WIPAC_10yr_dir / ref_10yr_filename
        logger.info(f"Loading datasets from {icecube_dataset_dir} (WIPAC)")
    else:
        raise ImportError(
            "No IceCube data directory found. Run: \n"
            "export FLARESTACK_DATASET_DIR=/path/to/IceCube/data"
        )


def get_dataset_dir() -> str:
    """
    Returns the path to the IceCube dataset directory. This ensures compatibility with all modules still using the path as a string.
    """
    dataset_dir = str(icecube_dataset_dir)
    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    return dataset_dir


def get_published_sens_ref_dir() -> tuple[Path, Path]:
    """
    Returns the paths to reference sensitivities.
    """
    if (ref_dir_7yr is not None) and (ref_10yr is not None):
        return ref_dir_7yr, ref_10yr
    else:
        error_msg = f"The reference sensitivities were not found. Please set FLARESTACK_DATASET_DIR and ensure it contains the required data."
        raise RuntimeError(error_msg)


@TimePDF.register_subclass("icecube_on_off_list")
class IceCubeRunList(DetectorOnOffList):
    """Custom TimePDF class designed to constructed a pdf from an IceCube
    GoodRunList.
    """

    def parse_list(self):
        """Parses the GoodRunList to build a TimePDF

        Returns:
            t0, t1: min and max time of the GRL
            full_livetime: livetime
            season_f: interpolating function returning 1 or 0 as a function of time
            mjd_to_livetime: function to convert a mjd to livetime [s]
            livetime_to_mjd: function to convert a livetime [s] to mjd

        """
        if list(self.on_off_list["run"]) != sorted(list(self.on_off_list["run"])):
            logger.error("Error in ordering GoodRunList!")
            logger.error("Runs are out of order!")
            self.on_off_list = np.sort(self.on_off_list, order="run")

        if self.t_dict.get("expect_gaps_in_grl", True):
            mask = self.on_off_list["stop"][:-1] == self.on_off_list["start"][1:]

            if np.sum(mask) > 0:
                first_run = self.on_off_list["run"][:-1][mask][0]

                logger.warning(
                    "\nMaybe the IceCube GoodRunList was not produced correctly. \n"
                    "Some runs in the GoodRunList start immediately after the preceding run ends. \n"
                    "For older files, there should be gaps between every run due to detector downtime, "
                    "but some are missing here. \n"
                    f"The first missing gap is between runs {first_run} and {first_run+1}. \n"
                    "Any livetime estimates using this GoodRunList will not be accurate. \n"
                    "This is a known problem affecting older IceCube GoodRunLists. \n"
                    "You should use a newer, corrected GoodRunList. \n"
                    "Flarestack will attempt to stitch these runs together. \n"
                    "However, livetime estimates may be off by several percentage points, "
                    "or even more for very short timescales. \n"
                    "You have been warned!"
                )

                while np.sum(mask) > 0:
                    index = list(mask).index(True)

                    self.on_off_list[index]["stop"] = self.on_off_list[index + 1][
                        "stop"
                    ]
                    self.on_off_list[index]["length"] += self.on_off_list[index + 1][
                        "length"
                    ]
                    self.on_off_list[index]["events"] += self.on_off_list[index + 1][
                        "events"
                    ]

                    mod_mask = np.arange(len(self.on_off_list)) == index + 1

                    self.on_off_list = self.on_off_list[~mod_mask]

                    mask = (
                        self.on_off_list["stop"][:-1] == self.on_off_list["start"][1:]
                    )

            mask = self.on_off_list["stop"][:-1] < self.on_off_list["start"][1:]

            if np.sum(~mask) > 0:
                first_run = self.on_off_list["run"][:-1][~mask][0]

                logger.error("The IceCube GoodRunList was not produced correctly.")
                logger.error(
                    "Some runs in the GoodRunList start before the preceding run has ended."
                )
                logger.error("Under no circumstances should runs overlap.")
                logger.error(
                    f"The first overlap is between runs {first_run} and {first_run+1}."
                )
                logger.error(
                    "Any livetime estimates using this GoodRunList will not be accurate."
                )
                logger.error(
                    "This is a known problem affecting older IceCube GoodRunLists."
                )
                logger.error("You should use a newer, corrected GoodRunList.")
                logger.error("Flarestack will attempt to stitch these runs together.")
                logger.error(
                    "However, livetime estimates may be off by several percentage points, "
                    "or even more for very short timescales."
                )
                logger.error("You have been warned!")

                while np.sum(~mask) > 0:
                    index = list(~mask).index(True)

                    self.on_off_list[index]["stop"] = self.on_off_list[index + 1][
                        "stop"
                    ]
                    self.on_off_list[index]["length"] += self.on_off_list[index + 1][
                        "length"
                    ]
                    self.on_off_list[index]["events"] += self.on_off_list[index + 1][
                        "events"
                    ]

                    mod_mask = np.arange(len(self.on_off_list)) == index + 1

                    self.on_off_list = self.on_off_list[~mod_mask]

                    mask = self.on_off_list["stop"][:-1] < self.on_off_list["start"][1:]

        t0 = min(self.on_off_list["start"])
        t1 = max(self.on_off_list["stop"])

        full_livetime = np.sum(self.on_off_list["length"])

        step = 1e-12

        t_range = [t0 - step]

        f = [0.0]

        # MJD timestaps marking start and stop time of each run
        mjd = [0.0]
        # cumulative livetime at each timestamp [unit to be checked]
        livetime = [0.0]
        # cumulative sum of run lengths [unit to be checked]
        total_t = 0.0

        for i, run in enumerate(self.on_off_list):
            mjd.append(run["start"])
            livetime.append(total_t)
            total_t += run["length"]
            mjd.append(run["stop"])
            livetime.append(total_t)

            # extends t_range and f with a box function of value 1 between the start and stop of the run
            # adds zero values at times immediately adjacent to the start and stop of the run
            t_range.extend(
                [run["start"] - step, run["start"], run["stop"], run["stop"] + step]
            )
            f.extend([0.0, 1.0, 1.0, 0.0])

        stitch_t = t_range
        stitch_f = f

        if stitch_t != sorted(stitch_t):
            logger.error("Error in ordering GoodRunList somehow!")
            logger.error("Runs are out of order!")

            for i, t in enumerate(stitch_t):
                if t != sorted(stitch_t)[i]:
                    print(t, sorted(stitch_t)[i])
                    print(stitch_t[i - 1 : i + 2])
                    print(sorted(stitch_t)[i - 1 : i + 2])
                    key = int((i - 1) / 4)
                    print(self.on_off_list[key : key + 2])
                    input("????")

            raise Exception(
                f"Runs in GoodRunList are out of order for {self.on_off_list}. Check that!"
            )

        # end of the livetime function domain
        mjd.append(1e5)

        livetime.append(total_t)

        season_f = interp1d(stitch_t, np.array(stitch_f), kind="linear")
        # cumulative livetime [[unit to be checked]] as a function of the date [mjd]
        mjd_to_livetime = interp1d(mjd, livetime, kind="linear")
        # date [mjd] at which a given livetime [unit to be checked] is reached
        livetime_to_mjd = interp1d(livetime, mjd, kind="linear")
        return t0, t1, full_livetime, season_f, mjd_to_livetime, livetime_to_mjd


class IceCubeDataset(Dataset):
    pass

    """
    Just a placeholder in case this __init__ method is needed in the future. 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    """


class IceCubeSeason(SeasonWithMC):
    def __init__(
        self,
        season_name,
        sample_name,
        exp_path,
        mc_path,
        grl_path,
        sin_dec_bins,
        log_e_bins,
        expect_gaps_in_grl=True,
        **kwargs,
    ):
        SeasonWithMC.__init__(
            self, season_name, sample_name, exp_path, mc_path, **kwargs
        )
        self.grl_path = grl_path
        self.all_paths.append(grl_path)
        self.sin_dec_bins = sin_dec_bins
        self.log_e_bins = log_e_bins
        self._expect_gaps_in_grl = expect_gaps_in_grl

    def check_data_quality(self):
        verify_grl_with_data(self)

    def get_grl(self):
        return grl_loader(self)

    def load_data(self, path, **kwargs):
        return data_loader(path, **kwargs)

    def build_time_pdf_dict(self):
        """Function to build a pdf for the livetime of the season. By
        default, this exploits the good run list (GRL)

        :return: Time pdf dictionary
        """

        t_pdf_dict = {
            "time_pdf_name": "icecube_on_off_list",
            "on_off_list": self.get_grl(),
            "expect_gaps_in_grl": self._expect_gaps_in_grl,
        }

        return t_pdf_dict
