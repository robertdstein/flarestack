import copy
import json
import logging
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional

from flarestack.shared import (
    fs_dir,
    log_dir,
    fs_scratch_dir,
    make_analysis_pickle,
    host_server,
    inj_dir_name,
    name_pickle_output_dir,
    cluster_dir,
)
from flarestack.core.multiprocess_wrapper import run_multiprocess
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.results import ResultsHandler


logger = logging.getLogger(__name__)


class Submitter(object):
    # in other modules, such attribute is called 'subclass'
    # is this functionally different?
    submitter_dict: dict[str, object] = dict()

    def __init__(
        self,
        mh_dict,
        use_cluster,
        n_cpu=None,
        do_sensitivity_scale_estimation=False,
        remove_old_results=False,
        **cluster_kwargs,
    ):
        """
        A class that takes care of submitting the trial calculations.
        Also can estimate the sensitivity scale before submitting.
        :param mh_dict: dict, MinimisationHandler dictionary
        :param use_cluster: bool, whether to run the trials locally or on the cluster
        :param n_cpu: int, number of cores to use
        :param do_sensitivity_scale_estimation: str, containing 'asimov', 'quick_injections' or both
        :param remove_old_results: bool, if True will delete directories containing injection values and pickled
                                         results from previous trials
        :param cluster_kwargs: keyword arguments used by the cluster
        """
        self.mh_dict = copy.deepcopy(mh_dict)
        self.use_cluster = use_cluster
        self.n_cpu = os.cpu_count() - 1 if isinstance(n_cpu, type(None)) else n_cpu
        self.job_id = None
        self.remove_old_results = remove_old_results
        self.do_sensitivity_scale_estimation = do_sensitivity_scale_estimation
        self.sens_guess = self.disc_guess = None
        self.successful_guess_by_quick_injections = False
        self.cluster_kwargs = cluster_kwargs

    def __str__(self):
        s = (
            f'\n----- Submitter for {self.mh_dict["name"]} -----\n'
            f'{"" if self.use_cluster else "not "}using cluster \n'
            f"using {self.n_cpu} CPUs locally\n"
            f"job-id: {self.job_id} \n"
            f'{self.do_sensitivity_scale_estimation if self.do_sensitivity_scale_estimation else "no"} '
            f"scale estimation \n"
        )

        if self.cluster_kwargs:
            s += "cluster kwargs: \n"
            for k, v in self.cluster_kwargs.items():
                s += f"  {k}: {v} \n"
        return s

    def submit_cluster(self, mh_dict):
        """Splits the trials into jobs and submits them to be calculated on the cluster"""
        raise NotImplementedError

    def submit_local(self, mh_dict):
        """Uses the MultiprocessWrapper to split the trials into jobs and run them locally"""
        # max CPU number is all but one
        make_analysis_pickle(mh_dict)
        n_cpu = min(self.n_cpu, os.cpu_count() - 1)
        run_multiprocess(n_cpu=n_cpu, mh_dict=mh_dict)

    def submit(self, mh_dict):
        if self.remove_old_results:
            self._clean_injection_values_and_pickled_results(self.mh_dict["name"])
        if self.use_cluster:
            self.submit_cluster(mh_dict)
        else:
            self.submit_local(mh_dict)

    def wait_for_job(self):
        """Waits until the cluster is finished processing the job with the ID self.job_id"""
        raise NotImplementedError

    # @staticmethod
    # def _wait_for_cluster(job_ids=None):
    #     raise NotImplementedError

    @staticmethod
    def get_pending_ids():
        raise NotImplementedError

    @staticmethod
    def wait_for_cluster(job_ids=None):
        """
        Waits until the cluster is done. Wait for all jobs if job_ids is None or give a list of IDs

        :param job_ids: list, optional, if given, specifies the IDs of the obs that will be waited on
        """

        # If no job IDs are specified, get all IDs currently listed for this user
        cls = Submitter.get_submitter_class()
        if not job_ids:
            # job_ids = np.unique(cls.get_ids(DESYSubmitter.status_cmd))
            job_ids = cls.get_pending_ids()

        for id in job_ids:
            logger.info(f"waiting for job {id}")
            # create a submitter, it does not need the mh_dict when no functions are calles
            s = cls(None, None)
            s.job_id = id  # set the right job_id
            s.wait_for_job()  # use the built-in function to wait for completion of that job

    @property
    def _quick_injections_name(self):
        name = self.mh_dict["name"]
        return f"{name if not name.endswith(os.sep) else name[:-1]}_quick_injection/"

    def run_quick_injections_to_estimate_sensitivity_scale(self):
        """
        Roughly estimates the injection scale in order to find a better scale range.
        The quick injection trials are run locally.
        Note that a scale still has to be given in the mh_dict as a first estimate.
        """
        logger.info(f"doing quick trials to estimate scale")
        if self.mh_dict["mh_name"] == "fit_weights":
            raise NotImplementedError(
                "This method does not work with the fit_weights MinimizationHandler "
                "because it assumes a background TS distribution median of zero! "
                "Be the hero to think of something!"
            )

        # The given scale will serve as an initial guess
        guess = self.mh_dict["scale"] if not self.disc_guess else self.disc_guess

        # make sure
        self._clean_injection_values_and_pickled_results(self._quick_injections_name)

        # repeat the guessing until success:
        while not self.successful_guess_by_quick_injections:
            quick_injections_mh_dict = dict(self.mh_dict)
            quick_injections_mh_dict["name"] = self._quick_injections_name
            quick_injections_mh_dict["background_ntrials_factor"] = 1
            quick_injections_mh_dict["n_trials"] = 10
            quick_injections_mh_dict["scale"] = guess
            self.submit_local(quick_injections_mh_dict)

            # collect the quick injections
            quick_injections_rh = ResultsHandler(
                quick_injections_mh_dict, do_sens=False, do_disc=False
            )

            # guess the disc and sens scale
            (
                self.disc_guess,
                self.sens_guess,
            ) = quick_injections_rh.estimate_sens_disc_scale()

            if any((g < 0) or (g > guess) for g in [self.disc_guess, self.sens_guess]):
                logger.info(
                    f"Could not perform scale guess because "
                    f"at least one guess outside [0, {guess}]! "
                    f"Adjusting accordingly."
                )
                guess = abs(max((self.sens_guess, self.disc_guess)) * 1.5)

            elif guess > 5 * self.disc_guess:
                logger.info(
                    f"Could not perform scale guess beause "
                    f"initial scale guess {guess} much larger than "
                    f"disc scale guess {self.disc_guess}. "
                    f"Adjusting initial guess to {4 * self.disc_guess} and retry."
                )
                guess = 4 * abs(self.disc_guess)

            else:
                logger.info("Scale guess successful. Adjusting injection scale.")
                self.successful_guess_by_quick_injections = True

            self._clean_injection_values_and_pickled_results(quick_injections_rh.name)

    @staticmethod
    def _clean_injection_values_and_pickled_results(name):
        """
        Removes directories containing injection values and pickled results
        :param name: str, the path used in the minimisation handler dictionary (mh_dict)
        """
        directories = [name_pickle_output_dir(name), inj_dir_name(name)]
        for d in directories:
            if os.path.isdir(d):
                logger.debug(f"removing {d}")
                shutil.rmtree(d)
            else:
                logger.warning(f"Can not remove {d}! It is not a directory!")

    def do_asimov_scale_estimation(self):
        """estimate the injection scale using Asimov estimation"""
        logger.info("doing asimov estimation")
        mh = MinimisationHandler.create(self.mh_dict)
        scale_estimate = mh.guess_scale()
        logger.debug(f"estimated scale: {scale_estimate}")
        self.disc_guess = scale_estimate
        self.sens_guess = 0.3 * self.disc_guess

    def analyse(self, do_disc=False):
        """
        Submits the minimisation handler dictionary (self.mh_dict) to be analysed.
        This happens locally if self.use_cluster == False.
        :param do_disc: bool, if True, use the estimated discovery potential as
                the injection scale instead of the sensitivity.
        """

        if self.do_sensitivity_scale_estimation:
            if "asimov" in self.do_sensitivity_scale_estimation:
                self.do_asimov_scale_estimation()

            if "quick_injections" in self.do_sensitivity_scale_estimation:
                self.run_quick_injections_to_estimate_sensitivity_scale()

            if not do_disc:
                self.mh_dict["scale"] = self.sens_guess / 0.5
            else:
                self.mh_dict["scale"] = self.disc_guess / 0.5

        self.submit(self.mh_dict)

    @classmethod
    def register_submitter_class(cls, server_name):
        """Adds a new subclass of Submitter, with class name equal to "server_name"."""

        def decorator(subclass):
            cls.submitter_dict[server_name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_submitter(cls, *args, **kwargs):
        """
        Get an initialised instance of the Submitter class suited for the
        used server.
        :param args: arguments passed to teh submitter
        :param kwargs: keyword arguments passed to the submitter
        :return: instance of Submitter subclass
        """
        return Submitter.get_submitter_class()(*args, **kwargs)

    @classmethod
    def get_submitter_class(cls):
        """Get the Submitter class suited for the used server."""
        if host_server not in cls.submitter_dict:
            logger.warning(
                f"No submitter implemented for host server {host_server}! "
                f"Using LocalSubmitter but you wont't be able to use cluster operations!"
            )
            return cls.submitter_dict["local"]

        return cls.submitter_dict[host_server]


@Submitter.register_submitter_class("local")
class LocalSubmitter(Submitter):
    def __init__(
        self,
        mh_dict,
        use_cluster,
        n_cpu=None,
        do_sensitivity_scale_estimation=False,
        **cluster_kwargs,
    ):
        if use_cluster:
            raise NotImplementedError(
                "No cluster operation implemented because you are using the LocalSubmitter!"
            )

        super(LocalSubmitter, self).__init__(
            mh_dict,
            use_cluster,
            n_cpu,
            do_sensitivity_scale_estimation,
            **cluster_kwargs,
        )


class HTCondorSubmitter(Submitter):
    home_dir = os.environ["HOME"]
    username = os.path.basename(home_dir)
    status_cmd = "condor_q"
    submit_cmd = "condor_submit"
    root_dir = os.path.dirname(fs_dir[:-1])

    # Log path
    log_path = Path(log_dir)
    override_log_path: Optional[Path] = None

    def __init__(self, *args, **kwargs):
        """
        A class that takes care of submitting the trial calculations.
        Also can estimate the sensitivity scale before submitting.

        :param args: arguments to be passed to Submitter class
        :param kwargs: keyword arguments used by the cluster, available are:
            manual_submit: bool, if True only prints out the location of the submit file without actually
            submitting to the cluster
            trials_per_task: int, how many trials to run per cluster job
            cluster_cpu: int, how many CPUs to use per cluster job
            ram_per_core: float, how much RAM for each cluster jobs in MB
        """
        super(HTCondorSubmitter, self).__init__(*args, **kwargs)

        from flarestack.data.icecube import icecube_dataset_dir

        self.icecube_dataset_dir = icecube_dataset_dir

        logger.debug(f"cluster kwargs: {json.dumps(self.cluster_kwargs, indent=4)}")

        # Get the kwargs that are specific to cluster operation
        self.remove_old_logs = self.cluster_kwargs.get("remove_old_logs", False)
        self.trials_per_task = self.cluster_kwargs.get("trials_per_task", 1)
        self.cluster_cpu = self.cluster_kwargs.get("cluster_cpu", self.n_cpu)
        self.ram_per_core = self.cluster_kwargs.get("ram_per_core", "2000")
        self.manual_submit = self.cluster_kwargs.get("manual_submit", False)

        # Set up a directory for cluster files
        self.cluster_files_directory = os.path.join(
            cluster_dir, self.mh_dict["name"] if self.mh_dict else ""
        )
        self.submit_file = os.path.join(self.cluster_files_directory, "job.submit")
        self.executable_file = os.path.join(self.cluster_files_directory, "job.sh")

        # Check whether automatic submission is desired and possible
        if not self.manual_submit:
            if shutil.which(self.submit_cmd) is None:
                logger.warning(
                    f"Submit command {self.submit_cmd} is not available on the current host. "
                    f"Forcing 'manual_submit' mode."
                )
                self.manual_submit = True

        self._status_output = None

    def make_executable_file(self, path):
        """
        Produces the executable that will be submitted to the NPX cluster.
        :param path: str, path to the file
        """
        flarestack_scratch_dir = os.path.dirname(fs_scratch_dir[:-1]) + "/"

        txt = (
            f"#!/bin/sh \n"
            f"eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/setup.sh) \n"
            f"export PYTHONPATH={HTCondorSubmitter.root_dir}/ \n"
            f"export FLARESTACK_SCRATCH_DIR={flarestack_scratch_dir} \n"
            f"export HOME={HTCondorSubmitter.home_dir} \n"
            f"export FLARESTACK_DATASET_DIR={self.icecube_dataset_dir} \n"
            f"{sys.executable} {fs_dir}core/multiprocess_wrapper.py -f {path} -n {self.cluster_cpu}"
        )

        logger.debug("writing executable to " + self.executable_file)
        with open(self.executable_file, "w") as f:
            f.write(txt)

    @staticmethod
    def format_jobfile(extension: str):
        return f"job-$(cluster)-$(process).{extension}"

    def get_logfile_path(self, extension: str):
        if self.override_log_path is not None:
            log_path = self.override_log_path
        else:
            log_path = Path(log_dir)
        logfile_path = Path(log_path) / self.format_jobfile(extension)
        return str(logfile_path)

    def get_outfile_path(self, extension: str):
        """Formats an output filename for a job given an extension.

        Args:
            extension (str): file extension
        """
        outfile_path = Path(log_dir) / self.format_jobfile(extension)
        return str(outfile_path)

    def make_submit_file(self, n_tasks):
        """
        Produces the submit file that will be submitted to the NPX cluster.
        :param n_tasks: Number of jobs that will be created
        """
        log_file = self.get_logfile_path("log")
        stdout_file = self.get_outfile_path("out")
        stderr_file = self.get_outfile_path("err")

        text = (
            f"executable = {self.executable_file} \n"
            # f"arguments = $(process) \n"
            f"should_transfer_files   = YES \n"
            f"when_to_transfer_output = ON_EXIT \n"
            f"log = {log_file} \n"
            f"output = {stdout_file} \n"
            f"error = {stderr_file} \n"
            f"RequestMemory = {self.ram_per_core} \n"
            f"\n"
            f"queue {n_tasks}"
        )

        logger.debug("writing submitfile at " + self.submit_file)
        with open(self.submit_file, "w") as f:
            f.write(text)

    def submit_cluster(self, mh_dict):
        """Submits the job to the cluster"""

        if self.remove_old_logs:
            self.clear_log_dir()

        # Get the number of tasks that will have to be submitted in order to get ntrials
        ntrials = self.mh_dict["n_trials"]
        n_tasks = int(ntrials / self.trials_per_task)
        logger.debug(f"running {ntrials} trials in {n_tasks} tasks")

        # The mh_dict will be submitted n_task times and will perform mh_dict['n_trials'] each time.
        # Therefore we have to adjust mh_dict['n_trials'] in order to actually perform the number
        # specified in self.mh_dict['n_trials']
        mh_dict["n_trials"] = self.trials_per_task
        path = make_analysis_pickle(mh_dict)

        # make the executable and the submit file
        if not os.path.isdir(self.cluster_files_directory):
            logger.debug(f"making directory {self.cluster_files_directory}")
            os.makedirs(self.cluster_files_directory)

        self.make_executable_file(path)
        self.make_submit_file(n_tasks)

        if not self.manual_submit:
            cmd = [self.submit_cmd, self.submit_file]
            logger.debug(f"command is {cmd}")
            msg = subprocess.check_output(cmd).decode()
            logger.info(msg)

            self.job_id = str(msg).split("cluster ")[-1].split(".")[0]

        else:
            input(
                f"You selected manual submit mode: \n"
                f"\tThe submit file can be found here: \n"
                f"\t{self.submit_file} \n"
                f"\tPlease submit this to the cluster and hit enter when all jobs are done! \n"
                f"[ENTER]"
            )

    @classmethod
    def get_condor_status(cls):
        """
        Queries condor to get cluster status.
        :return: str, output of query command
        """
        cmd = cls.status_cmd
        return subprocess.check_output(cmd).decode()

    def collect_condor_status(self):
        """Gets the condor status and saves it to private attribute"""
        self._status_output = self.get_condor_status()

    @property
    def condor_status(self):
        """
        Get the status of jobs running on condor.
        :return: number of jobs that are done, running, waiting, total, held
        """
        status_list = [
            [y for y in ii.split(" ") if y]
            for ii in self._status_output.split("\n")[4:-6]
        ]
        done = running = waiting = total = held = None

        for li in status_list:
            if li[2] == self.job_id:
                done, running, waiting = li[5:8]
                held = 0 if len(li) == 10 else li[8]
                total = li[-2]

        return done, running, waiting, total, held

    def wait_for_job(self):
        if self.job_id:
            logger.info("waiting for job with ID " + str(self.job_id))
            time.sleep(5)

            self.collect_condor_status()
            j = 0
            while not np.all(np.array(self.condor_status) == None):
                d, r, w, t, h = self.condor_status
                logger.info(
                    f"{time.asctime(time.localtime())} - Job{self.job_id}: "
                    f"{d} done, {r} running, {w} waiting, {h} held of total {t}"
                )
                j += 1
                if j > 7:
                    logger.info(self._status_output)
                    j = 0
                time.sleep(90)
                self.collect_condor_status()

            logger.info("Done waiting for job with ID " + str(self.job_id))

        else:
            logger.info(f"No Job ID!")

    @staticmethod
    def get_pending_ids():
        condor_status = HTCondorSubmitter.get_condor_status()
        ids = np.array([ii.split(" ")[2] for ii in condor_status.split("\n")[4:-6]])
        return ids

    @staticmethod
    def clear_log_dir():
        """Removes all files from log_dir.
        Note that logs may exist outside of log_dir in case of a subclass overriding log_path.
        """
        for f in os.listdir(log_dir):
            ff = f"{log_dir}/{f}"
            logger.debug(f"removing {ff}")
            os.remove(ff)


@Submitter.register_submitter_class("WIPAC")
class WIPACSubmitter(HTCondorSubmitter):
    """
    override log_path so that logging is done on local storage of submit host (rather than network storage)
    this is the IceCube best recommended practice
    ref: https://wiki.icecube.wisc.edu/index.php/Condor/BestPractices#Local_storage_on_submit_host
    """

    override_log_path = Path("/scratch") / HTCondorSubmitter.username
    pass


@Submitter.register_submitter_class("DESY")
class DESYSubmitter(HTCondorSubmitter):
    pass
