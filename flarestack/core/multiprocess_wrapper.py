import pickle
import logging
import time
from logging.handlers import QueueHandler, QueueListener
import argparse
from flarestack.core.minimisation import MinimisationHandler, read_mh_dict
from multiprocessing import JoinableQueue, Process, Queue, Value
import random
from multiprocessing import set_start_method

logger = logging.getLogger(__name__)

try:
    set_start_method("fork")
except RuntimeError:
    pass


def generate_dynamic_mh_class(mh_dict):
    # mh_dict = read_mh_dict(mh_dict)

    try:
        mh_name = mh_dict["mh_name"]
    except KeyError:
        raise KeyError("No MinimisationHandler specified.")

    # Set up dynamic inheritance

    try:
        ParentMinimisationHandler = MinimisationHandler.subclasses[mh_name]
    except KeyError:
        raise KeyError("Parent class {} not found.".format(mh_name))

    class MultiProcessingMinimisationHandler(ParentMinimisationHandler):
        def add_injector(self, season, sources):
            pass

    return MultiProcessingMinimisationHandler(mh_dict)


class MultiProcessor:
    queue = None

    # this is not used within the class, maybe it can be safely dropped
    # results = dict()

    def __init__(self, n_cpu, **kwargs):
        self.queue = JoinableQueue()
        self.log_queue = Queue()
        self.n_tasks = Value("i", 0)
        kwargs["n_tasks"] = self.n_tasks

        self.processes = [
            Process(target=self.run_trial, kwargs=kwargs) for _ in range(int(n_cpu))
        ]

        self.mh = MinimisationHandler.create(kwargs["mh_dict"])
        for season in self.mh.seasons.keys():
            inj = self.mh.get_injector(season)
            inj.calculate_n_exp()
        self.mh_dict = kwargs["mh_dict"]
        self.scales = []

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s")
        )
        # ql gets records from the queue and sends them to the handler

        ql = QueueListener(self.log_queue, handler)
        ql.start()

        for p in self.processes:
            p.start()

    def add_to_queue(self, item):
        self.queue.put(item)

    def dump_all_injection_values(self):
        for scale in self.scales:
            self.mh.dump_injection_values(scale)

    def run_trial(self, **kwargs):
        qh = QueueHandler(self.log_queue)
        logger.addHandler(qh)

        mh_dict = kwargs["mh_dict"]

        mpmh = generate_dynamic_mh_class(mh_dict)

        n_tasks = kwargs["n_tasks"]

        while True:
            item = self.queue.get()
            if item is None:
                break

            (scale, seed) = item

            full_dataset = self.mh.prepare_dataset(scale, seed)

            mpmh.run_single(full_dataset, scale, seed)
            with n_tasks.get_lock():
                n_tasks.value -= 1
            self.queue.task_done()

    def fill_queue(self):
        scale_range, n_trials = self.mh.trial_params(self.mh_dict)

        self.scales = scale_range

        for scale in scale_range:
            for _ in range(n_trials):
                self.add_to_queue((scale, int(random.random() * 10**8)))

        n_tasks = len(scale_range) * n_trials
        with self.n_tasks.get_lock():
            self.n_tasks.value += n_tasks

        logger.info("Added {0} trials to queue. Now processing.".format(n_tasks))

        while self.n_tasks.value > 0.0:
            logger.info("{0} tasks remaining.".format(self.n_tasks.value))
            time.sleep(30)
        logger.info("Finished processing {0} tasks.".format(n_tasks))

    def terminate(self):
        """wait until queue is empty and terminate processes"""
        self.queue.join()
        for p in self.processes:
            p.terminate()

        self.dump_all_injection_values()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


def run_multiprocess(n_cpu, mh_dict):
    with MultiProcessor(n_cpu=n_cpu, mh_dict=mh_dict) as r:
        r.fill_queue()
        r.terminate()
        del r


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # os.cpu_count() may return None, deal safely with that
    cpu_count = os.cpu_count()
    available_cpus = cpu_count if cpu_count is not None else 1
    # 1 <= n_cpu <= 32
    n_cpu = min(max(1, available_cpus - 1), 32)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    parser.add_argument("-n", "--n_cpu", default=n_cpu)
    cfg = parser.parse_args()

    logger.info(f"N CPU available {os.cpu_count()}. Using {cfg.n_cpu}")

    with open(cfg.file, "rb") as f:
        mh_dict = pickle.load(f)

    run_multiprocess(n_cpu=cfg.n_cpu, mh_dict=mh_dict)
