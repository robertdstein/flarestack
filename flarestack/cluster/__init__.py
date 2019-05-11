import time
import os
from flarestack.shared import host_server
from flarestack.cluster.run_desy_cluster import submit_to_cluster
from flarestack.cluster.make_local_bash_script import local_submit_file,\
    make_local_submit_file

if host_server == "DESY":
    submit_cluster = submit_to_cluster

else:
    def submit_cluster(path, **kwargs):
        raise Exception("No cluster submission script recognised!")


if not os.path.isfile(local_submit_file):
    make_local_submit_file()


def submit_local(path, n_cpu):

    bashfile = local_submit_file

    submit_cmd = bashfile + " " + path + " " + str(n_cpu)

    os.system(submit_cmd)


def submit(path, cluster=False, n_cpu=2, **kwargs):
    if cluster:
        submit_cluster(path, n_cpu=n_cpu, **kwargs)
    else:
        submit_local(path, n_cpu=n_cpu)

