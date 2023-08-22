# type: ignore

# exclude this file from `mypy` checking since the dataset "nt_v002_p01" can no longer be imported
# this also implies that this file is currently untested, otherwise such a test would fail
# if in the future this script is updated to use a new data set, remove the `type: ignore` comment

import os
import matplotlib.pyplot as plt
from flarestack.data.icecube.northern_tracks.nt_v002_p01 import diffuse_8year
from flarestack.shared import plots_dir
from flarestack.icecube_utils.dataset_loader import data_loader, grl_loader

data_rate_dir = plots_dir + "data_rate/"

# for season in txs_sample_v2 + gfu_v002_p01:
for season in diffuse_8year:
    data = data_loader(season["exp_path"])[season["MJD Time Key"]]

    print(data_loader(season["exp_path"]).dtype.names)

    sample_dir = data_rate_dir + season["Data Sample"] + "/"

    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    grl = grl_loader(season)
    print(grl.dtype.names)

    print(min(grl["start"]), max(grl["stop"]))
    print(min(grl["run"]), max(grl["run"]))

    print(grl[-10:])

    plt.figure()
    plt.hist(data, bins=50, histtype="stepfilled")

    savepath = sample_dir + season["Name"] + ".pdf"
    print("Saving to", savepath)

    plt.savefig(savepath)
    plt.close()
