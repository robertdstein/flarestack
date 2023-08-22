# type: ignore

# exclude this file from `mypy` checking since IC86_1_dict can no longer be imported
# this also implies that this file is currently untested, otherwise such a test would fail
# if in the future its functionality is restored, remove the `type: ignore` comment


import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import dataset_plot_dir
from flarestack.icecube_utils.dataset_loader import data_loader
from numpy.lib.recfunctions import append_fields

season = IC86_1_dict


def custom_data_loader(path):
    dataset = data_loader(path, floor=False)
    percentile = np.ones_like(dataset["ra"]) * np.nan
    dataset = append_fields(
        dataset, "percentile", percentile, usemask=False, dtypes=[float]
    )

    return dataset


mc = custom_data_loader(season["mc_path"])

exp = custom_data_loader(season["exp_path"])

n_sin_dec_bins = 35 + 1

sin_dec_bins = np.percentile(mc["sinDec"], np.linspace(0.0, 100.0, n_sin_dec_bins))
sin_dec_bins[0] = -1.0
sin_dec_bins[-1] = 1.0
print(sin_dec_bins)

path = dataset_plot_dir + season["Data Sample"] + "/" + season["Name"] + "/"

try:
    os.makedirs(path)
except OSError:
    pass

for j, bins in enumerate([sin_dec_bins, IC86_1_dict["sinDec bins"]]):
    plt.figure()
    for x in bins:
        plt.axvline(x)

    print(path)

    plt.savefig(path + ["equal_MC", "old"][j] + "sin_dec_.pdf")
    plt.close()


def weighted_quantile(values, quantiles, weight):
    """
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weight: array-like of the same length as `array`
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def calculate_logE_percentile(data, n_slices):
    n_high_e = int(0.7 * n_slices)

    percentiles = (
        list(np.linspace(0.0, 80.0, n_slices + 1 - n_high_e))
        + list(np.linspace(80.0, 100.0, n_high_e + 1))[1:]
    )
    percentiles = np.array(percentiles)
    log_e_bins = np.linspace(0.0, 10.0, n_slices + 1)
    # log_e_bins = np.percentile(data["logE"], percentiles)
    log_e_bins[0] = min(0.0, min(min(mc["logE"]), min(exp["logE"])))
    log_e_bins[-1] = max(10.0, max(max(mc["logE"]), max(exp["logE"])))
    return log_e_bins, percentiles


n_log_e_bins = 3

log_e_bins = np.linspace(1.0, 10.0, 5)

# all_log_e_bins = []
# n_data = []


# plt.figure()
#
# for i, lower in enumerate(sin_dec_bins[:-1]):
#     upper = sin_dec_bins[i + 1]
#     mask = np.logical_and(
#         mc["sinDec"] > lower,
#         mc["sinDec"] < upper
#     )
#
#     plt.axvline(lower)
#     plt.axvline(upper)
#
#     cut_mc = mc[mask]
#
#     exp_mask = np.logical_and(
#         exp["sinDec"] > lower,
#         exp["sinDec"] < upper
#     )
#
#     cut_data = exp[exp_mask]
#
#     log_e_bins, log_e_percentiles = calculate_logE_percentile(cut_data,
#                                                               n_log_e_bins)
#     all_log_e_bins.append(log_e_bins)
#     for e in log_e_bins:
#         plt.plot([lower, upper], [e, e])
#
#     interp_bins, interp_percentiles = calculate_logE_percentile(cut_data, 1000)
#     convert = interp1d(interp_bins, interp_percentiles)
#
#     exp["percentile"][exp_mask] = convert(cut_data["logE"])
#     mc["percentile"][mask] = convert(cut_mc["logE"])
#
# plt.savefig(path + "bins.pdf")
# plt.close()


def make_plot(hist, savepath, normed=True):
    if normed:
        norms = np.sum(hist, axis=hist.ndim - 2)
        norms[norms == 0.0] = 1.0
        hist /= norms

    hist = np.log(np.array(hist))
    plt.figure()
    ax = plt.subplot(111)
    X, Y = np.meshgrid(sin_dec_bins, log_e_bins)
    if not normed:
        max_col = min(
            abs(min([min(row) for row in hist.T])), max([max(row) for row in hist.T])
        )
        cbar = ax.pcolormesh(X, Y, hist, cmap="seismic", vmin=-5, vmax=5)
        plt.colorbar(cbar, label="Log(Signal/Background)")
    else:
        hist[hist == 0.0] = np.nan
        cbar = ax.pcolormesh(X, Y, hist)
        plt.colorbar(cbar, label="Log(Column-normalised density)")
    plt.xlabel(r"$\sin(\delta)$")
    plt.ylabel("LogE")
    plt.savefig(savepath)
    plt.close()


bkg_hist, binedges = np.histogramdd(
    (exp["logE"], exp["sinDec"]),
    bins=(log_e_bins, sin_dec_bins),
)

make_plot(bkg_hist, path + "data.pdf")
bkg_hist /= np.sum(bkg_hist, axis=0)


# plt.figure()
# plt.hist2d(exp["sinDec"], exp["percentile"],
#            bins=(sin_dec_bins, log_e_percentiles)
# )
#
# plt.savefig(path + "data.pdf")
# plt.close()
for gamma in [2.0]:
    # for gamma in [1.0, 2.0, 3.0, 3.5, 3.7]:
    weights = mc["ow"] * mc["trueE"] ** -gamma

    all_log_e_bins = [log_e_bins for _, _ in enumerate(sin_dec_bins)]

    plt.figure()
    mc_hist, binedges = np.histogramdd(
        (mc["logE"], mc["sinDec"]), bins=(log_e_bins, sin_dec_bins), weights=weights
    )
    savepath = path + "MC_" + str(gamma) + ".pdf"
    make_plot(mc_hist, savepath=savepath, normed=True)

    x = []
    y = []
    z = []

    for k, row in enumerate(mc_hist.T):
        if not np.logical_and(min(row) > 0.0, min(bkg_hist.T[k]) > 0.0):
            row = list(row)[::-1]
            bkg_row = list(bkg_hist.T[k])[::-1]
            borders = list(all_log_e_bins[k])[::-1]

            for j, x in enumerate(row):
                if not np.logical_and(x > 0.0, bkg_row[j] > 0.0):
                    new = row[j - 1] * 0.5
                    row[j] = new
                    row[j - 1] = new
                    bkg_new = bkg_row[j - 1] * 0.5
                    bkg_row[j] = bkg_new
                    bkg_row[j - 1] = bkg_new
                    del borders[j]

            mc_hist.T[k] = np.array(row[::-1])
            bkg_hist.T[k] = np.array(bkg_row[::-1])
            all_log_e_bins[k] = np.array(borders[::-1])
            print(mc_hist.T[k], bkg_hist.T[k], all_log_e_bins[k])
            input("prompt")

        x += [sin_dec_bins[k] for _, _ in enumerate(all_log_e_bins[k][1:])]
        y += list(all_log_e_bins[k][1:] + all_log_e_bins[k][:-1])
        z += old_div(mc_hist.T[k], bkg_hist.T[k])

    mc_hist /= np.sum(mc_hist, axis=0)

    ratio = old_div(mc_hist, bkg_hist)
    savepath = path + "ratio_" + str(gamma) + ".pdf"
    make_plot(ratio, savepath=savepath, normed=False)

    # log_e_bin_center = 0.5 * (log_e_percentiles[1:] + log_e_percentiles[:-1])
    sin_bin_center = 0.5 * (sin_dec_bins[1:] + sin_dec_bins[:-1])

    # x = []
    # y = []
    # z = ratio.T.flatten()
    #
    # for i, s in enumerate(sin_bin_center):
    #     x += [s for _, _ in enumerate(all_log_e_bins[i][1:])]
    #     centers = all_log_e_bins[i][1:] + all_log_e_bins[i][:-1]
    #     y += list(centers)

    order = 1

    spline = scipy.interpolate.interp2d(x, y, np.log(ratio))

    for x_val in [2.0, 3.0, 7.0]:
        print(x_val, spline(0.0, x_val), spline(0.5, x_val))

    spline_perc = np.linspace(1.0, 8.0, 100)
    splin_sindec = np.linspace(-1.0, 1.0, 100)

    Z = []
    for s in splin_sindec:
        z_line = []
        for e in spline_perc:
            z_line.append(spline(e, s)[0])
        Z.append(z_line)

    Z = np.array(Z).T

    # max_col = min(abs(min([min(row) for row in Z])),
    #               max([max(row) for row in Z]))
    max_col = 5.0

    plt.figure()
    ax = plt.subplot(111)
    X, Y = np.meshgrid(splin_sindec, spline_perc)
    cbar = ax.pcolormesh(X, Y, Z, cmap="seismic", vmin=-max_col, vmax=max_col)
    plt.colorbar(cbar, label="Log(Signal/Background)")
    plt.xlabel(r"$\sin(\delta)$")
    plt.ylabel("log(Energy)")
    plt.savefig(path + "spline_" + str(gamma) + ".pdf")
    plt.close()

# hist, binedges = np.histogramdd(
#     (exp["percentile"], exp["sinDec"]),
#     bins=(sin_dec_bins, log_e_percentiles)
# )
#
# ax = plt.subplot(111)
# X, Y = np.meshgrid(sin_dec_bins, log_e_percentiles)
# # X, Y = np.meshgrid(binedges)
#
# max_col = min(abs(min([min(row) for row in hist.T])),
#               max([max(row) for row in hist.T]))
# cbar = ax.pcolormesh(X, Y, hist.T, cmap="seismic")
# plt.colorbar(cbar, label="Log(Signal/Background)")
#
# plt.xlabel(r"$\sin(\delta)$")
# plt.ylabel("Energy Proxy percentile")


# log_e_centers =
#
# spline = scipy.interpolate.RectBivariateSpline(
#     log_e_bin_center, sin_bin_center, np.log(ratio),
#     kx=order, ky=order, s=0)

# log_e_bins
