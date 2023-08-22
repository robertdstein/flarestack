import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flarestack.shared import illustration_dir

plt.xkcd()

spatial_path = illustration_dir + "spatial.pdf"

w = 5

n_sig = 50
n_bkg = 200

sig_x = np.random.normal(0, 1.0, n_sig)
bkg_x = np.random.uniform(-w, w, n_bkg)

sig_y = np.random.normal(0, 1.0, n_sig)
bkg_y = np.random.uniform(-w, w, n_bkg)

x = np.concatenate((sig_x, bkg_x))
y = np.concatenate((sig_y, bkg_y))

s = (0.5 / np.pi) * np.exp(-(x**2 + y**2))
bkg = 1.0 / (2 * w) ** 2

z = s / bkg

cmap = cm.get_cmap("jet")

plt.figure()

plt.scatter(x, y, c=z, cmap=cmap)
plt.scatter(0.0, 0.0, color="k", marker="x", s=50)
plt.xlim(-w, w)
plt.ylim(-w, w)
plt.title("Spatial")
plt.xlabel(r"$\Delta_{RA}$ [deg]")
plt.ylabel(r"$\Delta_{Dec}$ [deg]")
plt.savefig(spatial_path, bbox_inches="tight", pad_inches=0.5)
plt.close()

energy_path = illustration_dir + "energy.png"

min_e = 10.0**2
max_e = 10.0**7


def ic_power_law(f, gamma):
    p = 1 - gamma

    if p == -2.0:
        e = min_e * (max_e / min_e) ** f
    else:
        e = (f * (max_e**p - min_e**p) + min_e**p) ** (1.0 / p)

    return e


s_g = 2.1
b_g = 3.7


def energy_ratio(e):
    k = (max_e ** (s_g - 1) - min_e ** (s_g - 1)) / (
        max_e ** (b_g - 1) - min_e ** (b_g - 1)
    )

    ratio = e ** (b_g - s_g)

    return ratio


sig_e = [ic_power_law(x, gamma=s_g) for x in np.random.uniform(0, 1.0, n_sig)]
bkg_e = [ic_power_law(x, gamma=b_g) for x in np.random.uniform(0, 1.0, n_bkg)]

all_e = np.concatenate((sig_e, bkg_e))

fig = plt.figure()

# Plot histogram.
n, bins, patches = plt.hist(np.log(all_e) - 2.0, 10, color="green")
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, "facecolor", cmap(c))

plt.yscale("log")
plt.xlabel("Log(Reco Energy / GeV)")
plt.title("Energy")

fig.tight_layout(pad=2)

plt.savefig(energy_path, bbox_inches="tight", pad_inches=0.5)
plt.close()

n_sig = 4
n_bkg = 6


s_t_list = []

for i, t in enumerate(np.random.uniform(0, 1.0, n_sig)):
    s_t_list.append((t + float(i)) * 25)

s_t = np.array(s_t_list)
del s_t_list

bkg_t = np.random.uniform(0, 1.0, n_bkg) * 100

all_t = np.concatenate((s_t, bkg_t))

w_s = np.random.uniform(1.1, 3.5, n_sig)
w_b = np.random.uniform(0.2, 0.9, n_bkg)

all_sob = np.concatenate((w_s, w_b), axis=0)
col = cmap(all_sob / max(all_sob))

time_path = illustration_dir + "time.pdf"

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.bar(all_t, height=all_sob, width=3, color=col)
ax1.axhline(1.0, linestyle="--", color="k")
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("S/B")
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")

ax2.set_ylim(0, 0.5 * (n_sig * (n_sig - 1)))
# ax2.arrow(s_t[0], 1.5, s_t[1]-s_t[0], 0,
#           head_width=0.3, length_includes_head=True,
#           color="k")

h = 0.5

for i, t in enumerate(s_t):
    ax2.axvline(t, linestyle="-", color="k")
    ax2.axvline(t, linestyle="--", color=col[i])
    for j, t_prime in enumerate(s_t[s_t > t]):
        ax2.arrow(
            t,
            h,
            t_prime - t,
            0,
            head_width=0.3,
            head_length=1.0,
            length_includes_head=True,
            color="k",
            shape="full",
            lw=0.7,
        )
        ax2.arrow(
            t_prime,
            h,
            t - t_prime,
            0,
            head_width=0.3,
            head_length=1.0,
            length_includes_head=True,
            color="k",
            shape="full",
            lw=0.7,
        )
        h += 1.0

f.subplots_adjust(hspace=0)
ax2.set_axis_off()
# ax2.get_yaxis().set_visible(False)
plt.savefig(time_path, bbox_inches="tight", pad_inches=0.5)
plt.close()


fig = plt.figure()

ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)


def box_func(t, t1):
    """Box function that is equal to 1 between t0 and t1, and 0 otherwise.
    Equal to 0.5 at to and t1.

    :param t: Time to be evaluated
    :param t0: Start time of box
    :param t1: End time of box
    :return: Value of Box function at t
    """
    val = 0.5 * ((np.sign(t)) - (np.sign(t - t1))) / t1
    return val


x = np.linspace(-10, 150, 3001)

y1 = box_func(x, 30)
y2 = box_func(x, 105)

ax1.plot(x, y1, color="red")
ax2.plot(x, y2, color="blue")

# ax1.fill_between(x, 0, y1, facecolor="blue")
# ax2.fill_between(x, 0, y2, facecolor="red")

ax1.set_ylim(0, 1.2 * max(y1 + y2))
ax2.set_ylim(0, 1.2 * max(y1 + y2))
ax1.set_xlim(0.0, 150)
fig.subplots_adjust(hspace=0)
ax2.invert_yaxis()
plt.xlabel("Time (days)")
ax2.set_ylabel("Search Window")
ax1.set_ylabel("Injection PDF")

ax1.get_xaxis().set_visible(False)

search_path = illustration_dir + "search1.pdf"

fig.set_size_inches(8, 6)
plt.savefig(search_path, bbox_inches="tight", pad_inches=0.5)

for tmax in [50, 90]:
    ax1.plot(x, box_func(x, tmax), color="red")

ax1.arrow(
    35,
    0.04,
    65,
    -0.02,
    head_width=0.005,
    head_length=10,
    length_includes_head=True,
    color="k",
    shape="full",
    lw=0.7,
)

search_path = illustration_dir + "search2.pdf"
plt.savefig(search_path, bbox_inches="tight", pad_inches=0.5)
plt.close()
