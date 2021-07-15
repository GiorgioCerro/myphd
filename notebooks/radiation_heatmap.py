from jet_tools import ReadHepmc, Components, InputTools, TrueTag, FormShower
import scipy.stats
import jet_tools
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import awkward as ak
import os


def first_points(eventWise, mcpid):
    if not hasattr(mcpid, '__iter__'):
        mcpid = [mcpid]
    all_idxs = [i for i, pid in enumerate(eventWise.MCPID)
                if pid in mcpid]
    firsts = []
    for idx in all_idxs:
        if mcpid not in eventWise.MCPID[eventWise.Parents[idx]]:
            firsts.append(idx)
    return firsts


def get_bg_tags(eventWise, tag_idxs):
    stack = set(tag_idxs)
    tag_parents = set()
    while stack:
        particle = stack.pop()
        parents = list(eventWise.Parents[particle])
        tag_parents.update(parents)
        stack.update(parents)
    # go down from the root, looking for the first leaves that are not 
    background = set()
    stack = set(eventWise.Is_root)
    while stack:
        particle = stack.pop()
        children = list(eventWise.Children[particle])
        for child in children:
            if child not in tag_parents:
                background.add(child)
            elif child not in tag_idxs:
                stack.add(child)
    return background


def get_paired_tags(eventWise, mcpid=5):
    is_particle = first_points(eventWise, mcpid)
    is_particle = is_particle[np.argmax(eventWise.Energy[is_particle])]
    is_antiparticle = first_points(eventWise, -mcpid)
    is_antiparticle = is_antiparticle[np.argmax(eventWise.Energy[is_antiparticle])]
    # they need to equal
    if abs(eventWise.MCPID[is_particle]) != \
            abs(eventWise.MCPID[is_antiparticle]):
        lower_idx, higher_idx = sorted([is_particle, is_antiparticle], 
                                       key=lambda x:  eventWise.Energy[x])
        lower_pid = eventWise.MCPID[lower_idx]
        higher_pid = eventWise.MCPID[higher_idx]

        try:
            return get_paired_tags(eventWise, higher_pid)
        except ValueError:
            pass
        try:
            return get_paired_tags(eventWise, lower_pid)
        except ValueError:
            pass
        for pid in mcpid:
            if pid not in [lower_pid, higher_pid]:
                try:
                    return get_paired_tags(eventWise, pid)
                except ValueError:
                    pass
        raise RuntimeError("Cannot find any particle-antiparticle pairs in "
                           + f"{eventWise.save_name} event {eventWise.selected_index}"
                           + f" with MCPID in {mcpid}")
    return np.array([is_particle, is_antiparticle])



def bin_events(bins, x1_edges, x2_edges, var, x1, x2):
    var = np.array(var)
    n_x1_bins, n_x2_bins = bins.shape
    for x1_bin in range(0, n_x1_bins):
        x1_start = x1_edges[x1_bin]
        x1_end = x1_edges[x1_bin+1]
        x1_location = (x1 < x1_end) * (x2 >= x1_start)
        for x2_bin in range(0, n_x2_bins):
            here = (x1_location *
                    (x2 < x2_edges[x2_bin+1]) * (x2 >= x2_edges[x2_bin]))
            bins[x1_bin, x2_bin] += np.nansum(ak.to_numpy(var[here]))
    return bins


def center_image(image_2d, cyclic_coord, cyclic_range=(-np.pi, np.pi)):
    image_2d = ak.to_numpy(image_2d)
    cyclic_bins = image_2d.shape[cyclic_coord]
    half_bins = int(cyclic_bins/2)
    cyclic_angles = np.linspace(*cyclic_range, cyclic_bins)

    linear_coord = (cyclic_coord + 1) % 2
    mean_values = np.nanmean(image_2d, linear_coord)

    cyclic_x = np.nanmean(np.cos(cyclic_angles)*mean_values)
    cyclic_y = np.nanmean(np.sin(cyclic_angles)*mean_values)
    cyclic_mean = np.arctan2(cyclic_y, cyclic_x)
    mean_bin = np.argmin(np.abs(cyclic_mean - cyclic_angles))

    shift = half_bins - mean_bin
    new_image = np.roll(image_2d, shift, axis=cyclic_coord)
    bin_width = cyclic_angles[1] - cyclic_angles[0]
    new_range = (cyclic_range[0] - shift*bin_width, 
                 cyclic_range[1] - shift*bin_width)
    return new_range, new_image


def plt_bins(ax, x_lims, y_lims, xs, ys, weights, cmap='viridis'):
    n_bins = 40
    bins = np.zeros((n_bins, n_bins), dtype=float)
    x_edges = np.linspace(*x_lims, n_bins)
    y_edges = np.linspace(*y_lims, n_bins)
    bins = bin_events(bins, x_edges, y_edges, weights, xs, ys)

    ax.imshow(np.rot90(bins), cmap=cmap, extent=[*x_lims, *y_lims])


def plt_kde(ax, x_lims, y_lims, xs, ys, weights, cmap='viridis'):
    xmin, xmax = x_lims
    ymin, ymax = y_lims
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([xs, ys])
    kernel = scipy.stats.gaussian_kde(values, 0.1, weights=weights)
    Z = np.reshape(kernel(positions).T, X.shape)
    Z = np.log(Z)/np.log(2)

    # center_image??

    #ax.imshow(np.rot90(Z), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    ax.contourf(X, Y, Z, cmap=cmap)


def get_data_file(input_file):
    read_file = input_file[:-6] + ".parquet"
    if os.path.exists(read_file):
        input_file = read_file
        data = Components.EventWise.from_file(input_file)
        Components.add_all(data)
    else:
        data = ReadHepmc.Hepmc(input_file)
        data.save_name = data.save_name[:-6] + ".parquet"
    return data


def calculate_roots_showers(data):
    n_events = len(data.X)
    b_root_name = "Is_BRoot"
    l_root_name = "Is_lRoot"
    bg_root_name = "Is_BGRoot"
    name_pids = [(b_root_name, 5), (l_root_name, np.array([11, 13, 15]))]
    for name, pid in name_pids:
        if name in data.columns:
            continue
        print(f"\n\n{name}\n\n")
        root_idxs, shower_idxs = [], []
        for event_n in range(n_events):
            if event_n % 10 == 0:
                print(f"{event_n/n_events:.0%}   ", end="\r")
            data.selected_index = event_n
            roots = get_paired_tags(data, pid)
            shower = FormShower.descendant_idxs(data, *roots)
            root_idxs.append(list(roots))
            shower_idxs.append(list(shower))
        data.append(**{name: root_idxs,
                       name.replace("Root", "Shower"): shower_idxs})

    if bg_root_name not in data.columns:
        print(f"\n\n{bg_root_name}\n\n")
        root_idxs, shower_idxs = [], []
        for event_n in range(n_events):
            if event_n % 10 == 0:
                print(f"{event_n/n_events:.0%}   ", end="\r")
            data.selected_index = event_n
            tag_idxs = np.concatenate([getattr(data, name)
                                       for name, _ in name_pids])
            roots = get_bg_tags(data, tag_idxs)
            shower = FormShower.descendant_idxs(data, *roots)
            root_idxs.append(list(roots))
            shower_idxs.append(list(shower))
        data.append(**{bg_root_name: root_idxs,
                       bg_root_name.replace("Root", "Shower"): shower_idxs})



def get_vars_from_masks(eventWise, mask_names, var_names,
                        exclude_mask_names=None):
    if exclude_mask_names is None:
        exclude_mask_names = []

    var = [[] for _ in var_names]

    eventWise.selected_index = None
    n_events = len(eventWise.X)
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{event_n/n_events:.0%}   ", end="\r")
        data.selected_index = event_n
        mask = set()
        for name in mask_names:
            mask.update(getattr(eventWise, name))
        for name in exclude_mask_names:
            mask = mask.difference(getattr(eventWise, name))
        mask = list(mask)
        for i, name in enumerate(var_names):
            var[i] += list(getattr(eventWise, name)[mask])
    return var


fig, ax_arr = plt.subplots(3, 3, figsize=(10, 10))
ax_list = ax_arr.flatten()

data_dir = "/home/henry/Programs/pythia8304/reshowered_events/"
suffix = "_1k_9.hepmc"
data_types = [name.replace(suffix, "") for name in os.listdir(data_dir)
              if name.endswith(suffix)]
print(data_types)
data_type = InputTools.list_complete("Which data type? ", data_types).strip()


for i in range(0, 9):
    print(f"Event {i}\n")
    input_file = os.path.join(data_dir, data_type + f"_1k_{i}.hepmc")
    data = get_data_file(input_file)
    #calculate_roots_showers(data)

    mask_names = ["Is_BShower"]  #, "Is_BGShower", "Is_lShower"]
    #exclude_mask_names = ["Is_lShower"]
    var_names = ["Rapidity", "Phi", "PT"]
    rapidity, phi, pt = get_vars_from_masks(data, mask_names, var_names)

    rapidity_lims = [-2.5, 2.5]
    phi_lims = [-np.pi, np.pi]

    ax = ax_list[i]
    plt_bins(ax, rapidity_lims, phi_lims, rapidity, phi, pt)
    ax.set_xlabel("Rapidity")
    ax.set_ylabel("$\\phi$")

    matplotlib.rc('text', usetex=True)
    data.selected_index = 0
    b_quarks = data.Is_BRoot
    b_quarks = b_quarks[np.abs(data.Rapidity[b_quarks]) < 2.5]
    if len(b_quarks):
        ax.scatter(data.Rapidity[b_quarks], data.Phi[b_quarks],
                   marker="$b$", color='red')

    leptons = data.Is_lRoot
    leptons = leptons[np.abs(data.Rapidity[leptons]) < 2.5]
    if len(leptons):
        ax.scatter(data.Rapidity[leptons], data.Phi[leptons],
                   marker="$l$", color='orange')
            

matplotlib.rc('text', usetex=False)

fig.suptitle(f"{data_type} reshowered a thousand times")
fig.set_tight_layout(True)
