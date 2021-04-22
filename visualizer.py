import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


def sensitivity_plot(sensitivities, fname, width=4, height=3):
    fig, ax = plt.subplots()

    assert isinstance(ax, matplotlib.pyplot.Axes)

    ax.plot(sensitivities.keys(), sensitivities.values())

    assert isinstance(fig, plt.Figure)
    fig.savefig(fname)


def violin_plot(profiles, fname, keep_ratio=0.70, proper_ratio=0.1, ncols=3, wrow=3, wcol=4):
    # transform latency into macro second
    for i in range(len(profiles)):
        profiles[i][1] *= 1e6

    # keep the lowest part (keep_ratio * 100%) of profiles
    profiles = sorted(profiles, key=lambda p: p[1])
    profiles = profiles[0: int(len(profiles) * keep_ratio)]

    # find y_min, y_max
    y_min = 0
    y_max = profiles[-1][1]

    # groups: Dict[group_name -> (values, latency_set_list)]
    groups = []

    def add_group(name, value_latency_list, subspace_name, subspace_idx):
        values = list(sorted(set(vl[0] for vl in value_latency_list)))
        latency_set_list = []
        for value in values:
            latency_set_list.append([vl[1] for vl in value_latency_list if vl[0] == value])
        groups.append((name, values, latency_set_list, (subspace_name, subspace_idx)))

    subspace_names = profiles[0][0].keys()
    for subspace_name in subspace_names:
        if isinstance(profiles[0][0][subspace_name], (list, tuple)):
            l = len(profiles[0][0][subspace_name])
            for i in range(1, l):
                group_name = f'{subspace_name}_{i}'
                value_latency_list = [(p[0][subspace_name][i], p[1]) for p in profiles]
                add_group(group_name, value_latency_list, subspace_name, i)
        else:
            group_name = subspace_name
            value_latency_list = [(p[0][subspace_name], p[1]) for p in profiles]
            add_group(group_name, value_latency_list, subspace_name, -1)

    # proper values
    proper_threshold = profiles[int(len(profiles) * proper_ratio)][1]
    proper_percentage: Dict[str, Dict[int, float]] = {}
    for group_name, values, latency_set, (subspace_name, subspace_idx) in groups:
        pp = {}
        for value in values:
            pp[value] = 0
        for p in profiles:
            if p[1] < proper_threshold:
                if subspace_idx == -1:
                    pp[p[0][subspace_name]] += 1
                else:
                    pp[p[0][subspace_name][subspace_idx]] += 1
        total = sum(v[1] for v in pp.items())
        proper_percentage[group_name] = {v[0]: v[1] / total for v in pp.items()}

    # plot violin
    ngroups = len(groups)
    nrows = (ngroups + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wcol * ncols, wrow * nrows))
    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]
        if idx >= ngroups:
            ax.remove()
            continue
        group_name, values, latency_set_list, _ = groups[row * ncols + col]
        assert isinstance(ax, matplotlib.pyplot.Axes)
        ax.violinplot(latency_set_list, showmeans=False, showmedians=True, showextrema=True)
        ax.set_ylim(bottom=y_min - y_min / 2, top=y_max + y_min / 2)
        ax.set_xticks([x + 1 for x in range(len(values))])
        ax.set_xticklabels([f'{v}\n{proper_percentage[group_name][v] * 100:.0f}%' for v in values])
        ax.set_xlabel(group_name)
        if col == 0:
            ax.set_ylabel(r'Latency ($10^{-6}$ sec)')

    assert isinstance(fig, plt.Figure)
    fig.tight_layout()
    # fig.set_label(f'Proper ratio {proper_ratio:.2f} ({len(profiles) * proper_ratio:.0f})')
    print(f'Proper ratio {proper_ratio:.2f} ({len(profiles) * proper_ratio:.0f})')
    fig.savefig(fname)


