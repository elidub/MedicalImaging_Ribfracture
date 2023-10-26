from matplotlib import pyplot as plt
import numpy as np

def plot_results(data):
    for d_i, d in enumerate(data):

        zs = d['img'].shape[-1]
        n = 6
        fig, axs = plt.subplots(len(d), n, figsize=(2.*n, 5), tight_layout=True)
        titles = ['Image', 'Label', 'Prediction\nUnprocessed', 'Prediction\nPostprocessed']
        zs_arr = np.linspace(0+1, zs-1, n, dtype=int)
        for i in range(len(titles)):
            axs[i, 0].set_ylabel(titles[i])
        for j, z in enumerate(zs_arr):
            axs_i = axs[:, j]
            for i, ax in enumerate(axs_i):
                ax.imshow(list(d.values())[i][:, :, z], cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            axs_i[0].set_title(f'slice z={z}')

        plt.show()

        if d_i > 3:
            break

def plot_gridsearch(gs, gs_max):
    # for norm in [colors.Normalize(vmax=1), colors.LogNorm(vmax=1), None]:
    fig, axs = plt.subplots(1, 3, figsize=np.array([11, 3])/1.3, sharex=True, sharey=True, constrained_layout=True)

    for i, metric in enumerate(['accuracy', 'precision', 'recall']):
        ax = axs[i]
        im = ax.scatter(gs['prob_thresh'], gs['size_thresh'], c=gs[f'{metric}_mean'], cmap = 'viridis', marker='s', s=2, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title(metric)
        # ax.set_xscale('log')
        ax.scatter(gs_max.iloc[0]['prob_thresh'], gs_max.iloc[0]['size_thresh'], c = 'red')
    fig.supylabel('Size Threshold')
    fig.supxlabel('Probability Threshold')
    plt.show()


