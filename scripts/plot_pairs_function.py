import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to calculate the absolute difference and use it to color 
# the scatter plots
def scatter_with_color(x, y, s=3, cmap='viridis', 
                       vmin = None, vmax = None, 
                       color = None):
    abs_diff = np.abs(x - y)
    plt.scatter(x, y, s, c=abs_diff, cmap=cmap,
                vmin = vmin, vmax = vmax, lw = 0)


def plot_longitudinal(df, 
                      outpath = None, 
                      title = None,
                      corner = False,
                      kde = False,
                      color = False,
                      cmap='viridis_r',
                      vmin = 0, 
                      vmax = 0.25):

    grid = sns.PairGrid(data= df)
    if corner:
        for i in range(grid.axes.shape[0]):
            for j in range(grid.axes.shape[1]):
                if i < j:
                    grid.axes[i, j].set_axis_off()
    else:
        # Map a density plot to the lower triangle
        grid.map_upper(sns.kdeplot, shade=True, thresh=0.05, cmap = 'Blues',
                    cut = 0)

    # Map a histogram to the diagonal
    grid.map_diag(plt.hist, bins = np.linspace(0, 1, 101), alpha = 0.4, linewidth=None)
    if kde:
        grid.map_lower(sns.kdeplot, cut = 0, color = 'k', 
                       shade=False, thresh=0.05, levels = 4,
                       alpha=0.3)
    if color:
        grid.map_lower(scatter_with_color, s = 3, vmin = vmin, vmax = vmax,
                       cmap = cmap)
        # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Map a scatter plot to the upper triangle
        grid.map_lower(plt.scatter, s = 3)

    for i in range(grid.axes.shape[0]):
        for j in range(grid.axes.shape[1]):
            if i > j:
                grid.axes[1, 0].set_ylim([0, 1])
    
    if title is not None:
        grid.fig.suptitle(title)

    plt.tight_layout()
    if outpath is not None:
        grid.fig.savefig(outpath, transparent = True)
        plt.close()