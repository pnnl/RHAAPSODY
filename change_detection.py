"""Change detection algorithms for time series.
Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
"""
from pathlib import Path

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

from PIL import Image as im
import ruptures as rpt

from sklearn.decomposition import PCA

from kernel_matrix import KernelMatrix

import matplotlib.style as mplstyle
mplstyle.use('fast')

def format_fn(tick_val, tick_pos):
    if int(tick_val) in xs:
        return labels[int(tick_val)]
    else:
        return ''

def pelt_changepoints(data: np.ndarray, cost: callable, pen: float, **pelt_kwargs):
    """Get the pelt changepoints for data, given cost."""
    algo = rpt.Pelt(custom_cost=cost, **pelt_kwargs).fit(data[:])
    cp = algo.predict(pen=pen)[:-1]  # don't include final point
    return cp


def evaluate_rank_cost(data: np.ndarray, pen=300, **pelt_kwargs):
    """Evaluate the data using the cosine cost function."""
    c = rpt.costs.CostRank()
    return pelt_changepoints(data=data, cost=c, pen=pen, **pelt_kwargs)


def evaluate_cosine_cost(data: np.ndarray, pen=35, **pelt_kwargs):
    """Evaluate the data using the cosine cost function."""
    c = rpt.costs.CostCosine()
    return pelt_changepoints(data=data, cost=c, pen=pen, **pelt_kwargs)


def evaluate_continuous_linear(data: np.ndarray, pen=500, **pelt_kwargs):
    """Evaluate the data using the cosine cost function."""
    c = rpt.costs.CostCLinear()
    return pelt_changepoints(data=data, cost=c, pen=pen, **pelt_kwargs)


def plot_changepoints(data: np.ndarray, cp: list[float], num_channels: int):
    """Plot the changepoints agains the first "num_channels" components."""
    fig, axs = plt.subplots(
        data.shape[1], 1, figsize=(10, 2 * data.shape[1]), tight_layout=True
    )
    for i in range(data.shape[1]):
        axs[i].vlines(cp, ymin=min(data[:, i]), ymax=max(data[:, i]), color="r")
        axs[i].plot(data[:, i])
    return (fig, axs)


def display_changepoints(cp, original_data, pca_data, pca, center, reshape_param):
    fig, axs = plt.subplots(len(cp) + 1, 2, figsize=(8, 15), tight_layout=True)

    axs[0, 0].set_title("PCA reproduced")
    axs[0, 1].set_title("Original Data")

    for i, c in enumerate([1] + cp):
        image = pca.inverse_transform(pca_data[c]).reshape(reshape_param) + center / 255

        axs[i, 0].imshow(-image, interpolation="nearest", cmap="binary")

        axs[i, 0].set_yticklabels([])
        axs[i, 0].set_xticklabels([])

        axs[i, 0].set_yticks([])
        axs[i, 0].set_xticks([])

        axs[i, 0].set_ylabel(rf"$t={c}$")

        image = original_data[:, :, c - 1]

        axs[i, 1].imshow(1 - image / 255, interpolation="nearest", cmap="binary")

        axs[i, 1].set_yticklabels([])
        axs[i, 1].set_xticklabels([])

        axs[i, 1].set_yticks([])
        axs[i, 1].set_xticks([])
    return (fig, axs)


def display_changepoints_no_pca(cp, original_data):
    fig, axs = plt.subplots(len(cp) + 1, 1, figsize=(8, 15), tight_layout=True)

    axs[0].set_title("Original Data")

    for i, c in enumerate([1] + cp):
        image = original_data[:, :, c - 1]

        axs[i].imshow(-image, interpolation="nearest", cmap="binary")

        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])

        axs[i].set_yticks([])
        axs[i].set_xticks([])

        axs[i].set_ylabel(rf"$t={c}$")

    return (fig, axs)


def read_h5_data(h5_path: Path, h5_file_key: str = None):
    """Read in data from an h5 file.

    Use time slice to slice the time axis and bbox to slice the spacial dimensions.

    bbox should follow this format:
    bbox = (x0, y0, x1, y1)
    (x0, y0)
        +-----+
        |     |
        +-----+
            (x1, y1)
    """
    f = h5py.File(h5_path, "r")
    h5_file_key = list(f.keys())[0] if h5_file_key is None else h5_file_key
    data = f[h5_file_key][:, :, 1, :]
    return data


class ChangepointDetection:
    """Class for the changepoint detection."""

    def __init__(
        self,
        cost_threshold: float = 0.06,
        window_size=np.inf,
        min_time_between_changepoints:int=10,
    ):
        """Initialize self."""
        self.cost_threshold = cost_threshold
        self.window_size = window_size
        self.min_time_between_changepoints = min_time_between_changepoints

        # set plot defaults
        self.dpi = 72
        self.height = 7
        self.width = 9

        self._reset()

    def _reset(self):
        self.changepoints = [0]
        self.detected_times = [0]

    def set_plot_size(self, plot_size, dpi):
        self.dpi = dpi
        self.height = plot_size[0] / self.dpi
        self.width = plot_size[1] / self.dpi
        self.layout = {}
        self.first_plot = True

    @staticmethod
    def segmented_cost(matrix: np.ndarray, tau:int, start:int=0, end:int=None):
        """Return the cost of segmenting the square matrix at tau."""
        sub_matrix_1 = matrix[start:tau, start:tau]
        val1 = np.diagonal(sub_matrix_1).sum()
        val1 -= sub_matrix_1.sum() / (tau-start)

        sub_matrix_2 = matrix[tau:end, tau:end]
        val2 = np.diagonal(sub_matrix_2).sum()
        val2 -= sub_matrix_2.sum() / (end - tau)

        sub_matrix_3 = matrix[start:end, start:end]
        val3 = np.diagonal(sub_matrix_3).sum()
        val3 -= sub_matrix_3.sum() / (end - start)

        return (val3 - val1 - val2) / (end - start)

    def maximize_segmented_cost(self, matrix, current_time, window_start):
        """Maximize the segmented cost for the square matrix on an interval."""
        xs = np.arange(window_start + 1, current_time, 3).astype(int)
        vals = [self.segmented_cost(matrix, x, start=window_start, end=current_time) for x in xs]
        
        if len(vals) > 0:
            max_idx = np.argmax(vals)
            max_time = xs[max_idx]
            max_val = vals[max_idx]
            return (max_time, max_val)
        else:
            return []

    def get_changepoint(self, matrix: np.ndarray, step: int):
        """Get the changepoints on the given interval."""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Cannot do changepoint detection on a rectangular matrix, "
                f"({matrix.shape[0]}, {matrix.shape[1]})."
            )
        current_time = step 
        window_start = self.changepoints[-1] 
        vals = self.maximize_segmented_cost(
            matrix=matrix, current_time=current_time, window_start=window_start
        )

        actual_changepoint = False
        if len(vals) > 0:
            proposed_changepoint = vals[0]
            changepoint_amplitude = vals[1]

            if current_time - self.changepoints[-1] > self.min_time_between_changepoints:
                if changepoint_amplitude > self.cost_threshold:
                    self.changepoints.append(proposed_changepoint)
                    self.detected_times.append(current_time)
                    actual_changepoint = True
                else:
                    actual_changepoint = False
            else:
                actual_changepoint = False

        else:
            proposed_changepoint = np.nan
            changepoint_amplitude = np.nan
            current_time = current_time
            actual_changepoint = False

        return (
            proposed_changepoint,
            changepoint_amplitude,
            current_time,
            actual_changepoint,
        )

    def get_image(
        self,
        matrix: np.ndarray,
        step: int,
        savepath: str,
        proposed_changepoint=None,
        max_display_window: int = np.inf,
    ):
        """Get the image associated to the display window."""        
        fig,ax = plt.subplots(1,1,figsize=(self.width, self.height), dpi=self.dpi)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)

        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1, 1)),
            cax=cax, ticks=[-1, 0, 1],
            orientation="vertical",
        )
        cax.set_ylabel('Similarity', fontsize=24, labelpad=-8)
        cax.tick_params(axis='y', which='major', labelsize=22)

        window_start = max([0, step - max_display_window])

        ax.pcolorfast(matrix, vmin=-1, vmax=1) 
        ax.invert_yaxis()
            
        for cp in self.changepoints:
            if cp - window_start > 0:
                ax.axvline(cp - window_start, c="r", alpha=0.6)

        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_xlabel('RHEED Frame', fontsize=24)
        ax.set_ylabel('RHEED Frame', fontsize=24)
        
        if self.first_plot:
            plt.tight_layout(pad=3)
            self.layout = {par : getattr(fig.subplotpars, par) for par in ["left", "right", "bottom", "top", "wspace", "hspace"]}
            self.first_plot = False
        else:
            fig.subplots_adjust(**self.layout)

        plt.savefig(savepath, format='png', dpi=self.dpi, transparent=False, facecolor='white', pad_inches=1)
        plt.close()
    
    @property
    def current_changepoint(self):
        """Return the most recent changepoint"""
        return self.changepoints[-1]
