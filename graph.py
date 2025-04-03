"""Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
"""

from sknetwork.clustering import Louvain
from sknetwork.embedding import SVD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.style as mplstyle
import matplotlib.cm as cm
import math

mplstyle.use("fast")


def min_max_norm(x):
    """Calculate the min max norm of x."""
    return (x - x.min()) / (x.max() - x.min())


class GraphClustering:
    """Louvain graph clustering."""

    def __init__(self, resolution: float = 1.0, seed: int = 123, window: int = 0):
        self.cluster_method = Louvain(resolution=resolution, random_state=seed)

        # set plot defaults
        self.dpi = 72
        self.height = 7
        self.width = 9

        self.window = window
        self.graph_layout = SVD(n_components=2, normalized=False)
        self._reset()

    def _reset(self):
        self.mean = 0.0
        self.A = None
        self.pos_init = None

    def set(self, max_steps, starting_period):
        self.max_steps = max_steps
        self.starting_period = starting_period
        self.pos_store = np.zeros((max_steps, 2), dtype="float32")
        self.svd_prev = np.zeros(2, dtype="float32")
        self.jitter = np.random.uniform(low=-1.1, high=1.1, size=(2, max_steps))

    def set_plot_size(self, plot_size, dpi):
        self.dpi = dpi
        self.height = plot_size[0] / self.dpi
        self.width = plot_size[1] / self.dpi
        self.svd_layout = {}
        self.stab_layout = {}

    def cluster(self, similarity_matrix):
        similarity_matrix = similarity_matrix.copy()
        np.fill_diagonal(
            similarity_matrix, 0
        )  # Set diagonal weights to zero to remove self interaction
        # if True: #for goes rough Demo
        if self.mean == 0:
            self.mean = similarity_matrix[similarity_matrix > 0].mean()

        # only look at the current window
        # if self.window>0:
        #    similarity_matrix = similarity_matrix[-self.window:,-self.window:]
        self.A = np.where(similarity_matrix > self.mean, similarity_matrix, 0)

        colors = (
            100 * cm.Set2.colors
        )  # repeat a bunch of times so colors are always available
        self.clusters = [colors[c] for c in self.cluster_method.fit_predict(self.A)]

        # # SVD
        # self.pos = self.graph_layout.fit(self.A) # TODO: It appears this was done twice

        # TODO this will only work if max_embeddings == max_steps
        self.n = len(self.clusters) - self.starting_period

        # spectral gap
        if self.window > 0 and self.n > self.window:
            similarity_matrix = similarity_matrix[
                -self.window - 1 :, -self.window - 1 :
            ]
            A = np.where(similarity_matrix > self.mean, similarity_matrix, 0)
            pos = self.graph_layout.fit(A)
        else:
            pos = self.graph_layout.fit(self.A)

        self.pos_store[self.n - 1] = [
            len(self.clusters),
            pos.singular_values_[0] / pos.singular_values_[1],
        ]
        # self.pos_store[self.n-1] = [len(self.clusters), self.pos.singular_values_[0]/self.pos.singular_values_[1]]

        if self.window > 0 and self.n > self.window:
            self.pos = self.graph_layout.fit(self.A)
        else:
            self.pos = pos

    def SVD_plot(self, savepath):
        c = self.pos.embedding_
        x = c[:, 0]
        y = c[:, 1]

        if math.copysign(1, x[-1]) < 0:
            x *= -1
        if math.copysign(1, y[-1]) < 0:
            y *= -1

        fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        ax = fig.gca()

        plt.scatter(
            x + self.jitter[0][: len(x)],
            y + self.jitter[1][: len(x)],
            c=self.clusters,
            edgecolors="black",
            linewidth=0.1,
            s=np.linspace(25, 225, num=len(x)),
            zorder=1,
        )
        plt.scatter(
            x=x[-1] + self.jitter[0][-1],
            y=y[-1] + self.jitter[1][-1],
            s=250,
            c="#c30010",
            edgecolors="black",
            linewidth=0.2,
            zorder=2,
        )
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
        plt.xlabel("SV0 Embedding", fontsize=24)
        plt.ylabel("SV1 Embedding", fontsize=24)

        plt.tight_layout()
        fig.savefig(
            savepath,
            format="png",
            dpi=self.dpi,
            transparent=False,
            facecolor="white",
            pad_inches=1,
        )
        plt.close()

    def plot_stabilization(self, savepath):
        fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        ax = fig.gca()
        plt.scatter(
            x=self.pos_store[: self.n][..., 0],
            y=self.pos_store[: self.n][..., 1],
            c=self.clusters[self.starting_period :],
            edgecolors="black",
            linewidth=0.1,
            s=np.linspace(25, 225, num=self.n),
            zorder=1,
        )
        plt.scatter(
            x=self.pos_store[self.n - 1][..., 0],
            y=self.pos_store[self.n - 1][..., 1],
            s=250,
            c="#c30010",
            edgecolors="black",
            linewidth=0.2,
            zorder=2,
        )
        ax.tick_params(axis="both", which="major", labelsize=22)

        ax.set_xlabel("RHEED Frame", fontsize=24)
        ax.set_ylabel("Stabilization", fontsize=24, labelpad=10)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.yaxis.set_major_locator(MaxNLocator(6))

        if len(self.stab_layout) == 0:
            plt.tight_layout(pad=1.25)
            self.stab_layout = {
                par: getattr(fig.subplotpars, par)
                for par in ["left", "right", "bottom", "top", "wspace", "hspace"]
            }
        else:
            fig.subplots_adjust(**self.stab_layout)

        fig.savefig(
            savepath,
            format="png",
            dpi=self.dpi,
            transparent=False,
            facecolor="white",
            pad_inches=0.2,
        )
        plt.close()

    def write_to_csv(self, csv_savepath):
        df = pd.DataFrame(
            {
                "step": self.pos_store[: self.n][..., 0],
                "stabilization": self.pos_store[: self.n][..., 1],
                "cluster": self.clusters[self.starting_period :],
            }
        )
        df["cluster"] = df["cluster"].astype("category").cat.codes
        df.to_csv(csv_savepath, index=False)

    def cluster_and_plot(
        self, similarity_matrix, svd_savepath, stab_savepath, csv_savepath
    ):
        self.cluster(similarity_matrix)
        self.SVD_plot(svd_savepath)
        self.plot_stabilization(stab_savepath)
        self.write_to_csv(csv_savepath)
