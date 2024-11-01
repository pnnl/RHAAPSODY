from sknetwork.clustering import Louvain 
from sknetwork.embedding import SVD 
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import sys
import matplotlib.style as mplstyle
import matplotlib.cm as cm
import math
mplstyle.use('fast')

min_max_norm = lambda x: (x - x.min()) / (x.max() - x.min())

class GraphClustering:
    """Louvain graph clustering."""
    def __init__(self, resolution: float = 0.9, seed: int = 12321):
        self.cluster_method = Louvain(resolution=resolution, random_state=seed, modularity='Newman')
        
        # set plot defaults
        self.dpi = 72
        self.height = 7
        self.width = 9

        self.graph_layout = SVD(n_components=2, normalized=False)
        self._reset()

    def _reset(self):
        self.mean = 0.
        self.A = None
        self.pos_init = None
        
        
    def set(self, max_steps, starting_period):
        self.max_steps = max_steps
        self.starting_period = starting_period
        self.pos_store = np.zeros((max_steps,2), dtype='float32')
        self.svd_prev = np.zeros(2, dtype='float32')
        self.jitter = np.random.uniform(low=-1.1, high=1.1, size=(2,max_steps))

    def set_plot_size(self, plot_size, dpi):
        self.dpi = dpi
        self.height = plot_size[0] / self.dpi
        self.width = plot_size[1] / self.dpi
        self.svd_layout = {} 
        self.stab_layout = {} 

    def cluster(self, similarity_matrix):
        if self.mean == 0:
            self.mean = similarity_matrix[similarity_matrix>0].mean()
        self.A = np.where(similarity_matrix>self.mean, similarity_matrix, 0)
        # print(" ----- ",self.A)
        # print("A.shape", self.A.shape)
        # self.clusters =  [cm.Set2.colors[c] for c in self.cluster_method.fit_predict(self.A)]
        self.clusters =  [cm.Set2.colors[c] for c in np.zeros(len(self.A)).astype(int)]
        # self.clusters =  self.cluster_method.fit_predict(self.A)
        
        #Debug
        # exit()

        # SVD
        self.pos = self.graph_layout.fit(self.A)
        
        self.n = len(self.clusters)-self.starting_period

        # spectral gap
        self.pos_store[self.n-1] = [len(self.clusters), self.pos.singular_values_[0]/self.pos.singular_values_[1]]


        
    def SVD_plot(self, savepath):
        c = self.pos.embedding_
        x = c[:,0]
        y = c[:,1]

        if math.copysign(1, x[-1])<0:
            x*=-1
        if math.copysign(1, y[-1])<0:
            y*=-1
        
        fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        ax = fig.gca()
        
        plt.scatter(x+self.jitter[0][:len(x)],y+self.jitter[1][:len(x)], c=self.clusters,
                    edgecolors='black', linewidth=0.1,s=np.linspace(25,225,num=len(x)),zorder=1)
        plt.scatter(x=x[-1]+self.jitter[0][-1], y=y[-1]+self.jitter[1][-1], s=250, c='#c30010', 
                    edgecolors='black', linewidth=0.2, zorder=2)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
        plt.xlabel('SV0 Embedding', fontsize=24)
        plt.ylabel('SV1 Embedding', fontsize=24)
        
        plt.tight_layout()
        fig.savefig(savepath, format='png', dpi=self.dpi, transparent=False, facecolor='white', pad_inches=1)
        plt.close()  


    def plot_stabilization(self, savepath):
        fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        ax = fig.gca()
        plt.scatter(x=self.pos_store[:self.n][...,0],y=self.pos_store[:self.n][...,1], c=self.clusters[self.starting_period:],
                    edgecolors='black', linewidth=0.1,s=np.linspace(25,225,num=self.n),zorder=1)
        plt.scatter(x=self.pos_store[self.n-1][...,0], y=self.pos_store[self.n-1][...,1], s=250, c='#c30010', 
                    edgecolors='black', linewidth=0.2, zorder=2)
        ax.tick_params(axis='both', which='major', labelsize=22)
        
        ax.set_xlabel('RHEED Frame', fontsize=24)
        ax.set_ylabel('Stabilization', fontsize=24, labelpad=10)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.yaxis.set_major_locator(MaxNLocator(6))
        
        if len(self.stab_layout) == 0:
            plt.tight_layout(pad=1.25)
            self.stab_layout = {par : getattr(fig.subplotpars, par) for par in ["left", "right", "bottom", "top", "wspace", "hspace"]}
        else:
            fig.subplots_adjust(**self.stab_layout)

        fig.savefig(savepath, format='png', dpi=self.dpi, transparent=False, facecolor='white', pad_inches=0.2)
        plt.close() 
        
    def cluster_and_plot(self, similarity_matrix, svd_savepath, stab_savepath):
        self.cluster(similarity_matrix)
        self.SVD_plot(svd_savepath)
        self.plot_stabilization(stab_savepath)
        