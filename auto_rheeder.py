#%%
"""Code for the AutoRHEEDer analysis pipeline.

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

import sys
import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import skimage
from pathlib import Path
from datetime import timedelta
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.sparse as sp

sys.path.append(".")
from embeddings import EmbeddingModel
from change_detection import ChangepointDetection
from kernel_matrix import KernelMatrix
from graph import GraphClustering


# for csv writing
import csv
from datetime import datetime



class AutoRHEEDer:
    """A class for RHEED analysis pipeline."""

    def __init__(
        self,
        root: str,
        data_processor: callable,
        change_detector: callable,
        classifier: callable,
        starting_period: int = 60,
        max_embeddings: int = 2000,
        max_steps: int = 2000,
        message_type: str = 'Analysis',
        dpi: int = 72,
    ):
        """Initialize self.

        Args:
            root (str): Path to root directory
            data_processor (callable): Callable to map the data to feature vectors
            change_detector (callable): Callable to check for changepoints
            classifier (callable): Callable to graph module
            starting_period (int): Number of initial steps before change point and graph analysis
            max_embeddings (int): Maximum number of steps to consider at once
            max_steps (int): Maximum number of steps in the full experiment
            message_type (str): Name of directory to save the images to should be 
                                located as follows
                                /root/experiment/loop/message_type/ 
            dpi (int): dpi of output images


        Returns:
            
        """
        self.root = root
        self.data_processor = data_processor
        self.change_detector = change_detector
        self.classifier = classifier
        self.starting_period = starting_period
        self.message_type = message_type
        self.max_embeddings = max_embeddings
        self.max_steps = max_steps
        self.dpi = dpi

        # set initial params to 0
        self._reset()

    def _reset(self):

        self.embeddings = np.empty((self.max_embeddings,512), dtype='float32')
        self.n = 0
        self.change_detected = 0
        self.change_time = 0
        self.A = None
        self.mean_embedding = None
        self.reset = False
        self.pause = True

        self.classifier.set(max_steps=self.max_steps, starting_period=self.starting_period)

        gc.collect()

    def receive_message(self, incoming_message):
        self.parse_incoming_message(incoming_message)

        if self.reset:
            self._reset()
            self.change_detector._reset()
            self.classifier._reset()
            print('AutoRHEEDer Reset')
            return 

        self.analyze_image(self.imgpath)

        self.outgoing_message = self.assemble_outgoing_message()


        return self.outgoing_message

    def parse_incoming_message(self, message):
        self.filetag =  Path(message['parameters']['filename']).stem
        self.reset = True if  message['message']['msgType']=='Reset' else False
        self.uuid = message['message']['uuid']
        self.experiment = message['parameters']['experiment']
        self.loop = message['parameters']['loop']
        
        self.imgpath = os.path.join(self.root, message['parameters']['experiment'], message['parameters']['loop'], 
                                    message['parameters']['directory'], message['parameters']['filename'])
        self.savedir = os.path.join(self.root, message['parameters']['experiment'], message['parameters']['loop'], self.message_type)

        self.changefile = self.filetag + 'changepoint.png'
        self.graphfile = self.filetag + 'graph.png'
        self.stabilityfile = self.filetag + 'stability.png'


    def assemble_outgoing_message(self):
        message ={'message': {'msgType': self.message_type, 
                               'uuid': self.uuid,
                             },
                  'parameters': {'experiment': self.experiment, 
                               'loop': self.loop, 
                               'directory': self.message_type, 
                               'change_point_at': str(timedelta(seconds=self.change_time)),
                               'change_detected_at': str(timedelta(seconds=self.change_detected)),
                               'changefile': self.changefile, 
                               'graphfile': self.graphfile, 
                               'stabilityfile': self.stabilityfile,
                            }
                 }
        return message

        
    def _update_matrix(self, Ain, cdists):
        """Update similarity matrix."""
        if self.pause:
            n1 = self.n - 1
            out = np.empty((self.n,self.n))
            out[:n1,:n1] = Ain
            out[n1,:] = cdists
            out[:,n1] = cdists.T
            out[n1,n1] = 0
        else:
            n1 = self.max_steps
            out = np.empty((n1+1,n1+1))
            out[:n1,:n1] = Ain[-n1:,-n1:]
            out[n1,:] = cdists[-n1-1:]
            out[:,n1] = cdists.T[-n1-1:]
            out[n1,n1] = 0 

        return out
        
    def _load_image(self, im_path: Path):
        self.im_precrop = skimage.io.imread(im_path)

        # set output images to be the same size as input RHEED images
        if self.n == 0:
            self.image_size = self.im_precrop.shape
            self.change_detector.set_plot_size(plot_size=self.image_size, dpi=self.dpi)
            self.classifier.set_plot_size(plot_size=self.image_size, dpi=self.dpi)

        # hard coded cropping
        self.im = self.im_precrop[12:150,200:480]

    def analyze_image(self, im_path: Path):
        """Run the analysis pipeline on the given data."""
        # load image
        self._load_image(im_path)

        # extract embeddings from image
        self.embeddings[self.n] = self.data_processor.tiff_to_embedding(self.im)
        
        # At start compute mean embeddings
        if self.n == self.starting_period and self.pause:
            self.mean_embedding = np.mean(self.embeddings[:self.n], axis=0)
            similarity_scores = squareform(1 - pdist(self.embeddings[:self.n] - self.mean_embedding, metric="cosine"))
            self.A = KernelMatrix.from_matrix(similarity_scores).coo_matrix.A

        if self.n > self.starting_period:
            similarity_scores = 1 - cdist(
                    self.embeddings[:self.n] - self.mean_embedding,
                    np.expand_dims(self.embeddings[self.n] - self.mean_embedding, axis=0),
                    metric="cosine")[:, 0]
            self.A = self._update_matrix(self.A, similarity_scores)

            # Change point analysis
            (
                proposed_changepoint,
                changepoint_amplitude,
                current_time,
                actual_changepoint,
            ) = self.change_detector.get_changepoint(self.A, self.n)

            
            self.change_detector.get_image(self.A, self.n, os.path.join(self.savedir, self.changefile), max_display_window=self.max_steps, proposed_changepoint=proposed_changepoint)           

            
            # Graph clustering analysis
            self.classifier.cluster_and_plot(self.A, 
                                             os.path.join(self.savedir, self.stabilityfile), 
                                             os.path.join(self.savedir, self.graphfile))
            


            if actual_changepoint:
                # add back removed steps
                dif = max([0, self.n-self.max_steps])
                self.change_time = int(proposed_changepoint)+dif
                self.change_detected = self.n 


        if self.n == self.max_steps:
            self.pause = False

        self.n+=1

        return

        

if __name__ == "__main__":
    gc.disable()
        
    rheed_analysis = AutoRHEEDer(
        # root="/home/svcAtScaleNodes/RHEED_DEMO/",
        root="/Users/amat841/WORKSPACE/ATSCALE/rheed_data/raw_tiffs/",
        data_processor=EmbeddingModel(),
        change_detector=ChangepointDetection(cost_threshold=0.05, window_size=300, min_time_between_changepoints=10),
        classifier=GraphClustering(resolution=1, seed=123),
        starting_period=30,
        max_embeddings=2502,
        max_steps=2500,
        )

    # images should be located in /root/experiment/loop/directory/
    # image_paths = sorted(Path("/home/svcAtScaleNodes/RHEED_DEMO/111723B TiO2-STO goes rough/Loop1/Raw").rglob("Run*.tiff"))
    image_paths = sorted(Path("/Users/amat841/WORKSPACE/ATSCALE/rheed_data/raw_tiffs/111723B TiO2-STO goes rough/loop1/raw").rglob("Run*.tiff"))
    for p, image_path in enumerate(image_paths):

        if p>rheed_analysis.max_embeddings:
            break

        incoming_message={'message': {'msgType': 'Raw', 
                                        'uuid': 'xyz'},
                          'parameters': {'experiment': '111723B TiO2-STO goes rough', 
                                    #    'loop': 'Loop2', 
                                       'loop': 'loop1', 
                                    #    'directory': 'Raw', 
                                       'directory': 'raw',
                                       'filename': image_path}
                         }
        
        out_message = rheed_analysis.receive_message(incoming_message)
            
        print(out_message)
        print()

        gc.collect()



# %%
