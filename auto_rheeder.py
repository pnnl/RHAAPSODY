# %%
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

"""Code for the AutoRHEEDer analysis pipeline."""
import sys
import os

# disable tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
from pathlib import Path
import json
import numpy as np
import skimage
from datetime import timedelta
from scipy.spatial.distance import cdist, pdist, squareform
from datetime import datetime

sys.path.append(".")
from embeddings import EmbeddingModel
from change_detection import ChangepointDetection, parse_changepoint_data
from kernel_matrix import KernelMatrix
from graph import GraphClustering
from segment import Segment


# for csv writing
import csv

# from multiprocessing import Process

# from threading import Thread
# https://nitratine.net/blog/post/python-threading-basics/


class AutoRHEEDer:
    """A class for RHEED analysis pipeline."""

    def __init__(
        self,
        root: str,
        data_processor: callable,
        change_detector: callable,
        classifier: callable,
        segmenter: callable,
        starting_period: int = 60,
        max_embeddings: int = 2000,
        max_steps: int = 500,
        # min_time_between_changepoints: int = 1,
        rolling_window: int = 0,
        message_type: str = "Analysis",
        dpi: int = 72,
    ):
        """Initialize self.

        Args:
            root (str): Path to root directory
            data_processor (callable): Callable to map the data to feature vectors
            change_detector (callable): Callable to check for changepoints
            classifier (callable): Callable to calssify data


        Returns:
            Union[int, None]: The string of the chosen preferred_uuid. Returns None if no
                match could be found (i.e. possible_preferred_terms has length zero).
        """
        self.root = root
        self.data_processor = data_processor
        self.change_detector = change_detector
        self.classifier = classifier
        self.segmenter = segmenter
        self.starting_period = starting_period
        self.message_type = message_type
        self.max_embeddings = max_embeddings
        self.max_steps = max_steps
        # self.min_time_between_changepoints = min_time_between_changepoints
        self.window = rolling_window
        self.dpi = dpi

        # set initial params to 0
        self._reset()

    def _reset(self):

        self.embeddings = np.empty((self.max_embeddings, 512), dtype="float32")
        self.n = 0
        self.change_detected = 0
        self.change_time = 0
        self.A = None
        self.mean_embedding = None
        self.reset = False
        self.pause = True

        self.classifier.set(
            max_steps=self.max_steps, starting_period=self.starting_period
        )

        gc.collect()

    def receive_message(self, incoming_message):
        print("Parsing incoming message")
        self.parse_incoming_message(incoming_message)

        print("Message parsed")
        if self.reset:
            self._reset()
            self.change_detector._reset()
            self.classifier._reset()
            print("AutoRHEEDer Reset")
            return

        print("Analyzing image")
        self.analyze_image(self.imgpath)

        print("Composing outgoing message")
        self.outgoing_message = self.assemble_outgoing_message()

        print("Returning outgoing message")

        return self.outgoing_message

    def parse_incoming_message(self, message):
        self.filetag = Path(message["parameters"]["filename"]).stem
        self.reset = True if message["message"]["msgType"] == "Reset" else False
        self.uuid = message["message"]["uuid"]
        self.experiment = message["parameters"]["experiment"]
        self.loop = message["parameters"]["loop"]
        # self.incoming_time = message['message']['time']
        self.incoming_time = datetime.now().__str__()

        self.imgpath = os.path.join(
            self.root,
            message["parameters"]["experiment"],
            message["parameters"]["loop"],
            message["parameters"]["directory"],
            message["parameters"]["filename"],
        )
        self.savedir = os.path.join(
            self.root,
            message["parameters"]["experiment"],
            message["parameters"]["loop"],
            self.message_type,
        )

        self.changefile = self.filetag + "changepoint.png"
        self.graphfile = self.filetag + "graph.png"
        self.segmentfile = self.filetag + "segmentation.png"
        self.csvfile = self.filetag + "stabilization.csv"

    def assemble_outgoing_message(self):
        # current_time = datetime.now()
        message = {
            "message": {
                "msgType": self.message_type,
                "uuid": self.uuid,
                # 'time': current_time.__str__(),
                # 'processed_in': str((datetime.now()-datetime.fromisoformat(self.incoming_time)).total_seconds()),
            },
            "parameters": {
                "experiment": self.experiment,
                "loop": self.loop,
                "directory": self.message_type,
                "change_point_at": str(timedelta(seconds=self.change_time)),
                "change_detected_at": str(timedelta(seconds=self.change_detected)),
                "changefile": self.changefile,
                "graphfile": self.graphfile,
                "segmentfile": self.segmentfile,
            },
        }
        return message

    def _update_matrix(self, Ain, cdists):
        """Update similarity matrix."""
        if self.pause:
            n1 = self.n - 1
            out = np.empty((self.n, self.n))
            out[:n1, :n1] = Ain
            out[n1, :] = cdists
            out[:, n1] = cdists.T
            out[n1, n1] = 0
        else:
            n1 = self.max_steps
            out = np.empty((n1 + 1, n1 + 1))
            out[:n1, :n1] = Ain[-n1:, -n1:]
            out[n1, :] = cdists[-n1 - 1 :]
            out[:, n1] = cdists.T[-n1 - 1 :]
            out[n1, n1] = 0

        return out

    def _load_image(self, im_path: Path):
        self.im_precrop = skimage.io.imread(im_path)

        if self.n == 0:
            self.image_size = self.im_precrop.shape
            self.change_detector.set_plot_size(plot_size=self.image_size, dpi=self.dpi)
            self.classifier.set_plot_size(plot_size=self.image_size, dpi=self.dpi)
            self.segmenter.set_plot_size(plot_size=self.image_size, dpi=self.dpi)

        # TODO flexible cropping
        # self.crop = ((12,150),(200,480))
        # self.im = self.im[self.crop[0][0]:self.crop[0][1],self.crop[1][0]:self.crop[1][1]]
        self.im = self.im_precrop[12:150, 200:480]

    def analyze_image(self, im_path: Path):
        """Run the analysis pipeline on the given data."""
        # load image
        self._load_image(im_path)

        # segment image
        # p1 = Process(target=self.segmenter.plot_segmented_image, args=(self.im_precrop, os.path.join(self.savedir, self.segmentfile)))
        # p1.start()

        # extract embeddings from image
        self.embeddings[self.n] = self.data_processor.tiff_to_embedding(self.im)

        # At start compute mean embeddings
        if self.n == self.starting_period and self.pause:
            self.mean_embedding = np.mean(self.embeddings[: self.n], axis=0)
            similarity_scores = 1 - squareform(
                pdist(self.embeddings[: self.n] - self.mean_embedding, metric="cosine")
            )
            self.A = KernelMatrix.from_matrix(similarity_scores).coo_matrix.A

        if self.n > self.starting_period:
            similarity_scores = (
                1
                - cdist(
                    self.embeddings[: self.n] - self.mean_embedding,
                    np.expand_dims(
                        self.embeddings[self.n] - self.mean_embedding, axis=0
                    ),
                    metric="cosine",
                )[:, 0]
            )
            self.A = self._update_matrix(self.A, similarity_scores)
            self.A[self.n - 1, self.n - 1] = 1  # self similarity value
            # Change point analysis
            (
                proposed_changepoint,
                changepoint_amplitude,
                current_time,
                actual_changepoint,
            ) = self.change_detector.get_changepoint(self.A, self.n)

            self.change_detector.get_image(
                self.A,
                self.n,
                os.path.join(self.savedir, self.changefile),
                max_display_window=self.max_steps,
                proposed_changepoint=proposed_changepoint,
            )

            # Graph clustering analysis
            self.classifier.cluster_and_plot(
                self.A,
                os.path.join(self.savedir, self.segmentfile),
                os.path.join(self.savedir, self.graphfile),
                os.path.join(self.savedir, self.csvfile),
            )
            # p2 = Process(target=self.classifier.cluster_and_plot, args=(self.A, os.path.join(self.savedir, self.graphfile)))
            # p2.start()

            if (
                actual_changepoint
            ):  # and (proposed_changepoint - self.change_time)>self.min_time_between_changepoints:
                # add back removed steps
                dif = max([0, self.n - self.max_steps])
                self.change_time = int(proposed_changepoint) + dif
                self.change_detected = self.n  # int(current_time)

        if self.n == self.max_steps:
            self.pause = False

        self.n += 1

        # wait for processes to end before returning
        # p1.join()

        return


if __name__ == "__main__":
    gc.disable()
    root = "/home/svcAtScaleNodes/RHEED_DEMO/"
    experiment = "111723B TiO2-STO goes rough"
    loop = "JVSTA_example"  # 'JVSTA'

    # parameters
    changepoint_cost_threshold = 0.025
    changepoint_window = 300
    min_time_between_changepoints = 10
    graph_clustering_window = 0
    starting_period = 30
    max_embeddings = 2101
    max_steps = 2100

    fields = ["frame", "total"]
    with open(
        Path(root, experiment, loop, "timing-multiprocessing.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

    rheed_analysis = AutoRHEEDer(
        root=root,
        data_processor=EmbeddingModel(),
        change_detector=ChangepointDetection(
            cost_threshold=changepoint_cost_threshold,
            window_size=changepoint_window,
            min_time_between_changepoints=min_time_between_changepoints,
        ),
        classifier=GraphClustering(
            resolution=1,
            seed=123,
            window=graph_clustering_window,
        ),
        segmenter=Segment(),
        starting_period=starting_period,
        max_embeddings=max_embeddings,
        max_steps=max_steps,
    )

    image_paths = sorted(Path(root, experiment, loop, "Raw").rglob("Run*.tiff"))

    for p in image_paths[: rheed_analysis.max_embeddings]:
        idx = int(str(p).split("-")[-1].replace(".tiff", ""))
        incoming_message = {
            "message": {
                "msgType": "Raw",
                "uuid": "xyz",
            },
            "parameters": {
                "experiment": experiment,
                "loop": loop,
                "directory": "Raw",
                "filename": p,
            },
        }
        start = datetime.now()
        out_message = rheed_analysis.receive_message(incoming_message)

        total_time = (datetime.now() - start).total_seconds()
        out_message["message"]["processed_in"] = total_time
        log = {"frame": p, "total": total_time}
        with open(
            Path(root, experiment, loop, "timing-multiprocessing.csv"), "a"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writerow(log)

        with open(
            Path(root, experiment, loop)
            / "Analysis"
            / f"out_message-{str(idx).zfill(5)}.json",
            "w",
        ) as f:
            json.dump(out_message, f)

        print(out_message)
        print()

        gc.collect()

    np.savez(
        Path(root, experiment, loop, "graph.npz"),
        stab=rheed_analysis.classifier.pos_store,
    )
    parse_changepoint_data(root, experiment, loop)
