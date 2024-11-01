"""Define a class to handle the sparse similarity matrix.

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

import numpy as np

from scipy.sparse import coo_matrix


class KernelMatrix:
    """A class for the matrix of kernel values."""

    def __init__(self, data, row, col):
        """Initialize self."""
        self._data = data.copy()
        self._row = row.copy()
        self._col = col.copy()

        self._apply_updates()

    def __getitem__(self, index):
        """Slice the underlying matrix."""
        return self.coo_matrix.A[index]

    @classmethod
    @staticmethod
    def from_matrix(matrix):
        row, col = matrix.nonzero()
        data = matrix[row, col]

        return KernelMatrix(data, row, col)

    def _apply_updates(self):
        """Update self.coo_matrix with the current data."""
        self.coo_matrix = coo_matrix((self._data, (self._row, self._col)), copy=False)

    def update(self, new_array):
        """Update the matrix with new array.

        The last value in new_array is placed past the bottom-right corner of
        self.coo_matrix.
        """
        if (
            len(new_array) > self.coo_matrix.shape[0] + 1
            or len(new_array) > self.coo_matrix.shape[1] + 1
        ):
            raise ValueError(
                "The given array is too big! "
                f"Expected an array of maximum size {self.coo_matrix.shape[0] + 1}, "
                f"got an array of size {len(new_array)}."
            )
        # Append to the right of the matrix
        new_shape = (self.coo_matrix.shape[0] + 1, self.coo_matrix.shape[1] + 1)

        self._col = np.append(self._col, [new_shape[0] - 1] * len(new_array)).astype(
            int
        )
        self._row = np.append(
            self._row, list(range(new_shape[0] - len(new_array), new_shape[0]))
        ).astype(int)
        self._data = np.append(self._data, new_array)

        self._col = np.append(
            self._col, list(range(new_shape[0] - len(new_array), new_shape[0] - 1))
        ).astype(int)
        self._row = np.append(
            self._row, [new_shape[1] - 1] * (len(new_array) - 1)
        ).astype(int)
        self._data = np.append(self._data, new_array[:-1])

        self._apply_updates()

    def copy(self):
        """Return a copy of self."""
        return KernelMatrix(self._data, self._rows, self._cols)

    def to_json(self):
        return {
            "data": self._data.tolist(),
            "row": self._row.tolist(),
            "col": self._col.tolist(),
        }
