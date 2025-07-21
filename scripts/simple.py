import numpy as np

class SMPLXSimp:
    def __init__(self, indices_file):
        """
        Initialize the SMPLXSimp class by loading faces and indices from provided files.

        Parameters:
        faces_file (str): Path to the file containing face data.
        indices_file (str): Path to the file containing indices data.
        """
        self.indices, self.faces = self._load_file(indices_file, "indices")

    def _load_file(self, file_path, data_type):
        """
        Private method to load data from a file.

        Parameters:
        file_path (str): Path to the file to be loaded.
        data_type (str): Type of data being loaded (e.g., "faces", "indices").

        Returns:
        numpy.ndarray: Loaded data as a NumPy array.
        """

        try:
            read_dictionary = np.load(file_path, allow_pickle='TRUE').item()
            return read_dictionary['idx'], read_dictionary['faces']
        except Exception as e:
            raise ValueError(f"Error loading {data_type} from {file_path}: {e}")

    def get_faces(self):
        """
        Retrieve the faces data.

        Returns:
        numpy.ndarray: Faces data.
        """
        return self.faces

    def get_indices(self):
        """
        Retrieve the indices data.

        Returns:
        numpy.ndarray: Indices data.
        """
        return self.indices
    def apply_indices(self, vertices):
        return vertices[:, self.indices].squeeze(2)

# Example usage
# faces_file = "path_to_faces.txt"
# indices_file = "path_to_indices.txt"
# smplx = SMPLXSimp(faces_file, indices_file)
# print("Faces:", smplx.get_faces())
# print("Indices:", smplx.get_indices())

# import numpy as np
# # Example: new faces to add (using indices of vertices in the mesh)
# new_faces = np.array([
#     [8522, 8498, 8486],  # Example face using vertex indices
#     [8486, 8498, 8499 ],   # Another example face
#     [8486, 8499, 8526],
#     [8486, 8526, 8453],
#     [8486, 8453, 8485],
#     [8485, 8453, 8487],
#     [8453, 8452, 8487],
#     [8487, 8452, 8493],
#     [8501, 8493, 8452],
#     [8493, 8501, 8520],
#     [8500, 8520, 8501],
#     [8515, 8488, 8479],
#     [8479,8488, 8489],
#     [8479, 8489, 8480],
#     [8489, 8521, 8480],
#     [8480, 8521, 8440],
#     [8521, 8457, 8440],
#     [8440, 8457, 8441],
#     [8441, 8457, 8456],
#     [8441, 8517, 8483],
#     [8441, 8456, 8517],
#     [8517, 8456, 8483],
#     [8456, 8491, 8483],
#     [8483, 8491, 8484],
#     [8491, 8490, 8484],
#     [8484, 8490, 8518],
#     [8514, 8478, 8470],
#     [8470, 8478, 8477],
#     [8470, 8477, 8469],
#     [8469, 8477, 8439],
#     [8469, 8439, 8511],
#     [8511, 8439, 8442],
#     [8511, 8442, 8450],
#     [8450, 8442, 8447],
#     [8447, 8442, 8476],
#     [8476, 8442, 8516],
#     [8476, 8516, 8482],
#     [8482, 8475, 8476],
#     [8482, 8481, 8475],
#     [8475, 8481, 8512],
#     [8505, 8472,  8459],
#     [8459, 8472, 8471],
#     [8459, 8471, 8460],
#     [8460 , 8471, 8445],
#     [8445, 8471, 8513],
#     [8513, 8449, 8445],
#     [8449, 8448, 8445],
#     [8445, 8448, 8508],
#     [8508, 8448, 8467],
#     [8467, 8448, 8473],
#     [8474, 8467, 8473],
#     [8467, 8474, 8468],
#     [8468, 8474, 8509],
#     [5830, 5785, 5798],
#     [5798, 5785, 5786],
#     [5798, 5786, 5797],
#     [5797, 5786, 5838],
#     [5838, 5786, 5775],
#     [5786, 5771, 5775],
#     [5775, 5771, 5833],
#     [5775, 5833, 5774],
#     [5833, 5793, 5774],
#     [5774, 5793, 5799],
#     [5799, 5793, 5800],
#     [5800, 5793, 5794],
#     [5800, 5794, 5834],
#     [5839, 5796, 5804],
#     [5804, 5796, 5803],
#     [5803, 5796, 5795],
#     [5803, 5795, 5765],
#     [5765, 5795, 5768],
#     [5795, 5836, 5768],
#     [5768, 5836, 5776],
#     [5773, 5768, 5776],
#     [5773, 5802, 5768],
#     [5768, 5802, 5841],
#     [5808, 5841, 5802],
#     [5802, 5801, 5808],
#     [5808, 5801, 5807],
#     [5801, 5837, 5807],
#     [5840, 5805, 5813],
#     [5813, 5805, 5814],
#     [5814, 5805, 5806],
#     [5814, 5806, 5846],
#     [5846, 5806, 5783],
#     [5806, 5766, 5783],
#     [5766, 5767, 5783],
#     [5767, 5842, 5783],
#     [5842, 5809, 5783],
#     [5783, 5809, 5782],
#     [5809, 5810, 5782],
#     [5782, 5810, 5816],
#     [5810, 5843, 5816],
#     [5843, 5815, 5816],
#     [5847, 5812, 5823],
#     [5823, 5812, 5824],
#     [5824, 5812, 5851],
#     [5812, 5811, 5851],
#     [5851, 5811, 5779],
#     [5811, 5844, 5779],
#     [5779, 5844, 5778],
#     [5778, 5844, 5784],
#     [5784, 5781, 5778],
#     [5781, 5817, 5778],
#     [5817, 5818, 5778],
#     [5818, 5826, 5778],
#     [5845, 5825, 5826],
#     [5818, 5845, 5826]
# ])
#
# # Append new faces to the existing faces
# faces_tot = np.vstack((faces, new_faces))