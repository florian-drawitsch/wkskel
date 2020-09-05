import numpy as np
import pandas as pd
from collections import namedtuple
from typing import Tuple, Optional


# Define Parameters type
PARAMETERS_DEFAULTS = (
    ("name", str),
    ("scale", Tuple[float, float, float]),
    ("offset", Optional[Tuple[float, float, float]]),
    ("time", Optional[int]),
    ("editPosition", Optional[Tuple[float, float, float]]),
    ("editRotation", Optional[Tuple[float, float, float]]),
    ("zoomLevel", Optional[float]),
    ("taskBoundingBox", Optional[Tuple[int, int, int, int, int, int]]),
    ("userBoundingBox", Optional[Tuple[int, int, int, int, int, int]]),
)

Parameters = namedtuple(
    'Parameters',
    [fields[0] for fields in list(PARAMETERS_DEFAULTS)],
    defaults=[defaults[1] for defaults in list(PARAMETERS_DEFAULTS)]
)


# Define Nodes Type
class Nodes:

    NODES_DEFAULTS = {
        ('id', ''): [None, 'int32'],
        ('position', 'x'): [None, 'int32'],
        ('position', 'y'): [None, 'int32'],
        ('position', 'z'): [None, 'int32'],
        ('radius', ''): [100.0, 'float32'],
        ('rotation', 'x'): [0.0, 'float32'],
        ('rotation', 'y'): [0.0, 'float32'],
        ('rotation', 'z'): [0.0, 'float32'],
        ('inVp', ''): [1, 'int8'],
        ('inMag', ''): [1, 'int8'],
        ('bitDepth', ''): [8, 'int8'],
        ('interpolation', ''): [True, 'bool'],
        ('time', ''): [0, 'int64'],
        ('comment', ''): ['', 'str']
    }

    def __init__(self, data=None):

        columns = pd.MultiIndex.from_tuples(tuple(self.NODES_DEFAULTS.keys()))
        self.nodes = Nodes._set_dtypes(pd.DataFrame(data, columns=columns))

    def __repr__(self):
        return self.nodes.__repr__()

    def __call__(self):
        return self.nodes

    def __len__(self):
        return len(self.nodes)

    def __getattr__(self, key):
        return self.nodes.__getattr__(key)

    def __getitem__(self, key):
        return self.nodes.__getitem__(key)

    def __delitem__(self, key):
        self.nodes.__delattr__(key)

    def __setitem__(self, key, value):
        self.nodes.__setattr__(key, value)

    @staticmethod
    def _set_dtypes(nodes):
        [nodes[key].astype(Nodes.NODES_DEFAULTS[key][1]) for key in Nodes.NODES_DEFAULTS.keys()]
        return nodes

    @staticmethod
    def _sanitize_args(args):
        args = list(filter(None, args))
        [args[idx] if idx < len(args) else [value[0]] * len(args[0]) for (idx, value) in enumerate(Nodes.NODES_DEFAULTS.values())]
        return args

    @classmethod
    def from_list(cls, *args):
        """ Constructs Nodes object from list data

        Args:
            id: (Globally unique) Node id
            position_x: Node position x
            position_y: Node position y
            position_z: Node position z
            radius (optional): Node radius
            rotation_x (optional): Node rotation x
            rotation_y (optional): Node rotation y
            rotation_z (optional): Node rotation z
            inVp (optional): Viewport index in which node was placed
            inMag (optional): (De-)Magnification factor in which node was placed
            bitDepth (optional): Bit (Color) Depth in which node was placed
            interpolation (optional): Interpolation state in which node was placed
            time (optional): Time stamp at which node was placed
            comment (optional): Comment associated with node

        """
        args = Nodes._sanitize_args(args)
        data = {key: args[idx] if idx < len(args) else [val[0]] * len(args[0]) for idx, (key, val) in
                enumerate(cls.NODES_DEFAULTS.items())}

        return cls(data)

    def append_from_list(self, *args):
        """ Appends list data to Nodes object

        Args:
            *args: Variable number of argument lists. The arguments are expected in the same order as the
                input argument order to the Nodes.from_list class method. The first four columns are required, all
                remaining columns are optional

        Returns:
            self: Nodes object

        """
        args = Nodes._sanitize_args(args)
        self.append(Nodes.from_list(*args).nodes, ignore_index=True)

        return self

    def append_from_numpy(self, data: np.ndarray):
        """ Appends numpy data to Nodes object

        Args:
            data: Numpy array holding the nodes data. The columns of the array are expected in the same order as the
                input argument order to the Nodes.from_list class method. The first four columns are required, all
                remaining columns are optional

        Returns:
            self: Nodes object

        """
        self.append_from_list(*list(data.T))

        return self





