import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class Nodes(pd.DataFrame):
    """The Nodes class facilitates a consistent tabular representation of skeleton nodes

    It represents a subclass of the pandas DataFrame class. To ensure consistency, the columns of nodes objects are
    overridden with a pre-defined column configuration during construction. Furthermore, the Nodes class implements
    custom append methods which allow to add new node data in a well-defined and efficient way.

    Class Attributes:
        COLUMNS (tuple of tuple of str): Column names for multi index generation

    """
    COLUMNS = (
        ('id', ''),
        ('position', 'x'),
        ('position', 'y'),
        ('position', 'z'),
        ('radius', ''),
        ('rotation', 'x'),
        ('rotation', 'y'),
        ('rotation', 'z'),
        ('inVp', ''),
        ('inMag', ''),
        ('bitDepth', ''),
        ('interpolation', ''),
        ('time', ''),
        ('comment', '')
    )

    def __init__(self, *args, **kwargs):
        assert ('columns' not in kwargs), "The columns of a Nodes object cannot be modified"
        kwargs['columns'] = pd.MultiIndex.from_tuples(self.COLUMNS)
        super(Nodes, self).__init__(*args, **kwargs)

    def append_from_list(self,
                         id: List[int],
                         position_x: List[int],
                         position_y: List[int],
                         position_z: List[int],
                         radius: Optional[List[int]] = None,
                         rotation_x: Optional[List[float]] = None,
                         rotation_y: Optional[List[float]] = None,
                         rotation_z: Optional[List[float]] = None,
                         inVP: Optional[List[int]] = None,
                         inMag: Optional[List[int]] = None,
                         bitDepth: Optional[List[int]] = None,
                         interpolation: Optional[List[bool]] = None,
                         time: Optional[List[int]] = None,
                         comment: Optional[List[int]] = None):
        """ Appends data to Nodes object

        Args:
            id: (Globally unique) Node id
            position_x: Node position x
            position_y: Node position y
            position_z: Node position z
            radius (optional): Node radius
            rotation_x (optional): Node rotation x
            rotation_y (optional): Node rotation y
            rotation_z (optional): Node rotation z
            inVP (optional): Viewport index in which node was placed
            inMag (optional): (De-)Magnification factor in which node was placed
            bitDepth (optional): Bit (Color) Depth in which node was placed
            interpolation (optional): Interpolation state in which node was placed
            time (optional): Time stamp at which node was placed
            comment (optional): Comment associated with node

        Returns:
            self: Nodes object

        """
        data = {
            ('id', ''): id,
            ('position', 'x'): position_x,
            ('position', 'y'): position_y,
            ('position', 'z'): position_z,
            ('radius', ''): radius,
            ('rotation', 'x'): rotation_x,
            ('rotation', 'y'): rotation_y,
            ('rotation', 'z'): rotation_z,
            ('inVp', ''): inVP,
            ('inMag', ''): inMag,
            ('bitDepth', ''): bitDepth,
            ('interpolation', ''): interpolation,
            ('time', ''): time,
            ('comment', ''): comment
        }

        nodes = Nodes(data=data)
        self = self.append(nodes)

        return self

    def append_from_numpy(self, data: np.ndarray):
        """ Appends numpy data to Nodes object

        Args:
            data: Numpy array holding the nodes data. The columns of the array are expected in the same order as the
                input argument order to the Nodes.append_from_list method. The first four columns are required, all
                remaining columns are optional

        Returns:
            self: Nodes object

        """
        self = self.append_from_list(*list(data.T))

        return self
