import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D, art3d
from typing import Union, Sequence, List, Tuple, Optional

import wknml

from wkskel.types import Nodes, Parameters


class Skeleton:
    """The Skeleton class facilitates scientific analysis and manipulation of webKnossos tracings.

    It is designed as a high-level interface for working with nml files generated e.g with webKnossos. It makes use of
    the (low-level) `wknml` package mostly as an I/O interface to nml files.

    Class Attributes:
        DEFAULTS (dict): Global default parameters which are passed to each skeleton object instance

    """

    DEFAULTS = {
        'node': {
            'radius': 100,
            'comment': ''
        },
        'tree': {
            'color': (0.0, 0.0, 0.0, 1.0)
        }
    }

    def __init__(self, nml_path: str = None, parameters: Parameters = None, strict = True):
        """ The Skeleton constructor expects either a path to a nml file or a Parameters object as input arguments

        Args:
            nml_path: Path to nml file. If constructed via an nml file, the skeleton object is populated with all the
                trees and additional properties specified in the .nml file
            parameters (optional): Parameters (wkskel.types.Parameters) specifying the most rudimentary properties
                 of the skeleton.
            strict (optional): Controls assertions ensuring that resulting skeleton objects are compatible with
                webKnossos. Default: True

        Examples:
            Using nml_path:
                nml_path = '/path/to/example.nml'
                skel = Skeleton(nml_path)

            Using parameters:
                parameters = Skeleton.define_parameters(name="2017-01-12_FD0156-2", scale=(11.24, 11.24, 32))
                skel = Skeleton(parameters=parameters)
        """

        assert (nml_path is not None) ^ (parameters is not None), \
            'To construct a skeleton object, either a path to a nml file or the skeleton parameters need to passed'

        self.nodes = list()
        self.edges = list()
        self.names = list()
        self.colors = list()
        self.tree_ids = list()
        self.group_ids = list()
        self.groups = list()
        self.branchpoints = list()
        self.parameters = Parameters()
        self.nml_path = str()

        self.strict = strict
        self.defaults = self.DEFAULTS

        # Construct from nml file
        if nml_path is not None:
            assert os.path.exists(nml_path), \
                'not a valid path: {}'.format(nml_path)
            try:
                with open(nml_path, "rb") as f:
                    nml = wknml.parse_nml(f)
            except IOError:
                print('not a valid nml file: {}'.format(nml_path))

            self._nml_to_skeleton(nml)

        # Construct from parameters
        else:
            assert type(parameters) is Parameters, \
                'provided parameters must be of type wkskel.types.Parameters'

            self._parameters_to_skeleton(parameters)

    def add_tree(self,
                 nodes: Nodes = Nodes(),
                 edges: Union[List[Tuple[int, int]], np.ndarray] = None,
                 tree_id: int = None,
                 group_id: int = None,
                 name: str = '',
                 color: Tuple[float, float, float, float] = None):
        """ Appends new tree to skeleton.

        Args:
            nodes (optional): Nodes representing tree to be added
            edges (optional): Edges representing tree to be added
            tree_id (optional): Tree id to be used for new tree. Default: Highest current tree id + 1
            group_id (optional): Group id to be used for new tree. Default: None
            name (optional): Name to be used for new tree. Default: Empty str
            color (optional): Color to be used for new tree specified as (r, g, b, alpha). Default: (0, 0, 0, 1)
        """

        if edges is None:
            edges = np.empty((0, 2), dtype=np.uint32)
        elif type(edges) is list:
            edges = np.asarray(edges)

        if self.strict & (len(nodes) > 1):
            assert Skeleton._num_conn_comp(Skeleton._get_graph(nodes, edges)) == 1, \
                'Added tree consists of more than one connected component'

        if tree_id is None:
            tree_id = self.max_tree_id() + 1

        if (group_id is not None) & (group_id not in self.groups_ids()):
            self.add_group(id=group_id)

        if color is None:
            color = self.defaults['tree']['color']

        self.nodes.append(nodes)
        self.edges.append(edges)
        self.tree_ids.append(tree_id)
        self.group_ids.append(group_id)
        self.names.append(name)
        self.colors.append(color)

    def add_tree_from_skel(self,
                           skel: 'Skeleton',
                           tree_idx: int,
                           group_id: int = None,
                           name: str = None):
        """ Appends a specific tree contained in a different skeleton object to the skeleton.

        Args:
            skel: Source skeleton object (different from the one calling this method) to be added
            tree_idx: Source tree index of tree to be added
            group_id (optional): Target group id to which the added tree should be assigned. Default: None
            name (optional): Target name for the added tree
        """

        if group_id not in self.groups_ids():
            self.add_group(id=group_id)

        if name is None:
            name = skel.names[tree_idx]

        skel._reset_node_ids(self.max_node_id() + 1)
        skel._reset_tree_ids(self.max_tree_id() + 1)

        self.nodes = self.nodes + [skel.nodes[tree_idx]]
        self.edges = self.edges + [skel.edges[tree_idx]]
        self.tree_ids = self.tree_ids + [skel.tree_ids[tree_idx]]
        self.group_ids = self.group_ids + [group_id]
        self.names = self.names + [name]
        self.colors = self.colors + [skel.colors[tree_idx]]

        return self

    def add_trees_from_skel(self, skel: 'Skeleton'):
        """ Appends all trees contained in a different skeleton object to the skeleton.

        This method attempts to preserve the relative group structure found in the skeleton object to be added

        Args:
            skel: Source skeleton object (different from the one calling this method) to be added
        """

        skel._reset_node_ids(self.max_node_id() + 1)
        skel._reset_tree_ids(self.max_tree_id() + 1)

        max_group_id = self.max_group_id()
        if max_group_id is not None:
            skel._reset_group_ids(max_group_id + 1)

        self.nodes = self.nodes + skel.nodes
        self.edges = self.edges + skel.edges
        self.tree_ids = self.tree_ids + skel.tree_ids
        self.group_ids = self.group_ids + skel.group_ids
        self.groups = self.groups + skel.groups
        self.names = self.names + skel.names
        self.colors = self.colors + skel.colors

        return self

    def add_nodes_as_trees(self,
                           nodes: Nodes,
                           tree_ids: List[int] = None,
                           group_ids: List[int] = None,
                           names: List[str] = None,
                           colors: List[Tuple[float, float, float, float]] = None):
        """ Appends each of the specified nodes as separate trees to the skeleton (1 node each).

        Args:
            nodes: Nodes representing the trees to be added
            tree_ids (optional): Tree ids to be assigned to the newly added trees. Default: Global max + [1, n]
            group_ids (optional): Group ids to be assigned to the newly added trees. Default: None
            names (optional): Names to be assigned to the newly added trees.
            colors (optional): Colors to be used for the new trees specified as (r, g, b, alpha). Default: (0, 0, 0, 1)
        """

        if tree_ids is None:
            tree_id_start = self.max_tree_id() + 1
            tree_id_end = tree_id_start + len(nodes)
            tree_ids = list(range(tree_id_start, tree_id_end))

        if group_ids is None:
            group_ids = [None for x in range(len(nodes))]

        if names is None:
            names = ['' for x in range(len(nodes))]

        if colors is None:
            colors = [(0.0, 0.0, 0.0, 1.0) for x in range(len(nodes))]

        for node_idx, _ in nodes.iterrows():
            self.add_tree(
                nodes=nodes[node_idx:node_idx+1],
                tree_id=tree_ids[node_idx],
                group_id=group_ids[node_idx],
                name=names[node_idx],
                color=colors[node_idx]
            )

    def delete_tree(self, idx: int = None, id: int = None):
        """ Deletes tree with specified idx or id.

        Args:
            idx: Linear index of tree to be deleted
            id: Id of tree to be deleted

        """

        if id is not None:
            idx = self.tree_ids.index(id)

        self.nodes.pop(idx)
        self.edges.pop(idx)
        self.names.pop(idx)
        self.colors.pop(idx)
        self.tree_ids.pop(idx)
        self.group_ids.pop(idx)

    def add_group(self, parent_id: int = None, id: int = None, name: str = None):
        """ Adds a new group to skeleton object.

        Args:
            parent_id: Parent group id to which new group is added as a child. Default: None (root group)
            id: Id of new group to be added. Default: Current max group id + 1
            name: Name of new group to be added. Default: 'Group {}'.format(id)

        Returns:
            id: Id of added group
            name: Name of added group

        """
        if parent_id is not None:
            assert (parent_id in self.group_ids), ('Parent id does not exist')

        if id is None:
            id = int(np.nanmax(np.asarray(self.group_ids, dtype=np.float)) + 1)
        else:
            assert (id not in self.groups_ids()), ('Id already exists')

        if name is None:
            name = 'Group {}'.format(id)

        new_group = wknml.Group(id, name, [])
        if parent_id is None:
            self.groups.append(new_group)
        else:
            self.groups = Skeleton._group_append(self.groups, parent_id, new_group)

        return id, name

    def delete_group(self, id, target_id):
        # TODO
        pass

    def define_nodes(self,
                     position_x: List[int],
                     position_y: List[int],
                     position_z: List[int],
                     id: List[int] = None,
                     radius: Optional[List[int]] = None,
                     rotation_x: Optional[List[float]] = None,
                     rotation_y: Optional[List[float]] = None,
                     rotation_z: Optional[List[float]] = None,
                     inVP: Optional[List[int]] = None,
                     inMag: Optional[List[int]] = None,
                     bitDepth: Optional[List[int]] = None,
                     interpolation: Optional[List[bool]] = None,
                     time: Optional[List[int]] = None,
                     comment: Optional[List[int]] = None) -> Nodes:
        """ Generates new nodes table from data.

        Args:
            position_x: Node position x
            position_y: Node position y
            position_z: Node position z
            id (optional): (Globally unique) Node id. Default: New unique ids are generated
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
            nodes: Nodes object

        """

        if id is None:
            id_max = self.max_node_id()
            id = list(range(id_max+1, id_max+len(position_x)+1))

        if radius is None:
            radius = [self.defaults['node']['radius'] for x in range(len(position_x))]

        if comment is None:
            comment = [self.defaults['node']['comment'] for x in range(len(position_x))]

        nodes = Nodes().append_from_list(id, position_x, position_y, position_z, radius, rotation_x, rotation_y,
                                         rotation_z, inVP, inMag, bitDepth, interpolation, time, comment)

        return nodes

    def define_nodes_from_positions(self, positions: np.ndarray) -> Nodes:
        """ Generates new nodes table from positions only (node ids are generated automatically).

        Args:
            positions (N x 3): Numpy array holding the (x,y,z) positions to be returned as nodes in a Nodes table

        Returns:
            nodes: Nodes object

        """

        id_max = self.max_node_id()
        id = np.array(range(id_max + 1, id_max + positions.shape[0] + 1)).reshape(-1, 1)

        nodes = Nodes().append_from_numpy(np.append(id, positions, axis=1))

        return nodes

    def get_distances_to_node(self,
                              positions: Union[Sequence[Tuple[int, int, int]], np.ndarray],
                              node_id: int = None,
                              tree_idx: int = None,
                              node_idx: int = None,
                              unit: str = 'um') -> List[np.ndarray]:
        """ Get the (euclidean) distances from the specified node to the provided (x,y,z) positions

        Args:
            positions (N x 3): Target (x,y,z) positions to which the distances should be computed
            node_id: Node id of the node for which the distances should be computed
            tree_idx: Tree idx of the node for which the distances should be computed
            node_idx: Node idx of the node for which the distances should be computed
            unit (optional): Unit flag specifying in which unit the distances should be returned.
                Options: 'vx' (voxels), 'nm' (nanometer), 'um' (micrometer). Default: 'um' (micrometer)

        Returns:
            distances: Array holding distances

        """

        assert (node_id is not None) ^ ((tree_idx is not None) & (node_idx is not None)), \
            'Either provide node_id or both tree_idx and node_idx'

        if type(positions) is not np.ndarray:
            positions = np.array(positions)

        if node_id is not None:
            node_idx, tree_idx = self.node_id_to_idx(node_id)

        unit_factor = self._get_unit_factor(unit)
        distances = Skeleton.get_distance(positions, np.array(self.nodes[tree_idx].position.values[node_idx]), unit_factor)

        return distances

    def get_distance_to_nodes(self,
                              position: Union[Tuple[int, int, int], np.ndarray],
                              tree_idx: int,
                              unit: str = 'um') -> List[np.ndarray]:
        """ Get the (euclidean) distances from the nodes of the specified tree to the provided (x,y,z) position

        Args:
            position (1 x 3): Target (x,y,z) position to which the node distances should be computed
            tree_idx: Tree idx for which node distances should be computed
            unit (optional): Unit flag specifying in which unit the distances should be returned.
                Options: 'vx' (voxels), 'nm' (nanometer), 'um' (micrometer). Default: 'um' (micrometer)

        Returns:
            distances: Array holding distances

        """

        if type(position) is not np.ndarray:
            position = np.array(position)

        unit_factor = self._get_unit_factor(unit)
        distances = Skeleton.get_distance(np.array(self.nodes[tree_idx].position.values), position, unit_factor)

        return distances

    def get_graph(self, tree_idx):
        """ Returns the networkx graph representation of a tree.

        Args:
            tree_idx: Linear index of the tree to be returned as graph object

        Returns:
            graph: Graph object

        """

        nodes = self.nodes[tree_idx]
        edges = self.edges[tree_idx]
        graph = Skeleton._get_graph(nodes, edges)

        return graph

    def get_shortest_path(self, node_id_start: int, node_id_end: int) -> List[int]:
        """ Returns the shortest path between two nodes of a tree.

        Args:
            node_id_start: Node id of start node
            node_id_end: Node id of end node

        Returns:
            shortest_path: Node indices comprising the shortest path

        """

        _, tree_idx_start = self.node_id_to_idx(node_id_start)
        _, tree_idx_end = self.node_id_to_idx(node_id_end)

        assert tree_idx_start == tree_idx_end, 'Provided node ids need to be part of the same tree'

        graph = self.get_graph(tree_idx_start)
        shortest_path = nx.shortest_path(graph, node_id_start, node_id_end)

        return shortest_path

    def plot(self,
             tree_inds: Union[int, List[int]] = None,
             view: str = None,
             colors: Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]] = None,
             unit: str = 'um',
             show: bool = True,
             ax: plt.axes = None):
        """ Generates a (3D) line plot of the trees contained in the skeleton object.

        Args:
            tree_inds (optional): Tree indices to be plotted.
                Default: All trees are plotted
            view (optional): Plot as 2D projection on orthonormal plane.
                Options: 'xy', 'xz', 'yz'
                Default: Plot as 3D projection
            colors (optional): Colors in which trees should be plotted. If only one RGBA tuple is specified, it is
                broadcasted over all trees. Alternatively, a list providing RGBA tuples for each tree can be passed.
                Default: Skeleton colors (self.colors) are used
            unit (optional): Specifies in which unit the plot should be generated.
                Options: 'vx' (voxels), 'nm' (nanometer), 'um' (micrometer).
                Default: 'um' (micrometer)
            show (optional): Displays the plot in an interactive window. For repeatedly plotting on the same axes, set
                to False. Default: True
            ax: Axes to be plotted on.

        Returns:
            ax: Axes which was plotted on
        """

        if tree_inds is None:
            tree_inds = list(range(len(self.nodes)))
        elif tree_inds is int:
            tree_inds = [tree_inds]

        if colors is None:
            colors = self.colors
        elif type(colors[0]) is not Sequence:
            colors = [colors] * self.num_trees()

        unit_factor = self._get_unit_factor(unit)

        allowed_views = ['xy', 'xz', 'yz']
        if view is not None:
            assert (view in allowed_views), \
                'The passed view argument: {} is not among the allowed views: {}'.format(view, allowed_views)

        if ax is None:
            fig = plt.figure()
            if view is None:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111, projection='rectilinear')
        else:
            if view is None:
                assert (ax.name == '3d'), \
                    'To generate a 3D skeleton plot, the projection type of the passed axes must be 3D'
            else:
                assert (ax.name != '3d'), \
                    'To generate a 2D skeleton plot, the projection type of the passed axes must be rectilinear'

        lims_min = []
        lims_max = []

        for tree_idx in tree_inds:
            edges = self.edges[tree_idx].copy()
            nodes = self.nodes[tree_idx].copy()

            if len(nodes) > 0:
                nodes['position'] = nodes['position'].multiply(unit_factor)
                if view is 'xy':
                    nodes = nodes.drop([('position', 'z')], axis=1)
                elif view is 'xz':
                    nodes = nodes.drop([('position', 'y')], axis=1)
                elif view is 'yz':
                    nodes = nodes.drop([('position', 'x')], axis=1)
                lims_min.append(np.min(nodes['position'].values, axis=0))
                lims_max.append(np.max(nodes['position'].values, axis=0))

                segments = []
                for edge in edges:
                    n0 = nodes['position'][nodes.id == edge[0]].values[0]
                    n1 = nodes['position'][nodes.id == edge[1]].values[0]
                    segment = [[c for c in n0], [c for c in n1]]
                    segments.append(segment)

                if view is None:
                    line_collection = art3d.Line3DCollection(segments=segments, colors=colors[tree_idx])
                    ax.add_collection3d(line_collection)
                else:
                    line_collection = LineCollection(segments=segments, colors=colors[tree_idx])
                    ax.add_collection(line_collection)

        lim_min = np.min(np.array(lims_min), axis=0)
        lim_max = np.max(np.array(lims_max), axis=0)

        ax.set_xlim(lim_min[0], lim_max[0])
        ax.set_ylim(lim_min[1], lim_max[1])
        if view is None:
            ax.set_zlim(lim_min[2], lim_max[2])
        else:
            ax.set_aspect('equal')

        if show:
            plt.show()

        return ax

    def write_nml(self, nml_write_path):
        """ Writes the present state of the skeleton object to a .nml file.

        Args:
            nml_write_path: Path to which .nml file should be written

        """

        # If the object does not have any trees, construct an empty tree before writing to enable webKnossos import
        if self.num_trees() == 0:
            self.add_tree()

        nml = self._skeleton_to_nml()
        with open(nml_write_path, "wb") as f:
            wknml.write_nml(f, nml)

    # Convenience Methods
    def node_id_to_idx(self, node_id: int) -> (int, int):
        """ Returns the linear tree and node indices for the provided node id."""

        node_idx = None
        for tree_idx, nodes in enumerate(self.nodes):
            index_list = nodes[nodes['id'] == node_id].index.tolist()
            if index_list:
                node_idx = index_list[0]
                break

        assert (node_idx is not None), \
            'node id {} does not exist'.format(node_id)

        return node_idx, tree_idx

    def node_idx_to_id(self, node_idx: int, tree_idx: int) -> int:
        """ Returns the node id for the provided tree and node idx."""

        node_id = self.nodes[tree_idx].loc[node_idx, 'id'].values[0]

        return node_id

    def min_group_id(self) -> int:
        """ Returns lowest group id. If no groups are defined, return None"""

        group_ids = np.asarray(self.group_ids, dtype=np.float)
        if np.all(np.isnan(group_ids)):
            group_id = None
        else:
            group_id = int(np.nanmin(group_ids))

        return group_id

    def max_group_id(self) -> int:
        """ Returns highest group id. If no groups are defined, return None"""

        group_ids = np.asarray(self.group_ids, dtype=np.float)
        if np.all(np.isnan(group_ids)):
            group_id = None
        else:
            group_id = int(np.nanmax(group_ids))

        return group_id

    def min_node_id(self) -> int:
        """ Returns lowest global node id."""

        if len(self.nodes) > 0:
            min_node_id = min([min(nodes.id) if len(nodes) > 0 else 0 for nodes in self.nodes])
        else:
            min_node_id = 0

        return min_node_id

    def max_node_id(self) -> int:
        """ Returns highest global node id."""

        if len(self.nodes) > 0:
            max_node_id = max([max(nodes.id) if len(nodes) > 0 else 0 for nodes in self.nodes])
        else:
            max_node_id = 0

        return max_node_id

    def min_tree_id(self) -> int:
        """ Returns lowest global tree id."""

        return min(self.tree_ids) if len(self.tree_ids)>0 else 0

    def max_tree_id(self) -> int:
        """ Returns highest global tree id."""

        return max(self.tree_ids) if len(self.tree_ids)>0 else 0

    def num_trees(self) -> int:
        """Returns number of trees contained in skeleton object."""

        return len(self.nodes)

    def groups_ids(self) -> List[int]:
        """ Returns all ids defined in groups tree"""

        _, groups_ids = Skeleton._group_get_ids(self.groups)

        return groups_ids

    # Private Methods
    def _get_unit_factor(self, unit: str) -> np.ndarray:
        """ Returns factor for unit conversion

        Args:
            unit: Unit for which to return the conversion factor.
                Options: 'vx' (voxels), 'nm' (nanometer), 'um' (micrometer)

        Returns:
            unit_factor (shape=(3,)): Unit conversion factors
        """

        unit_factors = {
            'vx': np.array((1, 1, 1)),
            'nm': np.array(self.parameters.scale),
            'um': np.array(self.parameters.scale)/1000
        }
        assert unit in unit_factors.keys(), 'Invalid unit'
        unit_factor = unit_factors[unit]

        return unit_factor

    def _reset_node_ids(self, start_id: int):
        """ Resets node ids of skeleton to begin with start value.

        Args:
            start_id: Start value to which the lowest node id should be set.
        """

        add_id = start_id - self.min_node_id()
        for tree_idx, _ in enumerate(self.nodes):
            self.nodes[tree_idx].id += add_id
            self.edges[tree_idx] += add_id

    def _reset_tree_ids(self, start_id: int):
        """ Resets tree ids of skeleton to begin with start value.

        Args:
            start_id: Start value to which the lowest tree id should be set.
        """

        add_id = start_id - self.min_tree_id()
        self.tree_ids = [tree_id + add_id for tree_id in self.tree_ids]

    def _reset_group_ids(self, start_id: int):
        """ Resets group ids of skeleton to begin with start value.

        Args:
            start_id: Start value to which the lowest group id should be set.
        """

        min_group_id = self.min_group_id()
        if min_group_id is not None:
            add_id = start_id - min_group_id
            self.group_ids = [i + add_id if i is not None else i for i in self.group_ids]
            self.groups = [Skeleton._group_modify_id(group, id_modifier=lambda x: x + add_id) for group in self.groups]

    def _parameters_to_skeleton(self, parameters):
        """ Generates bare skeleton object from parameters."""

        self.parameters = parameters

    def _nml_to_skeleton(self, nml):
        """ Converts wknml to skeleton data structures."""

        self.groups = nml.groups
        self.branchpoints = nml.branchpoints
        self.parameters = Parameters(**nml.parameters._asdict())

        for tree in nml.trees:
            self.add_tree(
                nodes=Skeleton._nml_nodes_to_nodes(nml_nodes=tree.nodes, nml_comments=nml.comments),
                edges=np.array([(edge.source, edge.target) for edge in tree.edges]),
                group_id=tree.groupId,
                name=tree.name,
                color=tree.color
            )

    def _skeleton_to_nml(self):
        """ Converts skeleton to wknml data structures."""

        trees = []
        for tree_idx, tree_id in enumerate(self.tree_ids):
            nml_nodes = Skeleton._nodes_to_nml_nodes(self.nodes[tree_idx])
            nml_edges = Skeleton._edges_to_nml_edges(self.edges[tree_idx])
            tree = wknml.Tree(
                id=tree_id,
                color=self.colors[tree_idx],
                name=self.names[tree_idx],
                groupId=self.group_ids[tree_idx],
                nodes=nml_nodes,
                edges=nml_edges
            )
            trees.append(tree)

        nml = wknml.NML(
            parameters=wknml.NMLParameters(**self.parameters._asdict()),
            trees=trees,
            branchpoints=self.branchpoints,
            comments=self._skeleton_to_nml_comments(),
            groups=self.groups
        )

        return nml

    def _skeleton_to_nml_comments(self):
        """ Converts skeleton to wknml comments."""

        nml_comments = []
        for nodes in self.nodes:
            comment_nodes = nodes[nodes['comment'].notnull()]
            for _, row in comment_nodes.iterrows():
                nml_comment = wknml.Comment(
                    node=row['id'].values[0],
                    content=row['comment'].values[0]
                )
                nml_comments.append(nml_comment)

        return nml_comments

    # Static Methods
    @staticmethod
    def define_parameters(
            name: str,
            scale: Tuple[float, float, float],
            offset: Tuple[float, float, float] = (0, 0, 0),
            time: int = 0,
            editPosition: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            editRotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            zoomLevel: float = 1.0,
            taskBoundingBox: Tuple[int, int, int, int, int, int] = None,
            userBoundingBox: Tuple[int, int, int, int, int, int] = None) -> Parameters:

        parameters = Parameters(
            name=name,
            scale=scale,
            offset=offset,
            time=time,
            editPosition=editPosition,
            editRotation=editRotation,
            zoomLevel=zoomLevel,
            taskBoundingBox=taskBoundingBox,
            userBoundingBox=userBoundingBox
        )

        return parameters

    # Static Methods
    @staticmethod
    def get_distance(positions: np.ndarray, position: np.ndarray, unit_factor: np.ndarray = None):
        """ Get the (euclidean) distances between positions and a target position

        Args:
            positions (N x 3): Array holding (multiple) x, y, z positions
            position (1 x 3): Array holding x, y, z position to which the distances should be computed
            unit_factors (1 x 3 Array, optional): Conversion factors with which distances are multiplied. Default (1,1,1)

        Returns:
            distances: Arrays holding distances

        """

        if unit_factor is None:
            unit_factor = np.array([1, 1, 1])

        distances = np.sqrt(np.sum(((positions - position) * unit_factor.reshape(1, 3)) ** 2, axis=1))

        return distances

    # Static Private Methods
    @staticmethod
    def _nml_nodes_to_nodes(nml_nodes, nml_comments):
        """ Converts wknml nodes (list of named tuples) to skeleton nodes (DataFrame subclass)."""

        data = [(node.id, node.position[0], node.position[1], node.position[2], node.radius, node.rotation[0],
                 node.rotation[1], node.rotation[2], node.inVp, node.inMag, node.bitDepth, node.interpolation,
                 node.time, np.nan) for node in nml_nodes]

        nodes = Nodes(data=data)

        # Add comments to nodes table
        comment_node_ids = [comment.node for comment in nml_comments]
        comment_strings = [comment.content for comment in nml_comments]
        nodes_ids_comments = nodes.id[nodes.id.isin(comment_node_ids)]
        for id in nodes_ids_comments:
            id_comment = comment_strings[comment_node_ids.index(id)]
            nodes.loc[nodes.id == id, ('comment', '')] = id_comment

        return nodes

    @staticmethod
    def _nodes_to_nml_nodes(nodes):
        """ Converts skeleton nodes (DataFrame subclass) to wknml nodes (list of named tuples)."""

        nml_nodes = []
        for idx, row in nodes.iterrows():
            nml_node = wknml.Node(
                id=int(row.id),
                position=tuple(row.position.values),
                radius=float(row.radius),
                rotation=tuple(row.rotation.values),
                inVp=int(row.inVp),
                inMag=int(row.inMag),
                bitDepth=int(row.bitDepth),
                interpolation=bool(row.interpolation.values),
                time=int(row.time)
            )
            nml_nodes.append(nml_node)

        return nml_nodes

    @staticmethod
    def _edges_to_nml_edges(edges):
        """ Converts skeleton edges (numpy array) to wknml edges (list of named tuples)."""

        nml_edges = []
        for idx in range(edges.shape[0]):
            nml_edge = wknml.Edge(
                source=int(edges[idx, 0]),
                target=int(edges[idx, 1]),
            )
            nml_edges.append(nml_edge)

        return nml_edges

    @staticmethod
    def _group_append(groups, id, new_group):
        """ Appends new group as a child of existing group with specified id. Currently only works up to depth=3."""

        path_inds = []
        _, _, idx = Skeleton._group_parent(groups, id)
        while id is not None:
            path_inds.append(idx)
            id, idx, _ = Skeleton._group_parent(groups, id)

        path_inds = list(reversed(path_inds))

        if len(path_inds) == 1:
            groups[path_inds[0]]._replace(children=new_group)
        elif len(path_inds) == 2:
            groups[path_inds[0]].children[path_inds[1]]._replace(children=new_group)
        elif len(path_inds) == 3:
            groups[path_inds[0]].children[path_inds[1]].children[path_inds[2]]._replace(children=new_group)

        return groups

    @staticmethod
    def _group_parent(groups, id, parent_id=None, parent_idx=None, child_idx=None):
        """ Returns the id of the parent group for a (child) group with specified id."""

        for group in groups:
            if id in [x.id for x in group.children]:
                parent_id = group.id
                parent_idx = groups.index(group)
                child_idx = [x.id for x in group.children].index(id)
            else:
                parent_id, parent_idx, child_idx = Skeleton._group_parent(group.children, id, parent_id, parent_idx, child_idx)

        return parent_id, parent_idx, child_idx

    @staticmethod
    def _group_modify_id(group, id_modifier):
        """ Modifies group ids with the passed id_modifier (e.g. lambda) function."""

        group = group._replace(id=id_modifier(group.id))
        group = group._replace(children=list(map(lambda g: Skeleton._group_modify_id(g, id_modifier), group.children)))

        return group

    @staticmethod
    def _group_get_ids(groups, ids = []):

        for group in groups:
            ids.append(group.id)
            Skeleton._group_get_ids(group.children, ids)

        return groups, ids

    @staticmethod
    def _get_graph(nodes: Nodes, edges: np.ndarray):
        """ Returns the networkx graph representation of provided nodes and edges."""

        graph = nx.Graph()
        graph.add_nodes_from(nodes['id'])
        attrs = nodes.set_index('id').to_dict('index')
        nx.set_node_attributes(graph, attrs)
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def _num_conn_comp(graph):
        """ Returns number of connected components for graph"""

        return nx.number_connected_components(graph)


