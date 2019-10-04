import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from typing import List, Tuple, Optional

import wknml

from .nodes import Nodes

# TODO: Implement: Construct without nml, only with parameters provided
# TODO: Remove comments attribute
# TODO: Implement: Delete group (non trivial, need to attach all children of group to specified base)


class Skeleton:
    """The Skeleton class facilitates scientific analysis and manipulation of webKnossos tracings

    It is designed as a high-level interface for working with nml files generated e.g with webKnossos. It makes use of
    the (low-level) `wknml` package mostly as an I/O interface to nml files.

    Class Attributes:
        DEFAULTS (dict): Global default parameters which are passed to each skeleton object instance

    """

    DEFAULTS = {
        'node_radius': 100,
        'node_comment': ''
    }

    def __init__(self, nml_source: str = None, **kwargs):
        self.nodes = []
        self.edges = []
        self.names = []
        self.colors = []
        self.tree_ids = []
        self.group_ids = []
        self.groups = []
        self.branchpoints = []
        self.comments = []
        self.parameters = {}
        self.defaults = self.DEFAULTS

        if nml_source is not None:
            try:
                with open(nml_source, "rb") as f:
                    nml = wknml.parse_nml(f)
                    self._nml_to_skeleton(nml)
            except IOError:
                print(nml_source + ' does not seem to exist or is not a valid nml file')

    def add_tree(self,
                 nodes: Nodes = Nodes(),
                 edges: np.ndarray = np.empty((0, 2), dtype=np.uint32),
                 tree_id: int = None,
                 group_id: int = None,
                 name: str = '',
                 color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)):
        """ Appends new tree to skeleton

        Args:
            nodes (optional): Nodes representing tree to be added
            edges (optional): Edges representing tree to be added
            tree_id (optional): Tree id to be used for new tree. Default: Highest current tree id + 1
            group_id (optional): Group id to be used for new tree. Default: None
            name (optional): Name to be used for new tree. Default: Empty str
            color (optional): Color to be used for new tree specified as (r, g, b, alpha). Default: (0, 0, 0, 1)
        """

        if tree_id is None:
            tree_id = max(self.tree_ids) + 1

        self.nodes.append(nodes)
        self.edges.append(edges)
        self.tree_ids.append(tree_id)
        self.group_ids.append(group_id)
        self.names.append(name)
        self.colors.append(color)

    def add_nodes_as_trees(self,
                           nodes: Nodes,
                           tree_ids: List[int] = None,
                           group_ids: List[int] = None,
                           names: List[str] = None,
                           colors: List[Tuple[float, float, float, float]] = None):
        """ Appends each of the specified nodes as separate trees to the skeleton (1 node each)

        Args:
            nodes: Nodes representing the trees to be added
            tree_ids (optional): Tree ids to be assigned to the newly added trees. Default: Global max + [1, n]
            group_ids (optional): Group ids to be assigned to the newly added trees. Default: None
            names (optional): Names to be assigned to the newly added trees.
            colors (optional): Colors to be used for the new trees specified as (r, g, b, alpha). Default: (0, 0, 0, 1)
        """

        if tree_ids is None:
            tree_id_start = self.get_highest_tree_id() + 1
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

    def delete_tree(self, tree_id):
        """ Deletes tree with specified tree_id

        Args:
            tree_id: Tree_id of tree to be deleted

        """
        tree_idx = self.tree_ids.index(tree_id)
        self.nodes.pop(tree_idx)
        self.edges.pop(tree_idx)
        self.names.pop(tree_idx)
        self.colors.pop(tree_idx)
        self.tree_ids.pop(tree_idx)
        self.group_ids.pop(tree_idx)

    def add_group(self, parent_id: int = None, id: int = None, name: str = None):
        """ Adds a new group to skeleton object

        Args:
            parent_id: Parent group id to which new group is added as a child. Default: None (root group)
            id: Id of new group to be added. Default: Current max group id + 1
            name: Name of new group to be added. Default: 'Group {}'.format(id)

        Returns:
            id: Id of added group
            name: Name of added group

        """
        assert (parent_id in self.group_ids), ('Parent id does not exist')

        if id is None:
            id = int(np.nanmax(np.asarray(self.group_ids, dtype=np.float)) + 1)
        else:
            assert (id not in self.group_ids), ('Id already exists')

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

    def make_nodes(self,
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
        """ Generates new nodes table from data

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
            id_max = self.get_highest_node_id()
            id = list(range(id_max+1, id_max+len(position_x)+1))

        if radius is None:
            radius = [self.defaults['node_radius'] for x in range(len(position_x))]

        if comment is None:
            comment = [self.defaults['node_comment'] for x in range(len(position_x))]

        nodes = Nodes().append_from_list(id, position_x, position_y, position_z, radius, rotation_x, rotation_y,
                                         rotation_z, inVP, inMag, bitDepth, interpolation, time, comment)

        return nodes

    def make_nodes_from_positions(self, positions: np.ndarray) -> Nodes:
        """ Generates new nodes table from positions only

        Args:
            positions: Numpy array holding the x y z positions to be returned as nodes in a Nodes table

        Returns:
            nodes: Nodes object

        """
        nodes = self.nodes(*list(positions.T))

        return nodes

    def get_highest_tree_id(self):
        """ Returns highest global tree id

        Returns:
            tree_id (int): Highest global tree id

        """
        return max(self.tree_ids)

    def get_highest_node_id(self, tree_ids: List[int] = None) -> int:
        """ Returns highest (global) node_id for all trees (default) or specified trees

        Args:
            tree_ids (list of int, optional): tree_ids for which highest node_id should be returned
                Default: None, i.e. highest global node_id for all trees is returned

        Returns:
            node_id (int): Highest node id

        """
        if tree_ids is None:
            tree_ids = self.tree_ids

        node_id = 0
        for tree_id in self.tree_ids:
            node_id = max([node_id, self.nodes[self.tree_ids.index(tree_id)]['id'].max()])

        return node_id

    def get_idx_for_node_id(self, node_id: int) -> (int, int):
        """ Gets the linear tree and node indices for the provided node id

        Args:
            node_id: Node id for which linear tree and node indices should be returned

        Returns:
            node_idx: Node index corresponding to the provided node id
            tree_idx: Tree index corresponding to the provided node id
        """

        for tree_idx, nodes in enumerate(self.nodes):
            index_list = nodes[nodes['id'] == node_id].index.tolist()
            if index_list:
                node_idx = index_list[0]
                break

        return node_idx, tree_idx

    def get_shortest_path(self, node_id_start: int, node_id_end: int) -> List[int]:
        """ Gets the shortest path between two nodes of a tree

        Args:
            node_id_start: Node id of start node
            node_id_end: Node id of end node

        Returns:
            shortest_path: Node indices comprising the shortest path

        """

        _, skel_idx_start = self.get_idx_for_node_id(node_id_start)
        _, skel_idx_end = self.get_idx_for_node_id(node_id_end)

        assert skel_idx_start == skel_idx_end, 'Provided node ids need to be part of the same tree'

        edge_list = self.edges[skel_idx_start].tolist()
        g = nx.Graph(edge_list)
        shortest_path = nx.shortest_path(g, node_id_start, node_id_end)

        return shortest_path

    def plot(self,
             tree_inds: Optional[List[int]] = None,
             colors: Optional[List[Tuple[float, float, float, float]]] = None,
             um_scale: bool = True,
             ax: Optional[plt.axes] = None):
        """ Generates a (3D) line plot of the trees contained in the skeleton object

        Args:
            tree_inds: Tree indices to be plotted
            colors: Colors in which trees should be plotted
            um_scale: Plot on micrometer scale instead of voxel scale
            ax: Axes to be plotted on

        Returns:
            ax: Plot axes
        """

        if tree_inds is None:
            tree_inds = list(range(len(self.nodes)))

        if colors is None:
            cm = plt.get_cmap('Dark2', len(self.nodes))
            colors = cm.colors

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        for tree_idx in tree_inds:

            nodes = self.nodes[tree_idx]
            edges = self.edges[tree_idx]

            if um_scale:
                nodes['position'] = nodes['position'].multiply(self.parameters.scale).divide(1000)

            for edge in edges:
                n0 = nodes['position'][nodes.id == edge[0]].values[0]
                n1 = nodes['position'][nodes.id == edge[1]].values[0]

                ax.plot([n0[0], n1[0]],
                        [n0[1], n1[1]],
                        [n0[2], n1[2]],
                        color=colors[tree_idx])

        return ax

    def write_nml(self, nml_write_path):
        """ Writes the present state of the skeleton object to a .nml file

        Args:
            nml_write_path: Path to which .nml file should be written

        """
        nml = self._skeleton_to_nml()
        with open(nml_write_path, "wb") as f:
            wknml.write_nml(f, nml)

    # Convenience Methods
    def num_trees(self) -> int:
        """Returns number of trees contained in skeleton object

        Returns:
            num_trees: Number of trees contained in skeleton object

        """
        num_trees = len(self.nodes)

        return num_trees

    # Private Methods
    def _nml_to_skeleton(self, nml):

        for tree in nml.trees:
            nodes = Skeleton._nml_nodes_to_nodes(nml_nodes=tree.nodes, nml_comments=nml.comments)
            self.nodes.append(nodes)
            self.edges.append(np.array([(edge.source, edge.target) for edge in tree.edges], dtype=np.uint32))
            self.names.append(tree.name)
            self.colors.append(tree.color)
            self.tree_ids.append(tree.id)
            self.group_ids.append(tree.groupId)

        self.groups = nml.groups
        self.branchpoints = nml.branchpoints
        self.comments = nml.comments
        self.parameters = nml.parameters

    def _skeleton_to_nml(self):

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

        nml_comments = self._skeleton_to_nml_comments()

        nml = wknml.NML(
            parameters=self.parameters,
            trees=trees,
            branchpoints=self.branchpoints,
            comments=nml_comments,
            groups=self.groups
        )

        return nml

    def _skeleton_to_nml_comments(self):

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

    # Static Private Methods
    @ staticmethod
    def _nml_nodes_to_nodes(nml_nodes, nml_comments):

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

    @ staticmethod
    def _nodes_to_nml_nodes(nodes):

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

    @ staticmethod
    def _edges_to_nml_edges(edges):

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

        for group in groups:
            if id in [x.id for x in group.children]:
                parent_id = group.id
                parent_idx = groups.index(group)
                child_idx = [x.id for x in group.children].index(id)
            else:
                parent_id, parent_idx, child_idx = Skeleton._group_parent(group.children, id, parent_id, parent_idx, child_idx)

        return parent_id, parent_idx, child_idx

















