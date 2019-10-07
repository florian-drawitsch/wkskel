from wkskel import Nodes, Skeleton

nml_source = '../testdata/test_01.nml'

# Construct skeleton object from nml file
skel = Skeleton(nml_source)

# Return number of trees contained in the skeleton
num_trees = skel.num_trees()
print('Number of trees: {}'.format(num_trees))

# For all of those trees, print the number of contained nodes and edges
num_nodes_all = [len(x) for x in skel.nodes]
num_edges_all = [x.shape[0] for x in skel.edges]
print('Number of nodes in each tree: {}'.format(num_nodes_all))
print('Number of edges in each tree: {}'.format(num_edges_all))

skel.node_idx_to_id(3, 5)
skel.plot()

