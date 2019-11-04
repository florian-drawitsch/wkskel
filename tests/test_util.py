from wkskel import Skeleton


def get_distances_to_node():
    skel = Skeleton('testdata/01_ref.nml')
    positions = skel.nodes[5].position
    skel.get_distances_to_node(positions=positions, node_id=35370)
    skel.get_distances_to_node(positions=positions, tree_idx=0, node_idx=5)


def get_distance_to_nodes():
    skel = Skeleton('testdata/01_ref.nml')
    skel.get_distance_to_nodes(position=(40000, 38000, 1000), tree_idx=6)


if __name__ == '__main__':
    get_distances_to_node()
    get_distance_to_nodes()