import numpy as np
from wkskel import Skeleton


def add_tree_from_skel():

    # Test merging skeleton with multiple trees with a single tree from another skeleton
    skel = Skeleton('testdata/01_ref.nml')
    skel.add_tree_from_skel(Skeleton('testdata/02_ref.nml'), tree_idx=1, group_id=10)

    # Test for unique node and tree ids in merged nml
    _, c = np.unique(np.concatenate([nodes.id.values for nodes in skel.nodes]), return_counts=True)
    assert all(c < 2)
    _, c = np.unique(skel.tree_ids, return_counts=True)
    assert all(c < 2)

    # Test writing
    skel.write_nml('testdata/01_02_single_merged_gen.nml')


def add_trees_from_skel():

    # Test merging skeletons both having both root and (nested) group trees
    skel = Skeleton('testdata/01_ref.nml')
    skel.add_trees_from_skel(Skeleton('testdata/02_ref.nml'))

    # Test for unique node and tree ids in merged nml
    _, c = np.unique(np.concatenate([nodes.id.values for nodes in skel.nodes]), return_counts=True)
    assert all(c < 2)
    _, c = np.unique(skel.tree_ids, return_counts=True)
    assert all(c < 2)

    # Test writing
    skel.write_nml('testdata/01_02_merged_gen.nml')

    # Test reading
    skel_gen = Skeleton('testdata/01_02_merged_gen.nml')
    skel_ref = Skeleton('testdata/01_02_merged_ref.nml')

    # Test for equal numbers of nodes and edges in both generated and reference merge nml
    assert set([len(nodes) for nodes in skel_gen.nodes]) == set([len(nodes) for nodes in skel_ref.nodes])
    assert set([len(edges) for edges in skel_gen.edges]) == set([len(edges) for edges in skel_ref.edges])

    # Test merging skeleton having both root and group trees with one having only root trees
    skel = Skeleton('testdata/02_ref.nml')
    skel.add_trees_from_skel(Skeleton('testdata/03_ref.nml'))
    skel.write_nml('testdata/02_03_merged_gen.nml')

    # Test merging skeleton having only root trees with one having both root and group trees
    skel = Skeleton('testdata/03_ref.nml')
    skel.add_trees_from_skel(Skeleton('testdata/02_ref.nml'))
    skel.write_nml('testdata/03_02_merged_gen.nml')


if __name__ == '__main__':
    add_tree_from_skel()
    add_trees_from_skel()


