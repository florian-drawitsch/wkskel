import numpy as np
from wkskel import Skeleton


def add_tree_from_skel():

    skel = Skeleton('testdata/01_ref.nml')
    skel.add_tree_from_skel(Skeleton('testdata/02_ref.nml'))
    skel.write_nml('testdata/01_02_merged_gen.nml')

    skel_gen = Skeleton('testdata/01_02_merged_gen.nml')
    skel_ref = Skeleton('testdata/01_02_merged_ref.nml')

    # Test for unique node ids in generated merge nml
    _, c = np.unique(np.concatenate([nodes.id.values for nodes in skel_gen.nodes]), return_counts=True)
    assert all(c < 2)

    # Test for equal numbers of nodes and edges in both generated and reference merge nml
    assert set([len(nodes) for nodes in skel_gen.nodes]) == set([len(nodes) for nodes in skel_ref.nodes])
    assert set([len(edges) for edges in skel_gen.edges]) == set([len(edges) for edges in skel_ref.edges])


if __name__ == '__main__':
    add_tree_from_skel()


