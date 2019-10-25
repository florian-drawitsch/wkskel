import numpy as np
from wkskel import Skeleton


def construct_from_nml():

    skel = Skeleton('testdata/02_ref.nml')
    skel.write_nml('testdata/02_gen.nml')


def construct_from_parameters_empty():

    parameters = Skeleton.define_parameters('2017-01-12_FD0156-2', (11.24, 11.24, 32))
    skel = Skeleton(parameters=parameters)
    skel.write_nml('testdata/PE_gen.nml')


def construct_from_parameters_add_tree():

    parameters = Skeleton.define_parameters('2017-01-12_FD0156-2', (11.24, 11.24, 32))
    skel = Skeleton(parameters=parameters)
    nodes = skel.define_nodes([40000, 40100, 40200], [45000, 45100, 45200], [1000, 1100, 1200], [1, 2, 3])
    edges = [(1, 3), (2, 3)]
    skel.add_tree(nodes, edges)
    skel.write_nml('testdata/PA_gen.nml')


if __name__ == '__main__':
    construct_from_nml()
    construct_from_parameters_empty()
    construct_from_parameters_add_tree()
