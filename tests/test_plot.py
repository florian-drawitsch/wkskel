from wkskel import Skeleton


def plot():
    skel = Skeleton('testdata/01_02_merged_ref.nml')
    skel.plot(tree_inds=[0, 2], unit='um', view=None)


if __name__ == '__main__':
    plot()
