from wkskel import Skeleton


def plot():
    skel = Skeleton('testdata/01_02_merged_ref.nml')
    skel.plot(unit='um', view=None, colors='Dark2')


if __name__ == '__main__':
    plot()
