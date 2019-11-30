import numpy as np


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk] = 1
    # import pdb; pdb.set_trace()
    return buf

