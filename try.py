import numpy as np


A = np.mat(np.random.random((3,3)))
vecs = np.linalg.svd(A)
vec0 = vecs[0][:, 0]



