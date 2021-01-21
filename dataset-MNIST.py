import sklearn.datasets
import numpy as np
from tqdm import tqdm
import scipy.sparse
import pickle
import time

print('start')
data_dir = './datasets'

A,b = sklearn.datasets.load_svmlight_file('{}/MNIST/mnist'.format(data_dir))
A = A.tolil().astype(np.int8)


start = time.time()

for j in range(6):
    print('part {}'.format(j))
    A_new= scipy.sparse.dok_matrix((10000, 780*780)).tolil().astype(np.int8)
    for i in range(10000):
        if i %100 == 0:
            print('iter: {} time: {}'.format(i,time.time()-start))
        A_new[i,:]=scipy.sparse.kron(A[i+j*10000,:],A[i+j*10000,:])

    A_new = A_new.tocsr()
    print(A_new.nnz)
    A_new = A_new.astype(np.int8)

    with open('{}/MNIST/mnist_kron_{}.p'.format(data_dir,j),'wb') as f:
        pickle.dump([A_new,b[j*10000:(j+1)*10000]],f)
