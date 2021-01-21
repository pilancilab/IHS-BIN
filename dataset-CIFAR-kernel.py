import sklearn.datasets
import numpy as np
from tqdm import tqdm
import scipy.sparse
import pickle
import time

print('start')
data_dir = './datasets'

def build_kernel_matrix(A, B, kernel_type='Gaussian', kernel_opt = {}):
    if kernel_opt.get('bandwidth',-1)==-1:
        kernel_opt['bandwidth'] = -1
    n, d = A.shape
    A_sum = np.sum(A**2,axis=1)
    B_sum = np.sum(B**2,axis=1)
    if kernel_type == 'Gaussian':
        dist_mat = -2*np.matmul(B, A.T)+B_sum.reshape([-1,1])+A_sum.reshape([1,-1])
        bandwidth = kernel_opt['bandwidth']
        if bandwidth == -1:
            bandwidth = np.median(dist_mat)/2/np.log(d+1)
        K = np.exp(-dist_mat*0.5/bandwidth)
    return K, bandwidth

A,b = sklearn.datasets.load_svmlight_file('{}/CIFAR10/cifar10'.format(data_dir))
A = A/255
A = A.A
Atrain = A[:25000,:]
Atest = A[25000:,:]

kernel_opt = {'bandwidth':1000}

Ktrain, bandwidth = build_kernel_matrix(Atrain,Atrain, kernel_opt = kernel_opt)
print(bandwidth)
kernel_opt = {'bandwidth':bandwidth}
Ktest, _ =  build_kernel_matrix(Atrain,Atest, kernel_opt = kernel_opt)

for j in range(3):
    print(j)
    with open('{}/CIFAR10/cifar10_kernel_{}.p'.format(data_dir,j),'wb') as f:
        K_part = Ktrain[j*10000:(j+1)*10000,:]
        b_part = b[j*10000:(j+1)*10000]
        pickle.dump([K_part,b_part],f)

for j in range(3):
    print(j+3)
    with open('{}/CIFAR10/cifar10_kernel_{}.p'.format(data_dir,j+3),'wb') as f:
        K_part = Ktest[j*10000:(j+1)*10000,:]
        b_part = b[(j+3)*10000:(j+4)*10000]
        pickle.dump([K_part,b_part],f)

