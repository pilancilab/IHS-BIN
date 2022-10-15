# cmp_ridge_regression.py

import numpy as np
from ridge_regression import ridge_regression, ridge_regression_over, get_SJLT_matrix
import time
import os
import scipy.io
import scipy.sparse
import sklearn.datasets
from utils import one_hot
import argparse
import pickle
from IHS_double import IHS_double, IHS_double_over

def get_parser():
    parser = argparse.ArgumentParser(description='ridge regression')
    parser.add_argument("--data_name", type=str, default="random",
                        help="data name", choices=["random", "rcv1", "gisette",
                        "MNIST-kron", "tfidf", "realsim", "avazu-app", "CIFAR10-kernel"])
    parser.add_argument("--n", type=float, default=1e3, help="number of sample")
    parser.add_argument("--d", type=float, default=1e3, help="number of dimension")
    parser.add_argument("--m", type=float, default=1e3, help="sketch dimension")
    parser.add_argument("--noise_level", type=float, default=0, help="noise level")
    parser.add_argument("--lbd_list_len", type=int, default=50, help="number of lambda")
    parser.add_argument("--lbd_min", type=float, default=1, help="minimal lambda")
    parser.add_argument("--lbd_max", type=float, default=1, help="maximal lambda")
    parser.add_argument("--bin_num", type=int, default=100, help="maximal lambda")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--plot", action='store_true', help="whether to plot")
    parser.add_argument("--ihs_iter_max", type=int, default=5, help="maximal iteration number")
    parser.add_argument("--ihs_iter_max_over", type=int, default=5, help="maximal iteration number")
    parser.add_argument("--tau_factor", type=float, default=1, help="tau_factor")
    parser.add_argument("--svd", action='store_true', help="whether to test svd")
    parser.add_argument("--native", action='store_true', help="whether to test native")
    parser.add_argument("--cg", action='store_true', help="whether to test cg")
    parser.add_argument("--IHS_BIN", action='store_true', help="whether to test IHS-BIN")
    parser.add_argument("--svd_over", action='store_true', help="whether to test svd")
    parser.add_argument("--native_over", action='store_true', help="whether to test native")
    parser.add_argument("--cg_over", action='store_true', help="whether to test cg")
    parser.add_argument("--IHS_BIN_over", action='store_true', help="whether to test IHS-BIN")
    parser.add_argument("--IHS_double", action='store_true', help="whether to test IHS-double")
    parser.add_argument("--IHS_double_over", action='store_true', help="whether to test IHS-double-over")
    parser.add_argument("--sketch_dim_start", type=float, default=1, help="sketch dimension")
    parser.add_argument("--eigATA", action='store_true', help="whether to compute eigATA")
    parser.add_argument("--eigAAT", action='store_true', help="whether to compute eigAAT")
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--shuffle", action='store_true', help="whether to shuffle the index")
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--sparsity", type=int, default=1)

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()
    print(args)
    data_name = args.data_name
    n = int(args.n)
    d = int(args.d)
    m = int(args.m)
    noise_level = args.noise_level
    lbd_list_len = int(args.lbd_list_len)
    lbd_min = args.lbd_min
    lbd_max = args.lbd_max
    seed = args.seed


    # cpu_times = np.zeros([6])
    np.random.seed(seed)
    lbd_list = lbd_max*(lbd_min/lbd_max)**np.linspace(0,1,lbd_list_len)
    # lbd_list = lbd_list[::-1]
    print(lbd_list)

    print('load dataset')
    if data_name == 'random':
        aux = np.arange(d)
        alpha = 0.99
        test_name = './results/random-alpha{}-n{}-d{}/noise{}-min{}-max{}'.format(alpha,n,d,noise_level,lbd_min,lbd_max)
        cor_mat = np.abs(aux.reshape([-1,1])-aux.reshape([1,-1]))
        cor_mat = np.power(alpha,cor_mat)
        A_full = np.random.randn(2*n,d)@cor_mat/np.sqrt(d)/np.sqrt(n)*10
        ind = np.arange(2*n)
        if args.shuffle:
            np.random.shuffle(ind)
        A = A_full[ind[:n],:]
        Atest = A_full[ind[n:],:]

        x_star = np.random.randn(d,1)/np.sqrt(d)
        # x_star = A.T@(A@x_star) # make the problem well-conditioned
        x_star = x_star/np.linalg.norm(x_star)
        b = A@x_star+noise_level*np.random.randn(n,1)
        btest = Atest@x_star+noise_level*np.random.randn(n,1)
    elif data_name == 'rcv1': #20242 47236
        test_name = './results/rcv1-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        A,b = sklearn.datasets.load_svmlight_file('{}/rcv1/rcv1_train.binary'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain
    elif data_name == 'gisette':#6000 5000
        test_name = './results/gisette-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        A,b = sklearn.datasets.load_svmlight_file('{}/gisette/gisette_scale'.format(args.data_dir))
        # A = A/np.max(A)
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain
    elif data_name == 'tfidf':#16087 150360
        test_name = './results/tfidf-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        A,b = sklearn.datasets.load_svmlight_file('{}/tfidf/E2006.train'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain
    elif data_name == 'realsim':#72309 20958
        test_name = './results/realsim-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        A,b = sklearn.datasets.load_svmlight_file('{}/realsim/real-sim'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain
    elif data_name == 'MNIST-kron': #60000, 780*780
        test_name = './results/MNIST-kron-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        if n<=10000:
            A, _ = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,0),'rb'))
            indd = np.arange(780*780)
            if args.shuffle:
                np.random.shuffle(indd)
            A = A[:n,:][:,indd[:d]]
            A = A.astype(np.float32)
            A = A/255**2
            Atest, _ = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,3),'rb'))
            Atest = Atest[:n,:][:,indd[:d]]
            Atest = Atest.astype(np.float32)/255**2

            _,b = sklearn.datasets.load_svmlight_file('{}/MNIST/mnist'.format(args.data_dir))
            b = one_hot(b.astype(np.int8),10) # one-hot encoding
            btest = b[30000:30000+n,:]
            b = b[:n,:]

        else:
            num_split = (n-1)//10000+1
            indd = np.arange(780*780)
            if args.shuffle:
                np.random.shuffle(indd)
            def mv(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:(j+1)*10000,:] = A_part@x
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:,:] = A_part@x
                return Ax

            def rmv(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                ATx = np.zeros([d,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        A_part = A_part.astype(np.float32)/255**2
                        ATx = ATx+A_part.T@x[j*10000:(j+1)*10000,:]
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        A_part = A_part.astype(np.float32)/255**2
                        ATx = ATx+A_part.T@x[j*10000:,:]
                return ATx

            def mv_sparse(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:(j+1)*10000,:] = (A_part@x).todense()
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:,:] = (A_part@x).todense()
                return Ax


            def mv_test(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/MNIST/mnist_kron_{}.p'.format(args.data_dir,j+3),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:(j+1)*10000,:] = A_part@x
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        A_part = A_part.astype(np.float32)/255**2
                        Ax[j*10000:,:] = A_part@x
                return Ax

            A = scipy.sparse.linalg.LinearOperator((n,d), matvec=mv,matmat=mv,rmatvec = rmv, rmatmat = rmv)
            Atest = scipy.sparse.linalg.LinearOperator((n,d), matvec=mv_test, matmat=mv_test)
            _,b = sklearn.datasets.load_svmlight_file('{}/MNIST/mnist'.format(args.data_dir))
            b = one_hot(b.astype(np.int8),10) # one-hot encoding
            btest = b[30000:30000+n,:]
            b = b[:n,:]

            x = np.random.randn(d,1)
            test_resid = A@x-Atest@x
            print('residual is {:.2e}'.format(np.linalg.norm(test_resid)))

        print(b.shape)
    elif data_name == 'CIFAR10-kernel': #50000, 25000
        test_name = './results/CIFAR10-kernel-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        if n<=10000:
            A, _ = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,0),'rb'))
            indd = np.arange(25000)
            if args.shuffle:
                np.random.shuffle(indd)
            A = A[:n,:][:,indd[:d]]
            Atest, _ = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,3),'rb'))
            Atest = Atest[:n,:][:,indd[:d]]
            
            _,b = sklearn.datasets.load_svmlight_file('{}/CIFAR10/cifar10'.format(args.data_dir))
            b = one_hot(b.astype(np.int8),10) # one-hot encoding
            btest = b[25000:25000+n,:]
            b = b[:n,:]
            print(A.shape)

        else:
            num_split = (n-1)//10000+1
            print(num_split)
            indd = np.arange(25000)
            if args.shuffle:
                np.random.shuffle(indd)
            def mv(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:(j+1)*10000,:] = A_part@x
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:,:] = A_part@x
                return Ax

            def rmv(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                ATx = np.zeros([d,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        # A_part = A_part.astype(np.float32)
                        ATx = ATx+A_part.T@x[j*10000:(j+1)*10000,:]
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        # A_part = A_part.astype(np.float32)
                        ATx = ATx+A_part.T@x[j*10000:,:]
                return ATx

            def mv_sparse(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,j),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:(j+1)*10000,:] = (A_part@x).todense()
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:,:] = (A_part@x).todense()
                return Ax


            def mv_test(x):
                if len(x.shape)==1:
                    x = x.reshape([-1,1])
                num_class = x.shape[1]
                Ax = np.zeros([n,num_class])
                for j in range(num_split):
                    A_part, b_part = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,j+3),'rb'))
                    if j<num_split-1:
                        A_part = A_part[:,indd[:d]]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:(j+1)*10000,:] = A_part@x
                    else:
                        A_part = A_part[:,indd[:d]][:n-10000,:]
                        # A_part = A_part.astype(np.float32)
                        Ax[j*10000:,:] = A_part@x
                return Ax

            A = scipy.sparse.linalg.LinearOperator((n,d), matvec=mv,matmat=mv,rmatvec = rmv, rmatmat = rmv)
            Atest = scipy.sparse.linalg.LinearOperator((n,d), matvec=mv_test, matmat=mv_test)
            _,b = sklearn.datasets.load_svmlight_file('{}/MNIST/mnist'.format(args.data_dir))
            b = one_hot(b.astype(np.int8),10) # one-hot encoding
            btest = b[25000:25000+n,:]
            b = b[:n,:]

        print(b.shape)
    elif data_name == 'avazu-app': #19264097, 29890095
        test_name = './results/avazu-app-n-{}-d{}/min{}-max{}'.format(n,d,lbd_min,lbd_max)
        A,b = sklearn.datasets.load_svmlight_file('{}/avazu/avazu-app'.format(args.data_dir))
        A = A.astype(np.float32)
        b = b.astype(np.float32)
        p, q = A.shape
        print(p,q)
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        # A = A.todense().A
        # A[:,-1] = 1 # add bias
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain

    if len(args.id)>0:
        test_name = test_name+'-{}'.format(args.id)
    print(test_name)
    if not os.path.exists(test_name):
        os.makedirs(test_name)

    if args.eigATA:
        K = A.T@A
        if scipy.sparse.issparse(K):
            K = K.todense()
        eigATA = np.linalg.eig(K)[0]
        eigATA.sort()
        eigATA_sort = eigATA[::-1]
    else:
        eigATA_sort = []

    if args.eigAAT:
        K = A@A.T
        if scipy.sparse.issparse(K):
            K = K.todense()
        eigAAT = np.linalg.eig(K)[0]
        eigAAT.sort()
        eigAAT_sort = eigAAT[::-1]
    else:
        eigAAT_sort = []

    # elif data_type == 'rcv1':
    #   data = scipy.io.loadmat('/Users/zackwang/Documents/MATLAB/optode/datasets/rcv1/rcv1_labels.mat')
    #   b = data['b'].ravel()[:n]
    #   btest = data['b'].ravel()[-1-n:-1]

    alpha = 0.5
    # print(b.shape)

    state_dict = {'svd':args.svd,'svd-over':args.svd_over, 'native': args.native, 'native-over': args.native_over,
                 'cg': args.cg, 'cg-over': args.cg_over, 'IHS-BIN':args.IHS_BIN, 'IHS-BIN-over':args.IHS_BIN_over, 
                'eigATA': args.eigATA,'eigAAT': args.eigAAT}
    output_dict = {}

    if args.svd:
        print('n = {} d = {} SVD'.format(n,d))
        vk, time_svd, info_svd = ridge_regression(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd',lbd_list=lbd_list, solver='svd', debug=True)
        output_dict['svd'] = [time_svd, info_svd]

    if args.svd_over:
        print('n = {} d = {} SVD-over'.format(n,d))
        vk, time_svd, info_svd = ridge_regression_over(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd',lbd_list=lbd_list, solver='svd', debug=True)
        output_dict['svd-over'] = [time_svd, info_svd]

    if args.native:
        print('n = {} d = {} LIN-SYS'.format(n,d))
        vk, time_lin, info_lin = ridge_regression(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='lin_sys', debug=True)
        output_dict['native'] = [time_lin, info_lin]

    if args.native_over:
        print('n = {} d = {} LIN-SYS-over'.format(n,d))
        vk, time_lin, info_lin = ridge_regression_over(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='lin_sys', debug=True)
        output_dict['native-over'] = [time_lin, info_lin]

    if args.cg:
        print('n = {} d = {} CG'.format(n,d))
        vk, time_cg, info_cg = ridge_regression(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='warm_cg', debug=True)
        output_dict['cg'] = [time_cg, info_cg]

    if args.cg_over:
        print('n = {} d = {} CG-over'.format(n,d))
        vk, time_cg, info_cg = ridge_regression_over(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='warm_cg', debug=True)
        output_dict['cg-over'] = [time_cg, info_cg]

    if args.IHS_double:
        ihs_opt = {'thres':0.5,'m':int(args.sketch_dim_start),'tol':1e-3, 'backtrack':True, "sparsity": args.sparsity}
        _, m_double, _ = IHS_double(A,b,lbd_min, ihs_opt, debug=True)
        m = min(m,m_double)

    if args.IHS_double_over:
        ihs_opt = {'thres':0.5,'m':int(args.sketch_dim_start),'tol':1e-3, 'backtrack':True, "sparsity": args.sparsity}
        _, m_double, _ = IHS_double_over(A,b,lbd_min, ihs_opt, debug=True)
        m = min(m,m_double)

    if args.IHS_BIN:
        print('n = {} d = {} m = {} IHS-BIN'.format(n,d,m))
        bin_num = min(args.bin_num, round(2*np.log(lbd_max/lbd_min).item()))
        sub_iter_max = max(round(2*(lbd_max/lbd_min)**(1/bin_num)),args.ihs_iter_max)
        tau_factor = args.tau_factor
        ihs_opt = {'sketch_dim':m,  'sub_iter_max':sub_iter_max, 'bin_num':bin_num, 
                  'sketch_type':'SJLT', 'tau_factor': tau_factor, "sparsity": args.sparsity}
        if data_name == 'CIFAR10-kernel' and n>10000:
            tic = time.process_time()
            S = get_SJLT_matrix(m,n,args.sparsity)
            SA = np.zeros([m,d])
            for j in range(num_split):
                A_part, b_part = pickle.load(open('{}/CIFAR10/cifar10_kernel_{}.p'.format(args.data_dir,j),'rb'))
                if j<num_split-1:
                    A_part = A_part[:,indd[:d]]
                    S_part = S[:,j*10000:(j+1)*10000]
                    SA = SA+S_part@A_part
                else:
                    A_part = A_part[:,indd[:d]][:n-10000,:]
                    # A_part = A_part.astype(np.float32)
                    S_part = S[:,j*10000:]
                    SA = SA+S_part@A_part
            toc = time.process_time()
            ihs_opt['SA'] = SA
            time_sketch = toc-tic
        elif data_name == 'MNIST-kernel' and n>10000:
            tic = time.process_time()
            S = get_SJLT_matrix(m,n,args.sparsity)
            SA = np.zeros([m,d])
            for j in range(num_split):
                A_part, b_part = pickle.load(open('{}/MNIST/mnist_kernel_{}.p'.format(args.data_dir,j),'rb'))
                if j<num_split-1:
                    A_part = A_part[:,indd[:d]]
                    S_part = S[:,j*10000:(j+1)*10000]
                    SA = SA+S_part@A_part
                else:
                    A_part = A_part[:,indd[:d]][:n-10000,:]
                    S_part = S[:,j*10000:]
                    SA = SA+S_part@A_part
            toc = time.process_time()
            ihs_opt['SA'] = SA
            time_sketch = toc-tic
        else:
            time_sketch = 0
        vk, time_ihs_bin, info_ihs_bin = ridge_regression(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='ihs_bin',ihs_opt=ihs_opt, debug=True)
        time_ihs_bin = time_ihs_bin+time_sketch
        output_dict['IHS-BIN'] = [time_ihs_bin, info_ihs_bin]

    if args.IHS_BIN_over:
        print('n = {} d = {} m = {} IHS-BIN-over'.format(n,d,m))
        bin_num = min(args.bin_num, round(2*np.log(lbd_max/lbd_min).item()))
        sub_iter_max = max(round(2*(lbd_max/lbd_min)**(1/bin_num)),args.ihs_iter_max_over)
        tau_factor = args.tau_factor
        ihs_opt = {'sketch_dim':m,  'sub_iter_max':sub_iter_max, 'bin_num':bin_num, 
                   'sketch_type':'SJLT', 'tau_factor': tau_factor, "sparsity": args.sparsity}
        if data_name == 'MNIST-kron' and n>10000:
            tic = time.process_time()
            S = get_SJLT_matrix(m,d,args.sparsity)
            SA = mv_sparse(S.T)
            SAT = SA.T
            toc = time.process_time()
            ihs_opt['SAT'] = SAT
            time_sketch = toc-tic
        else:
            time_sketch = 0
        vk, time_ihs_bin, info_ihs_bin = ridge_regression_over(A,b,alpha,lbd_min,lbd_max,Atest,btest, 
            prob_type='multi_lbd', lbd_list=lbd_list, solver='ihs_bin',ihs_opt=ihs_opt, debug=True)
        time_ihs_bin = time_ihs_bin+time_sketch
        output_dict['IHS-BIN-over'] = [time_ihs_bin, info_ihs_bin]


    results = [state_dict, eigATA_sort, eigAAT_sort, lbd_list, output_dict]
    pickle.dump(results,open('{}/results.p'.format(test_name),'wb'))
    if args.plot:
        import matplotlib.pyplot as plt
        if state_dict['eigATA']:
            fig = plt.figure(figsize=(10,10))
            ax = fig.gca()
            ax.plot(eigATA_sort)
            ax.set_yscale('log')
            ax.set_title('eigenvalues of ATA')
            fig.savefig('{}/ATA_eig.png'.format(test_name))
        if state_dict['eigAAT']:
            fig = plt.figure(figsize=(10,10))
            ax = fig.gca()
            ax.plot(eigAAT_sort)
            ax.set_yscale('log')
            ax.set_title('eigenvalues of AAT')
            fig.savefig('{}/AAT_eig.png'.format(test_name))
        # time
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        for key in output_dict.keys():
            time_alg = output_dict[key][0]
            ax.plot(time_alg,label=key)
        ax.legend()
        ax.set_title('Time')
        fig.savefig('{}/time.png'.format(test_name))

        # sub iter
        if state_dict['IHS-BIN']:
            fig = plt.figure(figsize=(10,10))
            ax = fig.gca()
            info_ihs_bin = output_dict['IHS-BIN'][1]
            # if data_type=='random':
            #   ax.plot(info_ihs['sub_iters'],label='IHS')
            #   ax.plot(info_ihs_mom['sub_iters'],label='IHS-MOM')
            ax.plot(info_ihs_bin['sub_iters'],label='IHS-BIN')
            ax.legend()
            ax.set_title('Sub-iteration number')
            fig.savefig('{}/sub_iters.png'.format(test_name))


        # train loss
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        for key in output_dict.keys():
            info = output_dict[key][1]
            ax.plot(lbd_list,info['train_loss'],label=key)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_title('Train loss')
        fig.savefig('{}/train_loss.png'.format(test_name))

        # test loss
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        for key in output_dict.keys():
            info = output_dict[key][1]
            ax.plot(lbd_list,info['test_loss'],label=key)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_title('Test loss')
        fig.savefig('{}/test_loss.png'.format(test_name))

if __name__ == '__main__':
    main()