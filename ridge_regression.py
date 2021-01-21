import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import time
import scipy.sparse
import scipy.sparse.linalg
from utils import get_SJLT_matrix

def ridge_regression(A,b,alpha,lbd_min,lbd_max, Atest, btest, solver = 'svd',prob_type='trust_region',
                 lbd_list=[],gd_opt={}, ihs_opt={}, precision=1e-3,iter_max = 100,debug=False):
    # A can be a matrix or a linear operator
    # the type of A includes np.ndarray or np.mat or scipy.sparse.linalg.LinearOperator
    if prob_type == 'multi_lbd' and len(lbd_list)==0:
        raise Exception('For prob_type=multi_lbd, lbd_list shall not be empty')
    if prob_type == 'multi_lbd' and len(lbd_list)>0:
        iter_max = max(iter_max,len(lbd_list))
    # print(iter_max)

    n, d = A.shape
    # if type(A) = np.ndarray:
    #     A = np.mat(A)
    time_list = np.zeros(iter_max)
    b_nrm = np.linalg.norm(b)

    train_loss = np.zeros(iter_max)
    test_loss = np.zeros(iter_max)

    if len(b.shape)==1:
        b = b.reshape([-1,1])
    num_classes = b.shape[1]

    if solver == 'lin_sys' or solver == 'warm_cg':
        tic = time.process_time()
        ATA = A.T@A
        ATb = A.T@b
        toc = time.process_time()
        time_ATA = toc-tic
    if solver == 'svd':
        tic = time.process_time()
        if type(A)==scipy.sparse.linalg.interface._CustomLinearOperator:
            raise Exception('Error: svd solver does not support for linear operator')
        if scipy.sparse.issparse(A):
            U, Sigma, V = np.linalg.svd(A.todense(),full_matrices=False)
        else:
            U, Sigma, V = np.linalg.svd(A,full_matrices=False)
        try:
            V = V.A
        except:
            pass
        UTb = U.T@b
        toc = time.process_time()
        time_SVD = toc-tic
    if solver == 'gd' or solver == 'gd_comp':
        if gd_opt.get('sub_iter_max',-1)==-1:
            gd_opt['sub_iter_max'] = 1e3
        A_norm = np.linalg.norm(A,2)
        k = 1+np.ceil(A_norm**2/lbd_min)*10
        print(A_norm,k)
        k = min(k,gd_opt['sub_iter_max'])
        k = int(k)
        tau = 1/(A_norm**2+lbd_max)
    if solver == 'ihs' or solver == 'ihs_bin':
        if ihs_opt.get('sub_iter_max',-1)==-1:
            ihs_opt['sub_iter_max'] = 1e3
        if ihs_opt.get('warm_start',-1)==-1:
            ihs_opt['warm_start'] = False
        if ihs_opt.get('sketch_dim',-1) == -1:
            raise Exception('ihs_opt shall have key sketch_dim.')
        if ihs_opt.get('tol',-1) == -1:
            ihs_opt['tol'] = 1e-3
        if ihs_opt.get('sub_solver',-1) == -1:
            ihs_opt['sub_solver'] = 'svd'
        if ihs_opt.get('sketch_type',-1) == -1:
            ihs_opt['sketch_type'] = 'Gaussian'
        if ihs_opt.get('sparsity',-1) == -1:
            ihs_opt['sparsity'] = 1
        if ihs_opt.get('backtrack',-1) == -1:
            ihs_opt['backtrack'] = False
        if ihs_opt.get('bls_alpha',-1) == -1:
            ihs_opt['bls_alpha'] = 1e-3
        if ihs_opt.get('bls_beta',-1) == -1:
            ihs_opt['bls_beta'] = 0.5
        if ihs_opt.get('bin_num',-1) == -1:
            ihs_opt['bin_num'] = 1
        if ihs_opt.get('tau_factor',-1) == -1:
            ihs_opt['tau_factor'] = 1

        m = ihs_opt['sketch_dim']
        tau = 1
        tol = ihs_opt['tol']
        sub_solver = ihs_opt['sub_solver']
        sketch_type = ihs_opt['sketch_type']
        sparsity = ihs_opt['sparsity']
        backtrack = ihs_opt['backtrack']
        bls_alpha = ihs_opt['bls_alpha']
        bls_beta = ihs_opt['bls_beta']
        # lbd0 = ihs_opt['lbd0']
        bin_num = ihs_opt['bin_num']
        tau_factor = ihs_opt['tau_factor']

        k = ihs_opt['sub_iter_max']
        k = int(k)
        tic = time.process_time()
        try:
            SA = ihs_opt['SA']
        except:
            if sketch_type == 'Gaussian':
                S = np.random.randn(m,n)/np.sqrt(m)
            elif sketch_type == 'SJLT':
                S = get_SJLT_matrix(m,n,sparsity)
            SAT = A.T@S.T
            SA = SAT.T
        toc = time.process_time()
        time_sketch = toc-tic
        
    if solver == 'ihs' or solver == 'ihs_bin':
        if sub_solver=='svd':
            tic = time.process_time()
            if scipy.sparse.issparse(SA):
                US, SigmaS, VS = np.linalg.svd(SA.todense(),full_matrices=False)
            else:
                US, SigmaS, VS = np.linalg.svd(SA,full_matrices=False)
            try:
                VS = VS.A
            except:
                pass
            toc = time.process_time()
            time_SVD_ihs = toc-tic
        
    if solver == 'gd_comp':
        ATbs = np.zeros([d,k])
        ATbs[:,0] = (A.T@b).squeeze()
        for i in range(1,k):
            tmp = ATbs[:,i-1]
            ATbs[:,i] = tmp-tau*A.T@(A@tmp)
        bi_mat = np.zeros([k,k])
        for j in range(k):
            for l in range(k-j):
                bi_mat[l,j] = (-1)**j*comb(l+j,j)*tau
        ATbs = ATbs@bi_mat

    def ihs_bin_build(lbd_max_sub, lbd_min_sub):
        tic = time.process_time()
        ratio = np.sqrt(lbd_min_sub/lbd_max_sub)
        tau = 2*tau_factor/(ratio+1/ratio) #2/(ratio+1/ratio)
        lbd0_sub = np.sqrt(lbd_min_sub*lbd_max_sub)
        ujs = np.zeros([d,num_classes,k])
        tujs = np.zeros([d,num_classes,k])
        grad = A.T@b # d by num_classes matrix
        if sub_solver == 'smw':
            SAAS_inv = np.linalg.inv(np.eye(m)+SA@SA.T/lbd0_sub)
            ASSA_inv = lambda x: (x/lbd0_sub-SA.T@(SAAS_inv@(SA@x))/(lbd0**2))      
        elif sub_solver == 'svd':
            SigmaS_bar = 1/(SigmaS**2+lbd0_sub)-1/lbd0_sub
            ASSA_inv = lambda x:(VS.T*SigmaS_bar)@(VS@x)+x/lbd0_sub
        ujs[:,:,0] = ASSA_inv(grad)
        tujs = tujs+ujs
        for i in range(1,k):
            j = i
            ujs[:,:,j] = -ASSA_inv(ujs[:,:,j-1])
            for j in range(i-1,0,-1):
                ujs[:,:,j] = -ASSA_inv(ujs[:,:,j-1]+tau*A.T@(A@ujs[:,:,j]))+ujs[:,:,j]
            ujs[:,:,0] = -ASSA_inv(tau*A.T@(A@ujs[:,:,0]))+ujs[:,:,0]
            tujs = tujs+ujs
        tujs = tujs*tau
        toc = time.process_time()
        time_bin = toc-tic
        return tujs, tau, time_bin
        
    if prob_type == 'trust_region':
        lbd = lbd_min
    elif prob_type == 'multi_lbd':
        lbd = lbd_list[0]

    cstop = 0
    info = {}
    info['status'] = -1

    if solver == 'ihs' or solver == 'ihs_bin':
        info['sub_iters'] = np.zeros(iter_max)
    # update vk
    if solver == 'svd':
        tic = time.process_time()
        Sigma_bar = Sigma/(Sigma**2+lbd)
        print((V.T*Sigma_bar).shape)
        vk = (V.T*Sigma_bar)@(UTb)
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'gd':
        vk = np.zeros(d)
        for i in range(k):
            vk = vk-tau*(A.T@(A@vk-b)+lbd*vk)
    if solver == 'ihs':
        tic = time.process_time()
        if ihs_opt['warm_start']:
            # warm start
            vk = np.linalg.solve(A.T@A+lbd*np.eye(d),A.T@b)
            toc = time.process_time()
            time_list[0] = toc-tic
        else:
            vk = np.zeros(d)
            if sub_solver == 'smw':
                SAAS_inv = np.linalg.inv(np.eye(m)+SA@SA.T/lbd)
                ASSA_inv = lambda x: (x/lbd-SA.T@(SAAS_inv@(SA@x))/(lbd**2))
            for j in range(k):
                grad = A.T@(A@vk-b)+lbd*vk
                grad_nrm = np.linalg.norm(grad)
                if debug and j%5==0:
                    print('iter: {:2d} siter: {} grad: {:.2e}'.format(1,j,grad_nrm))
                if grad_nrm<tol*b_nrm:
                    break
                if grad_nrm>1e12*b_nrm:
                    print('IHS diverge!')
                    info['status'] = -1
                    break
                if sub_solver == 'svd':
                    SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                    dk = (VS.T*SigmaS_bar)@(VS@grad)+grad/lbd
                if sub_solver == 'smw':
                    dk = ASSA_inv(grad)

                # backtracking line search
                if j==0 and backtrack:
                    vk_trial = vk-tau*dk
                    f_old = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
                    f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
                    graddk = np.dot(grad,dk)

                    while f_new-f_old>-bls_alpha*graddk:
                        tau = tau*bls_beta
                        vk_trial = vk-tau*dk
                        f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
                    print('backtraking line search terminates with tau={}'.format(tau))
                    vk = vk_trial
                else:
                    vk = vk-tau*dk
            info['sub_iters'][0] = j+1
            toc = time.process_time()
            time_list[0] = toc-tic
    if solver == 'lin_sys':
        tic = time.process_time()
        if type(ATA) == scipy.sparse.linalg.interface._ProductLinearOperator:
            raise Exception('Error: native solver does not support for linear operator')
        if scipy.sparse.issparse(ATA):
            vk = scipy.sparse.linalg.spsolve(ATA+lbd*np.eye(d),ATb)
        else:
            vk = np.linalg.solve(ATA+lbd*np.eye(d),ATb)
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'warm_cg':
        tic = time.process_time()
        vk = np.zeros([d,num_classes])
        for j in range(num_classes):
            if type(ATA)==scipy.sparse.linalg.interface._ProductLinearOperator:
                vk_sub, _ = scipy.sparse.linalg.cg(ATA+lbd*scipy.sparse.linalg.aslinearoperator(np.eye(d)),ATb[:,j])
            else:
                vk_sub, _ = scipy.sparse.linalg.cg(ATA+lbd*np.eye(d),ATb[:,j])
            vk[:,j] = vk_sub 
        toc = time.process_time()
        time_list[0] = toc-tic
        print(vk.shape)
    if solver == 'gd_comp':
        lbd_vec = np.power(lbd*tau,np.arange(k))
        vk = ATbs@lbd_vec
    if solver == 'ihs_bin':
        lbd_ihs_bin = lbd_max*(lbd_min/lbd_max)**np.linspace(0,1,bin_num+1)
        print(lbd_ihs_bin)
        j = 0
        tujs, tau, time_bin = ihs_bin_build(lbd_ihs_bin[j],lbd_ihs_bin[j+1])
        tic = time.process_time()
        lbd_vec = np.power(lbd*tau,np.arange(k))
        vk = tujs.reshape([d*num_classes,k])@lbd_vec
        vk = vk.reshape([d,num_classes])
        toc = time.process_time()
        time_list[0] = toc-tic+time_bin
        info['sub_iters'][0] = k
    vk_norm = np.linalg.norm(vk)   

    # print(vk.shape)
    # evaluate train loss and test loss
    train_loss[0] = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
    test_loss[0] = 0.5*np.linalg.norm(Atest@vk-btest)**2

    if debug:
        print('iter: {:2d} lbd: {:.2e} vk_norm: {:.2e}'.format(1,lbd,vk_norm))
    if prob_type=='trust_region':
        if vk_norm<alpha:
            cstop = 1
    i = 0
    if prob_type=='trust_region':
        lbd = lbd_max
    elif prob_type=='multi_lbd':
        lbd = lbd_list[i+1]

    while i<=iter_max and cstop==0:
        i = i+1
        # update vk
        if solver == 'svd':
            tic = time.process_time()
            Sigma_bar = Sigma/(Sigma**2+lbd)
            vk = (V.T*Sigma_bar)@(UTb)
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'gd':
            vk = np.zeros(d)
            for j in range(k):
                vk = vk-tau*(A.T@(A@vk-b)+lbd*vk)
        if solver == 'gd_comp':
            lbd_vec = np.power(lbd*tau,np.arange(k))
            vk = ATbs@lbd_vec
        if solver == 'ihs':
            tic = time.process_time()
#             vk = np.zeros([d,1])
            if sub_solver == 'smw':
                SAAS_inv = np.linalg.inv(np.eye(m)+SA@SA.T/lbd)
                ASSA_inv = lambda x: (x/lbd-SA.T@(SAAS_inv@(SA@x))/(lbd**2))
            for j in range(k):
                grad = A.T@(A@vk-b)+lbd*vk
                grad_nrm = np.linalg.norm(grad)
                if debug and j%5==0:
                    print('iter: {:2d} siter: {} grad: {:.2e}'.format(i+1,j,grad_nrm))
                if grad_nrm<tol*b_nrm:
                    break
                if grad_nrm>1e12:
                    print('IHS diverge!')
                    info['status'] = -1
                    break
                SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                if sub_solver == 'svd':
                    SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                    vk = vk-tau*((VS.T*SigmaS_bar)@(VS@grad)+grad/lbd)
                if sub_solver == 'smw':
                    vk = vk-tau*ASSA_inv(grad)
            toc = time.process_time()
            info['sub_iters'][i] = j+1
            time_list[i] = toc-tic
        if solver == 'lin_sys':
            tic = time.process_time()
            vk = np.linalg.solve(ATA+lbd*np.eye(d),ATb)
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'warm_cg':
            tic = time.process_time()
            # vk = np.zeros([d,num_classes])
            for j in range(num_classes):
                if type(ATA)==scipy.sparse.linalg.interface._ProductLinearOperator:
                    vk_sub, _ = scipy.sparse.linalg.cg(ATA+lbd*scipy.sparse.linalg.aslinearoperator(np.eye(d)),ATb[:,j], x0 = vk[:,j])
                else:
                    vk_sub, _ = scipy.sparse.linalg.cg(ATA+lbd*np.eye(d),ATb[:,j], x0 = vk[:,j])
                vk[:,j] = vk_sub 
            # print(vk.shape)
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'ihs_bin':
            if lbd<lbd_ihs_bin[j+1]:
                # update basis
                j = j+1
                # if j==bin_num//2+1:
                #     k = k+1
                print('shift to [{:.2e}, {:.2e}]'.format(lbd_ihs_bin[j+1],lbd_ihs_bin[j]))
                tujs, tau, time_bin = ihs_bin_build(lbd_ihs_bin[j],lbd_ihs_bin[j+1])
                tic = time.process_time()
                lbd_vec = np.power(lbd*tau,np.arange(k))
                vk = tujs.reshape([d*num_classes,k])@lbd_vec
                vk = vk.reshape([d,num_classes])
                toc = time.process_time()
                time_list[i] = toc-tic+time_bin
                info['sub_iters'][i] = k
            else:
                tic = time.process_time()
                lbd_vec = np.power(lbd*tau,np.arange(k))
                vk = tujs.reshape([d*num_classes,k])@lbd_vec
                vk = vk.reshape([d,num_classes])
                toc = time.process_time()
                time_list[i] = toc-tic
                info['sub_iters'][i] = k
        vk_norm = np.linalg.norm(vk)
        if debug:
            print('iter: {:2d} lbd: {:.2e} vk_norm: {:.2e}'.format(i+1,lbd,vk_norm))
        if prob_type=='trust_region':
            if abs(vk_norm-alpha)<precision:
                info['status'] = 1
                cstop = 1
            if vk_norm>alpha:
                if i==1:
                    info['status'] = 0
                    cstop = 1
                lbd_min = lbd
            else:
                lbd_max = lbd
            lbd = 0.5*(lbd_max+lbd_min)
        elif prob_type=='multi_lbd':
            if i>=len(lbd_list)-1:
                cstop = 1
            else:
                lbd = lbd_list[i+1]
        # evaluate train loss and test loss
        train_loss[i] = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
        test_loss[i] = 0.5*np.linalg.norm(Atest@vk-btest)**2

    time_list = time_list[:i+1]
    time_list = np.cumsum(time_list)
    info['train_loss'] = train_loss[:i+1]
    info['test_loss'] = test_loss[:i+1]
    if solver == 'ihs' or solver == 'ihs_bin':
        info['sub_iters'] = info['sub_iters'][:i+1]
    if solver == 'lin_sys' or solver == 'warm_cg':
        time_list = time_list+time_ATA
    if solver == 'svd':
        time_list = time_list+time_SVD
    if solver == 'ihs' and sub_solver == 'svd':
        time_list = time_list+time_SVD_ihs+time_sketch
    if solver == 'ihs_bin':
        time_list = time_list+time_sketch
    if solver == 'ihs_bin' and sub_solver == 'svd':
        time_list = time_list+time_SVD_ihs+time_sketch

        
    return vk, time_list, info

def ridge_regression_over(A,b,alpha,lbd_min,lbd_max, Atest, btest, solver = 'svd',prob_type='trust_region',
                 lbd_list=[],gd_opt={}, ihs_opt={}, precision=1e-3,iter_max = 100,debug=False):
    if prob_type == 'multi_lbd' and len(lbd_list)==0:
        raise Exception('For prob_type=multi_lbd, lbd_list shall not be empty')
    if prob_type == 'multi_lbd' and len(lbd_list)>0:
        iter_max = max(iter_max,len(lbd_list))
    n, d = A.shape
    if n>=d:
        raise Exception('n shall be smaller than d.') 
    time_list = np.zeros(iter_max)
    b_nrm = np.linalg.norm(b)

    train_loss = np.zeros(iter_max)
    test_loss = np.zeros(iter_max)

    if len(b.shape)==1:
        b = b.reshape([-1,1])
    num_classes = b.shape[1]
    
    if solver == 'lin_sys' or solver == 'eig' or solver == 'warm_cg':
        tic = time.process_time()
        AAT = A@A.T
        toc = time.process_time()
        time_AAT = toc-tic
    if solver == 'eig':
        tic = time.process_time()
        if scipy.sparse.issparse(AAT):
            Sigma, U = np.linalg.eig(AAT.todense())
        else:
            Sigma, U = np.linalg.eig(AAT)
        toc = time.process_time()
        time_eig = toc-tic
    if solver == 'svd':
        if type(A)==scipy.sparse.linalg.interface._CustomLinearOperator:
            raise Exception('Error: svd solver does not support for linear operator')
        tic = time.process_time()
        if scipy.sparse.issparse(A):
            U, Sigma, V = np.linalg.svd(A.todense(),full_matrices=False)
        else:
            U, Sigma, V = np.linalg.svd(A,full_matrices=False)
        try:
            V = V.A
        except:
            pass
        UTb = U.T@b
        toc = time.process_time()
        time_SVD = toc-tic
    if solver == 'ihs' or solver == 'ihs_bin':
        if ihs_opt.get('sub_iter_max',-1)==-1:
            ihs_opt['sub_iter_max'] = 1e3
        if ihs_opt.get('warm_start',-1)==-1:
            ihs_opt['warm_start'] = False
        if ihs_opt.get('sketch_dim',-1) == -1:
            raise Exception('ihs_opt shall have key sketch_dim.')
        if ihs_opt.get('tol',-1) == -1:
            ihs_opt['tol'] = 1e-3
        if ihs_opt.get('sub_solver',-1) == -1:
            ihs_opt['sub_solver'] = 'svd'
        if ihs_opt.get('sketch_type',-1) == -1:
            ihs_opt['sketch_type'] = 'Gaussian'
        if ihs_opt.get('sparsity',-1) == -1:
            ihs_opt['sparsity'] = 1
        if ihs_opt.get('backtrack',-1) == -1:
            ihs_opt['backtrack'] = False
        if ihs_opt.get('bls_alpha',-1) == -1:
            ihs_opt['bls_alpha'] = 1e-3
        if ihs_opt.get('bls_beta',-1) == -1:
            ihs_opt['bls_beta'] = 0.5
        if ihs_opt.get('bin_num',-1) == -1:
            ihs_opt['bin_num'] = 1
        # if ihs_opt.get('lbd0',-1) == -1:
        #     ihs_opt['lbd0'] = 1

        m = ihs_opt['sketch_dim']
        tau = 1
        tol = ihs_opt['tol']
        sub_solver = ihs_opt['sub_solver']
        sketch_type = ihs_opt['sketch_type']
        sparsity = ihs_opt['sparsity']
        backtrack = ihs_opt['backtrack']
        bls_alpha = ihs_opt['bls_alpha']
        bls_beta = ihs_opt['bls_beta']
        # lbd0 = ihs_opt['lbd0']
        bin_num = ihs_opt['bin_num']

        k = ihs_opt['sub_iter_max']
        k = int(k)
        try:
            SAT = ihs_opt['SAT']
            time_sketch = 0
        except:
            tic = time.process_time()
            if sketch_type == 'Gaussian':
                S = np.random.randn(m,d)/np.sqrt(m)
            elif sketch_type == 'SJLT':
                S = get_SJLT_matrix(m,d,sparsity)
            SA = A@S.T
            SAT = SA.T
            toc = time.process_time()
            time_sketch = toc-tic
        
    if solver == 'ihs' or solver == 'ihs_bin':
        if sub_solver=='svd':
            tic = time.process_time()
            if scipy.sparse.issparse(SAT):
                US, SigmaS, VS = np.linalg.svd(SAT.todense(),full_matrices=False)
            else:
                US, SigmaS, VS = np.linalg.svd(SAT,full_matrices=False)
            try:
                VS = VS.A
            except:
                pass
            toc = time.process_time()
            time_SVD_ihs = toc-tic
        
    if solver == 'gd_comp':
        ATbs = np.zeros([d,k])
        ATbs[:,0] = (A.T@b).squeeze()
        for i in range(1,k):
            tmp = ATbs[:,i-1]
            ATbs[:,i] = tmp-tau*A.T@(A@tmp)
        bi_mat = np.zeros([k,k])
        for j in range(k):
            for l in range(k-j):
                bi_mat[l,j] = (-1)**j*comb(l+j,j)*tau
        ATbs = ATbs@bi_mat

    def ihs_bin_build(lbd_max_sub,lbd_min_sub):
        ratio = np.sqrt(lbd_min_sub/lbd_max_sub)
        tau = 2/(ratio+1/ratio)
        lbd0 = np.sqrt(lbd_min_sub*lbd_max_sub)
        tic = time.process_time()
        ujs = np.zeros([n,num_classes,k])
        tujs = np.zeros([n,num_classes,k])
        grad = b
        if sub_solver == 'smw':
            SAAS_inv = np.linalg.inv(np.eye(m)+SAT@SAT.T/lbd0)
            ASSA_inv = lambda x: (x/lbd0-SAT.T@(SAAS_inv@(SAT@x))/(lbd0**2))      
        elif sub_solver == 'svd':
            SigmaS_bar = 1/(SigmaS**2+lbd0)-1/lbd0
            ASSA_inv = lambda x:(VS.T*SigmaS_bar)@(VS@x)+x/lbd0
        ujs[:,:,0] = ASSA_inv(grad)
        tujs = tujs+ujs
        for i in range(1,k):
            j = i
            ujs[:,:,j] = -ASSA_inv(ujs[:,:,j-1])
            for j in range(i-1,0,-1):
                ujs[:,:,j] = -ASSA_inv(ujs[:,:,j-1]+tau*A@(A.T@ujs[:,:,j]))+ujs[:,:,j]
            ujs[:,:,0] = -ASSA_inv(tau*A@(A.T@ujs[:,:,0]))+ujs[:,:,0]
            tujs = tujs+ujs
        tujs = tujs*tau
        tujs = A.T@tujs.reshape([n,num_classes*k])
        tujs = tujs.reshape([d,num_classes,k])
        toc = time.process_time()
        time_bin = toc-tic
        return tujs, tau, time_bin
        
    if prob_type == 'trust_region':
        lbd = lbd_min
    elif prob_type == 'multi_lbd':
        lbd = lbd_list[0]

    cstop = 0
    info = {}
    info['status'] = -1

    if solver == 'ihs' or solver == 'ihs_bin':
        info['sub_iters'] = np.zeros(iter_max)
    # update vk
    if solver == 'svd':
        tic = time.process_time()
        Sigma_bar = Sigma/(Sigma**2+lbd)
        vk = (V.T*Sigma_bar)@(UTb)
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'eig':
        tic = time.process_time()
        Sigma_bar = 1/(Sigma+lbd)
        vk = A.T@((U*Sigma_bar)@(U.T@b))
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'ihs':
        tic = time.process_time()
        if ihs_opt['warm_start']:
            # warm start
            zk = np.linalg.solve(A@A.T+lbd*np.eye(n),b)
            vk = A.T@zk
            toc = time.process_time()
            time_list[0] = toc-tic
        else:
            zk = np.zeros(n)
            if sub_solver == 'smw':
                SAAS_inv = np.linalg.inv(np.eye(m)+SAT@SAT.T/lbd)
                ASSA_inv = lambda x: (x/lbd-SAT.T@(SAAS_inv@(SAT@x))/(lbd**2))
            for j in range(k):
                grad = A@(A.T@zk)-b+lbd*zk
                grad_nrm = np.linalg.norm(grad)
                if debug and j%5==0:
                    print('iter: {:2d} siter: {} grad: {:.2e}'.format(1,j,grad_nrm))
                if grad_nrm<tol*b_nrm:
                    break
                if grad_nrm>1e12*b_nrm:
                    print('IHS diverge!')
                    info['status'] = -1
                    break
                if sub_solver == 'svd':
                    SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                    dk = (VS.T*SigmaS_bar)@(VS@grad)+grad/lbd
                if sub_solver == 'smw':
                    dk = ASSA_inv(grad)

                # # backtracking line search
                # if j==0 and backtrack:
                #     vk_trial = vk-tau*dk
                #     f_old = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
                #     f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
                #     graddk = np.dot(grad,dk)

                #     while f_new-f_old>-bls_alpha*graddk:
                #         tau = tau*bls_beta
                #         vk_trial = vk-tau*dk
                #         f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
                #     print('backtraking line search terminates with tau={}'.format(tau))
                #     vk = vk_trial
                # else:
                zk = zk-tau*dk
                vk = A.T@zk
            info['sub_iters'][0] = j+1
            toc = time.process_time()
            time_list[0] = toc-tic
    if solver == 'lin_sys':
        if type(A)==scipy.sparse.linalg.interface._CustomLinearOperator:
            raise Exception('Error: native solver does not support for linear operator')
        tic = time.process_time()
        vk = A.T@np.linalg.solve(AAT+lbd*np.eye(n),b)
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'warm_cg':
        tic = time.process_time()
        zk = np.zeros([n,num_classes])
        for j in range(num_classes):
            if type(AAT)==scipy.sparse.linalg.interface._ProductLinearOperator:
                zk_sub, _ = scipy.sparse.linalg.cg(AAT+lbd*scipy.sparse.linalg.aslinearoperator(np.eye(n)),b[:,j])
            else:
                zk_sub, _ = scipy.sparse.linalg.cg(AAT+lbd*np.eye(n),b[:,j])
            zk[:,j] =zk_sub 
        vk = A.T@zk
        toc = time.process_time()
        time_list[0] = toc-tic
    if solver == 'gd_comp':
        lbd_vec = np.power(lbd*tau,np.arange(k))
        vk = ATbs@lbd_vec
    if solver == 'ihs_bin':
        lbd_ihs_bin = lbd_max*(lbd_min/lbd_max)**np.linspace(0,1,bin_num+1)
        print(lbd_ihs_bin)
        j = 0
        tujs, tau, time_bin = ihs_bin_build(lbd_ihs_bin[j],lbd_ihs_bin[j+1])
        tic = time.process_time()
        lbd_vec = np.power(lbd*tau,np.arange(k))
        vk = tujs.reshape([d*num_classes,k])@lbd_vec
        vk = vk.reshape([d,num_classes])
        toc = time.process_time()
        time_list[0] = toc-tic+time_bin
        info['sub_iters'][0] = k
    vk_norm = np.linalg.norm(vk)

    # evaluate train loss and test loss
    train_loss[0] = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
    test_loss[0] = 0.5*np.linalg.norm(Atest@vk-btest)**2

    if debug:
        print('iter: {:2d} lbd: {:.2e} vk_norm: {:.2e}'.format(1,lbd,vk_norm))
    if prob_type=='trust_region':
        if vk_norm<alpha:
            cstop = 1
    i = 0
    if prob_type=='trust_region':
        lbd = lbd_max
    elif prob_type=='multi_lbd':
        lbd = lbd_list[i+1]
    # print(iter_max)
    while i<=iter_max and cstop==0:
        i = i+1
        # update vk
        if solver == 'svd':
            tic = time.process_time()
            Sigma_bar = Sigma/(Sigma**2+lbd)
            vk = (V.T*Sigma_bar)@(UTb)
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'eig':
            tic = time.process_time()
            Sigma_bar = 1/(Sigma+lbd)
            vk = A.T@((U*Sigma_bar)@(U.T@b))
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'ihs':
            tic = time.process_time()
#             vk = np.zeros([d,1])
            if sub_solver == 'smw':
                SAAS_inv = np.linalg.inv(np.eye(m)+SA@SA.T/lbd)
                ASSA_inv = lambda x: (x/lbd-SA.T@(SAAS_inv@(SA@x))/(lbd**2))
            for j in range(k):
                grad = A@(A.T@zk)-b+lbd*zk
                grad_nrm = np.linalg.norm(grad)
                if debug and j%5==0:
                    print('iter: {:2d} siter: {} grad: {:.2e}'.format(i+1,j,grad_nrm))
                if grad_nrm<tol*b_nrm:
                    break
                if grad_nrm>1e12:
                    print('IHS diverge!')
                    info['status'] = -1
                    break
                SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                if sub_solver == 'svd':
                    SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                    zk = zk-tau*((VS.T*SigmaS_bar)@(VS@grad)+grad/lbd)
                if sub_solver == 'smw':
                    zk = zk-tau*ASSA_inv(grad)
                vk = A.T@zk
            toc = time.process_time()
            info['sub_iters'][i] = j+1
            time_list[i] = toc-tic
        if solver == 'lin_sys':
            tic = time.process_time()
            vk = A.T@np.linalg.solve(AAT+lbd*np.eye(n),b)
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'warm_cg':
            tic = time.process_time()
            zk = np.zeros([n,num_classes])
            for j in range(num_classes):
                if type(AAT)==scipy.sparse.linalg.interface._ProductLinearOperator:
                    zk_sub, _ = scipy.sparse.linalg.cg(AAT+lbd*scipy.sparse.linalg.aslinearoperator(np.eye(n)),b[:,j],x0 = zk[:,j])
                else:
                    zk_sub, _ = scipy.sparse.linalg.cg(AAT+lbd*np.eye(n),b[:,j],x0 = zk[:,j])
                zk[:,j] =zk_sub 
            vk = A.T@zk
            toc = time.process_time()
            time_list[i] = toc-tic
        if solver == 'ihs_bin':
            if lbd<lbd_ihs_bin[j+1]:
                # update basis
                j = j+1
                print('shift to [{:.2e}, {:.2e}]'.format(lbd_ihs_bin[j+1],lbd_ihs_bin[j]))
                tujs, tau, time_bin = ihs_bin_build(lbd_ihs_bin[j],lbd_ihs_bin[j+1])
                tic = time.process_time()
                lbd_vec = np.power(lbd*tau,np.arange(k))
                vk = tujs.reshape([d*num_classes,k])@lbd_vec
                vk = vk.reshape([d,num_classes])
                toc = time.process_time()
                time_list[i] = toc-tic+time_bin
                info['sub_iters'][i] = k
            else:
                tic = time.process_time()
                lbd_vec = np.power(lbd*tau,np.arange(k))
                vk = tujs.reshape([d*num_classes,k])@lbd_vec
                vk = vk.reshape([d,num_classes])
                toc = time.process_time()
                time_list[i] = toc-tic
                info['sub_iters'][i] = k
        vk_norm = np.linalg.norm(vk)
        if debug:
            print('iter: {:2d} lbd: {:.2e} vk_norm: {:.2e}'.format(i+1,lbd,vk_norm))
        if prob_type=='trust_region':
            if abs(vk_norm-alpha)<precision:
                info['status'] = 1
                cstop = 1
            if vk_norm>alpha:
                if i==1:
                    info['status'] = 0
                    cstop = 1
                lbd_min = lbd
            else:
                lbd_max = lbd
            lbd = 0.5*(lbd_max+lbd_min)
        elif prob_type=='multi_lbd':
            if i>=len(lbd_list)-1:
                cstop = 1
            else:
                lbd = lbd_list[i+1]
        # evaluate train loss and test loss
        train_loss[i] = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
        test_loss[i] = 0.5*np.linalg.norm(Atest@vk-btest)**2

    time_list = time_list[:i+1]
    time_list = np.cumsum(time_list)
    info['train_loss'] = train_loss[:i+1]
    info['test_loss'] = test_loss[:i+1]
    if solver == 'ihs' or solver == 'ihs_bin':
        info['sub_iters'] = info['sub_iters'][:i+1]
    if solver == 'lin_sys' or solver=='warm_cg':
        time_list = time_list+time_AAT
    if solver == 'svd':
        time_list = time_list+time_SVD
    if solver == 'eig':
        time_list = time_list+time_AAT+time_eig
    if solver == 'ihs' and sub_solver == 'svd':
        time_list = time_list+time_SVD_ihs+time_sketch
    if solver == 'ihs_bin':
        time_list = time_list+time_sketch
    if solver == 'ihs_bin' and sub_solver == 'svd':
        time_list = time_list+time_sketch+time_SVD_ihs

        
    return vk, time_list, info