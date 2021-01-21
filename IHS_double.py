import numpy as np
import time
import scipy.sparse
import scipy.sparse.linalg
from utils import get_SJLT_matrix

def IHS_double(A,b,lbd, ihs_opt, debug = False):
    if ihs_opt.get('sub_iter_max',-1)==-1:
        ihs_opt['sub_iter_max'] = 1e3
    if ihs_opt.get('warm_start',-1)==-1:
        ihs_opt['warm_start'] = False
    if ihs_opt.get('tol',-1) == -1:
        ihs_opt['tol'] = 1e-3
    if ihs_opt.get('sub_solver',-1) == -1:
        ihs_opt['sub_solver'] = 'svd'
    if ihs_opt.get('sketch_type',-1) == -1:
        ihs_opt['sketch_type'] = 'Gaussian'
    if ihs_opt.get('sparsity',-1) == -1:
        ihs_opt['sparsity'] = 1
    if ihs_opt.get('tau_factor',-1) == -1:
        ihs_opt['tau_factor'] = 1
    if ihs_opt.get('thres',-1) == -1:
        ihs_opt['thres'] = 0.5
    if ihs_opt.get('m',-1)==-1:
        ihs_opt['m'] = 1
    if ihs_opt.get('backtrack',-1) == -1:
        ihs_opt['backtrack'] = False
    if ihs_opt.get('bls_alpha',-1) == -1:
        ihs_opt['bls_alpha'] = 1e-3
    if ihs_opt.get('bls_beta',-1) == -1:
        ihs_opt['bls_beta'] = 0.5

    if len(b.shape)==1:
        b = b.reshape([-1,1])

    n, d = A.shape
    m = ihs_opt['m']
    tol = ihs_opt['tol']
    sub_solver = ihs_opt['sub_solver']
    sketch_type = ihs_opt['sketch_type']
    sparsity = ihs_opt['sparsity']
    thres = ihs_opt['thres']
    tau_factor = ihs_opt['tau_factor']

    backtrack = ihs_opt['backtrack']
    bls_alpha = ihs_opt['bls_alpha']
    bls_beta = ihs_opt['bls_beta']

    tau = tau_factor

    k = ihs_opt['sub_iter_max']
    k = int(k)
    tic = time.process_time()

    if sketch_type == 'Gaussian':
        S = np.random.randn(m,n)/np.sqrt(m)
    elif sketch_type == 'SJLT':
        S = get_SJLT_matrix(m,n,sparsity)
    SAT = A.T@S.T
    SA = SAT.T
    toc = time.process_time()
    time_sketch = toc-tic

    b_nrm = np.linalg.norm(b)

    cstop = 0
    info = {}
    info['status'] = -1

    if sub_solver=='svd':
        if scipy.sparse.issparse(SA):
            US, SigmaS, VS = np.linalg.svd(SA.todense(),full_matrices=False)
        else:
            US, SigmaS, VS = np.linalg.svd(SA,full_matrices=False)
        try:
            VS = VS.A
        except:
            pass

    vk = np.zeros([d,1])
    if sub_solver == 'smw':
        SAAS_inv = np.linalg.inv(np.eye(m)+SA@SA.T/lbd)
        ASSA_inv = lambda x: (x/lbd-SA.T@(SAAS_inv@(SA@x))/(lbd**2))
    for j in range(k):
        grad = A.T@(A@vk-b)+lbd*vk
        if sub_solver == 'svd':
            SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
            dk = (VS.T*SigmaS_bar)@(VS@grad)+grad/lbd
        if sub_solver == 'smw':
            dk = ASSA_inv(grad)
        if j>0:
            grad_nrm_p = grad_nrm
        grad_nrm = np.sqrt(dk.T@grad).item()
        if grad_nrm<tol*b_nrm:
            break
        if grad_nrm>1e12*b_nrm:
            print('IHS diverge!')
            info['status'] = -1
            break
        if j>0 and grad_nrm>thres*grad_nrm_p and m<d/3:
            m = m*2
            print('change sketching dimension to {}'.format(m))
            if sketch_type == 'Gaussian':
                S = np.random.randn(m,n)/np.sqrt(m)
            elif sketch_type == 'SJLT':
                S = get_SJLT_matrix(m,n,sparsity)
            SAT = A.T@S.T
            SA = SAT.T
            if sub_solver=='svd':
                if scipy.sparse.issparse(SA):
                    US, SigmaS, VS = np.linalg.svd(SA.todense(),full_matrices=False)
                else:
                    US, SigmaS, VS = np.linalg.svd(SA,full_matrices=False)
                try:
                    VS = VS.A
                except:
                    pass
            if sub_solver == 'svd':
                SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
                dk = (VS.T*SigmaS_bar)@(VS@grad)+grad/lbd
            if sub_solver == 'smw':
                dk = ASSA_inv(grad)
            j = j-1

        if debug and j%1==0:
            print('iter: {} grad: {:.2e}'.format(j,grad_nrm))
        if backtrack:
            vk_trial = vk-tau*dk
            f_old = 0.5*np.linalg.norm(A@vk-b)**2+0.5*lbd*np.linalg.norm(vk)**2
            f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
            graddk = grad.T@dk

            while f_new-f_old>-bls_alpha*graddk:
                tau = tau*bls_beta
                vk_trial = vk-tau*dk
                f_new = 0.5*np.linalg.norm(A@vk_trial-b)**2+0.5*lbd*np.linalg.norm(vk_trial)**2
            print('backtraking line search terminates with tau={}'.format(tau))
            vk = vk_trial
            tau = tau_factor
        else:
            vk = vk-tau*dk
    return vk, m, info

def IHS_double_over(A,b,lbd, ihs_opt, debug = False):
    if ihs_opt.get('sub_iter_max',-1)==-1:
        ihs_opt['sub_iter_max'] = 1e3
    if ihs_opt.get('warm_start',-1)==-1:
        ihs_opt['warm_start'] = False
    if ihs_opt.get('tol',-1) == -1:
        ihs_opt['tol'] = 1e-3
    if ihs_opt.get('sub_solver',-1) == -1:
        ihs_opt['sub_solver'] = 'svd'
    if ihs_opt.get('sketch_type',-1) == -1:
        ihs_opt['sketch_type'] = 'Gaussian'
    if ihs_opt.get('sparsity',-1) == -1:
        ihs_opt['sparsity'] = 1
    if ihs_opt.get('tau_factor',-1) == -1:
        ihs_opt['tau_factor'] = 1
    if ihs_opt.get('thres',-1) == -1:
        ihs_opt['thres'] = 0.5
    if ihs_opt.get('m',-1)==-1:
        ihs_opt['m'] = 1
    if ihs_opt.get('backtrack',-1) == -1:
        ihs_opt['backtrack'] = False
    if ihs_opt.get('bls_alpha',-1) == -1:
        ihs_opt['bls_alpha'] = 1e-3
    if ihs_opt.get('bls_beta',-1) == -1:
        ihs_opt['bls_beta'] = 0.5

    if len(b.shape)==1:
        b = b.reshape([-1,1])

    n, d = A.shape
    m = ihs_opt['m']
    tol = ihs_opt['tol']
    sub_solver = ihs_opt['sub_solver']
    sketch_type = ihs_opt['sketch_type']
    sparsity = ihs_opt['sparsity']
    thres = ihs_opt['thres']
    tau_factor = ihs_opt['tau_factor']

    backtrack = ihs_opt['backtrack']
    bls_alpha = ihs_opt['bls_alpha']
    bls_beta = ihs_opt['bls_beta']

    tau = tau_factor

    k = ihs_opt['sub_iter_max']
    k = int(k)

    if sketch_type == 'Gaussian':
        S = np.random.randn(m,d)/np.sqrt(m)
    elif sketch_type == 'SJLT':
        S = get_SJLT_matrix(m,d,sparsity)
    SA = A@S.T
    SAT = SA.T

    b_nrm = np.linalg.norm(b)

    cstop = 0
    info = {}
    info['status'] = -1

    if sub_solver=='svd':
        if scipy.sparse.issparse(SAT):
            US, SigmaS, VS = np.linalg.svd(SAT.todense(),full_matrices=False)
        else:
            US, SigmaS, VS = np.linalg.svd(SAT,full_matrices=False)
        try:
            VS = VS.A
        except:
            pass

    zk = np.zeros([n,1])
    if sub_solver == 'smw':
        SAAS_inv = np.linalg.inv(np.eye(m)+SAT@SAT.T/lbd)
        ASSA_inv = lambda x: (x/lbd-SAT.T@(SAAS_inv@(SAT@x))/(lbd**2))
    for j in range(k):
        grad = A@A.T@zk-b+lbd*zk
        if sub_solver == 'svd':
            SigmaS_bar = 1/(SigmaS**2+lbd)-1/lbd
            dk = (VS.T*SigmaS_bar)@(VS@grad)+grad/lbd
        if sub_solver == 'smw':
            dk = ASSA_inv(grad)
        if j>0:
            grad_nrm_p = grad_nrm
        grad_nrm = np.sqrt(dk.T@grad).item()
        if grad_nrm<tol*b_nrm:
            break
        if grad_nrm>1e12*b_nrm:
            print('IHS diverge!')
            info['status'] = -1
            break
        if j>0 and grad_nrm>thres*grad_nrm_p and m<n/3:
            m = m*2
            print('change sketching dimension to {}'.format(m))
            if sketch_type == 'Gaussian':
                S = np.random.randn(m,d)/np.sqrt(m)
            elif sketch_type == 'SJLT':
                S = get_SJLT_matrix(m,d,sparsity)
            SA = A@S.T
            SAT = SA.T
            if sub_solver=='svd':
                if scipy.sparse.issparse(SAT):
                    US, SigmaS, VS = np.linalg.svd(SAT.todense(),full_matrices=False)
                else:
                    US, SigmaS, VS = np.linalg.svd(SAT,full_matrices=False)
                try:
                    VS = VS.A
                except:
                    pass

        if debug:
            print('iter: {} grad: {:.2e}'.format(j,grad_nrm))
        if backtrack:
            zk_trial = zk-tau*dk
            f_old = 0.5*np.linalg.norm(A.T@zk)**2+0.5*lbd*np.linalg.norm(zk)**2-b.T@zk
            f_new = 0.5*np.linalg.norm(A.T@zk_trial)**2+0.5*lbd*np.linalg.norm(zk_trial)**2-b.T@zk_trial
            graddk = grad.T@dk

            while f_new-f_old>-bls_alpha*graddk:
                tau = tau*bls_beta
                zk_trial = zk-tau*dk
                f_new = 0.5*np.linalg.norm(A.T@zk_trial)**2+0.5*lbd*np.linalg.norm(zk_trial)**2-b.T@zk_trial
            print('backtraking line search terminates with tau={}'.format(tau))
            zk = zk_trial
            tau = tau_factor
        else:
            zk = zk-tau*dk
        vk = A.T@zk
    return vk, m, info
