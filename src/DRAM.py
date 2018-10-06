from __future__ import division, print_function, absolute_import
import numpy as np
#from scipy.stats import multivariate_normal as mvn
from multiprocessing import Pool

def sampler(floglike, D, xlb, xub, Xinit = None, flogprior = None, \
            T = 1, B = 10000, N = 10000, M = 5, \
            parallel = False, processes = 4):
    ''' Delayed Rejection Adaptive Metropolis sampler, Markov Chain Monte Carlo
        Adaptively generate multiple Markov Chain (with delayed rejection) 
        to sample the posterior distribution
        floglike: -2log likelihood function, floglike.evaluate(X)
        D: dimension of input X
        xlb: lower bound of input
        xub: upper bound of input
        Xinit: initial value of X, D-dim vector (single start point)
        flogprior: -2log prior distribution function, use uniform distribution as default
        T: temperature, default is 1
        B: length of burn-in period
        N: Markov Chain length (after burn-in)
        M: number of Markov Chain
        parallel: evaluate MChain parallelly or not
        processes: number of parallel processes
    '''
    Chain = np.zeros([M,N,D]) # array of Markov Chain
    LogPost = np.zeros([M,N]) # array of log post
    ChainMerged = np.zeros([M*N,D]) # merged chain
    LogPostMerged = np.zeros(M*N) # merged log post
    ACC = np.zeros(M)  # total acceptance rate after burn-in
    beta = 1.0/T # inverse temperature
    
    sd      = (2.4**2)/D   # scalling factor of Adaptive Metropolis
    epsilon = 0.001        # to make sure the cov-matrix is positive definite
    nstep   = 100          # update covariance for every nstep iterations
    cmat    = np.eye(D)*sd # initial covariance matrix
    for i in range(D):
        cmat[i,i] *= (xub[i] - xlb[i])
    drscale = 3            # delayed rejection scale was set to 3

    if Xinit is None:
    # default init state of Markov Chain, uniform distribution in [xlb,xub]
        X = np.zeros([M,D])
        for i in range(M):
            X[i,:] = np.random.rand(D) * (xub - xlb) + xlb
    else:
        X = Xinit

    if not parallel:
        # burn-in, update cmat in every nstep
        nbatch = np.int(np.ceil(B/nstep))
        B = nbatch * nstep
        bChainMerged = np.zeros([0,D])
        for j in range(nbatch):
            for i in range(M):
                bChain, bAccept, bLogPost = \
                    MChain(floglike, flogprior, beta, nstep, D, xlb, xub, X[i,:], cmat, drscale)
                X[i,:] = bChain[-1,:]
                bChainMerged = np.append(bChainMerged, bChain, axis = 0)
            cmat = sd * np.cov(bChainMerged.transpose()) + sd * epsilon * np.eye(D)
        
        # sampling from Markov Chain with constant cmat
        for i in range(M):
            iChain, iAccept, iLogPost = \
                MChain(floglike, flogprior, beta, N, D, xlb, xub, X[i,:], cmat, drscale)
            Chain[i,:,:] = iChain
            ACC[i] = iAccept
            LogPost[i,:] = iLogPost
            ChainMerged[(i*N):((i+1)*N),:] = iChain
            LogPostMerged[(i*N):((i+1)*N)] = iLogPost
    else:
        # burn-in, update cmat in every nstep
        nbatch = np.int(np.ceil(B/nstep))
        B = nbatch * nstep
        bChainMerged = np.zeros([0,D])

        p = Pool(processes = processes)
        for j in range(nbatch):
            bpara = []
            for i in range(M):
                bpara.append({'floglike': floglike, 'flogprior': flogprior, \
                        'beta': beta, 'N': nstep, 'D': D, 'xlb': xlb, 'xub': xub, \
                        'X': X[i,:], 'cmat': cmat, 'drscale': drscale})
            bres = p.map(ParaMC, bpara)
            for i in range(M):
                X[i,:] = bres[i]['Chain'][-1,:]
                bChainMerged = np.append(bChainMerged, bres[i]['Chain'], axis = 0)
            cmat = sd * np.cov(bChainMerged.transpose()) + sd * epsilon * np.eye(D)

        # sampling from Markov Chain with constant cmat
        ipara = []
        for i in range(M):
            ipara.append({'floglike': floglike, 'flogprior': flogprior, \
                    'beta': beta,'N': N, 'D': D, 'xlb': xlb, 'xub': xub, \
                    'X': X[i,:], 'cmat': cmat, 'drscale': drscale})
        ires = p.map(ParaMC, ipara)
        for i in range(M):
            Chain[i,:,:] = ires[i]['Chain']
            ACC[i] = ires[i]['Accept']
            LogPost[i,:] = ires[i]['LogPost']
            ChainMerged[(i*N):((i+1)*N),:] = ires[i]['Chain']
            LogPostMerged[(i*N):((i+1)*N)] = ires[i]['LogPost']

    GRB = GRBfactor(Chain)
    
    # only save the merged chain and log post
    return ChainMerged, LogPostMerged, ACC, GRB

def MChain(floglike, flogprior, beta, N, D, xlb, xub, X, cmat, drscale):
    """ Single Markov Chain evaluation
    """
    # define posterior distribution function
    if flogprior is None:
    # default -2log prior distribution function, uniform distribution in [xlb,xub]
        flogpost = lambda X: floglike.evaluate(X)*beta - 2.0*np.sum(np.log(xub-xlb))
    else:
        flogpost = lambda X: floglike.evaluate(X)*beta + flogprior.evaluate(X)

    cmat2 = cmat/drscale
    cholcmat = np.linalg.cholesky(cmat)
    cholcmat2 = np.linalg.cholesky(cmat2)
    Accept = 0.0
    Chain = np.zeros([N,D])
    LogPost = np.zeros(N)
    pX = flogpost(X)
    for i in range(N):
        # step 1: generate proposed point
        #Xt = mvn.rvs(X, cmat)
        Xt = np.dot(np.random.randn(D),cholcmat) + X
        Xt = np.clip(Xt, xlb, xub)
        pXt = flogpost(Xt)
        # step 2: compute the acceptance ratio
        # r12 = min(1, np.exp(0.5*(pX - pXt)))
        if pX > pXt:
            r12 = 1.
        else:
            r12 = np.exp(0.5*(pX - pXt))
        # step 3: accept or decline
        u = np.random.rand()
        if u <= r12: # accept
            X = Xt
            pX = pXt
            Accept += 1
        else:
            # Delayed Rejection, stage = 3
            # not reject, but generate another candidate point
            #Xt2 = mvn.rvs(X, cmat2)
            Xt2 = np.dot(np.random.randn(D),cholcmat2) + X
            Xt2 = np.clip(Xt2, xlb, xub)
            pXt2 = flogpost(Xt2)
            #r32 = min(1, np.exp(0.5*(pXt2 - pXt)))
            if pXt2 > pXt:
                r32 = 1.
            else:
                r32 = np.exp(0.5*(pXt2 - pXt))
            if abs(pX - pXt2) > 700:
                r13 = 0.
            else:
                r13 = min(1., drscale**D * np.exp(0.5*(pX - pXt2)
                        + 0.5*(np.dot(np.dot(Xt-X, np.linalg.inv(cmat)), Xt-X)\
                        - np.dot(np.dot(Xt-Xt2, np.linalg.inv(cmat2)), Xt-Xt2))) \
                        * (1. - r32) / (1. - r12)) 
#                r13 = min(1, np.exp(0.5*(pX - pXt2)) \
#                      * mvn.pdf(Xt,Xt2,cmat2) \
#                      / mvn.pdf(Xt,X,cmat) \
#                      * (1 - r32) / (1 - r12))
            u2 = np.random.rand()
            if u2 <= r13: # accept
                X = Xt2
                pX = pXt2
                Accept += 1
        Chain[i,:] = X
        LogPost[i] = pX
        
    Accept /= N
    return Chain, Accept, LogPost

def ParaMC(xpara):
    ''' Parallel evaluation of Markov Chain
    '''
    res = {}
    res['Chain'], res['Accept'], res['LogPost'] = \
            MChain(xpara['floglike'], xpara['flogprior'], xpara['beta'], \
                xpara['N'], xpara['D'], xpara['xlb'], xpara['xub'], \
                xpara['X'], xpara['cmat'], xpara['drscale'])
    return res

def GRBfactor(Chain):
    """ Gelman-Rubin-Brooks multivariate potential scale
        reduction factor MCMC convergence diagnostic.
        S. Brooks and G. Roberts, Assessing Convergence of Markov Chain 
        Monte Carlo Algorithms, Statistics and Computing 8, 319-335, 1998.
        
        nchain: number of chains
        nmax: length of each chain
        ndim: dimension of the distribution
        Chain: array of Markov Chain, size(Chain) = [nchain, nmax, ndim]
    """
    nchain, nmax, ndim = Chain.shape
    W = np.zeros(ndim) # W is the mean value of within-chain variances
    B = np.zeros(ndim) # B is the variance between the mean values of each chain
    M = np.zeros([nchain, ndim]) # mean values of each chain
    V = np.zeros([ndim, ndim]) # within-chain variances
    for i in range(nchain):
        M[i,:] = np.mean(Chain[i,:,:], axis = 0)
        V += np.cov(Chain[i,:,:].T)
    V1 = V/nchain
    V2 = np.cov(M.T)
    L = np.max(np.linalg.eig(np.dot(np.linalg.inv(V1),V2))[0])
    GRB = np.sqrt((nmax-1)/nmax + (nchain+1)/nchain*L)
    return GRB
