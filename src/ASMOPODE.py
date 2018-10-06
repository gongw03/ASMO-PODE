# Adaptive Surrogate Modeling-based Optimization for Parameter 
#        Optimization and Distribution Estimation (ASMO-PODE)
from __future__ import division, print_function, absolute_import
import sampling
#import gp
import gwgp
import Metropolis
import AM
import DRAM
import numpy as np

def sampler(floglike, D, xlb, xub, \
            Xinit = None, Yinit = None, flogprior = None, \
            niter = 10, nhist = 5, resolution = 0.0001, \
            T = 1, B = 10000, N = 10000, M = 5, \
            parallel = False, processes = 4, sampler = None):
    ''' An adaptive surrogate modeling-based sampling strategy for parameter 
        optimization and distribution estimation (ASMO-PODE)
        use Metropolis/AM/DRAM sampler, Markov Chain Monte Carlo
        Parameters for ASMO-PODE
            floglike: -2log likelihood function, floglike.evaluate(X)
            D: dimension of input X
            xlb: lower bound of input
            xub: upper bound of input
            Xinit: initial value of X, Ninit x D matrix
            Yinit: initial value of Y, Ninit dim vector
            flogprior: -2log prior distribution function, 
                should be very simple that do not need surrogate
                use uniform distribution as default
            niter: total number of iteration
            nhist: number of histograms in each iteration
            resolution: use uniform sampling if the nearest neighbour distance 
                        is smaller than resolution 
                        (parameter space normalized to [0,1])
        Parameters for MCMC:
            T: temperature, default is 1
            B: length of burn-in period
            N: Markov Chain length (after burn-in)
            M: number of Markov Chain
            parallel: evaluate MChain parallelly or not
            processes: number of parallel processes
            sampler: name of sampler, one of Metropolis/AM/DRAM
    '''
    nbin = int(np.floor(N/(nhist-1)))
    if (Xinit is None and Yinit is None):
        Ninit = D * 10
        Xinit = sampling.glp(Ninit, D)
        for i in range(Ninit):
            Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
        Yinit = np.zeros(Ninit)
        for i in range(Ninit):
            Yinit[i] = floglike.evaluate(Xinit[i,:])
    else:
        Ninit = Xinit.shape[0]
        if len(Yinit.shape) == 2:
            Yinit = Yinit[:,0]
    x = Xinit.copy()
    y = Yinit.copy()
    ntoc = 0

    if sampler is None:
        sampler = 'Metropolis'
    
    resamples = []
    
    for i in range(niter):
        print('Surrogate Opt loop: %d' % i)
        
        # construct surrogate model
        #sm = gp.GPR_Matern(x, y, D, 1, x.shape[0], xlb, xub)
        sm = gwgp.MOGPR('CovMatern5', x, y.reshape((-1,1)), D, 1, xlb, xub, \
                mean = np.zeros(1), noise = 1e-3)
        # for surrogate-based MCMC, use larger value for noise, i.e. 1e-3, to smooth the response surface
        
        # run MCMC on surrogate model
        if sampler == 'AM':
            [Chain, LogPost, ACC, GRB] = \
                AM.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, \
                    parallel, processes)
        elif sampler == 'DRAM':
            [Chain, LogPost, ACC, GRB] = \
                DRAM.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, \
                    parallel, processes)
        elif sampler == 'Metropolis':
            [Chain, LogPost, ACC, GRB] = \
                Metropolis.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, None, \
                    parallel, processes)
        else:
            [Chain, LogPost, ACC, GRB] = \
                Metropolis.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, None, \
                    parallel, processes)
        
        # sort -2logpost with ascending order
        lidx = np.argsort(LogPost)
        Chain = Chain[lidx,:]
        LogPost = LogPost[lidx]
        
        # store result of MCMC on surrogate
        resamples.append({'Chain': Chain.copy(), \
                          'LogPost': LogPost.copy(),'ACC': ACC, 'GRB': GRB})
        
        # normalize the data
        xu = (x - xlb) / (xub - xlb)
        xp = (Chain - xlb) / (xub - xlb)
        
        # resampling
        xrf = np.zeros([nhist,D])
        for ihist in range(nhist-1):
            xpt = xp[nbin*ihist:nbin*(ihist+1),:].copy()
            xptt, pidx = np.unique(xpt.view(xpt.dtype.descr * xpt.shape[1]),\
                                   return_index=True)
            xpt = xpt[pidx,:]
            [xtmp, mdist] = maxmindist(xu,xpt)
            if mdist < resolution:
                [xtmp, mdist] = maxmindist(xu,np.random.random([10000,D]))
                ntoc += 1
            xrf[ihist,:] = xtmp
            xu = np.vstack((xu,xtmp))
        xrf[nhist-1,:] = xp[0,:]
        xu = np.vstack((xu,xrf[nhist-1,:]))
        resamples[i]['ntoc'] = ntoc
        
        # run dynamic model
        xrf = xrf * (xub - xlb) + xlb
        yrf = np.zeros(nhist)
        for i in range(nhist):
            yrf[i] = floglike.evaluate(xrf[i,:])
        x = np.concatenate((x,xrf.copy()), axis = 0)
        y = np.concatenate((y,yrf.copy()), axis = 0)
    
    bestidx = np.argmin(y)
    bestx = x[bestidx,:]
    besty = y[bestidx]
    
    return Chain, LogPost, ACC, GRB, bestx, besty, x, y, resamples


def onestep(D, xlb, xub, Xinit, Yinit, flogprior = None, \
            nhist = 5, resolution = 0.01, \
            T = 1, B = 10000, N = 10000, M = 5, \
            parallel = False, processes = 4, sampler = None):
    """
    An adaptive surrogate modeling-based sampling strategy for parameter 
    optimization and distribution estimation (ASMO-PODE)
    use Metropolis/AM/DRAM sampler, Markov Chain Monte Carlo
    One-step mode for offline optimization
    Do NOT call the model evaluation function
    Parameters for ASMO-PODE
        D: dimension of input X
        xlb: lower bound of input
        xub: upper bound of input
        Xinit: initial value of X, Ninit x D matrix
        Yinit: initial value of Y, Ninit dim vector
        flogprior: -2log prior distribution function, 
            should be very simple that do not need surrogate
            use uniform distribution as default
        nhist: number of histograms in each iteration
        resolution: use uniform sampling if the nearest neighbour distance 
                    is smaller than resolution 
                    (parameter space normalized to [0,1])
    Parameters for MCMC:
        T: temperature, default is 1
        B: length of burn-in period
        N: Markov Chain length (after burn-in)
        M: number of Markov Chain
        parallel: evaluate MChain parallelly or not
        processes: number of parallel processes
        sampler: name of sampler, one of Metropolis/AM/DRAM
    """
    nbin = int(np.floor(N/(nhist-1)))
    x = Xinit.copy()
    y = Yinit.copy()
    ntoc = 0
            
    # construct surrogate model
    #sm = gp.GPR_Matern(x, y, D, 1, x.shape[0], xlb, xub)
    sm = gwgp.MOGPR('CovMatern5', x, y.reshape((-1,1)), D, 1, xlb, xub, \
            mean = np.zeros(1), noise = 1e-3)
    # for surrogate-based MCMC, use larger value for noise, i.e. 1e-3, to smooth the response surface
    
    # run MCMC on surrogate model
    if sampler == 'AM':
        [Chain, LogPost, ACC, GRB] = \
            DRAM.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, \
                parallel, processes)
    elif sampler == 'DRAM':
        [Chain, LogPost, ACC, GRB] = \
            AM.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, \
                parallel, processes)
    else:
        [Chain, LogPost, ACC, GRB] = \
            Metropolis.sampler(sm, D, xlb, xub, None, flogprior, T, B, N, M, None, \
                parallel, processes)
    
    # sort -2logpost with ascending order
    lidx = np.argsort(LogPost)
    Chain = Chain[lidx,:]
    LogPost = LogPost[lidx]
    
    # normalize the data
    xu = (x - xlb) / (xub - xlb)
    xp = (Chain - xlb) / (xub - xlb)
    
    # resampling
    xrf = np.zeros([nhist,D])
    for ihist in range(nhist-1):
        xpt = xp[nbin*ihist:nbin*(ihist+1),:]
        xptt, pidx = np.unique(xpt.view(xpt.dtype.descr * xpt.shape[1]),\
                               return_index=True)
        xpt = xpt[pidx,:]
        [xtmp, mdist] = maxmindist(xu,xpt)
        if mdist < resolution:
            [xtmp, mdist] = maxmindist(xu,np.random.random([10000,D]))
            ntoc += 1
        xrf[ihist,:] = xtmp
        xu = np.vstack((xu,xtmp))
    xrf[nhist-1,:] = xp[0,:]
    xu = np.vstack((xu,xrf[nhist-1,:]))
    
    # return resample points
    x_resample = xrf * (xub - xlb) + xlb
    
    return x_resample, Chain, LogPost, ACC, GRB


def maxmindist(A, B):
    """ 
    maximize the minimum distance from point set B to A
    A is the referene point set
    for each point in B, compute its distance to its nearest neighbor of A
    find the point in B that has largest min-dist
    P: the coordinate of point
    D: the maxmin distance
    """
    T1 = A.shape[0]
    T2 = B.shape[0]
    
    Dist = np.zeros([T1,T2])
    for i in range(T1):
        for j in range(T2):
            Dist[i,j] = np.sqrt(np.sum((A[i,:]-B[j,:])**2))

    mindist = np.min(Dist, axis = 0)
    idx = np.argmax(mindist)
    P = B[idx,:]
    D = mindist[idx]

    return P, D
