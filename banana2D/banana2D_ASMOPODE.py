from __future__ import division, print_function, absolute_import
import os
import sys
sys.path.append('../src')
import ASMOPODE
import sampling
import numpy as np
import matplotlib.pyplot as plt
import util
#import cPickle
import pickle
import mpdf

# model name
modelname = 'banana2D'
model = __import__(modelname)

# result path
respath = '../../UQ-res/MCMC/%s' % modelname
if not os.path.exists(respath):
    os.makedirs(respath)

# load parameter name and range
pf = util.read_param_file('%s.txt' % modelname)
bd = np.array(pf['bounds'])
xlb = bd[:,0]
xub = bd[:,1]
D = pf['num_vars']

# parameters for ASMO-PODE
T = 1
B = 10000
N = 10000
M = 4
niter = 16
nhist = 5
resolution = 0.05
parallel = True
processes = 4
sampler = 'DRAM'

# initial sampling
Ninit = 20
Xinit = sampling.glp(Ninit, D, 5)
for i in range(Ninit):
    Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
Yinit = np.zeros(Ninit)
for i in range(Ninit):
    Yinit[i] = model.evaluate(Xinit[i,:])

# run ASMO-PODE
[Chain, LogPost, ACC, GRB, bestx, besty, x, y, resamples] = \
    ASMOPODE.sampler(model, D, xlb, xub, Xinit, Yinit, None, \
                     niter, nhist, resolution, T, B, N, M, \
                     parallel, processes, sampler)

# plot results
#print(ACC)
#print(GRB)
plt.plot(Chain[:,0],Chain[:,1],'b.')
plt.xlabel('x1')
plt.ylabel('x2')
# save figure
plt.savefig('%s/banana2D_ASMOPODE.png' % respath)

# save results to bin file
# with open('%s/banana2D_ASMOPODE.bin' % respath,'w') as f:
#     cPickle.dump({'D':D, \
#                   'niter': niter, 'nhist': nhist, 'resolution': resolution, \
#                   'T': 1, 'B': B, 'N': N, 'M': M, \
#                   'Chain': Chain, 'LogPost': LogPost, 'ACC': ACC, 'GRB': GRB, \
#                   'bestx': bestx, 'besty': besty, 'resamples': resamples, \
#                   'x': x, 'y': y}, f)

with open('%s/banana2D_ASMOPODE.bin' % respath,'wb') as f:
    pickle.dump({'D':D, \
                 'niter': niter, 'nhist': nhist, 'resolution': resolution, \
                 'T': 1, 'B': B, 'N': N, 'M': M, \
                 'Chain': Chain, 'LogPost': LogPost, 'ACC': ACC, 'GRB': GRB, \
                 'bestx': bestx, 'besty': besty, 'resamples': resamples, \
                 'x': x, 'y': y}, f)
