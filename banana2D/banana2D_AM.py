from __future__ import division, print_function, absolute_import
import os
import sys
sys.path.append('../src')
import AM
import numpy as np
import util
#import cPickle
import pickle
import matplotlib.pyplot as plt

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

# parameters for Metropolis
T = 1
B = 10000
N = 10000
M = 4
parallel = False

# run Metropolis
[Chain, LogPost, ACC, GRB] = \
    AM.sampler(model, D, xlb, xub, None, None, T, B, N, M, \
        parallel)

# plot results
#print(ACC)
#print(GRB)
plt.plot(Chain[:,0],Chain[:,1],'b.')
plt.xlabel('x1')
plt.ylabel('x2')
# save figure
plt.savefig('%s/banana2D_AM.png' % respath)

# save results to bin file
# with open('%s/banana2D_AM.bin' % respath,'w') as f:
#     cPickle.dump({'D':D, 'T': 1, 'B': B, 'N': N, 'M': M, \
#                   'Chain': Chain, 'LogPost': LogPost, \
#                   'ACC': ACC, 'GRB': GRB}, f)
with open('%s/banana2D_AM.bin' % respath,'wb') as f:
    pickle.dump({'D':D, 'T': 1, 'B': B, 'N': N, 'M': M, \
                 'Chain': Chain, 'LogPost': LogPost, \
                 'ACC': ACC, 'GRB': GRB}, f)
