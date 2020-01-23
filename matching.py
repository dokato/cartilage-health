import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def match_shapes(mat_a, mat_b):
    """
    For two matrices where *mat_a* has more rows (first dim) it padds the
    *mat_b* with vectors of zero above and below.
    """
    assert mat_a.shape[0] >= mat_b.shape[0], "first matrix should be bigger"
    cols = mat_a.shape[1]
    zerosvec = np.zeros((1,cols))
    while mat_a.shape[0] - mat_b.shape[0] > 1:
        mat_b = np.r_[zerosvec, mat_b, zerosvec]
    if mat_a.shape[0] == mat_b.shape[0]:
        return mat_b
    assert mat_a.shape[0] - mat_b.shape[0] == 1, \
        'sth wrong in difference a:{} b:{}'.format(mat_a.shape[0], mat_b.shape[0])
    mat_b_up = np.r_[zerosvec, mat_b]
    mat_b_down = np.r_[zerosvec, mat_b]
    if np.sum((mat_a>0)*(mat_b_up>0)) > np.sum((mat_a>0)*(mat_b_down>0)):
        return mat_b_up
    else:
        return mat_b_down

def detect_separate(mata, matb):
    assert mata.shape == matb.shape, "Matrices must be of the same shape"
    sepmat = np.zeros(mata.shape)
    sepmat[np.where((1*(mata>0)+1*(matb>0))==1)] = 1
    return sepmat


cnt = 0
for ff in os.listdir('data/t4'):
    try:
        t4data = np.load(os.path.join('data/t4', ff))
        t8data = np.load(os.path.join('data/t8', ff))
    except FileNotFoundError as e:
        print(e)
        print('Not such file: ' + ff)
        continue
    vizname = ff.split('.')[0]
    t4viz = t4data['visualization'] 
    t8viz = t8data['visualization']
    if t4viz.shape == t8viz.shape:
        title = 'Matched'
        plt.figure()
        plt.imshow(-1*(t4viz>0),vmin=-1,vmax=1, cmap='seismic')
        plt.imshow(1*(t8viz>0),alpha=0.5,vmin=-1,vmax=1, cmap='seismic')
        plt.savefig('matching_' + vizname+'.png')
        plt.close()
    else:
        title = 'No matched'
        print('Different sizes: ' + ff + ' t4: ' + str(t4viz.shape) + ' t8: ' + str(t8viz.shape))
        cnt += 1
        plt.figure(figsize=(9,4))
        plt.subplot(121)
        plt.imshow(t4viz>0)
        plt.subplot(122)
        plt.imshow(t8viz>0)
        plt.savefig('nonmatching_' + vizname+'.png')
        plt.close()
        if t4viz.shape[0] < t8viz.shape[0]:
            t4viz = match_shapes(t8viz, t4viz)
        else:
            t8viz = match_shapes(t4viz, t8viz)
    plt.figure(figsize=(13,2.5))
    plt.subplot(141)
    plt.imshow(-1*(t4viz>0),vmin=-1,vmax=1, cmap='seismic')
    plt.imshow(1*(t8viz>0),alpha=0.5,vmin=-1,vmax=1, cmap='seismic')
    plt.subplot(142)
    be = ndimage.morphology.binary_erosion((1*(t4viz>0)+1*(t8viz>0))>0,
                                    structure=np.ones((3,3))).astype(np.int)
    plt.imshow(be, vmin=-1,vmax=1, cmap='seismic')
    plt.subplot(143)
    viz = t4viz-t8viz
    rg_ = np.max([abs(np.max(viz)), abs(np.min(viz))])
    plt.imshow(viz, cmap='RdBu', vmin=-rg_, vmax=rg_)
    plt.colorbar()
    plt.subplot(144)
    viz = (t4viz-t8viz)*be
    rg_ = np.max([abs(np.max(viz)), abs(np.min(viz))])
    plt.imshow(viz, cmap='RdBu', vmin=-rg_, vmax=rg_)
    plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('erosion_' + vizname+'.png')
    plt.close()

print('Non matching sizes:' + str(cnt))


