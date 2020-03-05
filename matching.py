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

def remove_empty_rows(image):
    "Removes empty rows from image (2d numpy.array)"
    mask = image == 0
    rows = np.flatnonzero((~mask).sum(axis=1))
    cropped = image[rows.min():rows.max()+1, :]
    return cropped

def eroded_and_mask(mata, matb, kernel = np.ones((2,2))):
    """
    Creatives mask for matrices *mata* and *matb* ( each 2D numpy.array)
    with positive elements.
    Erosion is applied first to each of matrix with *kernel* default(2x2)
    Then pointwise logical and is taken between these two masks.
    """
    assert mata.shape == matb.shape, "Shapes don't match"
    assert (mata>=0).all() and (matb>=0).all(), "Works only for positive matrices"
    ermask_a = ndimage.morphology.binary_erosion(1*(mata > 0),
                                    structure = kernel).astype(np.int)
    ermask_b = ndimage.morphology.binary_erosion(1*(matb > 0),
                                    structure = kernel).astype(np.int)
    return ermask_a*ermask_b

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
    t4viz = remove_empty_rows(t4data['visualization'])
    t8viz = remove_empty_rows(t8data['visualization'])
    # in super and deep there's some empty slices, this handles it
    # but TODO: maybe account for it in projection?
    for i in np.argwhere(np.isnan(t4viz)): 
        t4viz[tuple(i)]=0
    for i in np.argwhere(np.isnan(t8viz)):
        t8viz[tuple(i)]=0
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
    be = eroded_and_mask(t4viz, t8viz)
    plt.imshow(be, vmin=-1,vmax=1, cmap='seismic')
    plt.subplot(143)
    viz = t4viz-t8viz
    rg_ = np.max([abs(np.max(viz)), abs(np.min(viz))])
    plt.imshow(viz, cmap='RdBu', vmin=-rg_, vmax=rg_)
    plt.colorbar()
    plt.subplot(144)
    diff_masked = (t4viz-t8viz)*be
    rg_ = np.max([abs(np.max(diff_masked)), abs(np.min(diff_masked))])
    plt.imshow(diff_masked, cmap='RdBu', vmin=-rg_, vmax=rg_)
    plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('erosion_' + vizname+'.png')
    plt.close()
    subj_ind = vizname.split('_')[0]
    np.savez(os.path.join('diff', '{}_diff'.format(subj_ind)), diff = diff_masked, mask=be)

print('Non matching sizes:' + str(cnt))
