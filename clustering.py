import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from skimage import measure

difference_maps_folder = 'diff/'

def threshold_matrix(mat, thr = 1):
    "Makes two - sided thresholding of 2D matrix *mat* based on threshold *thr* (defulat=1)"
    matthresh = mat.copy()
    matthresh[np.where((thr > mat) & (mat > 0))] = 0
    matthresh[np.where((-thr < mat) & (mat < 0))] = 0
    return matthresh

def threshold_matrix_twosided(mat, thr1 = 1, thr2 = -1):
    "Makes two - sided thresholding of 2D matrix *mat* based on threshold *thr* (defulat=1)"
    matthresh = mat.copy()
    matthresh[np.where((thr1 > mat) & (mat > 0))] = 0
    matthresh[np.where((thr2 < mat) & (mat < 0))] = 0
    return matthresh

def make_mask_from_coords_list(mat, coords_list, val = 1):
    """
    Puts *val* (int) on *mat* (2D numpy.array) given coordinates
    that are list of lists with pairs of values
    """
    for coords in coords_list:
        for (x,y) in coords:
            mat[x,y] = val
    return mat

list_diff_values = []
list_diff_stds   = []
for ff in glob.glob(difference_maps_folder + '*.npz'):
    name = os.path.split(ff)[-1].split('.')[0]
    loader = np.load(ff)
    diffmap = loader['diff']
    mask    = loader['mask']
    plt.hist(diffmap[np.where(mask>0)])
    plt.xlim([-100,100])
    plt.savefig(os.path.join(difference_maps_folder, 'hists', '{}'.format(name)))
    plt.close()
    list_diff_values.extend(list(diffmap[np.where(mask>0)]))
    list_diff_stds.append(diffmap[np.where(mask>0)].std())

sigma = np.std(list_diff_values)
print(sigma)

sigma2 = np.mean(list_diff_stds)
print(sigma2)


for ff in glob.glob(difference_maps_folder + '*.npz'):
    name = os.path.split(ff)[-1].split('.')[0]
    loader = np.load(ff)
    diffmap = loader['diff']
    mask    = loader['mask']
    sigma_in = diffmap[np.where(mask>0)].std()
    mean_in = diffmap[np.where(mask>0)].mean()
    # Here's threshold based on sigma
    diff_thresh = threshold_matrix_twosided(diffmap, mean_in + 1*sigma_in, mean_in - 1*sigma_in)
    rg_ = np.max([abs(np.max(diff_thresh)), abs(np.min(diff_thresh))])
    plt.imshow(diff_thresh, cmap='RdBu',vmin=-rg_, vmax=rg_)
    plt.imshow(mask, cmap='binary', alpha=0.2)
    plt.savefig(os.path.join(difference_maps_folder, 'thresh', '{}_in2sig'.format(name)))

    arrlabeled = measure.label(diff_thresh != 0)
    regions = measure.regionprops(arrlabeled)
    areas = [region['area'] for region in regions]
    plt.figure()
    plt.subplot(311)
    plt.hist(areas, bins=10)
    plt.title('Histogram of areas')
    for sc in [80, 85, 90, 95]:
        plt.axvline(x = ss.scoreatpercentile(areas, sc), color='r')
    plt.xlabel('Areas size (px)')
    plt.subplot(312)
    plt.imshow(arrlabeled)
    plt.title('Pixel clusters')
    plt.subplot(313)
    # Here's threshold based on size    
    cutoff_size = ss.scoreatpercentile(areas, 80)
    coords_list = [reg.coords for reg, ar in zip(regions, areas) if ar > cutoff_size]
    mask_sized = make_mask_from_coords_list(np.zeros(diff_thresh.shape), coords_list) 
    diff_thresh_size = diff_thresh * mask_sized
    rg_ = np.max([abs(np.max(diff_thresh_size)), abs(np.min(diff_thresh_size))])
    plt.imshow(diff_thresh_size * mask_sized, cmap='RdBu', vmin=-rg_, vmax=rg_)
    plt.colorbar()
    plt.imshow(mask, cmap='binary', alpha=0.2)
    plt.title(r"Difference values after whole clustering (80% area cut-off)")
    plt.tight_layout()
    plt.savefig(os.path.join(difference_maps_folder, 'histsize', '{}_areas'.format(name)))
    plt.close()

