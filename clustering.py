import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from skimage import measure

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

def get_values_std(list_of_diffs, mode = 0):
    '''
    Iterates over difference values and returns standard deviation.

    IN:
        list_of_diffs - list of files with difference and masks (check *make_difference* for details)
        model - 0 computes STD of all values, 1 return mean of individual STDs (default: 0)
    OUT:
        sigma - standard deviation of values from difference maps.
    Example:
        > import glob
        > list_diffs = glob.glob(difference_maps_folder + '*.npz')
        > get_values_std(list_diffs, 1)
    '''
    list_diff_values = []
    list_diff_stds   = []
    for fname in list_of_diffs:
        loader = np.load(fname)
        diffmap = loader['diff']
        mask    = loader['mask']
        list_diff_values.extend(list(diffmap[np.where(mask>0)]))
        list_diff_stds.append(diffmap[np.where(mask>0)].std())
    if mode == 0:
        sigma = np.std(list_diff_values)
    else:
        sigma2 = np.mean(list_diff_stds)
    return sigma

def threshold_diffmaps(diffmap, mask, sigma_threshold = None, area_threshold = 80, plot = False):
    '''
    Performs clustering in two steps:
      1) Based on values - must be below, above sigma.
      2) Based on size of pixel area.

    IN:
        diffmap - matrix with difference map (n, m)
        mask - binary matrix with cartilage region annotation as 1 (n, m)
        sigma_threshold - value of sigma; if None (default) sigma is calculated based
               from diffmap and +/- 1.5 * sigma threshold is taken
               (values above sigma_threshold survive)
        area_threshold - percentile of cluste areas to survive, eg. 80 (default)
               is 80-th percentile
        plot - if string is given it saves plot with steps of thresholding to this path
               (default is None - doesn't plot)
    OUT:
        diff_thresh_size - difference map after thresholding
    '''
    sigma_in = diffmap[np.where(mask>0)].std()
    mean_in = diffmap[np.where(mask>0)].mean()
    # Step (1) - Thresholding based on sigma
    val_threshold = sigma_threshold if sigma_threshold else 1.5 * sigma_in
    diff_thresh = threshold_matrix_twosided(diffmap, \
                                            mean_in + val_threshold, mean_in - val_threshold)
    # Determining area sizes
    arrlabeled = measure.label(diff_thresh != 0)
    regions = measure.regionprops(arrlabeled)
    areas = [region['area'] for region in regions]
    # Step (2) - Thresholding based on area
    cutoff_size = ss.scoreatpercentile(areas, area_threshold)
    coords_list = [reg.coords for reg, ar in zip(regions, areas) if ar > cutoff_size]
    mask_sized = make_mask_from_coords_list(np.zeros(diff_thresh.shape), coords_list) 
    diff_thresh_size = diff_thresh * mask_sized
    if plot:
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
        rg_ = np.max([abs(np.max(diff_thresh_size)), abs(np.min(diff_thresh_size))])
        plt.imshow(diff_thresh_size * mask_sized, cmap='RdBu', vmin=-rg_, vmax=rg_)
        plt.colorbar()
        plt.imshow(mask, cmap='binary', alpha=0.2)
        plt.title(r"Difference values after whole clustering (80% area cut-off)")
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
    return diff_thresh_size

if __name__ == "__main__":
    difference_maps_folder = 'diff/'
    for fname in glob.glob(difference_maps_folder + '*.npz'):
        name    = os.path.split(fname)[-1].split('.')[0]
        loader  = np.load(fname)
        diffmap = loader['diff']
        mask    = loader['mask']
        plot_path = os.path.join(difference_maps_folder, 'hist_xxx', '{}_areas'.format(name))
        threshold_diffmaps(diffmap, mask, area_threshold = 80, plot = plot_path)
