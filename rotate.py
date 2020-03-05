import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import CubicSpline
import cv2
from PIL import Image

from compT2 import * 

def cv_stretch_axis(matrix, factor = 2):
    """
    Stretches first dimension of the matrix by factor (default: 2).
    """
    d, h, w = matrix.shape
    new_matrix = np.zeros((int(factor * d), h, w))
    for i in range(h):
        im_ = matrix[:, i, :]
        im_ = cv2.resize(im_, dsize = (w, int(factor * d)), interpolation = cv2.INTER_NEAREST )
        new_matrix[:, i, :] = np.array(im_)
    return new_matrix

def map_on_plane(mat, func = np.max, axis = 1):
    'Flattend the image by mapping matrix using *func* over *axis*'
    return func(mat, axis=axis)

def rotate(mat, rotation_mat, rotation_point):
    '''
    Rotates *mat* (over dimension =1) using *rotation_mat* around *rotation_point*.
    '''
    mat_tran = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        x_s, y_s = np.nonzero(mat[:, i, :])
        if len(x_s) == 0 or len(y_s) == 0:
            continue
        x_sp = x_s - xm
        y_sp = y_s - ym
        coords_s = np.vstack([x_sp, y_sp])
        trans_mat_s = rotation_mat * coords_s
        x_t_s, y_t_s = trans_mat_s.A
        x_t_s = x_t_s + xm
        y_t_s = y_t_s + ym
        mat_tran[np.round(x_t_s).astype(np.int), i, np.round(y_t_s).astype(np.int)] = mat[x_s, i, y_s]
    return mat_tran

def make_rotation_matrix(angle, radian = False):
    '''
    Creates rotation matrix for *angle* (default in degrees).
    '''
    if not radian:
        angle = math.radians(angle)
    theta = angle
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return rotation_mat

def get_mirror_correlation(mat, binarise = True):
    '''
    From 2D matrix *mat* computes correlation with it's mirror flip (over dimension = 0).
    If you care only about area, set binarise to True.
    '''
    if binarise:
        mat = 1. * (mat > 0)
    return np.sum(mat * np.flipud(mat))

def get_area_in_region(mat, x_low, x_high):
    '''
    Get area size (by terms of number of non zero pixels) in the region limited
    '''
    mat = 1. * (mat > 0)
    area = np.where(mat[:,x_low:x_high]!=0)[0].size
    return area

if len(sys.argv) == 1:
    filename = '9518827'
else:
    filename = sys.argv[1]

segm = np.load('{}_t2.npy'.format(filename))
segm = cv_stretch_axis(segm, 4)

flattened = map_on_plane(segm)
x, y = np.nonzero(flattened)
xm, ym = np.mean(x), np.mean(y)

max_corr = get_mirror_correlation(flattened)
ang_max = None
for ang in np.linspace(-4,4,40):
    rot_mat = make_rotation_matrix(ang)
    segm_tran = rotate(segm, rot_mat, (xm, ym))
    mc_ = get_mirror_correlation(map_on_plane(segm_tran))
    if mc_ > max_corr:
        ang_max = ang
        max_corr = mc_

if ang_max is None:
    segm_tran = segm
    print(0)
else:
    print(ang_max)
    rot_mat = make_rotation_matrix(ang_max)
    segm_tran = rotate(segm, rot_mat, (xm, ym))

plt.subplot(121)
im1 = Image.fromarray(segm.max(1)).resize((512,512))
plt.imshow(im1)
plt.subplot(122)
im2 = Image.fromarray(segm_tran.max(1)).resize((512,512))
plt.imshow(im2)
#plt.show()
plt.savefig(filename+'_symm.png')

def plot_points(mat):
    x, y = np.nonzero(mat)
    plt.plot(x, y, 'b.')


plt.figure()
plt.suptitle(filename)
plt.subplot(221)
plt.title('before')
plot_points(segm.max(1))
plt.subplot(223)
plot_points(segm.max(0))
plt.subplot(222)
plt.title('after')
plot_points(segm_tran.max(1))
plt.subplot(224)
plot_points(segm_tran.max(0))
plt.savefig('scat_' + filename + '_symm.png')