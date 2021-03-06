import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import CubicSpline
import cv2
from PIL import Image

from compT2 import * 

def stretch_axis(matrix, factor = 2):
    """
    Stretches first dimension of the matrix by factor (default: 2).
    """
    d, h, w = matrix.shape
    new_matrix = np.zeros((int(factor * d), h, w))
    for i in range(h):
        im_ = Image.fromarray(matrix[:, i, :])
        im_ = im_.resize((w, int(factor * d)))
        new_matrix[:, i, :] = np.array(im_)
    return new_matrix

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

if len(sys.argv) == 1:
    filename = '9518827'
else:
    filename = sys.argv[1]
segm = np.load('{}_t2.npy'.format(filename))

segm = cv_stretch_axis(segm, 4)

flattened = segm.max(1)
x, y = np.nonzero(flattened)

xm, ym = np.mean(x), np.mean(y)
x = x - xm
y = y - ym
coords = np.vstack([x, y])

cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]
x_v2, y_v2 = evecs[:, sort_indices[1]]

scale = 20
# plt.plot(x, y, 'k.')
# plt.plot([x_v1*-scale*2, x_v1*scale*2],
#          [y_v1*-scale*2, y_v1*scale*2], color='red')
# plt.plot([x_v2*-scale, x_v2*scale],
#          [y_v2*-scale, y_v2*scale], color='blue')
# #plt.axis('equal')
# plt.show()

theta = np.arctan((x_v1)/(y_v1))
rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat * coords
# plot the transformed blob
x_transformed, y_transformed = transformed_mat.A

segm_tran = np.zeros(segm.shape)

i = 256
x_s, y_s = np.nonzero(segm[:, i, :])
x_s = x_s - xm
y_s = y_s - ym
coords_s = np.vstack([x_s, y_s])
trans_mat_s = rotation_mat * coords_s
x_t_s, y_t_s = trans_mat_s.A
x_t_s = x_t_s + xm
y_t_s = y_t_s + ym
segm_tran[np.round(x_t_s).astype(np.int), i, np.round(y_t_s).astype(np.int)] = 1

# x_s, y_s = np.nonzero(segm[:, i, :])
# plt.figure()
# plt.plot(x_s,y_s,'bo')
# plt.plot(x_t_s, y_t_s,'mo')
# plt.plot(np.round(x_t_s), np.round(y_t_s),'go')
# plt.show()

for i in range(segm.shape[1]):
    x_s, y_s = np.nonzero(segm[:, i, :])
    if len(x_s) == 0 or len(y_s) == 0:
        continue
    x_sp = x_s - xm
    y_sp = y_s - ym
    coords_s = np.vstack([x_sp, y_sp])
    trans_mat_s = rotation_mat * coords_s
    x_t_s, y_t_s = trans_mat_s.A
    x_t_s = x_t_s + xm
    y_t_s = y_t_s + ym
    segm_tran[np.round(x_t_s).astype(np.int), i, np.round(y_t_s).astype(np.int)] = segm[x_s, i, y_s]

# plt.figure()
# x_s, y_s = np.nonzero(segm[:, 157, :])
# x_t_s, y_t_s = np.nonzero(segm_tran[:, 157, :])
# plt.plot(x_s,y_s,'bo')
# plt.plot(x_t_s, y_t_s,'mo')
# plt.plot(np.round(x_t_s), np.round(y_t_s),'go')
# plt.show()

plt.figure()
plt.subplot(121)
im1 = Image.fromarray(segm.max(1)).resize((512,512))
plt.imshow(im1)
plt.subplot(122)
im2 = Image.fromarray(segm_tran.max(1)).resize((512,512))
plt.imshow(im2)
#plt.show()
plt.savefig(filename+'_pca.png')

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
plt.savefig('scat_' + filename + '_pca.png')

print('Angle:', math.degrees(theta))