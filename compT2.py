import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.io import loadmat
import time

def get_slice(filename):
    'Reads from dicom file basic information and returns image as array + aquisition time'
    ds = pydicom.dcmread(filename)
    image_2d = ds.pixel_array.astype(float)
    return image_2d, np.float(ds.AcquisitionTime)

def get_t2(path, nr_slices = 29, time_steps = 7):
    '''
    Gets T2 weighted image and its acquisition times.
    IN:
        path - path with T2 weighted images
        nr_slices - nr of slices in image
        time_steps - nr of echo times per slice
    OUT:
        matrix - with T2 weighted images in numpy array (nr_slices, time_steps, width, heigth)
        time_lists - list of aquisition times
    '''
    slices = []
    time_slices = []
    for slice_idx in range(1, nr_slices+1): # iterates over all slices
        temp_slice = []
        time_slice = []
        for i in range(time_steps): # iterates over all time steps
            file_path = dirname + '%03d' % (i * nr_slices + slice_idx)
            image, aq_time = get_slice(file_path)
            temp_slice.append(image)
            time_slice.append(aq_time)
        temp_slice = np.array(temp_slice)
        time_slices.append(time_slice)
        slices.append(temp_slice)
    slices = np.array(slices)

    vmax = np.max(slices)
    # Rescaling grey scale between 0-255
    slices = (np.maximum(slices,0) / vmax) * 255.0
    slices = np.float32(slices) #uint8
    return slices, time_slices

def get_segmentation_mask(path, nr_slices = 29):
    '''
    Returns binary mask of the region of interest.
    IN:
        path - string with formatting indication number of slice, eg.
               "data/9003126/T2BinarySegmentation/9003126_4_{}.mat"
        nr_slices - integer 
    OUT:
        segmentation matrix (nr_slices, width, heigth)
    '''
    segmentation = []
    for i in range(nr_slices):
        mat = loadmat(path.format(i))
        segmentation.append(mat['femoral_cartilage'])
    segmentation = np.array(segmentation)
    return segmentation

def fit_t2(t2imgs, t2times, segmentation = None):
    '''
    Fits T2 curves to the T2_weighted images in each slice.
    IN:
        t2imgs - with T2 weighted images in numpy array (nr_slices, time_steps, width, heigth)
        t2times - list with aquisition times
        segmentation - segmentation matrix (nr_slices, width, heigth)
    OUT:
        matrix (nr_slices, width, heigth) with T2 values
    '''
    t2_tensor = np.zeros((t2imgs.shape[0], t2imgs.shape[2], t2imgs.shape[3]))
    for slice_idx in range(t2imgs.shape[0]):
        scan = t2imgs[slice_idx,:,:,:]
        mri_time = np.array(t2times[slice_idx]) - t2times[slice_idx][0]
        if segmentation:
            segmentation_mask = segmentation[slice_idx,:,:]
        data = np.log(scan + 0.0000000001) # to avoid log(0)
        x = np.concatenate((np.ones_like(mri_time[..., np.newaxis]), -mri_time[..., np.newaxis]), 1)
        t2_matrix = np.zeros((data.shape[1], data.shape[2]))
        res_matrix = np.zeros((data.shape[1], data.shape[2]))
        for ix in range(data.shape[2]):
            for iy in range(data.shape[2]):
                if all(data[:,ix,iy] == data[0,ix,iy]): # if constant value, decay is 0 
                    continue
                beta, res, _, _ = la.lstsq(x[1:], data[1:,ix,iy])
                t2_ = 1./beta[1]
                t2_matrix[ix, iy] = t2_
                res_matrix[ix, iy] = res
        res_matrix[np.where(res_matrix > np.percentile(res_matrix.flatten(), 98.))]=0
        res_matrix[np.where(res_matrix < 0)] = 0
        t2_matrix[np.where(t2_matrix > np.percentile(t2_matrix.flatten(), 97.))] = 0
        t2_matrix[np.where(t2_matrix < 0)] = 0
        t2_matrix[np.where(res_matrix > 0.1)] = 0
        if segmentation:
            t2_matrix[np.where(segmentation_mask != 1)] = 0
        t2_tensor[slice_idx,:,:] = t2_matrix * 1000 # in ms
    return t2_tensor

if __name__ == "__main__":
    dirname = "data/9003126/T2/"
    t2imgs, t2times = get_t2(dirname)
    t0 = time.time()
    t2matrix = fit_t2(t2imgs, t2times)
    print(time.time() - t0)