import math
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
from PIL import Image

fig = False

def calc_R(x,y, xc, yc):
    "distance of each 2D points from the center (xc, yc)"
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    "algebraic distance between the data points and the mean circle centered at c=(xc, yc)"
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    """
    Computes leas squared circle fit based on coordinatex *x*, *y*.
    Adapted from http://wiki.scipy.org/Cookbook/Least_Squares_Circle
    """
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_region_value(regiondict, binned_super, binned_deep, angle, slice):
    if angle <=-35:
        if slice <= 40:
            regiondict['SLA'].extend(list(binned_super[:,2]))
            regiondict['DLA'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUA'].extend(list(binned_super[:,2]))
            regiondict['DUA'].extend(list(binned_deep[:,2]))
    elif -35 < angle <= 25:
        if slice <= 40:
            regiondict['SLC'].extend(list(binned_super[:,2]))
            regiondict['DLC'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUC'].extend(list(binned_super[:,2]))
            regiondict['DUC'].extend(list(binned_deep[:,2]))
    elif angle > 25:
        if slice <= 40:
            regiondict['SLP'].extend(list(binned_super[:,2]))
            regiondict['DLP'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUP'].extend(list(binned_super[:,2]))
            regiondict['DUP'].extend(list(binned_deep[:,2]))

masked = np.load('9003126_t2.npy') 
thickness_div = 0.5

nr_slices = masked.shape[0]

masked[masked > 100] = 0
mask_proj = np.max(masked, axis = 0)

x_coord, y_coord = np.nonzero(mask_proj)

xc, yc, R, _ = leastsq_circle(x_coord, y_coord)

if fig:
    plt.imshow(mask_proj)
    theta_fit = np.linspace(-np.pi, np.pi, 180)
    x_fit = xc + R * np.cos(theta_fit)
    y_fit = yc + R * np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_coord, y_coord, 'g.', label='data', mew=1)
    plt.show()

visualization       = np.zeros((nr_slices,72))
super_visualization = np.zeros((nr_slices,72))
deep_visualization  = np.zeros((nr_slices,72))
superficial_values  = []
deep_values         = []
regions = ['SLA','SLC','SLP','SUA','SUC','SUP','DLA','DLC','DLP','DUA','DUC', 'DUP']
regiondict = dict((i, []) for i in regions)

for i in range(nr_slices):
    if np.all(masked[i, :, :]==0):
        continue
    x_inds, y_inds = np.nonzero(np.squeeze(masked[i,:,:]))
    x_ind_c, y_ind_c = x_inds - xc, y_inds - yc
    values = masked[i, x_inds, y_inds] 

    rho, theta = cart2pol(x_ind_c, y_ind_c)
    theta = np.array(list(map(math.degrees, theta))) # transf. to degrees
    print(i, np.min(theta), np.max(theta))
    polar_matrix = np.vstack((theta, rho, values)).T

    angles = np.arange(-180,180,5)
    for e, angle in enumerate(angles):
        bottom_bin, top_bin = angle, angle + 5
        splice_matrix = np.logical_and(theta > bottom_bin, theta <= top_bin)
        if np.all(splice_matrix == False):
            continue
        if i==12 and angle==-90:
            import pdb;pdb.set_trace()
        polar_in_cone = polar_matrix[splice_matrix]
        max_radius = np.max(polar_in_cone[:,1])
        min_radius = np.min(polar_in_cone[:,1])

        cart_thickness = max_radius - min_radius
        rad_division = min_radius + cart_thickness * thickness_div

        splice_deep = polar_in_cone[:,1] <= rad_division
        binned_deep  = polar_in_cone[splice_deep]
        binned_super = polar_in_cone[~splice_deep]

        superficial_values.extend(list(binned_super[:,2]))
        deep_values.extend(list(binned_deep[:,2]))
        get_region_value(regiondict, binned_super, binned_deep, angle, i)
        visualization[i, e] = np.mean(polar_in_cone[:,2])
        super_visualization[i, e] = np.mean(binned_super[:,2])
        deep_visualization[i, e] = np.mean(binned_deep[:,2])


averagedT2 = masked.sum()/np.count_nonzero(masked)

print(averagedT2)
print(np.mean(superficial_values))
print(np.mean(deep_values))
for k in regiondict:
    print(k, np.mean(regiondict[k]))

if fig:
    plt.subplot(2,2,1)
    im1 = Image.fromarray(visualization)                                    
    im2 = im1.resize((512,512))                                           
    plt.imshow(im2)
    plt.colorbar()
    plt.title('All')
    plt.subplot(2,2,2)
    im1 = Image.fromarray(super_visualization)                                    
    im2 = im1.resize((512,512))                                           
    plt.imshow(im2)
    plt.colorbar()
    plt.title('superficial')
    plt.subplot(2,2,3)
    im1 = Image.fromarray(deep_visualization)                                    
    im2 = im1.resize((512,512))                                           
    plt.imshow(im2)
    plt.colorbar()
    plt.title('deep')
