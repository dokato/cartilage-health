# Cartilage Health

## Requirements

Install the following packages:

```
pydicom
matplotlib
numpy
scipy
joblib
cv2
```

The script should be compatible with python `2.7` and `>3.3`.

## How to use it?

The whole pipeline consists of the following steps:

1. `compT2.py` - calulcation of T2 image from echo responses;
2. `projection.py` - Femur projection of cartilage (after that optional `rotation.py`);
3. `matching.py` - matching projected images from two time points (eg. T4 and T8) and computing difference between them;
4. `clustering.py` - clustering regions of the cartilage based on the values difference;

Below you'll find details of each step.

### Calculation of T2 image

Example:
 
```python
from compT2 import *
dirname = "data/{}/T2/".format(file_name)
nr_slices = estimate_nr_slices(dirname)
path_segmentation = "data/{}/T2BinarySegmentation/{}_4_".format(file_name, file_name) + "{}.mat"
segmentation = get_segmentation_mask(path_segmentation, nr_slices) # segmentation is optional
t2imgs, t2times = get_t2(dirname, nr_slices = nr_slices)
t2matrix = fit_t2(t2imgs, t2times, segmentation=segmentation)
```

#### Command line

Alternatively, you may call this command line tool. For more details on how to use it, check the manual:

```
python compT2-cli.py -h
```

### Projection

For a **masked** matrix call:

```python
visualization, super_visualization, deep_visualization, avg_vals_dict = projection(t2masked)
```

### Matching & difference calculations

Assuming that you have projections from the two time steps, created using a function above, you may call:

```python
diff_masked, mask = make_difference(visualization_t4, visualization_t8)
```

### Clustering

For two sided clustering:

```python
threshold_diffmaps(diff_masked, mask, one_sided = 0, area_threshold = 80)
```

For negative and positive thresholding set argument `one_sided` to -1 or 1 respectly:

```python
threshold_diffmaps(diff_masked, mask, one_sided = 1, area_threshold = 80)
```

Check docs for more settings.
