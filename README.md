# Cartilage Health

## Requirements

Install the following packages:

```
pydicom
matplotlib
numpy
scipy
joblib
```

The script should be compatible with python `2.7` and `>3.3`.

## How to use it?

Without segmentation:

```python
from compT2 import *
dirname = "data/{}/T2/".format(file_name)
nr_slices = estimate_nr_slices(dirname)
t2imgs, t2times = get_t2(dirname, nr_slices = nr_slices)
t2matrix = fit_t2(t2imgs, t2times)
```

With segmentation:

```python
from compT2 import *
dirname = "data/{}/T2/".format(file_name)
nr_slices = estimate_nr_slices(dirname)
path_segmentation = "data/{}/T2BinarySegmentation/{}_4_".format(file_name, file_name) + "{}.mat"
segmentation = get_segmentation_mask(path_segmentation, nr_slices)
t2imgs, t2times = get_t2(dirname, nr_slices = nr_slices)
t2matrix = fit_t2(t2imgs, t2times, segmentation=segmentation)
```

As standalone script (for now it works without segmentation) for data recording `XXXXXXX`

```
$ python compT2.py XXXXXXX
```

assuming that you have the following catalogue structure:

```
data/
   9999999/
      T2/
         001
         002
         ...
      DESS/
   0000000/
   ....
compT2.py
```

It will save results of T2 computations as `.npy` matrix file.

## Command line

Alternatively, you may call this command line tool. For more details on how to use it, check the manual:

```
python compT2-cli.py -h
```
