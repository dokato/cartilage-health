from compT2 import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("id", help="Subject ID (should be a name of a folder from *data_folder*)")
parser.add_argument("-d", "--data_folder", type=str, default = "data/",
                    help="Folder with T2-weighted echo measurments (default: 'data/')")
parser.add_argument("-n", "--nr_slices", type=int, default = None,
                    help="""Number of slices. By default estimated based on
                    '*data_folder*/*id*/T2' folder content""")
parser.add_argument("-s", "--segmentation", type=str, default = None,
                    help="""Path to reach segmentation masks. It should use formatting
                    curly brackets to put there a slice number. Default None.
                    Eg. 'data/11111/T2BinarySegmentation/11111_4_\{\}.mat' """)
parser.add_argument("-f", "--save_file", type=str, default = None,
                    help="Name of file to save T2 results. Default: '*id*_t2.npy'")
parser.add_argument("-j", "--jobs", type=int, default = 4,
                    help="Number of processor jobs. Default: 4.")
args = parser.parse_args()


file_name = args.id
data_folder = args.data_folder
dirname = "{}/{}/T2/".format(data_folder, file_name)

if args.nr_slices is None:
    nr_slices = estimate_nr_slices(dirname)
else:
    nr_slices = args.nr_slices

t2imgs, t2times = get_t2(dirname, nr_slices = nr_slices)

segmentation = None
if not args.segmentation is None:
    segmentation = get_segmentation_mask(args.segmentation, nr_slices = nr_slices)

t2matrix = fit_t2(t2imgs, t2times, segmentation = segmentation, n_jobs = args.jobs)

save_file_name = '{}_t2.npy'.format(file_name) if args.save_file is None else args.save_file
with open(save_file_name, 'wb') as ff:
    np.save(ff, t2matrix)
