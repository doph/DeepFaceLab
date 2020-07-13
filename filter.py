import os
import argparse

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import gaussian, convolve2d

from core.cv2ex import *
from core import pathex
from core.leras import nn
from DFLIMG import *
from facelib import FaceType
from mainscripts.Extractor import ExtractSubprocessor


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


p = argparse.ArgumentParser()
p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir",
               help="Input directory. A directory containing the files you wish to process.")
p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir",
               help="Output directory. This is where the warped input files will be saved.")
p.add_argument('--aligned-dir', required=True, action=fixPathAction, dest="aligned_dir", default=None,
               help="Aligned directory. This is where the extracted of dst faces stored.")
p.add_argument('--radius', type=int, dest="radius", default=1,
               help="Number of frames before and after current over which to integrate")
p.add_argument('--sigma', type=int, dest="sigma", default=1,
               help="Number of frames before and after current over which to integrate")
p.add_argument('--output-debug', action="store_true", dest="debug", default=False,
               help="Output aligned debug images")

args = p.parse_args()

input_path = Path(args.input_dir)
if not input_path.exists():
    raise Exception('Input directory not found. Please ensure it exists.')

aligned_path = Path(args.aligned_dir)
if not aligned_path.exists():
    raise Exception('Aligned directory not found. Please ensure it exists.')

output_path = Path(args.output_dir)
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)

if args.debug:
    debug_dir = Path(args.output_dir + '_debug')
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True, exist_ok=True)
else:
    debug_dir = None


def fill_missing(mat, mask):
    grid_y, grid_x = np.mgrid[:mat.shape[0], :mat.shape[1]]
    mat_filled_lin = griddata((grid_x[mask].ravel(), grid_y[mask].ravel()), mat[mask].ravel(), (grid_x, grid_y),
                              method='linear')
    mat_filled_near = griddata((grid_x[mask].ravel(), grid_y[mask].ravel()), mat[mask].ravel(), (grid_x, grid_y),
                               method='nearest')
    mat_filled_lin[np.isnan(mat_filled_lin)] = mat_filled_near[np.isnan(mat_filled_lin)]
    return mat_filled_lin


def alignments_generator():
    for filepath in io.progress_bar_generator(pathex.get_image_paths(aligned_path, return_Path_class=True),
                                              "Collecting landmarks"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_err(f"{dflimg.filename} is not a dfl image file")
            continue
        yield dflimg


# get images from input and aligned dirs
dst_image_paths = sorted(pathex.get_image_paths(input_path, return_Path_class=True))
lmrks_dict = {dflimg.get_source_filename(): {'rect': dflimg.get_source_rect(),
                                             'lmrks': dflimg.get_source_landmarks(),
                                             'face_type': dflimg.get_face_type()
                                             } for dflimg in alignments_generator()}

# vectorize rects and landmarks
rect_mat = np.zeros((len(dst_image_paths), 4))
lmrks_mat = np.zeros((len(dst_image_paths), 68 * 2))
aligned_mask = []
face_type = None
for i, path in enumerate(dst_image_paths):
    if path.name in lmrks_dict.keys():
        rect_mat[i] = lmrks_dict[path.name]['rect']
        lmrks_mat[i] = lmrks_dict[path.name]['lmrks'].ravel()
        if not face_type:
            face_type = FaceType.fromString(lmrks_dict[path.name]['face_type'])
        aligned_mask.append(i)

# interpolate missing values of landmark matrix
if len(dst_image_paths) > len(lmrks_dict.keys()):
    rect_mat = fill_missing(rect_mat, aligned_mask)
    lmrks_mat = fill_missing(lmrks_mat, aligned_mask)

# smooth landmark matrix
k_width = args.radius * 2 + 1
kernel = gaussian(k_width,sigma)
kernel /= np.sum(kernel)
kernel = kernel.reshape(k_width,1)
rect_mat_smoothed = convolve2d(rect_mat, kernel, mode='same', boundary='symm')
lmrks_mat_smoothed = convolve2d(lmrks_mat, kernel, mode='same', boundary='symm')

# run back through final stage of extract to write new aligned images
data = [ExtractSubprocessor.Data(filename,
                                 [tuple(rect.astype(int))],
                                 [lmrks.reshape(68, 2)]
                                 ) for filename, rect, lmrks in
        zip(dst_image_paths, rect_mat_smoothed, lmrks_mat_smoothed)]
image_size = 512 if face_type < FaceType.HEAD else 768
io.log_info('Writing smoothed alignments: ')
ret = ExtractSubprocessor(data,
                          'final',
                          image_size,
                          face_type,
                          output_debug_path=debug_dir,
                          final_output_path=output_path,
                          device_config=nn.DeviceConfig.CPU()
                          ).run()
io.log_info('Complete.')
