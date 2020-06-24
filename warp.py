import os
from pathlib import Path
import argparse

import numpy as np
from scipy.interpolate import griddata

from DFLIMG import *
from facelib import LandmarksProcessor
from core.cv2ex import *
from core import pathex

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

p = argparse.ArgumentParser()
p.add_argument('--input-dir', required=False, action=fixPathAction, dest="input_dir",
               help="Input directory. A directory containing the files you wish to process.")
p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir",
               help="Output directory. This is where the warped input files will be saved.")
p.add_argument('--aligned-dir', required=True, action=fixPathAction, dest="aligned_dir", default=None,
               help="Aligned directory. This is where the extracted of dst faces stored.")
p.add_argument('--deltas-file', required=True, action=fixPathAction, dest="deltas_file", default=None,
               help="Aligned directory. This is where the extracted of dst faces stored.")
p.add_argument('--border', type=int, dest="border", default=0,
               help="Amount of non-warped border around aligned face (in percent) - default 0")
p.add_argument('--blur', type=int, dest="blur", default=5,
               help="Size of blur to apply to warp map (in percent of image size) - default 5%")
p.add_argument('--raw', action="store_true", dest="raw", default=False,
               help="Operate on raw predictions (in aligned state)")
p.add_argument('--save-map', action="store_true", dest="save_map", default=False,
               help="Save warp map only.")

args = p.parse_args()

if args.input_dir:
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise Exception('Input directory not found. Please ensure it exists.')
else:
    if args.save_map: #if no input given, but save map is specified, we must assume raw mode working on aligned frames
        args.raw = True
    else:
        raise Exception("No input and map saving not specified, nothing to do.")

aligned_path = Path(args.aligned_dir)
if not aligned_path.exists():
    raise Exception('Aligned directory not found. Please ensure it exists.')

output_path = Path(args.output_dir)
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)
if args.save_map:
    warp_output_path = output_path / 'warpmap'
    if not warp_output_path.exists():
        warp_output_path.mkdir(parents=True, exist_ok=True)

deltas_path = Path(args.deltas_file)
if not deltas_path.exists():
    raise Exception('Deltas file not found. Please ensure it exists.')
else:
    landmarks_deltas = np.loadtxt(deltas_path, dtype=float)

edge_x = np.arange(0,1.1,.1)
edge_t = np.zeros(11)
edge_b = np.ones(11)
edges = np.vstack((edge_x,edge_t)).T # top edge
edges = np.concatenate((edges, np.vstack((edge_x,edge_b)).T)) # bottom edge
edges = np.concatenate((edges, np.flip(edges))) # left and right edges

border_scale = 1 - args.border/100.0
borders = LandmarksProcessor.transform_points(edges, np.array([[1,0,-0.5],[0,1,-0.5]]))
borders = LandmarksProcessor.transform_points(borders, np.array([[border_scale,0,0.5],[0,border_scale,0.5]]))

def generator():
    for filepath in io.progress_bar_generator(pathex.get_image_paths(aligned_path), "Processing alignments"):
        filepath = Path(filepath)
        yield DFLIMG.load(filepath)

source_images = []
for dflimg in generator():
    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        continue

    source_filename = dflimg.get_source_filename()
    if source_filename is None:
        continue
    source_filename = Path(source_filename)
    if source_filename.name in source_images:
        io.log_info("Warning: multiple faces detected. Only one alignment file should refer one source file.")
        continue
    source_images.append(source_filename.name)

    a_h, a_w, _ = dflimg.get_shape()
    src = dflimg.get_landmarks()
    dst = src.copy() + landmarks_deltas * [a_w, a_h]
    src = np.concatenate((src, edges * [a_w, a_h], borders * [a_w, a_h]), 0)
    dst = np.concatenate((dst, edges * [a_w, a_h], borders * [a_w, a_h]), 0)

    if args.input_dir:
        dst_path = input_path / source_filename
        if not dst_path.exists():
            io.log_err(f'Input file {dst_path} not found, skipping...')
            continue
        dst_bgr = cv2_imread(dst_path)
        h, w, _ = dst_bgr.shape
    else:
        h = a_h
        w = a_w

    if args.raw: #scale aligned coords to match raw image coords
        src = src * [w / a_w, h / a_h]
        dst = dst * [w / a_w, h / a_h]
    else: # #put points into dst image space
        mat = dflimg.get_image_to_face_mat()
        src = LandmarksProcessor.transform_points(src, mat, invert=True)
        dst = LandmarksProcessor.transform_points(dst, mat, invert=True)
        src = np.concatenate((src, edges * [w, h]))
        dst = np.concatenate((dst, edges * [w, h]))

    grid_y, grid_x = np.mgrid[0:h, 0:w]  # cross outputs for x,y addressing
    grid_z = griddata(dst, src, (grid_x, grid_y), method='linear')
    k = int(args.blur/100 * w)
    if k % 2 == 0:
        k += 1
    grid_z_blur = cv2.GaussianBlur(grid_z, (k, k), 0)
    grid_z[k:h - k, k:w - k, :] = grid_z_blur[k:h-k, k:w-k, :] # slice blur away from borders to avoid tearing

    map_x = grid_z[:, :, 0].astype('float32')
    map_y = grid_z[:, :, 1].astype('float32')

    if args.save_map:
        warp_map_bgr = np.dstack((np.zeros_like(map_x), (h - 1) - map_y, map_x)) #invert y-axis for nuke-style ordering
        warp_map_bgr /= (w - 1) #normalize
        warp_map_path = warp_output_path / source_filename.with_suffix('.exr')
        cv2_imwrite(warp_map_path, warp_map_bgr)

    elif args.input_dir:
        warped_dst_path = output_path / source_filename
        warped = cv2.remap(dst_bgr, map_x, map_y, cv2.INTER_CUBIC)
        cv2_imwrite(warped_dst_path, warped)
