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
p.add_argument('--aligned-dir', required=True, action=fixPathAction, dest="aligned_dir", default=None,
               help="Aligned directory. This is where the extracted of dst faces stored.")
p.add_argument('--output-dir', required=False, action=fixPathAction, dest="output_dir",
               help="Output directory. This is where the warped input files will be saved.")
p.add_argument('--raw', action="store_true", dest="raw", default=True,
               help="Output masks in aligned space, default is screen space")

args = p.parse_args()

aligned_path = Path(args.aligned_dir)
if not aligned_path.exists():
    raise Exception('Aligned directory not found. Please ensure it exists.')

if args.output_dir:
    mask_output_path = Path(args.output_dir) / 'masks'
else:
    mask_output_path = aligned_path / 'masks'

if not mask_output_path.exists():
    mask_output_path.mkdir(parents=True, exist_ok=True)

def generator():
    for filepath in io.progress_bar_generator(pathex.get_image_paths(aligned_path), "Processing alignments"):
        filepath = Path(filepath)
        yield filepath, DFLIMG.load(filepath)

source_images = []
for aligned_file, dflimg in generator():
    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{aligned_file.name} is not a dfl image file")
        continue

    mask_bgr = LandmarksProcessor.get_cmask(dflimg.get_shape(),dflimg.get_landmarks())

    ''' not implemented
    if not args.raw: # #put points into dst image space
        mat = dflimg.get_image_to_face_mat()
        src = LandmarksProcessor.transform_points(src, mat, invert=True)
    '''

    mask_path = mask_output_path / Path(aligned_file.stem).with_suffix('.png')
    print(mask_path)
    cv2_imwrite(mask_path, mask_bgr*255)
