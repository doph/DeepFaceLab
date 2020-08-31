import os
import numpy as np

from PIL import Image

OUTPUT_PATH = '/home/ubuntu/output_merge_test'
PRE_FIX = 'dst_dst'
NUM_IMAEG = 30
DIM = 512
GAP = 16

list_folders = ['/media/ubuntu/Data/unpacked_WholeFace512x512_Cleaned/test',
                OUTPUT_PATH + '/' + PRE_FIX + '_FP32_local_59k',
                OUTPUT_PATH + '/' + PRE_FIX + '_FP32_local_64k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_rmsprop_62k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_rmsprop_finetune_63k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_rmsprop_102k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_rmsprop_finetune_105k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_adam_59k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_adam_finetune_60k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_adam_107k',
                OUTPUT_PATH + '/' + PRE_FIX +'_AMP_adam_finetune_110k'
                ]

list_formats = ['.jpg',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',
                '.png',                                
                '.png'                
                ]


# create a big canvas
output = np.zeros(((DIM + GAP) * (NUM_IMAEG - 1) + DIM, (DIM + GAP) * (len(list_folders) - 1) + DIM, 3))

for i in range(NUM_IMAEG):
    image_name = str(i).zfill(5)

    for j, (folder, fmt) in enumerate(zip(list_folders, list_formats)):
        image_path = os.path.join(folder, image_name + fmt)

        img = np.asarray(Image.open(image_path))

        output[(DIM + GAP) * i:(DIM + GAP) * i + DIM, (DIM + GAP) * j:(DIM + GAP) * j + DIM, :] = img

img_out = Image.fromarray(output.astype(np.uint8))
img_out = img_out.save(OUTPUT_PATH + "/pretrain_compare_" + PRE_FIX + ".png") 
