import os
import numpy as np

from PIL import Image

POST_FIX = 'ABAB'
NUM_IMAEG = 30
DIM = 512
GAP = 16

list_folders = ['/media/ubuntu/Data/unpacked_WholeFace512x512_Cleaned/test',
                '/home/ubuntu/DeepFaceLab_dev/output_video_FP32_rmsprop_17k_'+POST_FIX,
                '/home/ubuntu/DeepFaceLab_dev/output_video_AMP_rmsprop_17k_'+POST_FIX,
                '/home/ubuntu/DeepFaceLab_dev/output_video_AMP_rmsprop_30k_'+POST_FIX,
                '/home/ubuntu/DeepFaceLab_dev/output_video_AMP_adam_17k_'+POST_FIX,
                '/home/ubuntu/DeepFaceLab_dev/output_video_AMP_adam_30k_'+POST_FIX                
                ]

list_formats = ['.jpg',
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
img_out = img_out.save("pretrain_compare_" + POST_FIX + ".png") 
