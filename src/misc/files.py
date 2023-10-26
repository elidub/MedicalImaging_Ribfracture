import nibabel as nib
import numpy as np
import glob

def read_image(file_path):
    nii_img = nib.load(file_path)
    image_data = nii_img.get_fdata()
    header = nii_img.header
    return image_data, header

class SetupArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_boxes(boxes_dir):
    return [np.load(filename)['arr_0'] for filename in glob.iglob(boxes_dir + '**/**/*.npz', recursive=True)]