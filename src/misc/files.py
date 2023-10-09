import nibabel as nib

def read_image(file_path):
    nii_img = nib.load(file_path)
    image_data = nii_img.get_fdata()
    header = nii_img.header
    return image_data, header

class SetupArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)