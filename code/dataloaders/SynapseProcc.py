import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

mask_path = sorted(glob.glob("/home/ubuntu/Downloads/Synapse/Synapse/train/image/*.nii"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("image", "label")
    if os.path.exists(msk_path):
        print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        if image.shape != mask.shape:
            print("Error")
        print(item)
        f = h5py.File(
            '/home/ubuntu/Downloads/SynapseCombined/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
print("Combined the SynapseOld volumes and the segmentation into h5 files")
