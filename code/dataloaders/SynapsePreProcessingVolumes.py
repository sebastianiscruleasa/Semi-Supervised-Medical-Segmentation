import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

slice_num = 0
mask_path = sorted(glob.glob("/home/ubuntu/Downloads/SynapseOficial/RawData/Training/img/*.nii.gz"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("img", "label")
    if os.path.exists(msk_path):
        print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)
        # image = np.resize(image, (image.shape[0], 224, 224))
        # mask = np.resize(mask, (mask.shape[0], 224, 224))
        print(image.shape)
        item = case.split("/")[-1].split(".")[0].replace("img", "case")
        if image.shape != mask.shape:
            print("Error")
        print(item)
        f = h5py.File(
            '/home/ubuntu/Licenta/Semi Supervised Medical Segmentation/data/Synapse/data/volumes/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
print("Converted all Synapse volumes to 2D volumes")
