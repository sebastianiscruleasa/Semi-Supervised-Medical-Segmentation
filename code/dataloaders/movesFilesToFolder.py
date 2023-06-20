import os
import shutil

import numpy as np
import nibabel as nib


def convert_npy_to_nii(npy_file):
    case_number = npy_file.split("/")[-2].replace("case", "")
    slice_number = case_number.split("slice")[-1]
    case_number = case_number.split("slice")[0].replace("_", "")

    file_name = npy_file.split("/")[-1]
    aux = npy_file.split("/")[-2]
    if "label" in file_name:
        new_file_name = npy_file.replace(file_name, "case{}_slice{}_label.nii".format(case_number, slice_number))
        new_file_name = new_file_name.replace(aux + "/", "label/")
    else:
        new_file_name = npy_file.replace(file_name, "case{}_slice{}_image.nii".format(case_number, slice_number))
        new_file_name = new_file_name.replace(aux + "/", "image/")


    # Load the data from the .npy file
    data = np.load(npy_file)

    # Create a NIfTI image object from the data
    nifti_img = nib.Nifti1Image(data, affine=np.eye(4))

    # Save the NIfTI image to a .nii file
    nib.save(nifti_img, new_file_name)

    print("NIfTI file created:", new_file_name)


def convert_all_images(path):
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith(".npy"):
                convert_npy_to_nii(file_path)


if __name__ == "__main__":
    convert_all_images("/home/ubuntu/Downloads/Synapse/Synapse/train")
