import os
import h5py
import nibabel as nib


def combine_nii_files_to_h5(file_name):
    folder_path = "/home/ubuntu/Downloads/LITS17/"
    # Generate the image and label file paths
    image_file = folder_path + "volume" + file_name
    label_file = folder_path + "segmentation" + file_name

    # Read the image and label NIfTI files
    image_data = nib.load(image_file).get_fdata()
    label_data = nib.load(label_file).get_fdata()

    output_file = "/home/ubuntu/Downloads/LITS17Edited/volume" + file_name.replace("nii", "h5")
    # Create a new HDF5 file
    with h5py.File(output_file, 'w') as h5_file:
        # Create datasets for image and label
        h5_file.create_dataset('image', data=image_data)
        h5_file.create_dataset('label', data=label_data)

    print("NIfTI files combined and saved as HDF5:", output_file)


def combine_nii_files_in_folder():
    folder_path = "/home/ubuntu/Downloads/LITS17"
    # Get a list of files in the specified folder
    file_list = os.listdir(folder_path)

    # Process each file in the folder
    for file_name in file_list:
        if file_name.startswith("segmentation"):
            base_name = file_name[12:]

            # Combine the image and label files
            combine_nii_files_to_h5(base_name)


if __name__ == '__main__':
    combine_nii_files_in_folder()