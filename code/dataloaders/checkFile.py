import h5py
import nibabel
import numpy as np
import SimpleITK as sitk

hfile = h5py.File("/home/ubuntu/Licenta/Semi Supervised Medical Segmentation/data/Synapse/data/volumes/case0001.h5", "r");
file = hfile["label"]
# Read the dataset into a Numpy array
print(file.shape)
label_data = np.array(file)
print("Label data shape:", label_data.shape)

# # Check if the dataset is full of zeros
# is_full_of_zeros = np.all(label_data == 0)
#
# if is_full_of_zeros:
#     print("The 'label' dataset is full of zeros.")
# else:
#     print("The 'label' dataset contains values other than zero.")

# Close the HDF5 file
hfile.close()
