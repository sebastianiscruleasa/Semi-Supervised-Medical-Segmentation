import nibabel as nib

# Load the NIfTI file
nii_file = nib.load('/home/ubuntu/Downloads/database/training/patient001/patient001_frame12.nii.gz')

# Get the image data array
image_data = nii_file.get_fdata()

# Get the shape of the image data
image_shape = image_data.shape

# Print the shape
print("Image shape:", image_shape)