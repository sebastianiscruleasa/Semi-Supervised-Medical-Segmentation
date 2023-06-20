import os
import shutil

source_folder = '/home/ubuntu/Downloads/BratsEdited'
destination_folder = '/home/ubuntu/Downloads/BratsScripted'
matching_folder = '/home/ubuntu/Licenta/Semi Supervised Medical Segmentation/data/BraTS2019/data'


# Get a list of files in the matching folder
matching_files = [filename.split('.')[0] for filename in os.listdir(matching_folder) if
                    os.path.isfile(os.path.join(matching_folder, filename))]
print(matching_files)
# Iterate over the folders in the source folder
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)

    # Check if the folder name is in the list of matching files
    if folder_name in matching_files:
        destination_path = os.path.join(destination_folder, folder_name)

        # Move the folder to the destination folder
        shutil.move(folder_path, destination_path)
        print(f"Moved folder '{folder_name}' to '{destination_folder}'.")
