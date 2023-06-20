import os


def create_file_list(folder_path, output_file):
    # Get a list of file names in the specified folder
    file_names = os.listdir(folder_path)

    # Create a new .list file
    with open(output_file, 'w') as file:
        # Write each file name to a separate line
        for name in file_names:
            file.write(name.replace('.h5', "\n"))

    print("File list created:", output_file)


if __name__ == '__main__':
    # Create a file list for the training dataset
    create_file_list("/home/ubuntu/Licenta/Semi Supervised Medical Segmentation/data/Synapse/data/slices", "/home/ubuntu/Licenta/Semi Supervised Medical Segmentation/data/Synapse/train_slices.list")