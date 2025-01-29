import os
import shutil
import random
from math import ceil

# Paths to your datasets
existing_data_path = 'data/chest_xray'
new_data_path = 'new_data/'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Function to split files into train, val, test
def split_files(files, train_ratio, val_ratio, test_ratio):
    random.shuffle(files)
    total = len(files)
    train_end = ceil(total * train_ratio)
    val_end = train_end + ceil(total * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    print(f"Split {total} files into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    return train_files, val_files, test_files

def move_files(file_list, src_folder, dest_folder):
    # Ensure the destination directory exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")
    
    print(f"Moving {len(file_list)} files from {src_folder} to {dest_folder}")
    i=1

    for filename in file_list:
        src_file = os.path.join(src_folder, filename)
        # Generate a unique filename
        unique_filename = f"{i}_{filename}"
        dest_file = os.path.join(dest_folder, unique_filename)
        i=i+1
        try:
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
        except Exception as e:
            print(f"Error copying {src_file} to {dest_file}: {e}")


# Split and distribute the data
def split_and_merge(new_data_path, existing_data_path):

    for category in ['NORMAL', 'PNEUMONIA']:
        src_category_path = os.path.join(new_data_path, category)
        train_dest = os.path.join(existing_data_path, 'train', category)
        val_dest = os.path.join(existing_data_path, 'val', category)
        test_dest = os.path.join(existing_data_path, 'test', category)

        # Debug: Log the paths
        print(f"\nProcessing category: {category}")
        print(f"Source path: {src_category_path}")
        print(f"Train destination: {train_dest}")
        print(f"Validation destination: {val_dest}")
        print(f"Test destination: {test_dest}")

        # Get all files in the new data category folder
        files = os.listdir(src_category_path)
        print(f"Found {len(files)} files in {src_category_path}")
        
        # Split the files
        train_files, val_files, test_files = split_files(files, train_ratio, val_ratio, test_ratio)

        # Move files to the corresponding directories
        move_files(train_files, src_category_path, train_dest)
        move_files(val_files, src_category_path, val_dest)
        move_files(test_files, src_category_path, test_dest)

# Run the process
split_and_merge(new_data_path, existing_data_path)
