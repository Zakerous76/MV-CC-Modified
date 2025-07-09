import os
import shutil

# Source and target directories
src_root = './original/LEVIR-MCI-dataset'
dst_root = './LEVIR-MCI-dataset'

# Walk through the source directory
for dirpath, dirnames, filenames in os.walk(src_root):
    # Calculate corresponding destination path
    relative_path = os.path.relpath(dirpath, src_root)
    dst_path = os.path.join(dst_root, relative_path)

    # Create the directory structure
    os.makedirs(dst_path, exist_ok=True)

    # If there are files in the directory, copy the first one
    if filenames:
        for i in range(len(filenames)):
            src_file = os.path.join(dirpath, filenames[i])
            dst_file = os.path.join(dst_path, filenames[i])
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
            if i == 100:
                break


