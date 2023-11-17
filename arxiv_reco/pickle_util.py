import pickle
import os
import glob

def delete_all_pkl_files(directory):
    # Find all .pkl files in the specified directory
    pkl_files = glob.glob(os.path.join(directory, '*.pkl'))

    # Iterate over the list of file paths & remove each file.
    for file_path in pkl_files:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e.strerror}")

def write_temp_data(temp_file, data):
    with open(temp_file, 'wb') as file:
        pickle.dump(data, file)

def read_temp_data(temp_file):
    if os.path.exists(temp_file):
        with open(temp_file, 'rb') as file:
            return pickle.load(file)
    return None

def delete_temp_file(temp_file):
    if os.path.exists(temp_file):
        os.remove(temp_file)


