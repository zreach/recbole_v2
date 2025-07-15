import os
import shutil
import argparse
from tqdm import tqdm

def find_and_copy_missing_files(audio_source_dir, npy_dir, destination_dir, todo_list_path):
    """
    Checks for audio files in a todo list that haven't been processed into .npy files,
    and copies them to a destination directory for further processing.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    print(f"Destination directory '{destination_dir}' is ready.")

    # 1. Read the list of files to be processed
    try:
        with open(todo_list_path, 'r', encoding='utf-8') as f:
            todo_list = set(line.strip() for line in f if line.strip())
        print(f"Found {len(todo_list)} items in the todo list.")
    except FileNotFoundError:
        print(f"Error: Todo list file not found at '{todo_list_path}'")
        return

    # 2. Get the set of already processed files by checking for .npy files
    if not os.path.isdir(npy_dir):
        print(f"Warning: NPY directory not found at '{npy_dir}'. Assuming no files are processed.")
        processed_files = set()
    else:
        processed_files = {os.path.splitext(f)[0] for f in os.listdir(npy_dir) if f.endswith('.npy')}
    print(f"Found {len(processed_files)} already processed .npy files.")

    # 3. Determine which files need to be processed
    files_to_process = todo_list - processed_files
    print(f"Found {len(files_to_process)} files that need to be copied.")

    if not files_to_process:
        print("No new files to copy. Exiting.")
        return

    # 4. Find and copy the corresponding audio files
    copied_count = 0
    print(f"Scanning '{audio_source_dir}' for audio files to copy...")
    for root, _, files in os.walk(audio_source_dir):
        for file in tqdm(files, desc="Copying files"):
            file_prefix = os.path.splitext(file)[0]
            if file_prefix in files_to_process and file.endswith('.mp3'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)
                try:
                    shutil.copy2(source_path, destination_path)
                    copied_count += 1
                    # Remove from set to avoid repeated copies if filenames are not unique across subdirs
                    files_to_process.remove(file_prefix) 
                except Exception as e:
                    print(f"Error copying file {source_path}: {e}")

    print(f"\nFinished copying. Total files copied: {copied_count}")
    if files_to_process:
        print(f"Warning: {len(files_to_process)} files from the todo list were not found in the source directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy audio files that need feature extraction.')
    # parser.add_argument('--audio_source_dir', type=str, required=True, help='Path to the folder containing original audio files.')
    # parser.add_argument('--npy_dir', type=str, required=True, help='Path to the folder containing processed .npy feature files.')
    # parser.add_argument('--destination_dir', type=str, required=True, help='Path to the folder where missing audio files will be copied.')
    # parser.add_argument('--todo_list_path', type=str, required=True, help='Path to the file with a list of items to process.')

    # args = parser.parse_args()

    find_and_copy_missing_files(
        audio_source_dir="/data2/zhouyz/rec/MSD",
        npy_dir="/data2/zhouyz/rec/MSD/muq-alllayers-mean/",
        destination_dir="/data2/zhouyz/rec/MSD/todo",
        todo_list_path="/user/zhouyz/rec/recbole_v2/dataset/m4a-fil/m4a-fil-msd.txt"
    )
