import os
from tqdm import tqdm

folder_path = "/data2/zhouyz/rec/MSD_old/muq-last-npy-10/"
output_file = "/data2/zhouyz/rec/MSD_old/already.txt"
prefix_counts = {}
file_list = []

for root, _, files in os.walk(folder_path):
    for file in files:
        file_list.append(os.path.join(root, file))

for file_path in tqdm(file_list, desc="Processing files"):
    if os.path.isfile(file_path):
        filename = os.path.basename(file_path)
        prefix = filename.split('.')[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

with open(output_file, 'w') as f:
    for prefix, count in prefix_counts.items():
        f.write(f"{prefix}\n")
# ...existing code...