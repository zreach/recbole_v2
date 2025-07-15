import torch, librosa
from muq import MuQ

device = 'cuda'
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import pickle
import argparse

# already_file = '/data2/zhouyz/already.txt'
# with open(already_file, 'r', encoding='utf-8') as f:
#     skip_list = set(line.strip() for line in f if line.strip())

todo_list_file = "/user/zhouyz/rec/recbole_v2/dataset/m4a-fil/m4a-fil-msd.txt"
with open(todo_list_file, 'r', encoding='utf-8') as f:
    todo_list = set(line.strip() for line in f if line.strip())

# print(len(skip_list))
def downsample_avg(arr, factor):
    # 假设T在第二维度
    original_length = arr.shape[1]
    new_length = original_length // factor
    arr_cliped = arr[:, :factor * new_length, :]
    # factor = original_length // new_length
    # assert original_length % new_length == 0, "原长度必须能被目标长度整除"
    new_shape = list(arr_cliped.shape)
    new_shape[1] = new_length
    new_shape.insert(2, factor)
    reshaped = arr_cliped.reshape(new_shape)
    return np.mean(reshaped, axis=2)

# Extract features and save as id:embedding in pkl format
def traverse_and_extract_features(folder_path, output_path, threshold_sec):
    # features_dict = {}
    # names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]
    # # names_ready += skip_list
    # names_ready = set(names_ready)
    # print(len(names_ready))
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.mp3') or file.endswith('.wav'):
                try:
                    file_prefix = os.path.splitext(file)[0]
                    # if file_prefix not in names_ready:
                    if file_prefix in todo_list:
                        file_path = os.path.join(root, file)
                        wav, sr = librosa.load(file_path, sr=24000)
                        duration = librosa.get_duration(y=wav, sr=sr)
        
                        if duration > threshold_sec:
                            # 计算中间段的起始和结束时间
                            start_time = (duration - threshold_sec) / 2
                            end_time = start_time + threshold_sec
                            
                            # 转换为样本索引
                            start_idx = int(start_time * sr)
                            end_idx = int(end_time * sr)
                            
                        #     # 截取中间段
                        #     wav = wav[start_idx:end_idx]
                        wavs = torch.tensor(wav).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = muq(wavs, output_hidden_states=True)
                        
                        all_features = []
                        for l in range(1, len(output.hidden_states)):
                            feature_i = output.hidden_states[l].detach().cpu().numpy().squeeze().mean(axis=0)
                            all_features.append(feature_i)

                        all_features = np.array(all_features)
                        assert all_features.shape == (12, 1024), f"Unexpected feature shape: {all_features.shape}"
                        # print('All features shape: ', all_features.shape) # (13, 1024)
                        
                        output_file = os.path.join(output_path, file_prefix + '.npy')
                        np.save(output_file, all_features)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
    
    # Save features_dict to pkl
    # pkl_file_path = os.path.join(output_path, 'muq-last.pkl')
    # with open(pkl_file_path, 'wb') as f:
    #     pickle.dump(features_dict, f)

# folder_path = ':/data2/zhouyz/rec/MSD'
# output_path = '/data2/zhouyz/MSD/muq-last-npy-10/'
# os.makedirs(output_path, exist_ok=True)

# threshold_sec = 360.0
# selected_layer = 12

# traverse_and_extract_features(folder_path, output_path, threshold_sec)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio features using MuQ model.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing audio files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder for saving features.')
    parser.add_argument('--todo_list_path', type=str, default='/user/zhouyz/rec/recbole_v2/dataset/m4a-fil/m4a-fil-msd.txt', help='Path to the file with a list of items to process.')
    parser.add_argument('--threshold_sec', type=float, default=360.0, help='Threshold in seconds for audio duration.')
    parser.add_argument('--selected_layer', type=int, default=12, help='Selected layer from the model (currently not used in feature extraction logic).')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    traverse_and_extract_features(args.folder_path, args.output_path, args.threshold_sec,)