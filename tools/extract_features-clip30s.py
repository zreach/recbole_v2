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

todo_list_file = "/user/zhouyz/rec/recbole_v2/dataset/m4a-fil/m4a-fil-msd.txt"
with open(todo_list_file, 'r', encoding='utf-8') as f:
    todo_list = set(line.strip() for line in f if line.strip())

def traverse_and_extract_features(folder_path, output_path, threshold_sec):
    # features_dict = {}
    # names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]

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
                        
                        # 定义切片参数
                        slice_duration_sec = 30
                        slice_samples = slice_duration_sec * sr
                        
                        # 存储每个切片的特征
                        layer_features_list = [[] for _ in range(12)]

                        # 对音频进行切片并处理
                        for i in range(0, len(wav), slice_samples):
                            chunk = wav[i:i + slice_samples]
                            
                            # 如果最后一个切片太短，可以跳过或填充，这里我们直接使用
                            if len(chunk) < 1 * sr: # 跳过短于1秒的片段
                                continue

                            wavs_chunk = torch.tensor(chunk).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = muq(wavs_chunk, output_hidden_states=True)
                            
                            # 提取并保存每个层的隐藏状态
                            for l_idx in range(1, len(output.hidden_states)):
                                # hidden_states[l] shape: (1, T, D)
                                feature_l = output.hidden_states[l_idx].detach().cpu() # Shape: (1, T, 1024)
                                layer_features_list[l_idx-1].append(feature_l)

                        # 如果没有处理任何切片，则跳过
                        if not layer_features_list[0]:
                            print(f"Warning: No valid chunks processed for file {file}. Skipping.")
                            continue

                        # 在时间维度上合并所有切片的特征
                        all_features = []
                        for l_idx in range(len(layer_features_list)):
                            # 拼接当前层所有切片的特征
                            # layer_features_list[l_idx] is a list of tensors with shape (1, T_chunk, D)
                            concatenated_features_l = torch.cat(layer_features_list[l_idx], dim=1) # Shape: (1, T_total, D)
                            
                            # 在时间维度上取平均
                            mean_feature_l = concatenated_features_l.mean(dim=1).squeeze().numpy() # Shape: (D,)
                            all_features.append(mean_feature_l)

                        all_features = np.array(all_features)
                        assert all_features.shape == (12, 1024), f"Unexpected feature shape: {all_features.shape}"
                        
                        output_file = os.path.join(output_path, file_prefix + '.npy')
                        np.save(output_file, all_features)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

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