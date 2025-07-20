import os
import librosa
import numpy as np
from tqdm import tqdm
import multiprocessing

# 多进程处理单个文件的函数
def process_file(args):
    file, root, output_path, threshold_sec = args
    try:
        if file.endswith('.mp3') or file.endswith('.wav'):
            file_prefix = os.path.splitext(file)[0]
            file_path = os.path.join(root, file)
            wav, sr = librosa.load(file_path, sr=24000)


            fbank_features = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, win_length=960, hop_length=576)  # n_mels为滤波器组数量
            fbank_features_db = librosa.power_to_db(fbank_features, ref=np.max)  # 转换为对数刻度
            fbank_features_mean = np.mean(fbank_features_db, axis=1)

            output_file = os.path.join(output_path, file_prefix + '.npy')
            np.save(output_file, fbank_features_mean)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 多进程遍历和提取特征
def traverse_and_extract_features(folder_paths, output_path, threshold_sec, num_processes=4):
    os.makedirs(output_path, exist_ok=True)
    names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]
    names_ready = set(names_ready)

    # 收集所有待处理的文件
    tasks = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_prefix = os.path.splitext(file)[0]
                    if file_prefix not in names_ready:
                        tasks.append((file, root, output_path, threshold_sec))

    # 使用多进程池处理文件
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_file, tasks), total=len(tasks)))

# 配置参数
folder_paths = ['/user/zhouyz/rec/recbole_v2/dataset/music4all/music4all/audios']
output_path = '/user/zhouyz/rec/recbole_v2/dataset/music4all/music4all/mfcc-mean'
threshold_sec = 360.0
num_processes = 4  # 设置进程数

# 调用多进程函数
traverse_and_extract_features(folder_paths, output_path, threshold_sec, num_processes)