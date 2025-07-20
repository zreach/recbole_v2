import torch, librosa
from muq import MuQMuLan

device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", )
mulan = mulan.to(device).eval()

import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import pickle
import json
import pandas as pd

data = []
# 假设列名为 col1, col2, col3, col4, tags
# 您可以根据实际情况修改列名

with open("/user/zhouyz/rec/recbole_v2/dataset/lfm2b/tags_all_music.tsv", 'r', encoding='utf-8') as f:
    column_names = f.readline().strip().split('\t')[:4] + ['tags']
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            # 前四列为固定列
            row_data = parts[:4]
            # 第五列及之后的所有部分合并为tags列表
            tags = parts[4:]
            row_data.append(tags)
            data.append(row_data)

tags_df = pd.DataFrame(data, columns=column_names)


output_path = '/user/zhouyz/rec/recbole_v2/dataset/lfm2b/lfm2b-muqmulan-texts/'
os.makedirs(output_path, exist_ok=True)
for row in tqdm(tags_df.iterrows(), total=len(tags_df)):
    track_id = row[1]['track_id']
    tags = row[1]['tags']

    texts = [', '.join(tags)]
    with torch.no_grad():
        text_embeds = mulan(texts = texts)
    text_embeds = text_embeds.cpu().numpy().squeeze(0)
    assert text_embeds.shape == (512,), f"Expected shape (1024,), got {text_embeds.shape}"
    output_file = os.path.join(output_path, track_id + '.npy')
    np.save(output_file, text_embeds)