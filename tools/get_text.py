import torch, librosa
from muq import MuQMuLan
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

m4a_items = pd.read_csv("/user/zhouyz/rec/recbole_v2/dataset/m4a/m4a.item", sep='\t')

output_path = '/user/zhouyz/rec/recbole_v2/dataset/m4a/m4a-muqmulan-texts/'
os.makedirs(output_path, exist_ok=True)
for row in tqdm(m4a_items.iterrows(), total=len(m4a_items)):
    track_id = row[1]['tracks_id:token']
    tags = row[1]['tags:float'].split(',')
    language = row[1]['lang:token']
    texts = [', '.join(tags) + ', ' + language]
    with torch.no_grad():
        text_embeds = mulan(texts = texts)
    text_embeds = text_embeds.cpu().numpy().squeeze(0)
    assert text_embeds.shape == (512,), f"Expected shape (1024,), got {text_embeds.shape}"
    output_file = os.path.join(output_path, track_id + '.npy')
    np.save(output_file, text_embeds)