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

def traverse_and_extract_features(folder_path, output_path):
    features_dict = {}
    names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]
    
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                file_prefix = os.path.splitext(file)[0]
                if file_prefix not in names_ready:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        info_json = json.load(f)

                    texts = [','.join(info_json[0]['genres'])]
                    with torch.no_grad():
                        text_embeds = mulan(texts = texts)

                    text_embeds = text_embeds.cpu().numpy().squeeze()
                    assert text_embeds.shape == (512,)
                    # if file_prefix in mapping_dict:
                    #     features_dict[mapping_dict[file_prefix]] = audio_embeds
                    output_file = os.path.join(output_path, file_prefix + '.npy')
                    np.save(output_file, text_embeds)

folder_path = 'rec/MSD/' 
output_path = 'rec/MSD-muqmulan-text/'  
os.makedirs(output_path, exist_ok=True)

traverse_and_extract_features(folder_path, output_path)