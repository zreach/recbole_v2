import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.multiprocessing as mp
from muq import MuQMuLan

def process_chunk(df_chunk, device_id, output_path):
    """
    每个进程处理一个数据块的函数。
    """
    device = f'cuda:{device_id}'
    # 在每个子进程中独立加载模型
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    # 使用 position 参数防止 tqdm 进度条互相覆盖
    for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"GPU {device_id}", position=device_id):
        track_id = row['track_id']
        tags = row['tags']

        texts = [', '.join(tags)]
        with torch.no_grad():
            text_embeds = mulan(texts=texts)
        
        text_embeds = text_embeds.cpu().numpy().squeeze(0)
        
        # 注意：原始代码中的断言 text_embeds.shape == (512,) 可能需要根据模型实际输出调整
        # MuQ-MuLan-large 的输出维度是 512
        if text_embeds.shape != (512,):
             print(f"Warning: Unexpected shape for track {track_id}. Expected (512,), got {text_embeds.shape}")
             continue

        output_file = os.path.join(output_path, str(track_id) + '.npy')
        np.save(output_file, text_embeds)

def main():
    # --- 数据加载 ---
    data = []
    with open("/user/zhouyz/rec/recbole_v2/dataset/lfm2b/tags_all_music.tsv", 'r', encoding='utf-8') as f:
        column_names = f.readline().strip().split('\t')[:4] + ['tags']
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5: # 确保至少有4个固定列和1个tag
                row_data = parts[:4]
                tags = parts[4:]
                row_data.append(tags)
                data.append(row_data)

    tags_df = pd.DataFrame(data, columns=column_names)
    
    output_path = '/user/zhouyz/rec/recbole_v2/dataset/lfm2b/lfm2b-muqmulan-texts/'
    os.makedirs(output_path, exist_ok=True)

    # --- 多进程设置 ---
    try:
        mp.set_start_method('spawn', force=True)
        print("Start method 'spawn' set for multiprocessing.")
    except RuntimeError:
        pass # Start method can only be set once

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Exiting.")
        return
    
    print(f"Found {num_gpus} GPUs. Starting processing...")

    # 将 DataFrame 切分成多个块，每个 GPU 一个
    df_chunks = np.array_split(tags_df, num_gpus)
    
    processes = []
    for i in range(num_gpus):
        if len(df_chunks[i]) > 0:
            p = mp.Process(target=process_chunk, args=(df_chunks[i], i, output_path))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    print("All processes finished.")

if __name__ == '__main__':
    main()