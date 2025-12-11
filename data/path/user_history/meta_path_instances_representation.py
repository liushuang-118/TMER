#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:Hongxu_ICDM
@file: meta_path_instances_representation.py
@time: 2020/06/08
"""
import sys
sys.path.append("../../../")
from gensim.models import Word2Vec
from data.data_utils import *
import torch
from torch import nn
import numpy as np
import pickle
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class Autoencoder(nn.Module):
    def __init__(self, d_in=2000, d_hid=800, d_out=100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(True),
            nn.Linear(d_hid, d_out),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_hid),
            nn.ReLU(True),
            nn.Linear(d_hid, d_in),
            nn.ReLU(True))

    def forward(self, x):
        self.embeddings = self.encoder(x)
        xx = self.decoder(self.embeddings)
        return xx

    def save_embeddings(self):
        return self.embeddings

def instance_emb(metapath_file, output_file):
    print(f"\nProcessing metapath file: {metapath_file}")
    if not Path(metapath_file).exists():
        raise FileNotFoundError(f"Metapath file not found: {metapath_file}")

    walks = get_instance_paths(str(metapath_file))
    path_dict = instance_paths_to_dict(str(metapath_file))
    print(f"Number of walks: {len(walks)}")
    print(f"Number of ui pairs: {len(path_dict)}")

    print("Training Word2Vec model...")
    model = Word2Vec(walks, vector_size=100, window=3, min_count=0, sg=1, hs=1, workers=1)

    # mean pooling to get path embeddings
    ui_path_vectors = {}
    for ui, ui_paths in path_dict.items():
        for path in ui_paths:
            nodes_vectors = []
            for nodeid in path:
                nodes_vectors.append(model.wv[nodeid])
            nodes_np = np.array(nodes_vectors)
            path_vector = np.mean(nodes_np, axis=0)
            if ui not in ui_path_vectors:
                ui_path_vectors[ui] = [path_vector]
            else:
                ui_path_vectors[ui].append(path_vector)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出文件夹
    pickle.dump(ui_path_vectors, open(output_file, 'wb'))
    print(f"Saved embedding file: {output_file}, number of ui keys: {len(ui_path_vectors)}")

if __name__ == '__main__':
    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    ii_metapaths_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']

    # 绝对路径方式，确保不会多一层 data
    base_dir = Path("D:/Thesis_Project/Models/TMER/data/Amazon_Music")
    print("Base directory:", base_dir)
    assert base_dir.exists(), f"Base directory does not exist: {base_dir}"

    # embed ui paths
    for metapath in ui_metapaths_list:
        metapath_file = base_dir / 'path' / 'all_ui_ii_instance_paths' / f'{metapath}.paths'
        output_file = base_dir / 'path' / 'meta_path_instances_representation' / f'{metapath}.wv'
        print(f"\n--- Processing UI metapath: {metapath} ---")
        print("Input file exists?", metapath_file.exists())
        instance_emb(str(metapath_file), str(output_file))

    # embed ii paths
    ii_instance_file = base_dir / 'path' / 'all_ui_ii_instance_paths' / 'ii_random_form.paths'
    output_ii_emb_file = base_dir / 'path' / 'meta_path_instances_representation' / 'ii_random_form.wv'
    print(f"\n--- Processing II metapath ---")
    print("Input file exists?", ii_instance_file.exists())
    instance_emb(str(ii_instance_file), str(output_ii_emb_file))
