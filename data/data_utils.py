# encoding=utf-8
import torch
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import os


def load_node_tensor(filename):
    """
    Load node embeddings from pickle and return as a tensor.
    """
    nodewv_dic = pickle.load(open(filename, 'rb'))
    # 直接用列表推导式构造 numpy array，再转 torch.Tensor
    nodewv_tensor = torch.tensor([nodewv_dic[node].numpy() for node in range(len(nodewv_dic))], dtype=torch.float)
    return nodewv_tensor


def instance_paths_to_dict(path_file) -> dict:
    """
    Convert meta-path instances file to a dictionary: {(user,item): [paths]}
    """
    ui_paths_dict = {}
    with open(path_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        ui, pathnum, path_list_str = line.strip().split('\t', 2)
        user, item = map(int, ui.split(','))
        path_list = [p.strip().split(' ') for p in path_list_str.split('\t')]
        ui_paths_dict[(user, item)] = path_list
    return ui_paths_dict


def get_instance_paths(path_file) -> list:
    """
    Load all paths as a flat list
    """
    paths_list = []
    with open(path_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        _, _, path_list_str = line.strip().split('\t', 2)
        for path in path_list_str.split('\t'):
            paths_list.append(path.strip().split(' '))
    return paths_list


def load_ui_seq_relation(uifile):
    """
    Load user -> list of items relation from file
    """
    ui_dict = {}
    with open(uifile, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            ui_dict[user] = items
    return ui_dict


def load_item_item_wv(filename):
    """
    Load item embeddings from word2vec format
    """
    item_item_wv_dic = defaultdict(torch.Tensor)
    with open(filename, 'r') as f:
        f.readline()  # skip header
        for line in f:
            s = line.split()
            item_id = int(s[0])
            fea = list(map(float, s[1:]))
            item_item_wv_dic[item_id] = torch.tensor(fea, dtype=torch.float)
    return item_item_wv_dic


# def load_ui_metapath_instances_emb(metapath_list, ui_metapath_emb_folder, user_num, ui_dict, user_item_direct_emb):
#     """
#     Load UI metapath instance embeddings
#     """
#     ui_metapath = [pickle.load(open(ui_metapath_emb_folder + metapath + '.wv', 'rb'))
#                    for metapath in metapath_list]

#     ui_instances_embs = defaultdict(dict)
#     for u in range(user_num):
#         for i in ui_dict[u]:
#             this_user_ui_instances_embs = []
#             for ele in ui_metapath:
#                 if (u, i) in ele:
#                     this_user_ui_instances_embs.extend(ele[(u, i)])
#             if not this_user_ui_instances_embs:
#                 this_user_ui_instances_embs = [user_item_direct_emb[u].unsqueeze(0)]
#             ui_instances_embs[u][i] = torch.tensor(this_user_ui_instances_embs, dtype=torch.float)
#     return ui_instances_embs
def load_ui_metapath_instances_emb(ui_metapaths_list, metapath_emb_folder, user_num, ui_dict, user_item_direct_emb):
    """
    加载所有用户-物品元路径实例的嵌入
    """
    from collections import defaultdict
    import numpy as np
    import torch
    
    ui_instances_embs = defaultdict(dict)
    
    # 首先，为每个用户-物品对收集所有元路径的嵌入
    for metapath in ui_metapaths_list:
        metapath_file = metapath_emb_folder + metapath + '.wv'
        if not os.path.exists(metapath_file):
            print(f"警告: 元路径文件不存在 {metapath_file}")
            continue
            
        try:
            ui_emb = pickle.load(open(metapath_file, 'rb'))
            print(f"加载 {metapath}.wv 成功，包含 {len(ui_emb)} 个用户-物品对")
        except Exception as e:
            print(f"加载 {metapath_file} 失败: {e}")
            continue
        
        # 为每个用户处理
        for u in range(user_num):
            if u not in ui_dict:
                continue
                
            for i in ui_dict[u]:
                key = (u, i)
                
                # 检查这个键是否存在
                if key in ui_emb:
                    emb_value = ui_emb[key]
                else:
                    # 如果不存在，跳过这个元路径
                    continue
                
                # 确保 emb_value 是列表格式
                if not isinstance(emb_value, list):
                    emb_value = [emb_value]
                
                # 为每个嵌入创建副本并确保是 numpy 数组
                processed_embs = []
                for emb in emb_value:
                    if isinstance(emb, torch.Tensor):
                        emb_np = emb.detach().cpu().numpy()
                    else:
                        emb_np = np.array(emb, dtype=np.float32)
                    
                    # 确保形状是 (1, 100) 而不是 (100,)
                    if emb_np.ndim == 1:
                        emb_np = emb_np.reshape(1, -1)
                    
                    processed_embs.append(emb_np)
                
                # 如果没有处理出任何嵌入，跳过
                if not processed_embs:
                    continue
                
                # 将所有路径堆叠成一个数组
                try:
                    if len(processed_embs) == 1:
                        combined_emb = processed_embs[0]  # 形状 (1, 100)
                    else:
                        combined_emb = np.vstack(processed_embs)  # 形状 (n, 100)
                    
                    # 保存到字典
                    ui_instances_embs[u][key] = combined_emb
                except Exception as e:
                    print(f"处理用户 {u} 物品 {i} 的嵌入时出错: {e}")
    
    # 对于没有找到任何元路径嵌入的用户-物品对，使用直接嵌入作为后备
    for u in range(user_num):
        if u not in ui_dict:
            continue
            
        for i in ui_dict[u]:
            key = (u, i)
            if u not in ui_instances_embs or key not in ui_instances_embs[u]:
                # 尝试使用直接嵌入
                if u in user_item_direct_emb:
                    try:
                        direct_emb = user_item_direct_emb[u]
                        if isinstance(direct_emb, torch.Tensor):
                            direct_emb_np = direct_emb.detach().cpu().numpy()
                        else:
                            direct_emb_np = np.array(direct_emb, dtype=np.float32)
                        
                        if direct_emb_np.ndim == 1:
                            direct_emb_np = direct_emb_np.reshape(1, -1)
                        
                        if u not in ui_instances_embs:
                            ui_instances_embs[u] = {}
                        ui_instances_embs[u][key] = direct_emb_np
                    except Exception as e:
                        print(f"为用户 {u} 物品 {i} 使用直接嵌入时出错: {e}")
    
    # 打印统计信息
    total_pairs = sum(len(items) for items in ui_dict.values())
    loaded_pairs = sum(len(items) for items in ui_instances_embs.values())
    print(f"元路径嵌入加载统计: 总共需要 {total_pairs} 个用户-物品对，成功加载 {loaded_pairs} 个")
    
    return ui_instances_embs



def load_ii_metapath_instances_emb(metapath_emb_folder, user_num, ui_dict, item_item_direct_emb, edges_id_dict):
    """
    Load II metapath instance embeddings
    """
    ii_metapath_emb = pickle.load(open(metapath_emb_folder + 'ii_random_form.wv', 'rb'))
    ii_instances_embs = defaultdict(dict)

    for u in range(user_num):
        items = ui_dict[u]
        for idx in range(len(items) - 1):
            i1, i2 = items[idx], items[idx + 1]
            if (i1, i2) in ii_metapath_emb:
                emb_list = ii_metapath_emb[(i1, i2)]
            else:
                emb_list = [item_item_direct_emb[edges_id_dict[(i1, i2)]].unsqueeze(0)]
            ii_instances_embs[u][(i1, i2)] = torch.tensor(emb_list, dtype=torch.float)
    return ii_instances_embs


def load_train_test_data(train_file, test_file):
    """
    Load train/test CSV files and convert to torch.LongTensor
    """
    train_data = torch.LongTensor(pd.read_csv(train_file, header=None).values)
    test_data = torch.LongTensor(pd.read_csv(test_file, header=None).values)
    return train_data, test_data
