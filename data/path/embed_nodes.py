#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:deepwalk-master
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: embed_nodes.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/13 
"""


import random
import os
import sys

from data.path import simple_walks as serialized_walks
from gensim.models import Word2Vec
import pickle
import torch
from collections import defaultdict
from pathlib import Path

# if __name__ == '__main__':

#     # 检查 graph.nx 是否存在
#     import os
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     graph_file = os.path.join(base_dir, '..', 'Amazon_Music', 'graph.nx')

#     print(f"graph.nx 文件路径: {graph_file}")
#     print(f"文件是否存在: {os.path.exists(graph_file)}")
#     if os.path.exists(graph_file):
#         print(f"文件大小: {os.path.getsize(graph_file)} 字节")

#     number_walks = 10
#     walk_length = 6 # length of path
#     workers = 2
#     representation_size = 100
#     window_size = 3
#     output = '../Amazon_Music/node.wv'

#     base_dir = Path(__file__).resolve().parent.parent  # 上一级目录 data
#     graph_file = base_dir / 'Amazon_Music' / 'graph.nx'

#     G = pickle.load(open(graph_file, 'rb'))
#     # G = pickle.load(open('../Amazon_Music/graph.nx', 'rb')) #node 包括 user/item/brand/category/also_bought
#     walks_filebase = "../Amazon_Music/path/node_path/walks.txt"
#     nodewv = '../Amazon_Music/nodewv.dic'
#     print("Number of nodes: {}".format(G.number_of_nodes()))
#     print("Number of edges: {}".format(G.number_of_edges()))
#     print("number_walks: {}".format(number_walks))
#     num_walks = G.number_of_nodes() * number_walks
#     print("Number of walks: {}".format(num_walks))
#     data_size = num_walks * walk_length
#     print("Data size (walks*length): {}".format(data_size))

#     print(type(G))
#     Path("../Amazon_Music/path/node_path").mkdir(parents=True, exist_ok=True)

#     walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
#                                                       path_length=walk_length, num_workers=workers, alpha=0.1,
#                                                       rand=random.Random(100), always_rebuild=True)  # , r=args.r)
#     # walk_files = ["../Amazon_Music/path/node_path/walks.txt.0", "../Amazon_Music/path/node_path/walks.txt.1"]
#     walks = serialized_walks.WalksCorpus(walk_files)


#     print("Training...")
#     # model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
#     #                  workers=workers)

#     model = Word2Vec(
#         sentences=walks,
#         vector_size=representation_size,  # 之前的 size
#         window=window_size,
#         min_count=0,
#         sg=1,
#         hs=1,
#         workers=workers
#     )

#     model.wv.save_word2vec_format(output)

#     nodewv_dic = defaultdict(torch.Tensor)
#     with open(output, 'r') as f:
#         f.readline()
#         for line in f:
#             s = line.split()
#             nodeid = int(s[0])
#             fea = [float(x) for x in s[1:]]
#             nodewv_dic[nodeid] = torch.Tensor(fea)

#     pickle.dump(nodewv_dic, open(nodewv, 'wb'))
    
    

if __name__ == '__main__':
    # 配置参数
    number_walks = 10
    walk_length = 6
    workers = 2
    representation_size = 100
    window_size = 3
    
    # 文件路径
    base_dir = Path(__file__).resolve().parent.parent  # 上一级目录 data
    amazon_dir = base_dir / 'Amazon_Music'
    
    graph_file = amazon_dir / 'graph.nx'
    walks_filebase = amazon_dir / 'path' / 'node_path' / 'walks.txt'
    output_wv = amazon_dir / 'node.wv'
    nodewv_dic_file = amazon_dir / 'nodewv.dic'
    
    print("=" * 60)
    print("DeepWalk 节点嵌入生成")
    print("=" * 60)
    
    # 1. 检查输入文件
    print("\n1. 检查输入文件...")
    print(f"图文件: {graph_file}")
    if not graph_file.exists():
        print(f"❌ 错误: 图文件不存在: {graph_file}")
        print("请先运行 data_process.py 生成图文件")
        sys.exit(1)
    
    # 2. 加载图
    print("\n2. 加载图...")
    try:
        G = pickle.load(open(graph_file, 'rb'))
        print(f"✅ 成功加载图")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")
        
        # 检查图是否是空的
        if G.number_of_edges() == 0:
            print("⚠️ 警告: 图中没有边，随机游走可能无法进行")
            print("检查数据预处理是否正确")
    except Exception as e:
        print(f"❌ 加载图文件失败: {e}")
        sys.exit(1)
    
    # 3. 创建输出目录
    print("\n3. 准备输出目录...")
    walks_dir = amazon_dir / 'path' / 'node_path'
    walks_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建目录: {walks_dir}")
    
    # 4. 生成随机游走
    print("\n4. 生成随机游走...")
    print(f"参数: 每个节点 {number_walks} 条游走, 长度 {walk_length}")
    num_walks = G.number_of_nodes() * number_walks
    data_size = num_walks * walk_length
    print(f"总游走数: {num_walks}")
    print(f"总数据量: {data_size}")
    
    try:
        walk_files = serialized_walks.write_walks_to_disk(
            G, str(walks_filebase), 
            num_paths=number_walks,
            path_length=walk_length, 
            num_workers=workers, 
            alpha=0.1,
            rand=random.Random(100), 
            always_rebuild=True
        )
        print(f"✅ 成功生成随机游走文件")
        
        # 检查生成的文件
        if isinstance(walk_files, list):
            for wf in walk_files:
                if os.path.exists(wf):
                    size = os.path.getsize(wf)
                    print(f"   文件: {wf}, 大小: {size} 字节")
        elif os.path.exists(walks_filebase):
            size = os.path.getsize(walks_filebase)
            print(f"   文件: {walks_filebase}, 大小: {size} 字节")
    except Exception as e:
        print(f"❌ 生成随机游走失败: {e}")
        print("检查 simple_walks 模块是否正确安装")
        sys.exit(1)
    
    # 5. 训练 Word2Vec 模型
    print("\n5. 训练 Word2Vec 模型...")
    try:
        walks = serialized_walks.WalksCorpus(walk_files)
        print("✅ 成功加载游走语料")
        
        model = Word2Vec(
            sentences=walks,
            vector_size=representation_size,
            window=window_size,
            min_count=0,
            sg=1,
            hs=1,
            workers=workers
        )
        print("✅ 成功训练 Word2Vec 模型")
    except Exception as e:
        print(f"❌ 训练 Word2Vec 失败: {e}")
        sys.exit(1)
    
    # 6. 保存模型
    print("\n6. 保存模型...")
    try:
        model.wv.save_word2vec_format(str(output_wv))
        print(f"✅ 保存 Word2Vec 格式到: {output_wv}")
        
        # 转换为字典格式
        nodewv_dic = defaultdict(torch.Tensor)
        with open(output_wv, 'r') as f:
            f.readline()  # 跳过第一行（头信息）
            for line in f:
                s = line.strip().split()
                if len(s) == representation_size + 1:
                    nodeid = int(s[0])
                    fea = [float(x) for x in s[1:]]
                    nodewv_dic[nodeid] = torch.Tensor(fea)
        
        pickle.dump(nodewv_dic, open(nodewv_dic_file, 'wb'))
        print(f"✅ 保存字典格式到: {nodewv_dic_file}")
        print(f"   嵌入向量数: {len(nodewv_dic)}")
        
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 DeepWalk 嵌入生成完成!")
    print("=" * 60)
    
    # 7. 验证生成的文件
    print("\n7. 验证生成的文件:")
    generated_files = [
        (output_wv, "Word2Vec格式嵌入"),
        (nodewv_dic_file, "字典格式嵌入"),
    ]
    
    for file_path, desc in generated_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {desc}: {file_path}, 大小: {size} 字节")
        else:
            print(f"❌ {desc}: 文件不存在")