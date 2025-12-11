#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:Hongxu_ICDM
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: user_item.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2020/05/04


"""

# import pandas as pd
# from collections import defaultdict, Counter
# import argparse
# import json
# import pickle
# import networkx as nx
# from random import choices
# from pathlib import Path


# def process_initial_ratings(args):
#     ratings = pd.read_csv(args.ratings_csv, header=None, sep=',')

#     # print(len(set(ratings[0])))
#     # print(len(set(ratings[1])))
#     topk = args.topk

#     """
#     music
#     339231 users    get 1450
#     83046   items   get 11457

#     automotive
#     851418 users    get 4600
#     320112 items

#     toy
#     1342911 users
#     327698 items
#     """

#     # get top 2200 user who are of high frequency.
#     c = Counter(ratings[0])
#     most_user = c.most_common(1450)  # each user has more than 12 items.
#     # print(most_user)

#     selected_users = [i[0] for i in most_user]

#     # items of selected_users
#     item_filter = ratings[ratings[0].isin(selected_users)]
#     c_item = Counter(item_filter[1])
#     most_item = c_item.most_common()
#     # print(most_item)
#     selected_items = [i[0] for i in most_item]

#     small_ratings = ratings[(ratings[0].isin(selected_users)) & (ratings[1].isin(selected_items))]
#     # print(small_ratings)

#     re_user = Counter(small_ratings[0])

#     # all user, and all its items, only select first 12 items.
#     selected_ratings = pd.DataFrame()
#     for userid in re_user:
#         user_items = small_ratings[(small_ratings[0] == userid)]
#         user_items.columns = ['userid', 'itemid', 'ratings', 'timestamp']
#         if len(user_items) <= topk:
#             selected_ratings = pd.concat([selected_ratings, new_user_items], ignore_index=True)
#             continue
#         new_user_items = user_items.nlargest(topk, 'timestamp')
#         selected_ratings = pd.concat([selected_ratings, new_user_items], ignore_index=True)

#     re_user = Counter(selected_ratings['userid'])
#     # print(len(re_user))
#     re_item = Counter(selected_ratings['itemid'])
#     # print(len(re_item))

#     selected_ratings.to_csv(args.output_ratings_csv, header=None, index=None, sep=',')
#     print(f'saved to {args.output_ratings_csv}')

# def get_item_meta(args):
#     item_category = []
#     item_brand = []
#     item_item = []
#     # with open('./old/meta_Musical_Instruments.json', 'r') as f:
#     metafile = args.metafile
#     with open(metafile, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line_json = json.loads(line)
#             if 'category' in line_json.keys():
#                 for cate in line_json['category']:
#                     item_category.append([line_json['asin'], 'c_'+cate])
#             if 'brand' in line_json.keys():
#                 item_brand.append([line_json['asin'], 'b_'+line_json['brand']])

#             if 'also_buy' in line_json.keys():
#                 for also_item in line_json['also_buy']:
#                     item_item.append([line_json['asin'], also_item])

#     item_category_df = pd.DataFrame(item_category)
#     item_brand_df = pd.DataFrame(item_brand)
#     item_item_df = pd.DataFrame(item_item)
#     outputfolder = args.outputoldfolder
#     ic_filename = outputfolder + 'item_category.csv'
#     ib_filename = outputfolder + 'item_brand.csv'
#     ii_filename = outputfolder + 'item_item.csv'
#     item_category_df.to_csv(ic_filename, header=None, index=None, sep=',')
#     item_brand_df.to_csv(ib_filename, header=None, index=None, sep=',')
#     item_item_df.to_csv(ii_filename, header=None, index=None, sep=',')
#     print(f'saved items meta to folder: {outputfolder}')

# def form_ids(args):

#     """
#     refine item_category.csv,item_brand.csv,item_item.csv according to user_rate_item.csv
#     """
#     outputfolder = args.outputoldfolder
#     ic_filename = outputfolder + 'item_category.csv'
#     ib_filename = outputfolder + 'item_brand.csv'
#     ii_filename = outputfolder + 'item_item.csv'
#     user_rate_item_df = pd.read_csv(args.output_ratings_csv, header=None, sep=',')
#     print(user_rate_item_df)
#     print(user_rate_item_df.head())
#     user_set = set(user_rate_item_df[0])
#     item_set = set(user_rate_item_df[1])

#     print("Number of items in user_rate_item_df:", len(item_set))
#     print("Some item IDs:", list(item_set)[:10])

#     item_category_df = pd.read_csv(ic_filename, header=None, sep=',')
#     item_category_df = item_category_df[item_category_df[0].isin(list(item_set))]

#     print("item_category_df head:")
#     print(item_category_df.head())

#     category_set = set(item_category_df[1])
#     # print(item_category_df)

#     item_brand_df = pd.read_csv(ib_filename, header=None, sep=',')
#     item_brand_df = item_brand_df[item_brand_df[0].isin(list(item_set))]
#     brand_set = set(item_brand_df[1])
#     # print(item_brand_df)

#     print("item_brand_df head:")
#     print(item_brand_df.head())

#     item_item_df = pd.read_csv(ii_filename, header=None, sep=',')
#     item_item_df = item_item_df[(item_item_df[0].isin(list(item_set))) & (item_item_df[1].isin(list(item_set)))]
#     # print(item_item_df)

#     refinefolder = args.refinefolder
#     ic_refine = refinefolder + 'item_category_refine.csv'
#     ib_refine = refinefolder + 'item_brand_refine.csv'
#     ii_refine = refinefolder + 'item_item_refine.csv'

#     item_category_df.to_csv(ic_refine, header=None, index=None, sep=',')
#     item_brand_df.to_csv(ib_refine, header=None, index=None, sep=',')
#     item_item_df.to_csv(ii_refine, header=None, index=None, sep=',')

#     """
#     generate maps, for further using
#     """
#     name2id = defaultdict(int)
#     id2name = defaultdict(str)
#     name2type = defaultdict(str)
#     id2type = defaultdict(str)
#     type2name = defaultdict(list)
#     type2id = defaultdict(list)

#     allnodes = list(user_set) + list(item_set) + list(category_set) + list(brand_set)

#     """
#     ('Players', 2), ('Latin Percussion', 2) category_set,brand_set
#     """
#     # print('Latin Percussion' in category_set)
#     # print('Latin Percussion' in brand_set)
#     cc = Counter(allnodes)
#     print(cc.most_common())
#     print('all', len(allnodes))
#     print(len(set(allnodes)))
#     print('item_set', len(item_set))
#     print('category_set', len(category_set))
#     print('brand_set', len(brand_set))

#     i = 0
#     for name in user_set:
#         name2id[name] = i
#         id2name[i] = name
#         name2type[name] = 'user'
#         id2type[i] = 'user'
#         type2name['user'].append(name)
#         type2id['user'].append(i)
#         i = i + 1

#     for name in item_set:
#         name2id[name] = i
#         id2name[i] = name
#         name2type[name] = 'item'
#         id2type[i] = 'item'
#         type2name['item'].append(name)
#         type2id['item'].append(i)
#         i = i + 1
#     for name in category_set:
#         name2id[name] = i
#         id2name[i] = name
#         name2type[name] = 'category'
#         id2type[i] = 'category'
#         type2name['category'].append(name)
#         type2id['category'].append(i)
#         i = i + 1
#     for name in brand_set:
#         name2id[name] = i
#         id2name[i] = name
#         name2type[name] = 'brand'
#         id2type[i] = 'brand'
#         type2name['brand'].append(name)
#         type2id['brand'].append(i)
#         i = i + 1

#     name2idfile = refinefolder + 'map.name2id'
#     id2namefile = refinefolder + 'map.id2name'
#     name2typefile = refinefolder + 'map.name2type'
#     id2typefile = refinefolder + 'map.id2type'
#     type2namefile = refinefolder + 'map.type2name'
#     type2idfile = refinefolder + 'map.type2id'
#     pickle.dump(name2id, open(name2idfile, 'wb'))
#     pickle.dump(id2name, open(id2namefile, 'wb'))
#     pickle.dump(name2type, open(name2typefile, 'wb'))
#     pickle.dump(id2type, open(id2typefile, 'wb'))
#     pickle.dump(type2name, open(type2namefile, 'wb'))
#     pickle.dump(type2id, open(type2idfile, 'wb'))

#     """
#     generate relation file, using new ids
#     """

#     ic_relation = refinefolder + 'item_category.relation'
#     ib_relation = refinefolder + 'item_brand.relation'
#     ii_relation = refinefolder + 'item_item.relation'
#     ui_relation = refinefolder + 'user_item.relation'
#     item_category = []
#     item_brand = []
#     item_item = []
#     user_item = []  # user_id, item_id, timestamp
#     for _, row in item_category_df.iterrows():
#         item_id = name2id[row[0]]
#         category_id = name2id[row[1]]
#         item_category.append([item_id, category_id])
#     item_category_relation = pd.DataFrame(item_category)
#     item_category_relation.to_csv(ic_relation, header=None, index=None, sep=',')

#     for _, row in item_brand_df.iterrows():
#         item_id = name2id[row[0]]
#         brand_id = name2id[row[1]]
#         item_brand.append([item_id, brand_id])
#     item_brand_relation = pd.DataFrame(item_brand)
#     item_brand_relation.to_csv(ib_relation, header=None, index=None, sep=',')

#     for _, row in item_item_df.iterrows():
#         item1_id = name2id[row[0]]
#         item2_id = name2id[row[1]]
#         item_item.append([item1_id, item2_id])
#     item_item_relation = pd.DataFrame(item_item)
#     item_item_relation.to_csv(ii_relation, header=None, index=None, sep=',')

#     for _, row in user_rate_item_df.iterrows():
#         user_id = name2id[row[0]]
#         item_id = name2id[row[1]]
#         timestamp = int(row[3])
#         user_item.append([user_id, item_id, timestamp])
#     user_item_relation = pd.DataFrame(user_item)
#     user_item_relation.to_csv(ui_relation, header=None, index=None, sep=',')
#     print(f'generic id finish')
#     return len(allnodes)


# def gen_graph(args, num_nodes):
#     refinefolder = args.refinefolder
#     ic_relation = refinefolder + 'item_category.relation'
#     ib_relation = refinefolder + 'item_brand.relation'
#     ii_relation = refinefolder + 'item_item.relation'
#     ui_relation = refinefolder + 'user_item.relation'
#     item_brand = pd.read_csv(ib_relation, header=None, sep=',')
#     item_category = pd.read_csv(ic_relation, header=None, sep=',')
#     item_item = pd.read_csv(ii_relation, header=None, sep=',')
#     user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

#     number_nodes = num_nodes

#     G = nx.Graph()
#     G.add_nodes_from(list(range(number_nodes)))
#     G.add_edges_from(item_brand.to_numpy())
#     G.add_edges_from(item_category.to_numpy())
#     G.add_edges_from(item_item.to_numpy())
#     G.add_edges_from(user_item.to_numpy())
#     print(len(G.edges))

#     databasefolder = args.databasefolder
#     graphfile = databasefolder + 'graph.nx'
#     pickle.dump(G, open(graphfile, 'wb'))

# def gen_ui_history(args):
#     refinefolder = args.refinefolder
#     ui_relation = refinefolder + 'user_item.relation'
#     user_item_relation = pd.read_csv(ui_relation, header=None, sep=',')
#     # print(user_item_relation)
#     users = set(user_item_relation[0])
#     # print(users)
#     user_history_file = args.output_userhistory_folder + 'user_history.txt'
#     with open(user_history_file, 'w') as f:
#         for user in users:
#             this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)
#             path = [user] + this_user[1].tolist()
#             for s in path:
#                 f.write(str(s) + ' ')
#             f.write('\n')

#     edges = set()
#     with open(user_history_file, 'r') as f:
#         for line in f.readlines():
#             s = line.split()
#             uid = s[0]
#             node_list = [int(x) for x in s[1:]]
#             for i in range(len(node_list) - 1):
#                 if node_list[i] <= node_list[i + 1]:
#                     t = (node_list[i], node_list[i + 1])
#                 else:
#                     t = (node_list[i + 1], node_list[i])
#                 edges.add(t)

#     print('edges: ',len(edges))
#     edges_id = defaultdict(int)
#     id_edges = defaultdict(tuple)
#     for i, edge in enumerate(edges):
#         edges_id[edge] = i
#         id_edges[i] = edge
#     edges2id_file = args.output_userhistory_folder + 'user_history.edges2id'
#     id2edges_file = args.output_userhistory_folder + 'user_history.id2edges'
#     pickle.dump(edges_id, open(edges2id_file, 'wb'))
#     pickle.dump(id_edges, open(id2edges_file, 'wb'))

#     edge_path = []

#     with open(user_history_file, 'r') as f:
#         for line in f.readlines():
#             path = []
#             node_list = [int(x) for x in line.split()[1:]]
#             for i in range(len(node_list) - 1):
#                 if node_list[i] <= node_list[i + 1]:
#                     t = (node_list[i], node_list[i + 1])
#                 else:
#                     t = (node_list[i + 1], node_list[i])
#                 path.append(edges_id[t])
#             edge_path.append(path)

#     user_history_edge_path_file = args.output_userhistory_folder + 'user_history_edge_path.txt'
#     with open(user_history_edge_path_file, 'w') as f:
#         for path in edge_path:
#             print(len(path), path)
#             for s in path:
#                 print(s)
#                 f.write(str(s) + ' ')
#             f.write('\n')

# def split_train_test(args):
#     bridge = 2
#     train = 4
#     test = 6
#     testing = []
#     training = []
#     user_history_file = args.output_userhistory_folder + 'user_history.txt'
#     with open(user_history_file, 'r') as f:  # 2200
#         for line in f:
#             s = line.split()
#             uid = int(s[0])
#             item_history = [int(x) for x in s[1:]]
#             if len(item_history) < (bridge + train):
#                 continue
#             else:
#                 # 划分 train 和 test
#                 training.append([uid] + item_history[bridge:(bridge + train)])
#                 testing.append([uid] + item_history[train:])
#     training_file = args.databasefolder + 'training'
#     testing_file = args.databasefolder + 'testing'
#     pickle.dump(training, open(training_file, 'wb'))
#     pickle.dump(testing, open(testing_file, 'wb'))

# def neg_sample(args):
#     NEGS = [5, 100, 500]
#     refinefolder = args.refinefolder
#     type2idfile = refinefolder + 'map.type2id'
#     type2id = pickle.load(open(type2idfile, 'rb'))
#     all_items = set(type2id['item'])

#     training_file = args.databasefolder + 'training'
#     testing_file = args.databasefolder + 'testing'
#     training = pickle.load(open(training_file, 'rb'))  # uid, item1, item2,
#     testing = pickle.load(open(testing_file, 'rb'))

#     user_history_dic = defaultdict(list)
#     user_history_file = args.output_userhistory_folder + 'user_history.txt'
#     with open(user_history_file) as f:
#         for line in f:
#             s = line.split()
#             uid = int(s[0])
#             user_history_dic[uid] = [int(item) for item in s[1:]]

#     for NEG in NEGS:
#         training_link = []
#         for user_record in training:
#             uid = user_record[0]
#             positive = [[uid, item, 1] for item in user_record[1:]]
#             bought = set(user_history_dic[uid])
#             remain = list(all_items.difference(bought))
#             negative = [[uid, item, 0] for item in choices(remain, k=len(positive) * NEG)]
#             training_link = training_link + positive + negative

#         training_link_tf = pd.DataFrame(training_link)
#         training_link_file = args.databasefolder +'links/training_neg_' + str(NEG) + '.links'
#         training_link_tf.to_csv(training_link_file, header=None, index=None, sep=',')

#         test_link = []
#         for user_record in testing:
#             uid = user_record[0]
#             positive = [[uid, item, 1] for item in user_record[1:]]
#             bought = set(user_history_dic[uid])
#             remain = list(all_items.difference(bought))
#             negative = [[uid, item, 0] for item in choices(remain, k=len(positive) * NEG)]
#             test_link = test_link + positive + negative

#         test_link_tf = pd.DataFrame(test_link)
#         testing_link_file = args.databasefolder +'links/testing_neg_' + str(NEG) + '.links'
#         test_link_tf.to_csv(testing_link_file, header=None, index=None, sep=',')
#         print(f'save neg {NEG} sampled links ... finish')


# if __name__ == '__main__':
#     # please remember to change most_common(2200)
#     parser = argparse.ArgumentParser(description='process initial ratings.csv, \n'
#                                                  '1. get users bought k=12 items\n'
#                                                  '2. based on 1, get top u users\n')
#     parser.add_argument('databasefolder', type=str, default='./Amazon_Music/', nargs='?',
#                         help='this data base folder')
#     parser.add_argument('ratings_csv', type=str, default='./Amazon_Music/old/ratings_Musical_Instruments.csv', nargs='?',
#                         help='the initial ratings amazon csv file to process')
#     parser.add_argument('topk', type=int, default=12, nargs='?',
#                         help='get users bought k=12 items')
#     parser.add_argument('output_ratings_csv', type=str, default="./Amazon_Music/old/user_rate_item.csv", nargs='?',
#                         help='processed filename, eg: only get users bought more than 12 items.')
#     # item meta
#     parser.add_argument('metafile', type=str, default='./Amazon_Music/old/meta_Musical_Instruments.json', nargs='?',
#                         help='downloaded meta json filename')
#     parser.add_argument('outputoldfolder', type=str, default='./Amazon_Music/old/', nargs='?',
#                         help='output to an old folder')
#     # form ids
#     parser.add_argument('refinefolder', type=str, default='./Amazon_Music/refine/', nargs='?',
#                         help='output to refine folder')
#     # gen_ui_history
#     parser.add_argument('output_userhistory_folder', type=str, default='./Amazon_Music/path/user_history/', nargs='?',
#                         help='generate user history items file')

#     args = parser.parse_args()

#     Path(args.refinefolder).mkdir(parents=True, exist_ok=True)
#     Path(args.output_userhistory_folder).mkdir(parents=True, exist_ok=True)
#     Path(args.databasefolder +'links/').mkdir(parents=True, exist_ok=True)

#     # 1. get users that have > 12 items; get these users and related top 12 items
#     process_initial_ratings(args)

#     # 2. get items metas
#     get_item_meta(args)

#     # # 3. form these nodes to generic ids
#     allnodes = form_ids(args)

#     # 4. generate graph
#     gen_graph(args, allnodes)

#     # 5. user history
#     gen_ui_history(args)

#     # 6. split_train_test
#     split_train_test(args)

#     # 7. negative sample
#     neg_sample(args)

import pandas as pd
from collections import defaultdict, Counter
import argparse
import json
import pickle
import networkx as nx
from random import choices
from pathlib import Path

def process_initial_ratings(args):
    if Path(args.output_ratings_csv).exists():
        print(f"{args.output_ratings_csv} already exists, skipping processing initial ratings.")
        return
    ratings = pd.read_csv(args.ratings_csv, header=None, sep=',')
    topk = args.topk
    c = Counter(ratings[0])
    most_user = c.most_common(1450)
    selected_users = [i[0] for i in most_user]
    item_filter = ratings[ratings[0].isin(selected_users)]
    c_item = Counter(item_filter[1])
    selected_items = [i[0] for i in c_item.most_common()]
    small_ratings = ratings[(ratings[0].isin(selected_users)) & (ratings[1].isin(selected_items))]
    re_user = Counter(small_ratings[0])
    selected_ratings = pd.DataFrame()
    for userid in re_user:
        user_items = small_ratings[small_ratings[0] == userid]
        user_items.columns = ['userid', 'itemid', 'ratings', 'timestamp']
        if len(user_items) <= topk:
            new_user_items = user_items
        else:
            new_user_items = user_items.nlargest(topk, 'timestamp')
        selected_ratings = pd.concat([selected_ratings, new_user_items], ignore_index=True)
    selected_ratings.to_csv(args.output_ratings_csv, header=None, index=None, sep=',')
    print(f'saved to {args.output_ratings_csv}')


def get_item_meta(args):
    ic_file = Path(args.outputoldfolder) / 'item_category.csv'
    ib_file = Path(args.outputoldfolder) / 'item_brand.csv'
    ii_file = Path(args.outputoldfolder) / 'item_item.csv'
    if ic_file.exists() and ib_file.exists() and ii_file.exists():
        print("Item meta files already exist, skipping get_item_meta.")
        return

    item_category, item_brand, item_item = [], [], []
    metafile = args.metafile
    with open(metafile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        if 'category' in line_json.keys():
            for cate in line_json['category']:
                item_category.append([line_json['asin'], 'c_' + cate])
        if 'brand' in line_json.keys():
            item_brand.append([line_json['asin'], 'b_' + line_json['brand']])
        if 'also_buy' in line_json.keys():
            for also_item in line_json['also_buy']:
                item_item.append([line_json['asin'], also_item])

    item_category_df = pd.DataFrame(item_category)
    item_brand_df = pd.DataFrame(item_brand)
    item_item_df = pd.DataFrame(item_item)
    outputfolder = args.outputoldfolder
    item_category_df.to_csv(Path(outputfolder) / 'item_category.csv', header=None, index=None, sep=',')
    item_brand_df.to_csv(Path(outputfolder) / 'item_brand.csv', header=None, index=None, sep=',')
    item_item_df.to_csv(Path(outputfolder) / 'item_item.csv', header=None, index=None, sep=',')
    print(f'saved items meta to folder: {outputfolder}')


def form_ids(args):
    ic_refine = Path(args.refinefolder) / 'item_category_refine.csv'
    ib_refine = Path(args.refinefolder) / 'item_brand_refine.csv'
    ii_refine = Path(args.refinefolder) / 'item_item_refine.csv'
    if ic_refine.exists() and ib_refine.exists() and ii_refine.exists():
        print("Refine files exist, loading mappings...")
        name2id = pickle.load(open(Path(args.refinefolder) / 'map.name2id', 'rb'))
        id2name = pickle.load(open(Path(args.refinefolder) / 'map.id2name', 'rb'))
        name2type = pickle.load(open(Path(args.refinefolder) / 'map.name2type', 'rb'))
        id2type = pickle.load(open(Path(args.refinefolder) / 'map.id2type', 'rb'))
        type2name = pickle.load(open(Path(args.refinefolder) / 'map.type2name', 'rb'))
        type2id = pickle.load(open(Path(args.refinefolder) / 'map.type2id', 'rb'))
        allnodes = len(name2id)
        return allnodes

    # 否则生成 refine 文件和映射
    outputfolder = args.outputoldfolder
    ic_filename = Path(outputfolder) / 'item_category.csv'
    ib_filename = Path(outputfolder) / 'item_brand.csv'
    ii_filename = Path(outputfolder) / 'item_item.csv'
    user_rate_item_df = pd.read_csv(args.output_ratings_csv, header=None, sep=',')

    user_set = set(user_rate_item_df[0])
    item_set = set(user_rate_item_df[1])

    # --- item_category ---
    item_category_df = pd.read_csv(ic_filename, header=None, sep=',')
    item_category_df = item_category_df[item_category_df[0].isin(item_set)]
    category_set = set(item_category_df[1])

    # --- item_brand ---
    item_brand_df = pd.read_csv(ib_filename, header=None, sep=',')
    item_brand_df = item_brand_df[item_brand_df[0].isin(item_set)]
    brand_set = set(item_brand_df[1])

    # --- item_item ---
    item_item_df = pd.read_csv(ii_filename, header=None, sep=',')
    item_item_df = item_item_df[(item_item_df[0].isin(item_set)) & (item_item_df[1].isin(item_set))]

    # --- 保存 refine CSV ---
    item_category_df.to_csv(ic_refine, header=None, index=None, sep=',')
    item_brand_df.to_csv(ib_refine, header=None, index=None, sep=',')
    item_item_df.to_csv(ii_refine, header=None, index=None, sep=',')

    # --- 生成 maps ---
    name2id = defaultdict(int)
    id2name = defaultdict(str)
    name2type = defaultdict(str)
    id2type = defaultdict(str)
    type2name = defaultdict(list)
    type2id = defaultdict(list)

    allnodes = list(user_set) + list(item_set) + list(category_set) + list(brand_set)

    i = 0
    for name in user_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'user'
        id2type[i] = 'user'
        type2name['user'].append(name)
        type2id['user'].append(i)
        i += 1
    for name in item_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'item'
        id2type[i] = 'item'
        type2name['item'].append(name)
        type2id['item'].append(i)
        i += 1
    for name in category_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'category'
        id2type[i] = 'category'
        type2name['category'].append(name)
        type2id['category'].append(i)
        i += 1
    for name in brand_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'brand'
        id2type[i] = 'brand'
        type2name['brand'].append(name)
        type2id['brand'].append(i)
        i += 1

    # --- 保存映射 ---
    pickle.dump(name2id, open(Path(args.refinefolder) / 'map.name2id', 'wb'))
    pickle.dump(id2name, open(Path(args.refinefolder) / 'map.id2name', 'wb'))
    pickle.dump(name2type, open(Path(args.refinefolder) / 'map.name2type', 'wb'))
    pickle.dump(id2type, open(Path(args.refinefolder) / 'map.id2type', 'wb'))
    pickle.dump(type2name, open(Path(args.refinefolder) / 'map.type2name', 'wb'))
    pickle.dump(type2id, open(Path(args.refinefolder) / 'map.type2id', 'wb'))

    return len(allnodes)


def gen_graph(args, num_nodes):
    refinefolder = Path(args.refinefolder)
    ic_relation = refinefolder / 'item_category.relation'
    ib_relation = refinefolder / 'item_brand.relation'
    ii_relation = refinefolder / 'item_item.relation'
    ui_relation = refinefolder / 'user_item.relation'

    graph_file = Path(args.databasefolder) / 'graph.nx'
    if graph_file.exists():
        print("Graph already exists, skipping gen_graph.")
        return

    item_brand = pd.read_csv(ib_relation, header=None, sep=',')
    item_category = pd.read_csv(ic_relation, header=None, sep=',')
    item_item = pd.read_csv(ii_relation, header=None, sep=',')
    user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(item_brand.to_numpy())
    G.add_edges_from(item_category.to_numpy())
    G.add_edges_from(item_item.to_numpy())
    G.add_edges_from(user_item.to_numpy())
    print(len(G.edges))

    databasefolder = Path(args.databasefolder)
    pickle.dump(G, open(databasefolder / 'graph.nx', 'wb'))


def gen_ui_history(args):
    user_history_file = Path(args.output_userhistory_folder) / 'user_history.txt'
    if user_history_file.exists():
        print("User history already exists, skipping gen_ui_history.")
        return
    refinefolder = Path(args.refinefolder)
    ui_relation = refinefolder / 'user_item.relation'
    user_item_relation = pd.read_csv(ui_relation, header=None, sep=',')
    users = set(user_item_relation[0])
    with open(user_history_file, 'w') as f:
        for user in users:
            this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)
            path = [user] + this_user[1].tolist()
            for s in path:
                f.write(str(s) + ' ')
            f.write('\n')

    # 生成 edges2id 和 id2edges
    edges = set()
    for line in open(user_history_file, 'r'):
        s = line.split()
        node_list = [int(x) for x in s[1:]]
        for i in range(len(node_list) - 1):
            t = tuple(sorted((node_list[i], node_list[i + 1])))
            edges.add(t)

    edges_id = {edge: i for i, edge in enumerate(edges)}
    id_edges = {i: edge for i, edge in enumerate(edges)}
    pickle.dump(edges_id, open(Path(args.output_userhistory_folder) / 'user_history.edges2id', 'wb'))
    pickle.dump(id_edges, open(Path(args.output_userhistory_folder) / 'user_history.id2edges', 'wb'))

    edge_path = []
    for line in open(user_history_file, 'r'):
        node_list = [int(x) for x in line.split()[1:]]
        path = [edges_id[tuple(sorted((node_list[i], node_list[i + 1])))] for i in range(len(node_list) - 1)]
        edge_path.append(path)

    with open(Path(args.output_userhistory_folder) / 'user_history_edge_path.txt', 'w') as f:
        for path in edge_path:
            for s in path:
                f.write(str(s) + ' ')
            f.write('\n')


def split_train_test(args):
    training_file = Path(args.databasefolder) / 'training'
    testing_file = Path(args.databasefolder) / 'testing'
    if training_file.exists() and testing_file.exists():
        print("Train/test split already exists, skipping split_train_test.")
        return
    bridge = 2
    train = 4
    test = 6
    testing = []
    training = []
    user_history_file = Path(args.output_userhistory_folder) / 'user_history.txt'
    for line in open(user_history_file, 'r'):
        s = line.split()
        uid = int(s[0])
        item_history = [int(x) for x in s[1:]]
        if len(item_history) < (bridge + train):
            continue
        training.append([uid] + item_history[bridge:(bridge + train)])
        testing.append([uid] + item_history[train:])
    pickle.dump(training, open(training_file, 'wb'))
    pickle.dump(testing, open(testing_file, 'wb'))


def neg_sample(args):
    NEGS = [5, 100, 500]
    refinefolder = Path(args.refinefolder)
    type2idfile = refinefolder / 'map.type2id'
    type2id = pickle.load(open(type2idfile, 'rb'))
    all_items = set(type2id['item'])

    training_file = Path(args.databasefolder) / 'training'
    testing_file = Path(args.databasefolder) / 'testing'
    training = pickle.load(open(training_file, 'rb'))
    testing = pickle.load(open(testing_file, 'rb'))

    user_history_dic = defaultdict(list)
    user_history_file = Path(args.output_userhistory_folder) / 'user_history.txt'
    for line in open(user_history_file):
        s = line.split()
        uid = int(s[0])
        user_history_dic[uid] = [int(item) for item in s[1:]]

    for NEG in NEGS:
        training_link_file = Path(args.databasefolder) / f'links/training_neg_{NEG}.links'
        testing_link_file = Path(args.databasefolder) / f'links/testing_neg_{NEG}.links'
        if training_link_file.exists() and testing_link_file.exists():
            print(f"Neg samples for {NEG} already exist, skipping...")
            continue

        training_link = []
        for user_record in training:
            uid = user_record[0]
            positive = [[uid, item, 1] for item in user_record[1:]]
            bought = set(user_history_dic[uid])
            remain = list(all_items.difference(bought))
            negative = [[uid, item, 0] for item in choices(remain, k=len(positive) * NEG)]
            training_link += positive + negative
        pd.DataFrame(training_link).to_csv(training_link_file, header=None, index=None, sep=',')

        test_link = []
        for user_record in testing:
            uid = user_record[0]
            positive = [[uid, item, 1] for item in user_record[1:]]
            bought = set(user_history_dic[uid])
            remain = list(all_items.difference(bought))
            negative = [[uid, item, 0] for item in choices(remain, k=len(positive) * NEG)]
            test_link += positive + negative
        pd.DataFrame(test_link).to_csv(testing_link_file, header=None, index=None, sep=',')
        print(f"Saved neg {NEG} sampled links.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process initial ratings.csv and generate graphs.')
    parser.add_argument('databasefolder', type=str, default='./Amazon_Music/', nargs='?')
    parser.add_argument('ratings_csv', type=str, default='./Amazon_Music/old/ratings_Musical_Instruments.csv', nargs='?')
    parser.add_argument('topk', type=int, default=12, nargs='?')
    parser.add_argument('output_ratings_csv', type=str, default="./Amazon_Music/old/user_rate_item.csv", nargs='?')
    parser.add_argument('metafile', type=str, default='./Amazon_Music/old/meta_Musical_Instruments.json', nargs='?')
    parser.add_argument('outputoldfolder', type=str, default='./Amazon_Music/old/', nargs='?')
    parser.add_argument('refinefolder', type=str, default='./Amazon_Music/refine/', nargs='?')
    parser.add_argument('output_userhistory_folder', type=str, default='./Amazon_Music/path/user_history/', nargs='?')

    args = parser.parse_args()

    Path(args.refinefolder).mkdir(parents=True, exist_ok=True)
    Path(args.output_userhistory_folder).mkdir(parents=True, exist_ok=True)
    Path(args.databasefolder + 'links/').mkdir(parents=True, exist_ok=True)

    # 1. get users that have > 12 items
    process_initial_ratings(args)

    # 2. get items metas
    get_item_meta(args)

    # 3. form these nodes to generic ids
    allnodes = form_ids(args)

    # 4. generate graph
    gen_graph(args, allnodes)

    # 5. user history
    gen_ui_history(args)

    # 6. split_train_test
    split_train_test(args)

    # 7. negative sample
    neg_sample(args)

