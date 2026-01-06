#encoding=utf-8
import time
import math
from torchnlp.nn import Attention
import torch.utils.data as Data
from rank_metrics import ndcg_at_k
from data.path.path_attention.att import *
from data.data_utils import *
import pickle
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'run.py device: {device}')

class Recommendation(nn.Module):
    def __init__(self, in_features):
        """

        :param in_features: mlp input latent: here 100
        :param outxfeatures:  mlp classification number, here neg+1
        """
        super(Recommendation, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(2, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(2))
        self.in_features = in_features
        self.attention1 = Attention(self.in_features)
        self.attention2 = Attention(self.in_features)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, item_emb, sequence_emb, return_att=False):
        """

        :param sequence_emb
        :return:
        """
        x, weights = self.attention1(item_emb, sequence_emb)
        output = F.linear(x, self.weight, self.bias)
        a, b, c = output.shape
        output = output.reshape((a, c))
        fe = F.log_softmax(output)

        if return_att:
            return fe, weights

        return fe

def instances_slf_att(model, input_tensor, device):
    with torch.no_grad():
        return model(input_tensor.to(device))

# def item_attention(model, item_input, ii_path, device):
#     with torch.no_grad():
#         return model(item_input.to(device), ii_path.to(device))
# 修改 run.py 中的调用
def item_attention(model, item_input, ii_path, device):
    with torch.no_grad():
        # 确保 item_input 是2维的 [batch_size, latent_dim]
        if item_input.dim() == 3:
            item_input = item_input.squeeze(1)
        return model(item_input.to(device), ii_path.to(device))
    
def build_user_history(train_data):
    user_hist = defaultdict(set)
    for u, i, l in train_data:
        if int(l) == 1:
            user_hist[int(u)].add(int(i))
    return user_hist

# def rec_net(train_loader, test_loader, node_emb, sequence_tensor, eval_every=10):
#     if isinstance(node_emb, np.ndarray):
#         node_emb = torch.tensor(node_emb, dtype=torch.float32).to(device)
#     else:
#         node_emb = node_emb.to(device)

#     recommendation = Recommendation(latent_size).to(device)
#     optimizer = torch.optim.Adam(recommendation.parameters(), lr=1e-3)

#     # ===== 构建用户训练历史 =====
#     user_history = build_user_history(train_data)

#     # ===== 构建 test positives（每个用户一个）=====
#     test_pos = defaultdict(list)
#     for u, i, l in test_data:
#         if int(l) == 1:
#             test_pos[int(u)].append(int(i))

#     all_items = list(range(user_num, user_num + item_num))

#     for epoch in range(100):
#         running_loss = 0.0

#         # ===== Training =====
#         for batch in train_loader:
#             batch = batch.long()
#             user_ids = batch[:, 0]
#             item_ids = batch[:, 1]
#             labels = batch[:, 2].to(device)

#             batch_seq = []
#             for u_id in user_ids:
#                 seq_len = sequence_tensor[u_id].shape[0]
#                 batch_seq.append(sequence_tensor[u_id].reshape(1, seq_len, latent_size))
#             batch_seq = torch.cat(batch_seq, dim=0).to(device)

#             batch_item_emb = node_emb[item_ids].unsqueeze(1)

#             optimizer.zero_grad()
#             pred = recommendation(batch_item_emb, batch_seq)
#             loss = F.nll_loss(pred, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         print(f"[Epoch {epoch}] Train loss: {running_loss:.4f}")

#         # ===== Evaluation =====
#         if (epoch + 1) % eval_every != 0:
#             continue

#         hr_10, recall_10, precision_10, ndcg_10 = 0, 0, 0, 0
#         user_cnt = 0

#         for u in test_pos:
#             pos_items = test_pos[u]
#             if len(pos_items) == 0:
#                 continue

#             # 候选集：所有未在训练集中出现的 item
#             candidate_items = [
#                 i for i in all_items if i not in user_history[u]
#             ]

#             scores = []
#             for item_id in candidate_items:
#                 seq_len = sequence_tensor[u].shape[0]
#                 i_emb = node_emb[item_id].reshape(1, 1, latent_size)
#                 seq_emb = sequence_tensor[u].reshape(1, seq_len, latent_size)

#                 with torch.no_grad():
#                     pred = recommendation(i_emb, seq_emb)
#                     score = pred[0, -1].item()

#                 scores.append(score)

#             scores = np.array(scores)
#             ranked_idx = np.argsort(-scores)
#             top10_idx = ranked_idx[:10]
#             top10_items = [candidate_items[i] for i in top10_idx]

#             hit = 0
#             dcg = 0
#             for pos_item in pos_items:
#                 if pos_item in top10_items:
#                     hit = 1
#                     rank = top10_items.index(pos_item)
#                     dcg += 1 / np.log2(rank + 2)

#             hr_10 += hit
#             recall_10 += hit / len(pos_items)
#             precision_10 += hit / 10
#             ndcg_10 += dcg

#             user_cnt += 1

#         hr_10 /= user_cnt
#         recall_10 /= user_cnt
#         precision_10 /= user_cnt
#         ndcg_10 /= user_cnt

#         print(
#             f"[Epoch {epoch}] HR@10: {hr_10:.4f} "
#             f"Recall@10: {recall_10:.4f} "
#             f"Precision@10: {precision_10:.4f} "
#             f"NDCG@10: {ndcg_10:.4f}"
#         )

def rec_net(train_loader, test_loader, node_emb, sequence_tensor, eval_every=10):
    if isinstance(node_emb, np.ndarray):
        node_emb = torch.tensor(node_emb, dtype=torch.float32).to(device)
    else:
        node_emb = node_emb.to(device)

    recommendation = Recommendation(latent_size).to(device)
    optimizer = torch.optim.Adam(recommendation.parameters(), lr=1e-3)

    # ===== 构建用户训练历史 =====
    user_history = build_user_history(train_data)

    # ===== 构建 test positives（一个用户可能多个正样本）=====
    test_pos = defaultdict(list)
    for u, i, l in test_data:
        if int(l) == 1:
            test_pos[int(u)].append(int(i))

    all_items = list(range(user_num, user_num + item_num))

    POS_CLASS = 1   # 正样本类别 index（⚠️非常重要）

    for epoch in range(100):
        running_loss = 0.0

        # ================== Training ==================
        recommendation.train()
        for batch in train_loader:
            batch = batch.long()
            user_ids = batch[:, 0]
            item_ids = batch[:, 1]
            labels = batch[:, 2].to(device)

            batch_seq = []
            for u_id in user_ids:
                seq_len = sequence_tensor[u_id].shape[0]
                batch_seq.append(sequence_tensor[u_id].reshape(1, seq_len, latent_size))
            batch_seq = torch.cat(batch_seq, dim=0).to(device)

            batch_item_emb = node_emb[item_ids].unsqueeze(1)

            optimizer.zero_grad()
            pred = recommendation(batch_item_emb, batch_seq)
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch}] Train loss: {running_loss:.4f}")

        # ================== Evaluation ==================
        if (epoch + 1) % eval_every != 0:
            continue

        recommendation.eval()

        hr_10, recall_10, precision_10, ndcg_10 = 0.0, 0.0, 0.0, 0.0
        user_cnt = 0

        for u in test_pos:
            pos_items = test_pos[u]
            if len(pos_items) == 0:
                continue

            # 候选集：所有用户没在训练集中交互过的 item
            candidate_items = [
                i for i in all_items if i not in user_history[u]
            ]

            scores = []

            seq_len = sequence_tensor[u].shape[0]
            seq_emb = sequence_tensor[u].reshape(1, seq_len, latent_size)

            with torch.no_grad():
                for item_id in candidate_items:
                    i_emb = node_emb[item_id].reshape(1, 1, latent_size)
                    pred = recommendation(i_emb, seq_emb)
                    score = pred[0, POS_CLASS].item()  # ✅ 正类得分
                    scores.append(score)

            scores = np.array(scores)
            ranked_idx = np.argsort(-scores)[:10]
            top10_items = [candidate_items[i] for i in ranked_idx]

            # ===== 多正样本 Top-K 评价 =====
            hits = len(set(pos_items) & set(top10_items))

            hr_10 += 1 if hits > 0 else 0
            recall_10 += hits / len(pos_items)
            precision_10 += hits / 10

            # ===== NDCG@10 =====
            dcg = 0.0
            for pos_item in pos_items:
                if pos_item in top10_items:
                    rank = top10_items.index(pos_item)
                    dcg += 1 / np.log2(rank + 2)

            ideal_hits = min(len(pos_items), 10)
            idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
            ndcg_10 += dcg / idcg if idcg > 0 else 0.0

            user_cnt += 1

        hr_10 /= user_cnt
        recall_10 /= user_cnt
        precision_10 /= user_cnt
        ndcg_10 /= user_cnt

        print(
            f"[Epoch {epoch}] "
            f"HR@10: {hr_10:.5f} | "
            f"Recall@10: {recall_10:.5f} | "
            f"Precision@10: {precision_10:.5f} | "
            f"NDCG@10: {ndcg_10:.5f}"
        )


if __name__ == '__main__':

    att_size = 100
    latent_size = 100
    negative_num = 100
    user_n_items = 4 # for each user, it has n items

    data_name = 'Amazon_Music'
    # split train and test data
    user_history_file = 'data/'+data_name+'/path/user_history/user_history.txt'

    # get train and test links
    N = negative_num  # for each user item, there are N negative samples.
    train_file = 'data/'+data_name+'/links/training_neg_' + str(N) + '.links'
    test_file= 'data/'+data_name+'/links/testing_neg_' + str(N) + '.links'
    train_data, test_data = load_train_test_data(train_file, test_file)

    # load users id and items id
    maptype2id_file = 'data/'+data_name+'/refine/map.type2id'
    type2id = pickle.load(open(maptype2id_file, 'rb'))
    users_list = type2id['user']
    user_num = len(users_list)
    items_list = type2id['item']
    item_num = len(items_list)

    # load node embeds
    node_emb_file = 'data/'+data_name+'/nodewv.dic'
    node_emb = load_node_tensor(node_emb_file)

    # load ui pairs
    ui_dict = load_ui_seq_relation(user_history_file)

    # load all ui embeddings and ii embeddings
    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    ii_metapaths_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
    metapath_emb_folder = 'data/'+data_name+'/path/meta_path_instances_representation/'
    user_item_direct_emb_file = 'data/'+data_name+'/representations/user_item_dic.wv'
    user_item_direct_emb = pickle.load(open(user_item_direct_emb_file, 'rb'))
    item_item_direct_emb_file = 'data/'+data_name+'/path/user_history/item_item.wv'
    item_item_direct_emb = load_item_item_wv(item_item_direct_emb_file)
    ui_all_paths_emb = load_ui_metapath_instances_emb(ui_metapaths_list, metapath_emb_folder, user_num, ui_dict, user_item_direct_emb)
    edges_id_dict_file = 'data/'+data_name+'/path/user_history/user_history.edges2id'
    edges_id_dict = pickle.load(open(edges_id_dict_file, 'rb'))
    ii_all_paths_emb = load_ii_metapath_instances_emb(metapath_emb_folder, user_num, ui_dict, item_item_direct_emb, edges_id_dict)
    labels = train_data[:, 2].to(device)
    print(f'labels.shape: {labels.shape}')
    print('loading node embedding, all user-item and item-item paths embedding...finished')

    # # 1. user-item instances self attention and for each user-item, get one instance embedding.
    # print('start training user-item instance self attention module...')
    # maxpool = Maxpooling()
    # ui_paths_att_emb = defaultdict()
    # t = time.time()

    # # 创建 self-attention 模型实例
    # slf_att_model = Self_Attention_Network(user_item_dim=latent_size).to(device)

    # for u in range(user_num):
    #     if u % 100 == 0:
    #         t_here = time.time() - t
    #         print('user ', u, 'time: ', t_here)

    #     user_item_paths_emb = ui_all_paths_emb[u]
    #     this_user_ui_paths_att_emb = defaultdict()

    #     for i in ui_dict[u]:
    #         paths_emb_list = ui_all_paths_emb[u][(u, i)]
    #         if len(paths_emb_list) == 1:
    #             this_user_ui_paths_att_emb[(u, i)] = torch.tensor(paths_emb_list[0], dtype=torch.float, device=device)
    #         else:
    #             slf_att_input = torch.tensor(paths_emb_list, dtype=torch.float, device=device).unsqueeze(0)
    #             # 调用 self-attention 时传入 model
    #             att_output = instances_slf_att(model=slf_att_model, input_tensor=slf_att_input, device=device)

    #             # max-pooling 取一个实例
    #             get_one_ui = maxpool(att_output).squeeze(0)
    #             this_user_ui_paths_att_emb[(u, i)] = get_one_ui

    #     ui_paths_att_emb[u] = this_user_ui_paths_att_emb

    # ui_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) +'_ui_batch_paths_att_emb.pkl'
    # pickle.dump(ui_paths_att_emb, open(ui_batch_paths_att_emb_pkl_file, 'wb'))


    # # 2. item-item instances slf attention
    # print('start training item-item instance self attention module...')
    # start_t_ii = time.time()
    # ii_paths_att_emb = defaultdict()

    # for u in range(user_num):
    #     if u % 100 == 0:
    #         t_here = time.time() - start_t_ii
    #         print('user ',u, 'time: ',t_here)

    #     item_item_paths_emb = ii_all_paths_emb[u]
    #     num_item = len(ui_dict[u])
    #     this_user_ii_paths_att_emb = defaultdict()

    #     for i_index in range(num_item - 1):
    #         i1 = ui_dict[u][i_index]
    #         i2 = ui_dict[u][i_index + 1]

    #         paths_emb_list = ii_all_paths_emb[u][(i1, i2)]
    #         if len(paths_emb_list) == 1:
    #             this_user_ii_paths_att_emb[(i1, i2)] = torch.tensor(paths_emb_list[0], dtype=torch.float, device=device)
    #         else:
    #             slf_att_input = torch.tensor(paths_emb_list, dtype=torch.float, device=device).unsqueeze(0)
    #             att_output = instances_slf_att(model=slf_att_model, input_tensor=slf_att_input, device=device)
    #             get_one_ii = maxpool(att_output).squeeze(0)
    #             this_user_ii_paths_att_emb[(i1, i2)] = get_one_ii

    #     ii_paths_att_emb[u] = this_user_ii_paths_att_emb

    # ii_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    # pickle.dump(ii_paths_att_emb, open(ii_batch_paths_att_emb_pkl_file, 'wb'))

    # # 3. user and item embedding
    # slf_att_model = Self_Attention_Network(user_item_dim=latent_size).to(device)
    # item_att_model = ItemAttention(latent_dim=latent_size, att_size=100).to(device)

    # # 加载之前生成的路径 embeddings
    # ii_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    # ui_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ui_batch_paths_att_emb.pkl'
    # ii_paths_att_emb = pickle.load(open(ii_batch_paths_att_emb_pkl_file, 'rb'))
    # ui_paths_att_emb = pickle.load(open(ui_batch_paths_att_emb_pkl_file, 'rb'))

    # print('start updating user and item embedding...')
    # start_t_u_i = time.time()
    # sequence_concat = []

    # for u in range(user_num):
    #     if u % 100 == 0:
    #         t_here = time.time() - start_t_u_i
    #         print('user ', u, 'time: ', t_here)

    #     user_sequence_concat = defaultdict()
    #     this_user_ui_paths_dic = ui_paths_att_emb[u]
    #     this_user_ii_paths_dic = ii_paths_att_emb[u]

    #     # user embedding
    #     u_emb = node_emb[u].reshape(1, -1).to(device)  # [1, latent_size]

    #     # ---- First item ----
    #     i1_id = ui_dict[u][0]

    #     # 转成 tensor 并确保 [1, num_paths, latent_size]
    #     u_i1_emb = torch.tensor(this_user_ui_paths_dic[(u, i1_id)], dtype=torch.float32, device=device)
    #     if u_i1_emb.dim() == 1:
    #         u_i1_emb = u_i1_emb.unsqueeze(0).unsqueeze(0)  # [1,1,latent_size]
    #     else:
    #         u_i1_emb = u_i1_emb.unsqueeze(0)  # [1,num_paths,latent_size]

    #     # 当前 item embedding
    #     item1_emb = node_emb[i1_id].reshape(1, -1).to(device)  # [1, latent_dim]
    #     item1_att = item_attention(item_att_model, item1_emb, u_i1_emb, device)  # 输出 [1, latent_dim]

    #     # 合并 user embedding
    #     user_sequence_concat[0] = torch.cat([u_emb, item1_att], dim=0)  # [2, latent_size]

    #     last_item_att = item1_att

    #     # ---- Remaining items ----
    #     for i_index in range(1, user_n_items):
    #         i1 = ui_dict[u][i_index - 1]
    #         i2 = ui_dict[u][i_index]

    #         # 转成 tensor 并确保 [1, num_paths, latent_size]
    #         item_att_input = torch.tensor(this_user_ii_paths_dic[(i1, i2)], dtype=torch.float32, device=device)
    #         if item_att_input.dim() == 1:
    #             item_att_input = item_att_input.unsqueeze(0).unsqueeze(0)  # [1,1,latent_size]
    #         else:
    #             item_att_input = item_att_input.unsqueeze(0)  # [1,num_paths,latent_size]

    #         # attention #1: 上一个 item embedding
    #         ii_1 = item_attention(item_att_model, last_item_att, item_att_input, device)  # [1, latent_dim]

    #         # attention #2: 当前 item embedding
    #         i2_emb_input = node_emb[i2].reshape(1, -1).to(device)  # [1, latent_dim]
    #         ii_2 = item_attention(item_att_model, i2_emb_input, item_att_input, device)  # [1, latent_dim]

    #         # 合并 user + ii_1 + ii_2
    #         user_sequence_concat[i_index] = torch.cat([u_emb, ii_1, ii_2], dim=0)  # [3, latent_size]
    #         last_item_att = ii_2

    #     # concatenate user sequence
    #     sequence_concat.append(torch.cat([user_sequence_concat[i] for i in range(user_n_items)], dim=0))

    # # stack all users
    # sequence_tensor = torch.stack(sequence_concat)  # [user_num, total_seq_len, latent_size]

    # sequence_tensor_pkl_name = data_name + '_' + str(negative_num) + '_sequence_tensor.pkl'
    # pickle.dump(sequence_tensor, open(sequence_tensor_pkl_name, 'wb'))
    # print("Sequence tensor saved:", sequence_tensor_pkl_name)

    # 4. recommendation
    print('start training recommendation module...')
    sequence_tensor_pkl_name = data_name + '_' + str(negative_num) + '_sequence_tensor.pkl'
    sequence_tensor = pickle.load(open(sequence_tensor_pkl_name, 'rb'))
    item_emb = node_emb[user_num:(user_num+item_num),:]
    BATCH_SIZE = 100

    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  #
        num_workers=5,  #
    )
    test_loader = Data.DataLoader(
        dataset=test_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  #
        num_workers=1,  #
    )
    rec_net(train_loader, test_loader, node_emb, sequence_tensor)

