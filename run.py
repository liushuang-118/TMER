#encoding=utf-8
import time
import math
from torchnlp.nn import Attention
import torch.utils.data as Data
from rank_metrics import ndcg_at_k
from data.path.path_attention.att import *
from data.data_utils import *
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'run.py device: {device}')

class Recommendation(nn.Module):
    def __init__(self, in_features):
        """

        :param in_features: mlp input latent: here 100
        :param out_features:  mlp classification number, here neg+1
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

def rec_net(train_loader, test_loader, node_emb, sequence_tensor, eval_every=10):
    if isinstance(node_emb, np.ndarray):
        node_emb = torch.tensor(node_emb, dtype=torch.float32).to(device)
    else:
        node_emb = node_emb.to(device)

    recommendation = Recommendation(latent_size).to(device)
    optimizer = torch.optim.Adam(recommendation.parameters(), lr=1e-3)

    user_recommendations = defaultdict(list)
    user_scores = defaultdict(dict)

    # 准备测试集正负样本
    all_pos, all_neg = [], []
    for idx in range(test_data.shape[0]):
        u, i, l = test_data[idx]
        u, i, l = int(u), int(i), int(l)
        if l == 1:
            all_pos.append((idx, u, i))
        else:
            all_neg.append((idx, u, i))

    for epoch in range(100):
        running_loss = 0.0
        start_train = time.time()

        # ===== Training =====
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

        print(f"Epoch {epoch} loss={running_loss:.4f}, train_time={time.time()-start_train:.2f}s")

        # ===== Evaluation =====
        if (epoch + 1) % eval_every != 0:
            continue

        hr_10, recall_10, precision_10, ndcg_10 = 0.0, 0.0, 0.0, 0.0

        for i, pos_entry in enumerate(all_pos):
            start = N * i
            end = N * i + N
            candidates = all_neg[start:end] + [pos_entry]

            scores = []
            for _, u_id, item_id in candidates:
                seq_len = sequence_tensor[u_id].shape[0]
                i_emb = node_emb[item_id].reshape(1, 1, latent_size)
                seq_emb = sequence_tensor[u_id].reshape(1, seq_len, latent_size)
                pred = recommendation(i_emb, seq_emb)
                score = pred[0, -1].item()
                scores.append(score)

            scores = np.array(scores)
            ranked_idx = np.argsort(-scores)
            top10 = ranked_idx[:10]

            # ===== 保存前20个推荐 =====
            top_k = min(20, len(ranked_idx))
            for rank in range(top_k):
                rec_item_id = candidates[ranked_idx[rank]][2]
                user_id = candidates[ranked_idx[rank]][1]
                user_recommendations[user_id].append(rec_item_id)
                user_scores[user_id][rec_item_id] = scores[ranked_idx[rank]]

            # ===== 计算指标 =====
            pos_idx = len(scores) - 1  # 正样本在最后
            if pos_idx in top10:
                hr_10 += 1
                recall_10 += 1
                precision_10 += 1 / 10
                rank_pos = np.where(top10 == pos_idx)[0][0]
                ndcg_10 += 1 / np.log2(rank_pos + 2)

        total_pos = len(all_pos)
        hr_10 /= total_pos
        recall_10 /= total_pos
        precision_10 /= total_pos
        ndcg_10 /= total_pos

        print(f"Epoch {epoch} | HR@10: {hr_10:.4f} Recall@10: {recall_10:.4f} "
              f"Precision@10: {precision_10:.4f} NDCG@10: {ndcg_10:.4f}")

    # ===== 保存推荐结果和分数 =====
    with open('tmer_user_recommendations.pkl', 'wb') as f:
        pickle.dump(user_recommendations, f)
    with open('tmer_user_scores.pkl', 'wb') as f:
        pickle.dump(user_scores, f)

    print("推荐结果和分数已保存！训练结束！")




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

    # 3. user and item embedding
    slf_att_model = Self_Attention_Network(user_item_dim=latent_size).to(device)
    item_att_model = ItemAttention(latent_dim=latent_size, att_size=100).to(device)

    # 加载之前生成的路径 embeddings
    ii_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    ui_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ui_batch_paths_att_emb.pkl'
    ii_paths_att_emb = pickle.load(open(ii_batch_paths_att_emb_pkl_file, 'rb'))
    ui_paths_att_emb = pickle.load(open(ui_batch_paths_att_emb_pkl_file, 'rb'))

    print('start updating user and item embedding...')
    start_t_u_i = time.time()
    sequence_concat = []

    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - start_t_u_i
            print('user ', u, 'time: ', t_here)

        user_sequence_concat = defaultdict()
        this_user_ui_paths_dic = ui_paths_att_emb[u]
        this_user_ii_paths_dic = ii_paths_att_emb[u]

        # user embedding
        u_emb = node_emb[u].reshape(1, -1).to(device)  # [1, latent_size]

        # ---- First item ----
        i1_id = ui_dict[u][0]

        # 转成 tensor 并确保 [1, num_paths, latent_size]
        u_i1_emb = torch.tensor(this_user_ui_paths_dic[(u, i1_id)], dtype=torch.float32, device=device)
        if u_i1_emb.dim() == 1:
            u_i1_emb = u_i1_emb.unsqueeze(0).unsqueeze(0)  # [1,1,latent_size]
        else:
            u_i1_emb = u_i1_emb.unsqueeze(0)  # [1,num_paths,latent_size]

        # 当前 item embedding
        item1_emb = node_emb[i1_id].reshape(1, -1).to(device)  # [1, latent_dim]
        item1_att = item_attention(item_att_model, item1_emb, u_i1_emb, device)  # 输出 [1, latent_dim]

        # 合并 user embedding
        user_sequence_concat[0] = torch.cat([u_emb, item1_att], dim=0)  # [2, latent_size]

        last_item_att = item1_att

        # ---- Remaining items ----
        for i_index in range(1, user_n_items):
            i1 = ui_dict[u][i_index - 1]
            i2 = ui_dict[u][i_index]

            # 转成 tensor 并确保 [1, num_paths, latent_size]
            item_att_input = torch.tensor(this_user_ii_paths_dic[(i1, i2)], dtype=torch.float32, device=device)
            if item_att_input.dim() == 1:
                item_att_input = item_att_input.unsqueeze(0).unsqueeze(0)  # [1,1,latent_size]
            else:
                item_att_input = item_att_input.unsqueeze(0)  # [1,num_paths,latent_size]

            # attention #1: 上一个 item embedding
            ii_1 = item_attention(item_att_model, last_item_att, item_att_input, device)  # [1, latent_dim]

            # attention #2: 当前 item embedding
            i2_emb_input = node_emb[i2].reshape(1, -1).to(device)  # [1, latent_dim]
            ii_2 = item_attention(item_att_model, i2_emb_input, item_att_input, device)  # [1, latent_dim]

            # 合并 user + ii_1 + ii_2
            user_sequence_concat[i_index] = torch.cat([u_emb, ii_1, ii_2], dim=0)  # [3, latent_size]
            last_item_att = ii_2

        # concatenate user sequence
        sequence_concat.append(torch.cat([user_sequence_concat[i] for i in range(user_n_items)], dim=0))

    # stack all users
    sequence_tensor = torch.stack(sequence_concat)  # [user_num, total_seq_len, latent_size]

    sequence_tensor_pkl_name = data_name + '_' + str(negative_num) + '_sequence_tensor.pkl'
    pickle.dump(sequence_tensor, open(sequence_tensor_pkl_name, 'wb'))
    print("Sequence tensor saved:", sequence_tensor_pkl_name)

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

