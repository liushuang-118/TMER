import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path


class ExplanationMetrics:
    def __init__(self, args):
        self.args = args
        self.refinefolder = Path(args['refinefolder'])
        self.explanations_folder = Path(args['explanations_folder'])
        self.recommendations_file = Path(args['recommendations_file'])

        self.load_data()

    # ===============================
    # 数据加载
    # ===============================
    def load_data(self):
        # 1. 用户历史交互
        self.user_item_relation = pd.read_csv(
            self.refinefolder / 'user_item.relation',
            header=None, sep=',',
            names=['user_id', 'item_id', 'timestamp', '_']
        )

        # 2. 节点类型映射
        with open(self.refinefolder / 'map.id2type', 'rb') as f:
            self.id2type = pickle.load(f)

        # 3. 推荐结果（✅ 修正路径）
        self.user_recommendations = self.load_user_recommendations()

        # 4. 实体流行度
        self.entity_popularity, self.entity_popularity_norm = self.calculate_entity_popularity()

        # 5. 推荐物品对应的解释路径
        self.user_explanations = self.load_user_explanations()

        # 6. 时间戳归一化
        self.prepare_timestamps_for_normalization()

    def load_user_recommendations(self):
        if not self.recommendations_file.exists():
            raise FileNotFoundError(
                f"❌ 推荐结果不存在: {self.recommendations_file}"
            )

        with open(self.recommendations_file, 'rb') as f:
            recs = pickle.load(f)

        print(f"✅ Loaded recommendations for {len(recs)} users")
        return recs

    # ===============================
    # 时间戳归一化
    # ===============================
    def prepare_timestamps_for_normalization(self):
        ts = self.user_item_relation['timestamp'].values
        self.min_timestamp = np.min(ts)
        self.max_timestamp = np.max(ts)
        self.timestamp_range = max(self.max_timestamp - self.min_timestamp, 1)

    def normalize_timestamp(self, ts):
        return (ts - self.min_timestamp) / self.timestamp_range

    # ===============================
    # 解释路径加载（仅推荐物品）
    # ===============================
    def load_user_explanations(self):
        user_explanations = defaultdict(list)

        for path_file in self.explanations_folder.glob('*.paths'):
            with open(path_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    try:
                        user_id, item_id = map(int, parts[0].split(','))
                        path_nodes = list(map(int, parts[-1].split()))
                    except ValueError:
                        continue

                    # ⭐ 只保留“被推荐物品”的路径
                    if user_id not in self.user_recommendations:
                        continue
                    if item_id not in self.user_recommendations[user_id]:
                        continue

                    user_explanations[user_id].append({
                        'target_item': item_id,
                        'path': path_nodes
                    })

        print(f"✅ Loaded explanations for {len(user_explanations)} users")
        return user_explanations

    # ===============================
    # LIR
    # ===============================
    def calculate_lir(self, beta=0.5):
        scores = []

        for user, exps in self.user_explanations.items():
            hist = self.user_item_relation[self.user_item_relation['user_id'] == user]
            if hist.empty:
                continue

            path_items = set()
            for e in exps:
                for node in e['path']:
                    if self.id2type.get(node) == 'item':
                        path_items.add(node)

            timestamps = []
            for item in path_items:
                ts = hist[hist['item_id'] == item]['timestamp']
                if not ts.empty:
                    timestamps.append(self.normalize_timestamp(ts.iloc[0]))

            if not timestamps:
                continue

            timestamps.sort()
            ewma = timestamps[0]
            for t in timestamps[1:]:
                ewma = (1 - beta) * ewma + beta * t

            scores.append(np.clip(ewma, 0, 1))

        return scores

    # ===============================
    # SEP
    # ===============================
    def calculate_sep(self, beta=0.5):
        scores = []

        for _, exps in self.user_explanations.items():
            entities = []
            for e in exps:
                entities.extend(e['path'][1:-1])

            if not entities:
                continue

            pops = [self.entity_popularity_norm.get(e, 0.0) for e in entities]
            pops.sort(reverse=True)

            ewma = pops[0]
            for p in pops[1:]:
                ewma = (1 - beta) * ewma + beta * p

            scores.append(np.clip(ewma, 0, 1))

        return scores

    # ===============================
    # ETD
    # ===============================
    def calculate_etd(self):
        scores = []

        for _, exps in self.user_explanations.items():
            patterns = set()
            for e in exps:
                rels = self.extract_relations(e['path'])
                patterns.add('→'.join(rels))

            if exps:
                scores.append(len(patterns) / len(exps))

        return scores

    # ===============================
    # 实体流行度
    # ===============================
    def calculate_entity_popularity(self):
        rel_files = [
            'item_category.relation',
            'item_brand.relation',
            'item_item.relation',
            'user_item.relation'
        ]

        degree = defaultdict(int)

        for rf in rel_files:
            p = self.refinefolder / rf
            if not p.exists():
                continue
            df = pd.read_csv(p, header=None)
            for col in [0, 1]:
                if col in df:
                    for k, v in df[col].value_counts().items():
                        degree[k] += v

        if not degree:
            return {}, {}

        vals = list(degree.values())
        mn, mx = min(vals), max(vals)
        norm = {k: (v - mn) / (mx - mn) if mx > mn else 0.0 for k, v in degree.items()}
        return dict(degree), norm

    # ===============================
    # 关系抽取
    # ===============================
    def extract_relations(self, path):
        rels = []
        for i in range(len(path) - 1):
            t1 = self.id2type.get(path[i], 'unknown')
            t2 = self.id2type.get(path[i + 1], 'unknown')
            rels.append(self.infer_relation(t1, t2))
        return rels

    def infer_relation(self, t1, t2):
        if (t1, t2) == ('user', 'item'):
            return 'purchase'
        if (t1, t2) == ('item', 'brand'):
            return 'produced_by'
        if (t1, t2) == ('item', 'category'):
            return 'belongs_to'
        if t1 == t2 == 'item':
            return 'also_viewed'
        return 'related'

    # ===============================
    # 总评估
    # ===============================
    def evaluate(self):
        lir = np.mean(self.calculate_lir())
        sep = np.mean(self.calculate_sep())
        etd = np.mean(self.calculate_etd())
        return lir, sep, etd


# ===============================
# main
# ===============================
def main():
    args = {
        'refinefolder': 'data/Amazon_Music/refine',
        'explanations_folder': 'data/Amazon_Music/path/all_ui_ii_instance_paths',
        'recommendations_file': 'D:/Thesis_Project/Models/TMER/tmer_user_recommendations.pkl'
    }

    evaluator = ExplanationMetrics(args)
    lir, sep, etd = evaluator.evaluate()

    print("\n===== Explainability Evaluation =====")
    print(f"LIR : {lir:.4f}")
    print(f"SEP : {sep:.4f}")
    print(f"ETD : {etd:.4f}")
    print("=====================================")


if __name__ == '__main__':
    main()
