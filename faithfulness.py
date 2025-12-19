"""
TMER Explanation Faithfulness Evaluation
"""

import numpy as np
import pickle
import random
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class TMERFaithfulnessEvaluator:
    def __init__(self, args):
        self.args = args

        self.refinefolder = Path(args['refinefolder'])
        self.explanations_folder = Path(args['explanations_folder'])
        self.recommendations_file = Path(args['recommendations_file'])
        self.scores_file = Path(args['scores_file'])
        self.sample_users = args.get('sample_users', 20)

        self.load_mappings()
        self.metapath_patterns = [
            'UIU', 'UIB', 'UIC', 'UIBI', 'UIBICI',
            'UICI', 'UICIBI', 'IBIBI', 'IBICI',
            'ICIUI', 'IUIUI', 'OTHER'
        ]
        self.type_mapping = {'user': 'U', 'item': 'I', 'brand': 'B', 'category': 'C'}

        self.path_cache = None
        self.recommendation_pairs = None
        self.user_scores = None

        self.results = {
            'JS_f': None,
            'JS_w': None,
            'sampled_users': [],
            'user_metrics': []
        }

    # --------------------------------------------------
    # Load node type mapping
    # --------------------------------------------------
    def load_mappings(self):
        print("Loading node type mappings...")
        try:
            with open(self.refinefolder / 'map.id2type', 'rb') as f:
                self.id2type = pickle.load(f)
            print(f"  Loaded {len(self.id2type)} node types")
        except Exception as e:
            print(f"  Failed to load id2type: {e}")
            self.id2type = {}

    # --------------------------------------------------
    # Meta-path extraction
    # --------------------------------------------------
    def extract_metapath_from_path(self, path_nodes):
        if not path_nodes or len(path_nodes) < 2:
            return 'OTHER'

        symbols = []
        for nid in path_nodes:
            ntype = self.id2type.get(nid, None)
            if ntype in self.type_mapping:
                symbols.append(self.type_mapping[ntype])
            else:
                symbols.append('X')

        pattern = ''.join(symbols)
        if pattern in self.metapath_patterns:
            return pattern

        for p in sorted(self.metapath_patterns, key=len, reverse=True):
            if p == 'OTHER':
                continue
            if pattern.find(p) != -1:
                return p
        return 'OTHER'

    # --------------------------------------------------
    # Load explanation paths
    # --------------------------------------------------
    def load_all_explanations(self):
        print("\nLoading explanation paths...")
        self.path_cache = defaultdict(list)
        files = list(self.explanations_folder.glob("*"))
        total = 0
        for f in tqdm(files, desc="Reading path files"):
            if f.stat().st_size == 0:
                continue
            with open(f, 'r', encoding='utf-8', errors='ignore') as fin:
                for line in fin:
                    line = line.strip()
                    if not line or '\t' not in line:
                        continue
                    ui_part, path_part = line.split('\t', 1)
                    if ',' not in ui_part:
                        continue
                    try:
                        u, i = map(int, ui_part.split(','))
                        path_nodes = list(map(int, path_part.split()))
                    except:
                        continue
                    if 2 <= len(path_nodes) <= 15:
                        self.path_cache[(u, i)].append({'path': path_nodes, 'prob': 1.0})
                        total += 1
        print(f"  Loaded {total} paths")
        print(f"  Covered {len(self.path_cache)} user-item pairs")

    # --------------------------------------------------
    # Load recommendations and scores
    # --------------------------------------------------
    def load_recommendations_and_scores(self):
        print("\nLoading recommendation pairs...")
        with open(self.recommendations_file, 'rb') as f:
            self.recommendation_pairs = pickle.load(f)  # {user_id: [item_id,...]}
        print(f"  Loaded recommendations for {len(self.recommendation_pairs)} users")

        print("\nLoading recommendation scores...")
        with open(self.scores_file, 'rb') as f:
            self.user_scores = pickle.load(f)  # {user_id: {item_id: score}}
        print(f"  Loaded scores for {len(self.user_scores)} users")

    # --------------------------------------------------
    # Get paths for a specific user-item
    # --------------------------------------------------
    def get_paths_for_user_item(self, user_id, item_id, max_paths=20):
        paths = self.path_cache.get((user_id, item_id), [])
        if not paths:
            return []
        if len(paths) > max_paths:
            paths = random.sample(paths, max_paths)
        return [(p['path'], p.get('prob', 1.0)) for p in paths]

    # --------------------------------------------------
    # Build F_u (historical)
    # --------------------------------------------------
    def build_F_u_for_user(self, user_id, max_paths=1000):
        all_paths, all_weights = [], []
        for (u, i), paths in self.path_cache.items():
            if u != user_id:
                continue
            for p in paths:
                all_paths.append(p['path'])
                all_weights.append(p.get('prob', 1.0))
        if not all_paths:
            return None
        if len(all_paths) > max_paths:
            idx = random.sample(range(len(all_paths)), max_paths)
            all_paths = [all_paths[i] for i in idx]
            all_weights = [all_weights[i] for i in idx]
        return self.calculate_rule_distribution(all_paths, all_weights)

    # --------------------------------------------------
    # Build Q_u (recommendation)
    # --------------------------------------------------
    def build_Q_u_for_user(self, user_id, max_paths=20):
        if user_id not in self.recommendation_pairs:
            return None
        all_paths, all_weights = [], []

        for item_id in self.recommendation_pairs[user_id]:
            paths = self.get_paths_for_user_item(user_id, item_id, max_paths)
            score = float(self.user_scores.get(user_id, {}).get(item_id, 1.0))
            for p, _ in paths:
                all_paths.append(p)
                all_weights.append(score)

        if not all_paths:
            return None

        if len(all_paths) > max_paths:
            idx = random.sample(range(len(all_paths)), max_paths)
            all_paths = [all_paths[i] for i in idx]
            all_weights = [all_weights[i] for i in idx]

        return self.calculate_rule_distribution(all_paths, all_weights)

    # --------------------------------------------------
    # Rule distribution
    # --------------------------------------------------
    def calculate_rule_distribution(self, paths, weights):
        freq = Counter()
        wfreq = Counter()
        weights = np.array(weights, dtype=float)
        weights = weights / (weights.sum() + 1e-10)
        for idx, path in enumerate(paths):
            rule = self.extract_metapath_from_path(path)
            freq[rule] += 1
            wfreq[rule] += weights[idx]
        eps = 1e-10
        freq_dist = {r: freq.get(r, eps) for r in self.metapath_patterns}
        wfreq_dist = {r: wfreq.get(r, eps) for r in self.metapath_patterns}
        # normalize
        s1 = sum(freq_dist.values())
        s2 = sum(wfreq_dist.values())
        freq_dist = {k: v / s1 for k, v in freq_dist.items()}
        wfreq_dist = {k: v / s2 for k, v in wfreq_dist.items()}
        return freq_dist, wfreq_dist

    # --------------------------------------------------
    # JS divergence
    # --------------------------------------------------
    def js_divergence(self, p, q):
        keys = list(set(p.keys()) | set(q.keys()))
        pvec = np.array([p.get(k, 1e-10) for k in keys])
        qvec = np.array([q.get(k, 1e-10) for k in keys])
        pvec /= pvec.sum()
        qvec /= qvec.sum()
        m = 0.5 * (pvec + qvec)
        js = 0.5 * np.sum(pvec * np.log2(pvec / (m + 1e-10))) + \
             0.5 * np.sum(qvec * np.log2(qvec / (m + 1e-10)))
        return float(js)

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------
    def run(self):
        print("\n===== TMER Faithfulness Evaluation =====")

        self.load_all_explanations()
        self.load_recommendations_and_scores()

        exp_users = set(u for (u, _) in self.path_cache.keys())
        rec_users = set(self.recommendation_pairs.keys())
        common_users = list(exp_users & rec_users)
        print(f"Common users: {len(common_users)}")

        sampled = random.sample(common_users, min(self.sample_users, len(common_users)))
        self.results['sampled_users'] = sampled

        JS_f, JS_w = [], []

        for u in tqdm(sampled, desc="Evaluating users"):
            Fu = self.build_F_u_for_user(u)
            Qu = self.build_Q_u_for_user(u)
            if Fu is None or Qu is None:
                continue
            Ff, Fw = Fu
            Qf, Qw = Qu
            JS_f.append(self.js_divergence(Ff, Qf))
            JS_w.append(self.js_divergence(Fw, Qw))

        self.results['JS_f'] = float(np.mean(JS_f)) if JS_f else None
        self.results['JS_w'] = float(np.mean(JS_w)) if JS_w else None

        print("\n===== Final Results =====")
        print(f"JS_f = {self.results['JS_f']:.4f}")
        print(f"JS_w = {self.results['JS_w']:.4f}")

        return self.results


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    base_dir = Path("D:/Thesis_Project/Models/TMER")
    args = {
        'refinefolder': str(base_dir / "data/Amazon_Music/refine/"),
        'explanations_folder': str(base_dir / "data/Amazon_Music/path/all_ui_ii_instance_paths/"),
        'recommendations_file': str(base_dir / "tmer_user_recommendations.pkl"),
        'scores_file': str(base_dir / "tmer_user_scores.pkl"),
        'sample_users': 50
    }

    evaluator = TMERFaithfulnessEvaluator(args)
    results = evaluator.run()
    print(results)


if __name__ == "__main__":
    main()
