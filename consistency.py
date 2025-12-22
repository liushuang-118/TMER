import json
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ExplanationConsistencyMetrics:
    """
    Explanation Consistency for RECOMMENDED ITEMS
    """

    def __init__(self, args):
        self.refinefolder = Path(args["refinefolder"])
        self.explanations_folder = Path(args["explanations_folder"])
        self.review_file = Path(args["review_file"])

        self.load_data()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    def load_data(self):
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)

        # id maps
        self.id2name = pickle.load(open(self.refinefolder / "map.id2name", "rb"))
        self.name2id = pickle.load(open(self.refinefolder / "map.name2id", "rb"))

        # explanation paths (ONLY for recommended items)
        self.user_explanations = self.load_user_explanations()

        # reviews
        self.reviews = self.load_reviews()

        # ground truth preference words
        self.ground_truth = self.build_ground_truth_tfidf()

        print("\nData loaded:")
        print(f"  Users with explanations: {len(self.user_explanations)}")
        print(f"  Reviews loaded: {len(self.reviews)}")
        print(f"  Users with ground truth: {len(self.ground_truth)}")

    # ------------------------------------------------------------------
    # Load explanation paths
    # ------------------------------------------------------------------
    def load_user_explanations(self):
        user_exps = defaultdict(list)

        files = list(self.explanations_folder.glob("*.paths"))
        print(f"Found {len(files)} path files")

        for f in tqdm(files, desc="Loading explanation paths"):
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue

                    try:
                        ui = parts[0]
                        path_nodes = list(map(int, parts[-1].split()))
                        u, i = map(int, ui.split(","))

                        user_exps[u].append(
                            {
                                "item": i,
                                "path": path_nodes,
                            }
                        )
                    except:
                        continue

        total = sum(len(v) for v in user_exps.values())
        print(f"Loaded {total} explanation paths for recommended items")
        return user_exps

    # ------------------------------------------------------------------
    # Load reviews
    # ------------------------------------------------------------------
    def load_reviews(self):
        reviews = []

        with open(self.review_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    r = json.loads(line)
                    uid = self.name2id.get(r.get("reviewerID"))
                    iid = self.name2id.get(r.get("asin"))
                    text = (r.get("summary", "") + " " + r.get("reviewText", "")).strip()

                    if uid is not None and iid is not None and text:
                        reviews.append(
                            {
                                "user": uid,
                                "item": iid,
                                "text": text.lower(),
                            }
                        )
                except:
                    continue

        return reviews

    # ------------------------------------------------------------------
    # Ground truth via TF-IDF (paper-consistent)
    # ------------------------------------------------------------------
    def build_ground_truth_tfidf(self):
        print("Building TF-IDF ground truth...")

        user_texts = defaultdict(list)
        for r in self.reviews:
            user_texts[r["user"]].append(r["text"])

        docs, users = [], []
        for u, texts in user_texts.items():
            docs.append(" ".join(texts))
            users.append(u)

        if not docs:
            return {}

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
        )

        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()

        ground_truth = {}
        for i, u in enumerate(users):
            row = X[i]
            words = set(
                vocab[idx]
                for idx in row.indices
                if row[0, idx] > 0.1
            )
            ground_truth[u] = words

        return ground_truth

    # ------------------------------------------------------------------
    # Extract explanation words from paths
    # ------------------------------------------------------------------
    def extract_words_from_path(self, path):
        words = set()
        for node in path:
            name = self.id2name.get(node, "")
            name = re.sub(r"^[cb]_", "", name.lower())
            parts = re.findall(r"[a-zA-Z]+", name)
            for p in parts:
                if len(p) > 2:
                    words.add(p)
        return words

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def evaluate(self):
        recalls, precisions, f1s = [], [], []
        valid_users = 0

        for u, exps in tqdm(self.user_explanations.items(), desc="Evaluating users"):
            G = self.ground_truth.get(u)
            if not G:
                continue

            S = set()
            for e in exps:
                S |= self.extract_words_from_path(e["path"])

            if not S:
                continue

            valid_users += 1
            inter = S & G

            recall = len(inter) / (len(G) + 1)
            precision = len(inter) / (len(S) + 1)

            if recall + precision > 0:
                f1 = 2 * recall * precision / (recall + precision + 1)
            else:
                f1 = 0.0

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        print("\n================ FINAL RESULTS ================")
        print(f"Valid users: {valid_users}")
        print(f"Recall:    {np.mean(recalls):.4f}")
        print(f"Precision: {np.mean(precisions):.4f}")
        print(f"F1-score:  {np.mean(f1s):.4f}")
        print("==============================================")

        return np.mean(recalls), np.mean(precisions), np.mean(f1s)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    base_dir = Path("D:/Thesis_Project/Models/TMER/data/Amazon_Music")

    args = {
        "refinefolder": base_dir / "refine",
        "explanations_folder": base_dir / "path/all_ui_ii_instance_paths",
        "review_file": Path(
            "D:/Thesis_Project/Models/TMER/Clothing_Shoes_and_Jewelry_5.json"
        ),
    }

    evaluator = ExplanationConsistencyMetrics(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()

