# import pandas as pd
# import numpy as np
# import json
# import pickle
# from collections import defaultdict, Counter
# from pathlib import Path
# import re
# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from tqdm import tqdm

# class ImprovedExplanationConsistencyMetrics:
#     def __init__(self, args):
#         self.args = args
#         self.refinefolder = Path(args['refinefolder'])
#         self.explanations_folder = Path(args['explanations_folder'])
#         self.review_file = Path(args['review_file'])
        
#         # 初始化NLP工具
#         self.stop_words = set(stopwords.words('english'))
#         self.stemmer = PorterStemmer()
#         self.lemmatizer = WordNetLemmatizer()
        
#         # 领域关键词（针对手机配件）
#         self.domain_keywords = {
#             'general': ['good', 'great', 'excellent', 'perfect', 'nice', 
#                        'awesome', 'amazing', 'love', 'like', 'best'],
#             'phone': ['phone', 'cell', 'mobile', 'smartphone', 'iphone', 
#                      'android', 'samsung', 'apple', 'huawei', 'xiaomi'],
#             'case': ['case', 'cover', 'protector', 'protection', 'shield',
#                     'bumper', 'shell', 'skin', 'wallet', 'holster'],
#             'screen': ['screen', 'display', 'glass', 'protector', 'film',
#                       'tempered', 'guard', 'shield', 'privacy', 'matte'],
#             'battery': ['battery', 'power', 'charge', 'charger', 'charging',
#                        'life', 'backup', 'portable', 'bank', 'capacity'],
#             'audio': ['headphone', 'earphone', 'earbud', 'wireless', 'bluetooth',
#                      'sound', 'audio', 'music', 'noise', 'cancelling'],
#             'cable': ['cable', 'cord', 'wire', 'charging', 'usb', 'lightning',
#                      'type-c', 'adapter', 'connector', 'extension'],
#             'mount': ['mount', 'holder', 'stand', 'car', 'dashboard', 'vent',
#                      'suction', 'magnetic', 'gps', 'navigation']
#         }
        
#         # 加载数据
#         self.load_data()
    
#     def load_data(self):
#         """加载数据"""
#         print("="*60)
#         print("加载数据中...")
#         print("="*60)
        
#         # 1. 加载映射文件
#         self.id2type = pickle.load(open(self.refinefolder / 'map.id2type', 'rb'))
#         self.id2name = pickle.load(open(self.refinefolder / 'map.id2name', 'rb'))
#         self.name2id = pickle.load(open(self.refinefolder / 'map.name2id', 'rb'))
        
#         # 2. 加载用户-物品交互
#         self.user_item_relation = pd.read_csv(
#             self.refinefolder / 'user_item.relation',
#             header=None, sep=',', names=['user_id', 'item_id', 'timestamp', '_']
#         )
        
#         # 3. 加载解释路径
#         self.user_explanations = self.load_user_explanations()
        
#         # 4. 加载review数据
#         self.reviews = self.load_review_data()
        
#         # 5. 构建改进的真实理由
#         self.ground_truth = self.build_ground_truth_improved()
        
#         print(f"\n数据加载完成:")
#         print(f"  - 用户数: {len(self.user_explanations)}")
#         print(f"  - Review数量: {len(self.reviews)}")
#         print(f"  - 构建了 {len(self.ground_truth)} 个用户的真实理由")
    
#     def load_user_explanations(self):
#         """加载用户解释路径"""
#         user_explanations = defaultdict(list)
        
#         if self.explanations_folder.exists():
#             files = list(self.explanations_folder.glob('*.paths'))
#             print(f"  发现 {len(files)} 个路径文件")
            
#             for metapath_file in tqdm(files, desc="  加载路径文件"):
#                 with open(metapath_file, 'r') as f:
#                     for line_num, line in enumerate(f, 1):
#                         parts = line.strip().split('\t')
#                         if len(parts) < 2:
#                             continue
                        
#                         try:
#                             # 解析用户-物品对
#                             ui_pair = parts[0]
#                             path_str = parts[-1]  # 取最后一个路径
                            
#                             user_id, item_id = map(int, ui_pair.split(','))
#                             path_nodes = list(map(int, path_str.split()))
                            
#                             user_explanations[user_id].append({
#                                 'target_item': item_id,
#                                 'path': path_nodes
#                             })
#                         except (ValueError, IndexError) as e:
#                             continue
        
#         # 统计信息
#         total_paths = sum(len(paths) for paths in user_explanations.values())
#         print(f"  加载了 {total_paths} 条路径，涉及 {len(user_explanations)} 个用户")
        
#         return user_explanations
    
#     def load_review_data(self):
#         """加载review数据"""
#         reviews = []
        
#         if self.review_file.exists():
#             print(f"  读取review文件: {self.review_file.name}")
            
#             with open(self.review_file, 'r', encoding='utf-8') as f:
#                 total_lines = sum(1 for _ in f)
            
#             print(f"  文件共有 {total_lines} 行")
            
#             with open(self.review_file, 'r', encoding='utf-8') as f:
#                 for line in tqdm(f, total=total_lines, desc="  读取review"):
#                     try:
#                         review = json.loads(line.strip())
                        
#                         reviewer_id = review.get('reviewerID', '')
#                         asin = review.get('asin', '')
#                         review_text = review.get('reviewText', '')
#                         summary = review.get('summary', '')
                        
#                         if reviewer_id and asin and (review_text or summary):
#                             user_id = self.name2id.get(reviewer_id)
#                             item_id = self.name2id.get(asin)
                            
#                             if user_id is not None and item_id is not None:
#                                 reviews.append({
#                                     'user_id': user_id,
#                                     'item_id': item_id,
#                                     'review_text': review_text,
#                                     'summary': summary,
#                                     'full_text': f"{summary} {review_text}".strip()
#                                 })
#                     except json.JSONDecodeError:
#                         continue
        
#         print(f"  成功加载 {len(reviews)} 条有效review")
#         return reviews
    
#     def preprocess_text_advanced(self, text):
#         """高级文本预处理"""
#         if not text:
#             return []
        
#         # 转换为小写
#         text = text.lower()
        
#         # 移除特殊字符和数字
#         text = re.sub(r'[^\w\s]', ' ', text)
#         text = re.sub(r'\d+', ' ', text)
        
#         # 分词
#         tokens = word_tokenize(text)
        
#         # 移除停用词和短词
#         tokens = [token for token in tokens 
#                  if token not in self.stop_words and len(token) > 2]
        
#         # 词干提取
#         tokens = [self.stemmer.stem(token) for token in tokens]
        
#         return tokens
    
#     def build_ground_truth_improved(self):
#         """改进的真实理由构建"""
#         print("  改进版：构建真实理由集合...")
        
#         # 按用户分组收集review文本
#         user_reviews = defaultdict(list)
#         for review in tqdm(self.reviews, desc="  分组用户review"):
#             user_id = review['user_id']
#             user_reviews[user_id].append(review['full_text'])
        
#         print(f"  有review的用户数: {len(user_reviews)}")
        
#         # 构建每个用户的真实理由集合
#         ground_truth = {}
        
#         for user_id in tqdm(user_reviews.keys(), desc="  处理用户"):
#             texts = user_reviews[user_id]
#             combined_text = ' '.join(texts)
#             tokens = self.preprocess_text_advanced(combined_text)
            
#             if tokens:
#                 # 计算词频
#                 word_counts = Counter(tokens)
                
#                 # 取前100个高频词
#                 top_words = [word for word, _ in word_counts.most_common(100)]
                
#                 # 添加领域关键词（如果相关）
#                 domain_words_to_add = set()
#                 for word in top_words[:20]:  # 只检查前20个词
#                     for category, keywords in self.domain_keywords.items():
#                         if any(keyword.startswith(word) or word.startswith(keyword) 
#                                for keyword in keywords):
#                             domain_words_to_add.update(keywords[:10])  # 添加前10个相关词
                
#                 # 合并所有词
#                 all_words = set(top_words)
#                 all_words.update(domain_words_to_add)
                
#                 # 限制为最多150个词
#                 all_words = set(list(all_words)[:150])
                
#                 ground_truth[user_id] = all_words
        
#         # 统计信息
#         if ground_truth:
#             word_counts = [len(words) for words in ground_truth.values()]
#             avg_words = np.mean(word_counts)
#             print(f"  平均每个用户的真实理由词数: {avg_words:.1f} (范围: {min(word_counts)}-{max(word_counts)})")
        
#         print(f"  构建了 {len(ground_truth)} 个用户的真实理由")
#         return ground_truth
    
#     def extract_explanation_words_extended(self, path):
#         """扩展的解释词汇提取"""
#         words = set()
        
#         # 1. 从节点名称中提取词
#         for node in path:
#             node_name = self.id2name.get(node, '')
#             if node_name:
#                 # 移除前缀
#                 if node_name.startswith('c_'):
#                     node_name = node_name[2:]
#                 elif node_name.startswith('b_'):
#                     node_name = node_name[2:]
                
#                 # 提取所有字母序列
#                 parts = re.findall(r'[a-zA-Z]+', node_name.lower())
#                 for part in parts:
#                     if len(part) > 2:
#                         # 词干提取
#                         stemmed = self.stemmer.stem(part)
#                         words.add(stemmed)
        
#         # 2. 添加基于路径模式的词汇
#         # 检查路径中的节点类型序列，推断可能的相关词汇
#         node_types = [self.id2type.get(node, '') for node in path]
        
#         # 如果路径包含品牌->物品模式，添加相关词汇
#         if 'brand' in node_types and 'item' in node_types:
#             for category, keywords in self.domain_keywords.items():
#                 if category != 'general':  # 不添加通用词
#                     words.update(keywords[:5])
        
#         # 3. 添加通用积极词汇（经常在解释中出现）
#         words.update(self.domain_keywords['general'])
        
#         return words
    
#     def calculate_similarity(self, word1, word2):
#         """计算两个词的相似度"""
#         # 1. 完全匹配
#         if word1 == word2:
#             return 1.0
        
#         # 2. 部分匹配（子字符串）
#         if word1 in word2 or word2 in word1:
#             return 0.8
        
#         # 3. 词干相同
#         stem1 = self.stemmer.stem(word1)
#         stem2 = self.stemmer.stem(word2)
#         if stem1 == stem2:
#             return 0.9
        
#         # 4. 同义词（简化版）
#         synonyms1 = self.get_synonyms(word1)
#         synonyms2 = self.get_synonyms(word2)
#         if word1 in synonyms2 or word2 in synonyms1:
#             return 0.7
        
#         # 5. 领域相关词匹配
#         for category, keywords in self.domain_keywords.items():
#             if word1 in keywords and word2 in keywords:
#                 return 0.6
        
#         return 0.0
    
#     def get_synonyms(self, word):
#         """获取同义词（简化版）"""
#         synonyms = set()
#         for syn in wordnet.synsets(word):
#             for lemma in syn.lemmas():
#                 synonym = lemma.name().replace('_', ' ')
#                 synonyms.add(synonym)
#         return synonyms
    
#     def calculate_metrics_improved(self):
#         """改进的指标计算"""
#         print("\n" + "="*60)
#         print("计算改进版Recall、Precision、F1指标...")
#         print("="*60)
        
#         recalls = []
#         precisions = []
#         f1_scores = []
        
#         valid_users = 0
        
#         for user_id, explanations in tqdm(self.user_explanations.items(), 
#                                          desc="  处理用户", 
#                                          total=len(self.user_explanations)):
            
#             # 获取真实理由
#             G_u = self.ground_truth.get(user_id, set())
#             if not G_u or not explanations:
#                 continue
            
#             # 提取模型解释单词
#             S_u = set()
#             for exp in explanations:
#                 words = self.extract_explanation_words_extended(exp['path'])
#                 S_u.update(words)
            
#             if not S_u:
#                 continue
            
#             valid_users += 1
            
#             # 计算改进的交集（使用相似度匹配）
#             intersection = set()
#             matched_pairs = []
            
#             for s_word in S_u:
#                 best_match = None
#                 best_similarity = 0.0
                
#                 for g_word in G_u:
#                     similarity = self.calculate_similarity(s_word, g_word)
#                     if similarity > 0.5 and similarity > best_similarity:  # 阈值0.5
#                         best_similarity = similarity
#                         best_match = g_word
                
#                 if best_match:
#                     intersection.add(s_word)
#                     matched_pairs.append((s_word, best_match, best_similarity))
            
#             # 计算指标
#             recall = len(intersection) / (len(G_u) + 1)  
#             precision = len(intersection) / (len(S_u) + 1)
            
#             # 计算F1
#             if precision + recall > 0:
#                 f1 = 2 * precision * recall / (precision + recall + 1)
#             else:
#                 f1 = 0.0
            
#             recalls.append(recall)
#             precisions.append(precision)
#             f1_scores.append(f1)
        
#         # 计算平均值
#         avg_recall = np.mean(recalls) if recalls else 0.0
#         avg_precision = np.mean(precisions) if precisions else 0.0
#         avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
#         print(f"\n改进版评估完成:")
#         print(f"  - 有效评估用户数: {valid_users}")
#         print(f"  - 平均Recall: {avg_recall:.4f}")
#         print(f"  - 平均Precision: {avg_precision:.4f}")
#         print(f"  - 平均F1: {avg_f1:.4f}")
        
#         return avg_recall, avg_precision, avg_f1, valid_users


# def main():
#     """主函数"""
#     print("="*60)
#     print("改进版解释一致性评估系统")
#     print("="*60)
    
#     # 配置参数
#     base_dir = Path("D:/Thesis_Project/Models/TMER/data/Amazon_Music")
    
#     args = {
#         'refinefolder': str(base_dir / 'refine/'),
#         'explanations_folder': str(base_dir / 'path/all_ui_ii_instance_paths/'),
#         'review_file': Path("D:/Thesis_Project/Models/TMER/Cell_Phones_and_Accessories_5.json"),
#     }
    
#     print(f"数据目录: {base_dir}")
#     print(f"精炼数据目录: {args['refinefolder']}")
#     print(f"解释路径目录: {args['explanations_folder']}")
#     print(f"Review文件: {args['review_file']}")
    
#     # 检查路径是否存在
#     for key, path in args.items():
#         if key != 'review_file' and not Path(path).exists():
#             print(f"❌ 错误: {key} 不存在: {path}")
#             return
    
#     if not args['review_file'].exists():
#         print(f"⚠️ 警告: review文件不存在: {args['review_file']}")
#         print("将无法计算基于文本的指标")
#         return
    
#     try:
#         # 初始化改进版评估器
#         evaluator = ImprovedExplanationConsistencyMetrics(args)
        
#         # 计算改进版指标
#         recall, precision, f1, valid_users = evaluator.calculate_metrics_improved()
        
#         # 打印最终结果
#         print("\n" + "="*60)
#         print("改进版最终评估结果:")
#         print("="*60)
#         print(f"有效评估用户数: {valid_users}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"F1-score:  {f1:.4f}")
#         print("="*60)
        
#     except Exception as e:
#         print(f"\n❌ 错误: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ExplanationConsistencyMetrics:
    def __init__(self, args):
        self.args = args
        self.refinefolder = Path(args['refinefolder'])
        self.explanations_folder = Path(args['explanations_folder'])
        self.review_file = Path(args['review_file'])
        
        # 初始化NLP工具
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
        
        self.stemmer = PorterStemmer()
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载数据中...")
        print("="*60)
        
        # 1. 加载映射文件
        self.id2type = pickle.load(open(self.refinefolder / 'map.id2type', 'rb'))
        self.id2name = pickle.load(open(self.refinefolder / 'map.id2name', 'rb'))
        self.name2id = pickle.load(open(self.refinefolder / 'map.name2id', 'rb'))
        
        # 2. 加载用户-物品交互
        self.user_item_relation = pd.read_csv(
            self.refinefolder / 'user_item.relation',
            header=None, sep=',', names=['user_id', 'item_id', 'timestamp', '_']
        )
        
        # 3. 加载解释路径
        self.user_explanations = self.load_user_explanations()
        
        # 4. 加载review数据
        self.reviews = self.load_review_data()
        
        # 5. 构建真实理由（按照论文方法）
        self.ground_truth = self.build_ground_truth_tfidf()
        
        print(f"\n数据加载完成:")
        print(f"  - 用户数: {len(self.user_explanations)}")
        print(f"  - Review数量: {len(self.reviews)}")
        print(f"  - 构建了 {len(self.ground_truth)} 个用户的真实理由")
    
    def load_user_explanations(self):
        """加载用户解释路径"""
        user_explanations = defaultdict(list)
        
        if self.explanations_folder.exists():
            files = list(self.explanations_folder.glob('*.paths'))
            print(f"  发现 {len(files)} 个路径文件")
            
            for metapath_file in tqdm(files, desc="  加载路径文件"):
                with open(metapath_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue
                        
                        try:
                            # 解析用户-物品对
                            ui_pair = parts[0]
                            path_str = parts[-1]  # 取最后一个路径
                            
                            user_id, item_id = map(int, ui_pair.split(','))
                            path_nodes = list(map(int, path_str.split()))
                            
                            user_explanations[user_id].append({
                                'target_item': item_id,
                                'path': path_nodes
                            })
                        except (ValueError, IndexError) as e:
                            continue
        
        # 统计信息
        total_paths = sum(len(paths) for paths in user_explanations.values())
        print(f"  加载了 {total_paths} 条路径，涉及 {len(user_explanations)} 个用户")
        
        return user_explanations
    
    def load_review_data(self):
        """加载review数据"""
        reviews = []
        
        if self.review_file.exists():
            print(f"  读取review文件: {self.review_file.name}")
            
            with open(self.review_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            print(f"  文件共有 {total_lines} 行")
            
            with open(self.review_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc="  读取review"):
                    try:
                        review = json.loads(line.strip())
                        
                        reviewer_id = review.get('reviewerID', '')
                        asin = review.get('asin', '')
                        review_text = review.get('reviewText', '')
                        summary = review.get('summary', '')
                        
                        if reviewer_id and asin and (review_text or summary):
                            user_id = self.name2id.get(reviewer_id)
                            item_id = self.name2id.get(asin)
                            
                            if user_id is not None and item_id is not None:
                                reviews.append({
                                    'user_id': user_id,
                                    'item_id': item_id,
                                    'review_text': review_text,
                                    'summary': summary,
                                    'full_text': f"{summary} {review_text}".strip()
                                })
                    except json.JSONDecodeError:
                        continue
        
        print(f"  成功加载 {len(reviews)} 条有效review")
        return reviews
    
    def preprocess_text(self, text):
        """文本预处理（按照论文方法）"""
        if not text:
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词和短词
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def build_ground_truth_tfidf(self):
        """按照论文方法使用TF-IDF构建真实理由"""
        print("  使用TF-IDF构建真实理由集合...")
        
        # 按用户分组收集review文本
        user_reviews = defaultdict(list)
        for review in tqdm(self.reviews, desc="  分组用户review"):
            user_id = review['user_id']
            user_reviews[user_id].append(review['full_text'])
        
        print(f"  有review的用户数: {len(user_reviews)}")
        
        # 创建用户文档
        user_docs = {}
        for user_id, texts in user_reviews.items():
            # 合并用户的所有review文本
            combined_text = ' '.join(texts)
            # 预处理但不进行词干提取（论文中未提及词干提取）
            tokens = self.preprocess_text(combined_text)
            user_docs[user_id] = ' '.join(tokens)  # 重新组合为字符串用于TF-IDF
        
        # 提取所有用户文档用于TF-IDF计算
        user_ids = list(user_docs.keys())
        documents = [user_docs[uid] for uid in user_ids]
        
        if not documents:
            print("  警告：没有可用的文档进行TF-IDF计算")
            return {}
        
        # 计算TF-IDF
        print("  计算TF-IDF...")
        vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,  # 使用相同的预处理
            token_pattern=None,
            max_df=5000,  # 过滤高频词（词频>5000）
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"  TF-IDF计算失败: {e}")
            return {}
        
        # 为每个用户构建真实理由集合
        ground_truth = {}
        
        print("  构建用户真实理由集合...")
        for idx, user_id in enumerate(tqdm(user_ids, desc="  处理用户")):
            # 获取该用户的TF-IDF向量
            tfidf_vector = tfidf_matrix[idx]
            
            # 提取TF-IDF值大于0.1的词
            words = []
            for word_idx in tfidf_vector.indices:
                tfidf_value = tfidf_vector[0, word_idx]
                if tfidf_value > 0.1:  # 过滤低TF-IDF词
                    word = feature_names[word_idx]
                    words.append(word)
            
            ground_truth[user_id] = set(words)
        
        # 统计信息
        if ground_truth:
            word_counts = [len(words) for words in ground_truth.values()]
            avg_words = np.mean(word_counts)
            print(f"  平均每个用户的真实理由词数: {avg_words:.1f} (范围: {min(word_counts)}-{max(word_counts)})")
        
        print(f"  构建了 {len(ground_truth)} 个用户的真实理由")
        return ground_truth
    
    def extract_explanation_words(self, path):
        """提取解释词汇（严格按照论文方法）"""
        words = set()
        
        # 从路径节点名称中提取词汇
        for node in path:
            node_name = self.id2name.get(node, '')
            if node_name:
                # 移除前缀
                if node_name.startswith('c_'):
                    node_name = node_name[2:]
                elif node_name.startswith('b_'):
                    node_name = node_name[2:]
                
                # 提取所有字母序列（不进行词干提取）
                parts = re.findall(r'[a-zA-Z]+', node_name.lower())
                for part in parts:
                    if len(part) > 2 and part not in self.stop_words:
                        words.add(part)
        
        return words
    
    def calculate_metrics(self):
        """计算指标（按照论文方法）"""
        print("\n" + "="*60)
        print("计算Recall、Precision、F1指标（论文方法）...")
        print("="*60)
        
        recalls = []
        precisions = []
        f1_scores = []
        
        valid_users = 0
        
        for user_id, explanations in tqdm(self.user_explanations.items(), 
                                         desc="  处理用户", 
                                         total=len(self.user_explanations)):
            
            # 获取真实理由
            G_u = self.ground_truth.get(user_id, set())
            if not G_u or not explanations:
                continue
            
            # 提取模型解释单词
            S_u = set()
            for exp in explanations:
                words = self.extract_explanation_words(exp['path'])
                S_u.update(words)
            
            if not S_u:
                continue
            
            valid_users += 1
            
            # 计算精确交集（不进行相似度匹配）
            intersection = S_u.intersection(G_u)
            
            # 计算指标（分母加1平滑）
            recall = len(intersection) / (len(G_u) + 1)
            precision = len(intersection) / (len(S_u) + 1)
            
            # 计算F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall + 1)
            else:
                f1 = 0.0
            
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
        
        # 计算平均值
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        print(f"\n评估完成:")
        print(f"  - 有效评估用户数: {valid_users}")
        print(f"  - 平均Recall: {avg_recall:.4f}")
        print(f"  - 平均Precision: {avg_precision:.4f}")
        print(f"  - 平均F1: {avg_f1:.4f}")
        
        # 详细统计
        if valid_users > 0:
            print(f"\n详细统计:")
            print(f"  - Recall范围: [{np.min(recalls):.4f}, {np.max(recalls):.4f}]")
            print(f"  - Precision范围: [{np.min(precisions):.4f}, {np.max(precisions):.4f}]")
            print(f"  - F1范围: [{np.min(f1_scores):.4f}, {np.max(f1_scores):.4f}]")
            print(f"  - Recall标准差: {np.std(recalls):.4f}")
            print(f"  - Precision标准差: {np.std(precisions):.4f}")
            print(f"  - F1标准差: {np.std(f1_scores):.4f}")
        
        return avg_recall, avg_precision, avg_f1, valid_users


def main():
    """主函数"""
    print("="*60)
    print("解释一致性评估系统（严格按照论文方法）")
    print("="*60)
    
    # 配置参数
    # 注意：请根据您的实际路径修改
    base_dir = Path("D:/Thesis_Project/Models/TMER/data/Amazon_Music")
    
    args = {
        'refinefolder': str(base_dir / 'refine/'),
        'explanations_folder': str(base_dir / 'path/all_ui_ii_instance_paths/'),
        'review_file': Path("D:/Thesis_Project/Models/TMER/Cell_Phones_and_Accessories_5.json"),
    }
    
    print(f"数据目录: {base_dir}")
    print(f"精炼数据目录: {args['refinefolder']}")
    print(f"解释路径目录: {args['explanations_folder']}")
    print(f"Review文件: {args['review_file']}")
    
    # 检查路径是否存在
    for key, path in args.items():
        if key != 'review_file' and not Path(path).exists():
            print(f"❌ 错误: {key} 不存在: {path}")
            return
    
    if not args['review_file'].exists():
        print(f"⚠️ 警告: review文件不存在: {args['review_file']}")
        print("将无法计算基于文本的指标")
        return
    
    try:
        # 初始化评估器
        evaluator = ExplanationConsistencyMetrics(args)
        
        # 计算指标
        recall, precision, f1, valid_users = evaluator.calculate_metrics()
        
        # 打印最终结果
        print("\n" + "="*60)
        print("最终评估结果:")
        print("="*60)
        print(f"有效评估用户数: {valid_users}")
        print(f"Recall:    {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("="*60)
        
        # 保存结果到文件
        results = {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'valid_users': valid_users,
            'total_users': len(evaluator.user_explanations),
            'users_with_reviews': len(evaluator.ground_truth)
        }
        
        output_file = Path("explanation_consistency_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()