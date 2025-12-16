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
        
        # 加载所需数据
        self.load_data()
        
    def load_data(self):
        """加载计算指标所需的所有数据"""
        # 1. 用户历史交互和时间戳
        self.user_item_relation = pd.read_csv(
            self.refinefolder / 'user_item.relation', 
            header=None, sep=',', names=['user_id', 'item_id', 'timestamp', '_']
        )
        
        # 2. 加载映射文件
        self.id2type = pickle.load(open(self.refinefolder / 'map.id2type', 'rb'))
        
        # 3. 计算实体流行度
        self.entity_popularity, self.entity_popularity_norm = self.calculate_entity_popularity()
        
        # 4. 加载解释路径
        self.user_explanations = self.load_user_explanations()
        
        # 5. 准备时间戳数据用于LIR归一化
        self.prepare_timestamps_for_normalization()
        
    def prepare_timestamps_for_normalization(self):
        """准备时间戳数据用于LIR归一化"""
        all_timestamps = self.user_item_relation['timestamp'].values
        
        if len(all_timestamps) > 0:
            self.min_timestamp = np.min(all_timestamps)
            self.max_timestamp = np.max(all_timestamps)
            self.timestamp_range = self.max_timestamp - self.min_timestamp
        else:
            self.min_timestamp = 0
            self.max_timestamp = 1
            self.timestamp_range = 1
            
    def load_user_explanations(self):
        """加载用户解释路径"""
        user_explanations = defaultdict(list)
        
        # 读取元路径文件
        if self.explanations_folder.exists():
            for metapath_file in self.explanations_folder.glob('*.paths'):
                with open(metapath_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue
                        
                        ui_pair = parts[0]
                        path_str = parts[-1]
                        
                        try:
                            user_id, item_id = map(int, ui_pair.split(','))
                            path_nodes = list(map(int, path_str.split()))
                            
                            user_explanations[user_id].append({
                                'target_item': item_id,
                                'path': path_nodes
                            })
                        except ValueError:
                            continue
        
        return user_explanations
    
    def normalize_timestamp(self, timestamp):
        """将时间戳归一化到[0,1]区间"""
        if self.timestamp_range == 0:
            return 0.0
        return (timestamp - self.min_timestamp) / self.timestamp_range
    
    def calculate_lir(self):
        """
        计算链接交互时效性（归一化到0-1）
        """
        lir_scores = []
        beta_lir = 0.5  # hyperparameter设为0.5
        
        for user_id, explanations in self.user_explanations.items():
            if len(explanations) == 0:
                continue
            
            # 获取用户的历史交互记录
            user_interactions = self.user_item_relation[self.user_item_relation['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                continue
            
            # 提取解释路径中涉及的历史物品
            path_items = set()
            for exp in explanations:
                for node in exp['path']:
                    node_type = self.id2type.get(node, '')
                    if node_type == 'item':
                        if node in user_interactions['item_id'].values:
                            path_items.add(node)
            
            if len(path_items) == 0:
                continue
            
            # 获取这些物品的时间戳并归一化
            normalized_timestamps = []
            for item in path_items:
                timestamps = user_interactions[user_interactions['item_id'] == item]['timestamp'].values
                if len(timestamps) > 0:
                    normalized_timestamp = self.normalize_timestamp(timestamps[0])
                    normalized_timestamps.append(normalized_timestamp)
            
            if len(normalized_timestamps) == 0:
                continue
            
            # 按时间排序（从最旧到最新）
            sorted_timestamps = sorted(normalized_timestamps)
            
            # 计算EWMA（归一化后）
            lir_value = sorted_timestamps[0]
            for i in range(1, len(sorted_timestamps)):
                lir_value = (1 - beta_lir) * lir_value + beta_lir * sorted_timestamps[i]
            
            # 确保值在[0,1]区间
            lir_value = max(0.0, min(1.0, lir_value))
            lir_scores.append(lir_value)
        
        return lir_scores
    
    def calculate_sep(self):
        """
        计算共享实体流行度（归一化到0-1）
        """
        sep_scores = []
        beta_sep = 0.5  # hyperparameter设为0.5
        
        for user_id, explanations in self.user_explanations.items():
            if len(explanations) == 0:
                continue
            
            # 提取解释路径中的所有共享实体（非用户、非目标物品）
            all_entities = []
            for exp in explanations:
                path = exp['path']
                # 跳过第一个（用户）和最后一个（目标物品）
                for node in path[1:-1]:
                    all_entities.append(node)
            
            if len(all_entities) == 0:
                continue
            
            # 获取实体的归一化流行度分数
            entity_popularities = []
            for entity in all_entities:
                norm_pop = self.entity_popularity_norm.get(entity, 0.0)
                entity_popularities.append(norm_pop)
            
            # 排序（从最流行到最不流行）
            sorted_popularities = sorted(entity_popularities, reverse=True)
            
            # 计算EWMA（使用归一化的流行度）
            sep_value = sorted_popularities[0]
            for i in range(1, len(sorted_popularities)):
                sep_value = (1 - beta_sep) * sep_value + beta_sep * sorted_popularities[i]
            
            # 确保值在[0,1]区间
            sep_value = max(0.0, min(1.0, sep_value))
            sep_scores.append(sep_value)
        
        return sep_scores
    
    def calculate_etd(self):
        """
        计算解释类型多样性（0-1值）
        """
        etd_scores = []
        
        for user_id, explanations in self.user_explanations.items():
            if len(explanations) == 0:
                continue
            
            # 提取所有路径模式
            path_patterns = set()
            
            for exp in explanations:
                path = exp['path']
                relations = self.extract_relations(path)
                if relations:
                    pattern = '→'.join(relations)
                    path_patterns.add(pattern)
            
            # 计算ETD分数（0-1之间）
            k = len(explanations)
            unique_patterns = len(path_patterns)
            
            if k > 0:
                etd_score = unique_patterns / k
                etd_score = max(0.0, min(1.0, etd_score))
            else:
                etd_score = 0.0
            
            etd_scores.append(etd_score)
        
        return etd_scores
    
    def calculate_entity_popularity(self):
        """
        计算实体的流行度（在知识图谱中的度数）并归一化
        """
        # 1. 加载关系文件
        relation_files = {
            'item_category': self.refinefolder / 'item_category.relation',
            'item_brand': self.refinefolder / 'item_brand.relation',
            'item_item': self.refinefolder / 'item_item.relation',
            'user_item': self.refinefolder / 'user_item.relation'
        }
        
        # 2. 统计每个实体的度数
        entity_degree = defaultdict(int)
        
        for rel_name, rel_file in relation_files.items():
            try:
                if rel_file.exists():
                    rel_df = pd.read_csv(rel_file, header=None)
                    
                    # 统计第一列实体的出度
                    entity_counts = rel_df[0].value_counts()
                    for entity, count in entity_counts.items():
                        entity_degree[entity] += count
                    
                    # 统计第二列实体的入度
                    if len(rel_df.columns) > 1:
                        entity_counts = rel_df[1].value_counts()
                        for entity, count in entity_counts.items():
                            entity_degree[entity] += count
            except Exception:
                continue
        
        # 3. 归一化到[0,1]区间
        if entity_degree:
            degrees = list(entity_degree.values())
            min_degree = min(degrees)
            max_degree = max(degrees)
            degree_range = max_degree - min_degree
            
            entity_degree_norm = {}
            for entity, degree in entity_degree.items():
                if degree_range == 0:
                    norm_degree = 0.0
                else:
                    norm_degree = (degree - min_degree) / degree_range
                entity_degree_norm[entity] = norm_degree
        else:
            entity_degree_norm = {}
        
        return dict(entity_degree), entity_degree_norm
    
    def extract_relations(self, path):
        """
        从路径中提取关系序列
        """
        relations = []
        
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            type1 = self.id2type.get(node1, 'unknown')
            type2 = self.id2type.get(node2, 'unknown')
            
            # 根据节点类型推断关系
            relation = self.infer_relation(type1, type2)
            relations.append(relation)
        
        return relations
    
    def infer_relation(self, type1, type2):
        """
        根据节点类型推断关系
        """
        if (type1, type2) == ('user', 'item'):
            return 'purchase'
        elif (type1, type2) == ('item', 'brand'):
            return 'produced_by'
        elif (type1, type2) == ('item', 'category'):
            return 'belongs_to'
        elif (type1, type2) == ('item', 'item'):
            return 'also_bought'
        elif (type1, type2) == ('brand', 'item'):
            return 'produces'
        elif (type1, type2) == ('category', 'item'):
            return 'contains'
        elif type1 == type2 == 'item':
            return 'also_viewed'
        else:
            return 'related'
    
    def evaluate_explanations(self):
        """评估所有用户的解释路径"""
        # 计算三个指标
        lir_scores = self.calculate_lir()
        sep_scores = self.calculate_sep()
        etd_scores = self.calculate_etd()
        
        # 计算平均值
        lir_avg = np.mean(lir_scores) if lir_scores else 0.0
        sep_avg = np.mean(sep_scores) if sep_scores else 0.0
        etd_avg = np.mean(etd_scores) if etd_scores else 0.0
        
        return lir_avg, sep_avg, etd_avg


def main():
    """主函数"""
    print("="*40)
    print("解释性指标评估系统")
    print("="*40)
    
    # 配置参数
    base_dir = Path("data/Amazon_Music")
    
    args = {
        'refinefolder': str(base_dir / 'refine/'),
        'explanations_folder': str(base_dir / 'path/all_ui_ii_instance_paths/'),
    }
    
    # 检查路径是否存在
    if not Path(args['refinefolder']).exists():
        print(f"❌ 错误: refinefolder 不存在: {args['refinefolder']}")
        return
    
    if not Path(args['explanations_folder']).exists():
        print(f"❌ 错误: explanations_folder 不存在: {args['explanations_folder']}")
        return
    
    try:
        # 初始化评估器
        evaluator = ExplanationMetrics(args)
        
        # 计算三个指标的平均值
        lir_avg, sep_avg, etd_avg = evaluator.evaluate_explanations()
        
        # 打印结果
        print(f"\n评估结果 (β=0.5):")
        print("-"*40)
        print(f"LIR: {lir_avg:.4f}")
        print(f"SEP: {sep_avg:.4f}")
        print(f"ETD: {etd_avg:.4f}")
        print("-"*40)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()