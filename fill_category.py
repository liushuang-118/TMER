# -*- coding: utf-8 -*-
import jsonlines
from collections import defaultdict
import re

# 输入输出文件路径
INPUT_JSON = r"D:\Thesis_Project\Models\TMER\data\Amazon_Music\old\meta_Musical_Instruments.json"
OUTPUT_JSON = r"D:\Thesis_Project\Models\TMER\data\Amazon_Music\old\meta_Beauty_filled.json"

# 关键词映射，用于从 title 或 description 提取细分类
CATEGORY_KEYWORDS = {
    'Makeup': ['lip', 'eye', 'mascara', 'foundation', 'blush', 'eyeliner', 'concealer', 'primer'],
    'Skincare': ['cream', 'lotion', 'serum', 'toner', 'mask', 'moisturizer', 'cleanser'],
    'Hair': ['shampoo', 'conditioner', 'hair', 'styling', 'treatment'],
    'Fragrance': ['perfume', 'cologne', 'scent', 'fragrance'],
    'Tools': ['brush', 'applicator', 'spoon', 'tool', 'comb', 'tweezers'],
    'Bath & Body': ['soap', 'bath', 'body', 'lotion', 'scrub', 'shower']
}

def extract_category_from_text(text):
    """根据关键词从文本中提取类别"""
    if not text:
        return None
    # 如果是列表，把列表拼成字符串
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return cat
    return None

# 1. 读取 JSON Lines 文件
data = []
with jsonlines.open(INPUT_JSON) as reader:
    for obj in reader:
        data.append(obj)

print(f"加载完成，总商品数: {len(data)}")
print("示例数据 keys:", list(data[0].keys()))

# 2. 构建 asin -> category 映射（已有 category 或 main_cat）
asin2cat = {}
for item in data:
    cat = item.get('category')
    if cat and isinstance(cat, list) and len(cat) > 0:
        asin2cat[item['asin']] = cat
    else:
        main_cat = item.get('main_cat')
        if main_cat:
            asin2cat[item['asin']] = [main_cat]

print(f"初步已知 category 数量: {len(asin2cat)}")

# 3. 填充 category
for item in data:
    cat = item.get('category')
    if cat and len(cat) > 0:
        continue  # 已有 category

    filled = False

    # 3.1 尝试 also_buy
    also_buy = item.get('also_buy', [])
    for asin in also_buy:
        if asin in asin2cat:
            item['category'] = asin2cat[asin]
            filled = True
            break

    if filled:
        continue

    # 3.2 尝试 title 和 description
    for field in ['title', 'description', 'feature']:
        text = item.get(field)
        cat_from_text = extract_category_from_text(text)
        if cat_from_text:
            item['category'] = [cat_from_text]
            filled = True
            break

    if filled:
        continue

    # 3.3 使用 main_cat
    main_cat = item.get('main_cat')
    if main_cat:
        item['category'] = [main_cat]
        filled = True

    # 3.4 最后保底空列表
    if not filled:
        item['category'] = []

# 4. 保存新的完整 JSON 文件（JSON Lines 格式）
with jsonlines.open(OUTPUT_JSON, mode='w') as writer:
    for item in data:
        writer.write(item)

print(f"新的完整 JSON 文件已保存: {OUTPUT_JSON}")

# 5. 简单统计
empty_cat = sum(1 for item in data if not item['category'])
print(f"补全后仍为空 category 的商品数: {empty_cat}")
