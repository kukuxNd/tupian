import pandas as pd
import json

# 读取JSONL文件
jsonl_file = 'd:\\我的文档\\Pictures\\[OCR]_Pictures_20250111_0124.jsonl'
data = []

with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 提取数据并转换为DataFrame
rows = []
for item in data:
    for entry in item['data']:
        row = {
            'code': item['code'],
            'box': entry['box'],
            'score': entry['score'],
            'text': entry['text'],
            'end': entry['end']
        }
        rows.append(row)

df = pd.DataFrame(rows)

# 将DataFrame保存为CSV文件
csv_file = 'output.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print(f"数据已成功转换并保存到 {csv_file}")