"""
@dingtian
Step3
用于合并common voice和fleurs的json文件，并划分5%出来当作验证集
"""

import json
import random

def merge_json_files(file_list):
    merged_data = []

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                # 如果是列表，直接扩展合并
                merged_data.extend(data)
            elif isinstance(data, dict):
                # 如果是字典，包装成列表后合并
                merged_data.append(data)
            else:
                print(f"文件 {file_path} 格式不支持，仅支持字典或列表")
    
    # 返回合并后的数据
    return merged_data

def split_data(data, dev_percentage=5):
    """将数据划分为训练集和验证集"""
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算验证集的大小
    dev_size = int(len(data) * dev_percentage / 100)
    
    # 划分数据
    dev_data = data[:dev_size]
    train_data = data[dev_size:]
    
    return train_data, dev_data

def save_json(data, output_file):
    """保存数据到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {output_file}")

if __name__ == "__main__":
    input_files = [
        "/mnt/data/dingtian/whisper-fintune-dt/it-ft/train_ts/train_cut.json",
        "/mnt/data/dingtian/whisper-fintune-dt/it-ft/train_ts/train.json",
    ]
    
    # 合并数据
    merged_data = merge_json_files(input_files)
    
    # 划分数据集
    train_data, dev_data = split_data(merged_data, dev_percentage=0)
    
    # 保存训练集和验证集
    save_json(train_data, "/mnt/data/dingtian/whisper-fintune-dt/it-ft/train_ts/train_all.json")
    save_json(dev_data, "/mnt/data/dingtian/whisper-fintune-dt/it-ft/d.json",)
