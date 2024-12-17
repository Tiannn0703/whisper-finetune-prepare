"""
@dingtian
Step1
用于从原始commonvoice、fleurs数据生成whisper微调的json文件，通过VAD加头尾时间戳并检测无活动的语音，同时通过whisper tokenizer检查超长的label sentence
dtconda：whisper-ft
用法：
更换最下面的路径，直接run即可
"""

import os
import json
import pandas as pd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import shutil
import torchaudio
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, AutoTokenizer, WhisperProcessor

model_path = "/mnt/data/hanzebei/asr/whisper-base"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, language="bo", task="transcribe", use_fast=False)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def generate_json_from_tsv(tsv_path, audio_dir, dataset, language):
    vad_model = load_silero_vad()
    # 读取 TSV 文件
    data = pd.read_csv(tsv_path, sep="\t")
    result = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        if dataset == "cv":
            sentence = row["sentence"]
            audio_file = os.path.join(audio_dir, row["path"])
        else:
            audio_path = row[1]
            audio_file = os.path.join(audio_dir,audio_path)
            sentence = row[2]

        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            continue

        # 检查labels是否超长
        labels = tokenizer(sentence).input_ids
        labels_length = len(labels)
        if labels_length > 448:
            print(f"Too long labels: {audio_file}")
            print(f"Sentence: {sentence}")
            continue

        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_file)

        # 计算音频总时长（秒）
        total_duration = waveform.size(1) / sample_rate

        # 使用 VAD 模型获取分段信息
        wav = read_audio(audio_file, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav,vad_model,sampling_rate=16000,return_seconds=True)
        if not speech_timestamps:
            print(f"No speech detected in audio: {audio_file}")
            print(f"Sentence: {sentence}")
            speech_timestamps = [{"start": 0, "end": total_duration}]

        sentences = []
        sentences.append({
            "start": speech_timestamps[0]["start"],
            "end": speech_timestamps[-1]["end"],
            "text": sentence  # 此处假设整个句子与每段音频对应
        })

        # 构建 JSON 数据
        result.append({
            "audio": {
                "path": audio_file
            },
            "language": language,
            "sentences": sentences,
            "duration": total_duration
        })

    return result

def process_dataset(base_audio_dir, transcript_dir, output_json_dir, dataset, language):
    combined_data = []
    # 遍历 dev, train, test 文件夹
    for subset in ["dev", "train", "test"]:
    #for subset in ["train"]:
        tsv_path = os.path.join(transcript_dir, f"{subset}.tsv")
        audio_subset_dir = os.path.join(base_audio_dir, subset)

        # 找到音频文件所在的唯一子文件夹
        subfolders = [d for d in os.listdir(audio_subset_dir) if os.path.isdir(os.path.join(audio_subset_dir, d))]
        if dataset=="cv" and len(subfolders) != 1:
            # 如果子文件夹不止一个，将它们合并到一个新的文件夹中
            merged_dir = os.path.join(audio_subset_dir, f"{subset}_merged")
            os.makedirs(merged_dir, exist_ok=True)
            print(f"Found multiple subfolders in {audio_subset_dir}: {subfolders}. Merging into {merged_dir}.")
            for folder in subfolders:
                folder_path = os.path.join(audio_subset_dir, folder)
                for item in os.listdir(folder_path):
                    source_path = os.path.join(folder_path, item)
                    target_path = os.path.join(merged_dir, item)
                    if os.path.isfile(source_path):
                        shutil.move(source_path, target_path)
                    elif os.path.isdir(source_path):
                        shutil.move(source_path, os.path.join(merged_dir, os.path.basename(source_path)))

            # 更新处理路径为合并后的文件夹
            audio_dir = merged_dir
        elif dataset=="cv" and len(subfolders) == 1:
            audio_dir = os.path.join(audio_subset_dir, subfolders[0])
        else: #fleurs
            audio_dir = os.path.join(base_audio_dir, subset)

        if not os.path.exists(tsv_path):
            print(f"TSV file not found for {subset}: {tsv_path}")
            continue

        print(f"Processing {subset}...")
        subset_data = generate_json_from_tsv(tsv_path, audio_dir, dataset, language)
        combined_data.extend(subset_data)

    # 保存为一个合并后的 JSON 文件
    with open(output_json_dir, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Combined JSON file saved to {output_json_dir}")

# 示例用法
base_audio_directory = "/mnt/data/zhengxuhui/common_voice_17_0/audio/vi"
transcript_directory = "/mnt/data/zhengxuhui/common_voice_17_0/transcript/vi"
output_json_directory = "/mnt/data/dingtian/whisper-fintune-dt/vi-ft/"
dataset = "cv" #cv, fleurs
language = "vi"

# base_audio_directory = "/mnt/data/hanzebei/asr/fleurs/data/vi_vn/audio"
# transcript_directory = "/mnt/data/hanzebei/asr/fleurs/data/vi_vn"
# output_json_directory = "/mnt/data/dingtian/whisper-fintune-dt/vi-ft/"
# dataset = "fleurs" #cv, fleurs
# language = "vi"

os.makedirs(output_json_directory, exist_ok=True)
output_json_path = os.path.join(output_json_directory, f"{dataset}.json")
process_dataset(base_audio_directory, transcript_directory, output_json_path, dataset, language)