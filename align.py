"""
@dingtian
Step2
用于给预处理过的初步json按照标点分段，并给每段增加头尾时间戳
需要python 3.10 环境来安装uroman
dtconda-whisperx
用法：
python align.py --json_file /mnt/data/dingtian/whisper-fintune-dt/it-ft/train_timestamp/dev.json --output_json /mnt/data/dingtian/whisper-fintune-dt/it-ft/dev.json --bundle_path /mnt/data/dingtian/prepare_whisper_ft
"""

import uroman as ur
import torch
import torchaudio
import re
import os
import json
import argparse
from typing import List

def normalize_uroman(text):
    """Normalize text for romanization"""
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', '', text)
    return text.strip()

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    """Compute alignment between waveform and transcript"""
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

# Compute average score weighted by the span length
def _score(spans):
    """Compute score for alignment spans"""
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def process_json_file(json_file):
    """Process JSON file and convert sentences into required format"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Define Italian punctuation marks (and similar marks for sentence splitting)
    punctuation_pattern = r'[.,!?;]'  # punctuation
    punctuation_ch = r'[，。！？“”‘’（）【】《》、；：–]'

    all_data = []

    # Process each audio entry in the JSON data
    for entry in data:
        audio_path = entry["audio"]["path"]
        sentence = entry["sentences"][0]["text"]
        sentence = sentence.replace("’", "'")
        sentence = sentence.replace("‘", "'")
        sentence = sentence.replace("“", "\"")
        sentence = sentence.replace("”", "\"")
        re.sub(punctuation_ch, '', sentence)
        duration = entry["duration"]

        start_time = entry["sentences"][0]['start']
        end_time = entry["sentences"][0]['end']

        # Split sentence by punctuation while keeping the punctuation
        raw_text = re.split(f'({punctuation_pattern})', sentence)  # Split and keep punctuation
        raw_text = [word.strip() for word in raw_text if word.strip()]  # Remove extra spaces

        # Avoid splitting on decimal points
        i = 0
        while i < len(raw_text):
            if re.match(r'[.]', raw_text[i]):
                if i > 0 and i < len(raw_text) - 1 and raw_text[i-1][-1].isdigit() and raw_text[i+1][0].isdigit():
                    # Merge decimal point with surrounding numbers
                    raw_text[i-1] = raw_text[i-1] + raw_text[i] + raw_text[i+1]
                    raw_text.pop(i)  # Remove the decimal point
                    raw_text.pop(i)  # Remove the next number
                else:
                    raw_text[i-1] += raw_text[i]
                    raw_text.pop(i)
            elif re.match(punctuation_pattern, raw_text[i]):
                # Append punctuation to the previous word
                if i > 0:
                    raw_text[i-1] += raw_text[i]
                    raw_text.pop(i)
                else:
                    i += 1
            elif not any(char.isalpha() for char in raw_text[i]):
                raw_text[i-1] += raw_text[i]
                raw_text.pop(i)
            else:
                i += 1
        print(raw_text)

        # 只有一句，下一个
        if len(raw_text)==1:
            sentences = [{"start": start_time, "end": end_time, "text": sentence}]
            data_entry = {
                "audio": {
                    "path": audio_path
                },
                "language": entry["language"],  # Language is already provided in the JSON
                "sentences": sentences,
                "duration": duration
            }
            all_data.append(data_entry)
            continue

        # Normalize the text
        text_normalized = [
            normalize_uroman(uroman.romanize_string(cut))
            for cut in raw_text if normalize_uroman(uroman.romanize_string(cut)).strip()
        ]

        # Load waveform
        print(f"Loading audio: {audio_path}")
        # Check if the file is mp3
        if audio_path.lower().endswith('.mp3'):
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            # Resample to 16000 Hz if it's an MP3 file
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                sr = 16000
        else:
            waveform, sr = torchaudio.load(audio_path)
        assert sr == bundle.sample_rate
        
        transcript = text_normalized
        tokens = tokenizer(transcript)

        # Add try-except block around alignment computation
        try:
            emission, token_spans = compute_alignments(waveform, transcript)
            num_frames = emission.size(1)

            # Prepare to create sentences with timestamps
            sentences = []
            for i in range(len(raw_text)):
                x0, x1 = preview_word(waveform, token_spans[i], num_frames)
                x0, x1 = round(x0, 2), round(x1, 2)
                if i == 0:
                    start = start_time
                    if raw_text[i][-2].isdigit() and i < len(raw_text) - 1:
                        x, y = preview_word(waveform, token_spans[i+1], num_frames)
                        x, y = round(x, 2), round(y, 2)
                        end = x
                    else:
                        end = x1
                elif i == len(raw_text) - 1:
                    end = end_time
                    if raw_text[i][0].isdigit():
                        x, y = preview_word(waveform, token_spans[i-1], num_frames)
                        x, y = round(x, 2), round(y, 2)
                        start = y
                    else:
                        start = x0
                elif raw_text[i][-2].isdigit() and i < len(raw_text) - 1:
                    x, y = preview_word(waveform, token_spans[i+1], num_frames)
                    x, y = round(x, 2), round(y, 2)
                    start = x0
                    end = x
                else:
                    start, end = x0, x1

                sentences.append({"start": start, "end": end, "text": raw_text[i]})

            # Create the full data structure for the current audio entry
            data_entry = {
                "audio": {
                    "path": audio_path
                },
                "sentence": sentence,
                "language": entry["language"],  # Language is already provided in the JSON
                "sentences": sentences,
                "duration": duration
            }

            all_data.append(data_entry)

        except RuntimeError as e:
            if "targets length is too long for CTC" in str(e):
                print(f"Skipping audio {audio_path} due to alignment error: {e}")
                continue
            else:
                raise e  # Re-raise other RuntimeErrors

    return all_data

def save_json(data, output_file):
    """Save the processed data into JSON format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    """Main function to process TSV file and generate JSON"""
    data = process_json_file(args.json_file)
    save_json(data, args.output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TSV file to generate JSON alignment data")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input TSV file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--bundle_path", type=str, required=True, help="Path to the model bundle")
    
    args = parser.parse_args()
    
    uroman = ur.Uroman()   # Load uroman data (takes about a second or so)
    bundle = torchaudio.pipelines.MMS_FA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.get_model(with_star=False, dl_kwargs={"model_dir":args.bundle_path}).to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    def preview_word(waveform, spans, num_frames, sample_rate=bundle.sample_rate):
        """Preview words and get time spans for each token"""
        ratio = waveform.size(1) / num_frames
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        return x0/sample_rate, x1/sample_rate

    main(args)

