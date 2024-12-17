import json

def filter_long_sentences(input_file, output_file):
    """
    Process the input JSON file and extract sentences longer than 1 second.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = []

    for entry in input_data:
        audio_path = entry.get("audio", {}).get("path")
        language = entry.get("language")
        sentences = entry.get("sentences", [])

        if len(sentences) == 1:
            continue

        for sentence in sentences:
            start = sentence.get("start")
            end = sentence.get("end")
            text = sentence.get("text")

            # Calculate duration of the sentence
            duration = end - start if start is not None and end is not None else 0
            duration = round(duration,2)

            # Only process sentences longer than 1 second
            if duration > 1:
                output_data.append({
                    "audio": {
                        "path": audio_path,
                        "start_time": start,
                        "end_time": end
                    },
                    "language": language,
                    "sentences": [
                        {
                            "start": 0.00,
                            "end": duration,
                            "text": text
                        }
                    ],
                    "duration": duration
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    input_file = "/mnt/data/dingtian/whisper-fintune-dt/it-ft/train_ts/train.json"
    output_file = "/mnt/data/dingtian/whisper-fintune-dt/it-ft/train_ts/train_cut.json"

    filter_long_sentences(input_file, output_file)
    print(f"Filtered data has been saved to {output_file}")
