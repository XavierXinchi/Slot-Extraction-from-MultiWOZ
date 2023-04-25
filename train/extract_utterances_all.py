import json
import os


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_utterances(data):
    utterances = []
    for item in data:
        dialogue_id = item['dialogue_id']
        services = ", ".join(item['services'])
        for turn in item['turns']:
            speaker = turn['speaker']
            utterance = turn['utterance']
            utterances.append(dialogue_id + "\t" + speaker +
                              "\t" + utterance + "\t" + services)
    return utterances


def save_utterances(utterances, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for utterance in utterances:
            f.write(utterance + '\n')


def create_tsv(filename, output_file):
    separator = "\t"
    with open(filename, "r") as file:
        data = [line.strip().split(separator) for line in file]
    labeled_data = [(dialogue_id, speaker, utterance, domain) if len(item) == 4 else (
        dialogue_id, speaker, utterance, None) for item in data for dialogue_id, speaker, utterance, *domain in [item]]
    with open(output_file, "w") as file:
        for dialogue_id, speaker, utterance, domain in labeled_data:
            file.write(
                f"{dialogue_id}{separator}{speaker}{separator}{utterance}{separator}{domain}\n")


def main():
    input_directory = './train'
    output_txt_file = './train/all_dialogues_utterances.txt'
    output_tsv_file = './train/all_dialogues_utterances.tsv'

    all_utterances = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            input_file = os.path.join(input_directory, filename)
            data = load_data(input_file)
            utterances = extract_utterances(data)
            all_utterances.extend(utterances)

    save_utterances(all_utterances, output_txt_file)
    print(f'Utterances extracted: {len(all_utterances)}')
    create_tsv(output_txt_file, output_tsv_file)
    
if __name__ == '__main__':
    main()
