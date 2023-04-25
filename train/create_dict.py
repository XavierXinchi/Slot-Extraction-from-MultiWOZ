import csv
from collections import defaultdict

# read tsv file


def read_tsv(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

# create dictionary


def create_dict(data):
    domain_dict = defaultdict(list)
    for row in data:
        speaker, utterance, domain_list = row
        domain_list = domain_list.strip('][').split(
            ', ')  # convert string to list
        if speaker == 'USER':
            for domain in domain_list:
                cleaned_domain = domain.strip("'")  # remove single quotes
                domain_dict[cleaned_domain].append(utterance)
    return domain_dict


def write_dict_to_file(domain_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for key, values in domain_dict.items():
            f.write(f"{key}: {', '.join(values)}\n")


def main():
    file_name = './train/dialogues_001_utterances.tsv'
    output_file = 'output.txt'
    data = read_tsv(file_name)
    domain_dict = create_dict(data)
    # write_dict_to_file(domain_dict, output_file)
    print(domain_dict)
    # for key, values in domain_dict.items():
    #     print(f"{key}: {len(values)}")

if __name__ == '__main__':
    main()
