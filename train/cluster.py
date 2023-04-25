from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import csv
from collections import defaultdict
from tqdm import tqdm
import re

# Loading pre-trained BERT models using the huggingface library
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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
        dialogue_id, speaker, utterance, domain_list = row
        domain_list = domain_list.strip('][').split(
            ', ')  # convert string to list
        if speaker == 'USER':
            for domain in domain_list:
                cleaned_domain = domain.strip("'")  # remove single quotes
                domain_dict[cleaned_domain].append(utterance)
    return domain_dict


def cluster_utterances(domain_dict):
    clustered_domains = {}

    # 1. Check if there are CUDA devices available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Move the model to GPU (if available)
    model.to(device)

    for domain, utterances in tqdm(domain_dict.items(), desc="Clustering domains"):
        encoded_utterances = tokenizer(
            utterances, padding=True, truncation=True, return_tensors="pt")

        # Move input data to GPU (if available)
        encoded_utterances = {key: value.to(
            device) for key, value in encoded_utterances.items()}

        with torch.no_grad():
            embeddings = model(
                **encoded_utterances).last_hidden_state[:, 0, :].cpu().numpy()

        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        clustered_domains[domain] = defaultdict(
            lambda: {'utterances': [], 'word_distribution': {}})
        for i, label in enumerate(labels):
            clustered_domains[domain][label]['utterances'].append(
                utterances[i])

        # Compute word distribution for each cluster in each domain
        for cluster_label, cluster_info in clustered_domains[domain].items():
            word_count = defaultdict(int)
            total_words = 0
            cluster_utterances = cluster_info['utterances']

            for utterance in cluster_utterances:
                # Extract words without punctuation
                words = re.findall(r'\b[a-zA-Z]+\b ', utterance) # \b\w+\b for words with numbers
                for word in words:
                    if word.lower() not in stop_words:  # Only count non-stopwords
                        word_count[word] += 1
                        total_words += 1

            # Sort word distribution by frequency in descending order
            sorted_word_distribution = {
                word: count / total_words for word, count in sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            }
            clustered_domains[domain][cluster_label]['word_distribution'] = sorted_word_distribution

    return clustered_domains


def main():
    file_name = './train/dialogues_001_utterances.tsv'
    data = read_tsv(file_name)
    domain_dict = create_dict(data)
    clustered_domains = cluster_utterances(domain_dict)

    # Print Clustering Results
    for domain, clusters in clustered_domains.items():
        print(f"Domain: {domain}")
        for cluster_label, cluster_info in clusters.items():
            print(f"\tCluster {cluster_label}:")
            print("\tUtterances:")
            for utterance in cluster_info['utterances']:
                print(f"\t\t{utterance}")
            print("\tWord Distribution:")
            for word, distribution in cluster_info['word_distribution'].items():
                print(f"\t\t{word}: {distribution}")


if __name__ == '__main__':
    main()
