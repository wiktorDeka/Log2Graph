from logparser.Drain import LogParser
import re
import json
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer

root_path = r'/home/ubuntu/bsc/BootDet/Log2Graph'


def parse_dataset(dataset_name, st, depth):
    input_dir = f'{root_path}/Data/{dataset_name}' # The input directory of log file
    output_dir = f'{root_path}/Data/{dataset_name}'  # The output directory of parsing results
    log_file = f'{dataset_name}.log'  # The input log file name
    log_format = '<GroupId> <DateTime> <Host> <Component>:<Content>' # Define log format to split message fields

    regex = [
        r'https?://(?:[\w\-]+\.)+[a-zA-Z]{2,}(?:/[\w\-./?%&=]*)?',  # URL
        r'\[mem\s+[^\]-]+-[^\]]+\]',                      # zakresy pamięci e820: [mem 0x...-0x...]
        r'\b[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b',      # PCI bus:device.function (BDF)
        r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b',  # UUID/GUID
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b(?::\d{1,5})?',      # IPv4 z opcjonalnym portem
        r'\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b',      # MAC
        r'/org/[A-Za-z0-9_/\.]+',                         # ścieżki obiektów D-Bus
        r'\b[\w\-\.]+\.(?:service|socket|target|mount|slice|timer)\b',  # jednostki systemd

        # --- ścieżki, wersje, hex: ---
        r'\b/(?:[\w\-.]+/)*[\w\-.]*\b',                   # ścieżki plików (dość agresywne)
        r'\b\d+(?:\.\d+){1,3}[-\w]*\b',                   # wersje typu 1.2.3-rc1
        r'0x[0-9a-fA-F]+',                                # liczby heksadecymalne
        r'\[mem\s+[^\]-]+-[^\]]+\]',  # memory address

        # --- wielkości i identyfikatory: ---
        r'\b\d+[KMG]?B\b',                                # rozmiary: 512KB, 4MB, 1GB
        r'\b(PID|TID|UID|GID)?\s*[:=]?\s*\d+\b',          # PID/TID/UID/GID lub gołe ID
    ]

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, maxChild=120)
    parser.parse(log_file)

def generate_embeddings(dataset_name):
    # ---------------------------------------------------
    # 1) Load CSV templates
    # ---------------------------------------------------
    csv_path = f'{root_path}/Data/{dataset_name}/{dataset_name}.log_templates.csv'
    df = pd.read_csv(csv_path, )
    texts = df['EventTemplate'].dropna().astype(str).tolist()

    print(f'Loaded {len(texts)} templates from {csv_path}')

    # ---------------------------------------------------
    # 2) Load GloVe vectors
    # ---------------------------------------------------

    d = 0
    if dataset_name == 'Linux':
        glove_input_file = f'{root_path}/Data/Gloves/glove.6B.200d.txt'
        word2vec_output_file = f'{root_path}/Data/Gloves/glove.6B.200d.w2v.txt'
        d = 200
    elif dataset_name == 'Windows':
        glove_input_file = f'{root_path}/Data/Gloves/glove.6B.50d.txt'
        word2vec_output_file = f'{root_path}/Data/Gloves/glove.6B.50d.w2v.txt'
        d = 50

    glove2word2vec(glove_input_file, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print(f'Loaded GloVe embeddings. ({d}D)')

    # ---------------------------------------------------
    # 3) TF-IDF setup
    # ---------------------------------------------------
    def tokenize(text):
        # Keep words only (drop <*> etc.)
        return [t.lower() for t in re.findall(r"\w+", text)]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.vocabulary_
    rev_vocab = {v: k for k, v in vocab.items()}
    print('TF-IDF model fitted on all templates.')

    # ---------------------------------------------------
    # 4) TF-IDF weighted GloVe embedding
    # ---------------------------------------------------
    def tfidf_glove_vector(text: str) -> np.ndarray:
        vec = np.zeros(model.vector_size, dtype=np.float32)
        weights_sum = 0.0

        tfidf_vec = vectorizer.transform([text]).tocoo()
        for idx, weight in zip(tfidf_vec.col, tfidf_vec.data):
            token = rev_vocab[idx]
            if token in model:
                vec += weight * model[token]
                weights_sum += weight

        if weights_sum > 0:
            vec /= weights_sum
        return vec

    # ---------------------------------------------------
    # 5) Compute embeddings and export JSON
    # ---------------------------------------------------
    def vector_to_json(vec: np.ndarray):
        return {str(i): float(v) for i, v in enumerate(vec)}

    result = {}
    for text in texts:
        result[text] = vector_to_json(tfidf_glove_vector(text))

    output_path = f'{root_path}/Data/Gloves/Results/{dataset_name}_embeddings.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'Saved embeddings for {len(texts)} templates to {output_path}')
