from typing import List
import os
import requests

def generate_character_lexicon(corpora_path: str) -> List[str]:
    with open(corpora_path, encoding='utf-8') as f:
        corpora = f.read()
    characters = sorted(list(set(corpora)))
    return characters

def download_tiny_shakespeare():
    input_file_path = os.path.join('./corpora_data/input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
