from utils import *
from encoder import EncoderandDecoder
from language_models import BigramLanguageModel
import torch
from train import train

def main():

    lexicon = generate_character_lexicon(corpora_path='./corpora_data/input.txt')
    ed = EncoderandDecoder(list_of_tokens=lexicon)
    bgm = BigramLanguageModel(vocab_size=len(lexicon))

    train(model=bgm,
          corpora_path='./corpora_data/input.txt',
          block_size=8,
          learning_rate=1e-3,
          batch_size=4,
          epoch=5)

if __name__ == "__main__":
    main()