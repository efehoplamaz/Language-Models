class EncoderandDecoder:
    def __init__(self, list_of_tokens) -> None:
        
        self.tokens = list_of_tokens
        self.basic_encoder_mapping = {character: i for i, character in enumerate(list_of_tokens)}
        self.basic_decoder_mapping = {i: character for i, character in enumerate(list_of_tokens)}
        pass

    def rudimentary_encoder(self, sentence):
        return [self.basic_encoder_mapping[char] for char in sentence]
    
    def rudimentary_decoder(self, encoded_sentence):
        return ''.join([self.basic_decoder_mapping[enc_int] for enc_int in encoded_sentence])

    def encode_subword_google_SentencePiece():
        pass

    def encode_openai_tiktoken():
        pass