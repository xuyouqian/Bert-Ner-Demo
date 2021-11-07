from transformers import AutoTokenizer


import torch
class Config:
    def __init__(self):
        self.label_list = ['NS', 'NR', 'NT', 'O']
        self.label_map = {tmp: idx for idx, tmp in enumerate(self.label_list)}
        self.max_seq_length = 128

        self.model_path = 'chinese_L-12_H-768_A-12'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
