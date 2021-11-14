from transformers import AutoTokenizer

import torch
import os



class Config:
    def __init__(self):
        self.label_list = ['NS', 'NR', 'NT', 'O']
        self.label_map = {tmp: idx for idx, tmp in enumerate(self.label_list)}
        self.max_seq_length = 150
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path = 'chinese_L-12_H-768_A-12'
        self.model_path = os.path.join(current_dir, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = 768
        self.dropout = 0.1
        self.learning_rate = 5e-5

        self.weight_end = 1
        self.weight_span = 1
        self.weight_start = 1
        self.num_train_epochs = 200

        self.clip_grad = 1
        self.n_gpu = 1
        self.entity_threshold = 0.5
        self.entity_sign = 'flat'

        self.batch_size = 2
        self.checkpoint = 15
        self.export_model = True

        output_dir = ''
if __name__ == '__main__':
    config = Config()
