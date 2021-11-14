from torch.utils.data import DataLoader
from utils import NerDataset, collate_fn
from config import Config

import json

config = Config()
tokenizer = config.tokenizer
label_list = config.label_list
label_map = config.label_map

with open('data.train', "r", encoding='utf8') as f:
    input_data = json.load(f)

# collate_fn=collate_fn
ner = NerDataset(input_data, label_map, tokenizer, 150)
loader = DataLoader(ner, batch_size=2, collate_fn=collate_fn)

if __name__ == '__main__':
    for i in ner:
        print(i)
    for i in loader:
        print(i[0].shape)
        print(i[-2].shape)
        print(i[-1].shape)
    pass
