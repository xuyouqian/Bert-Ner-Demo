from torch.utils.data import  DataLoader
from transformers import AutoTokenizer
from utils import NerDataset
import torch
import json

tokenizer = AutoTokenizer.from_pretrained('../chinese_L-12_H-768_A-12')
label_list = ['NS', 'NR', 'NT', 'O']
label_map = {tmp: idx for idx, tmp in enumerate(label_list)}

with open('data.train', "r", encoding='utf8') as f:
    input_data = json.load(f)

ner = NerDataset(input_data, label_map, tokenizer, 128)

for i in ner:
    print(i)


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[1], dtype=torch.long)
    input_mask = torch.tensor(batch[2], dtype=torch.long)
    segment_ids = torch.tensor(batch[3], dtype=torch.long)
    start_pos = torch.tensor(batch[4], dtype=torch.long)
    end_pos = torch.tensor(batch[5], dtype=torch.long)
    span_position = torch.tensor(batch[6], dtype=torch.long)
    entity_label = torch.tensor(batch[7], dtype=torch.long)

    del batch
    return input_ids, input_mask, segment_ids, start_pos, end_pos, span_position, entity_label


loader = DataLoader(ner, batch_size=2, collate_fn=collate_fn)
for i in loader:
    print(i[0].shape)
    print(i[-2].shape)
    print(i[-1].shape)
