from model import BertQueryNER
from config import Config
from test_data import loader
from utils import load_model

config = Config()

model, optimizer, sheduler, device, n_gpu = load_model(config)

if __name__ == ' __main__':
    for input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate in loader:
        loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                     start_positions=start_pos, end_positions=end_pos, span_positions=span_pos)
        print(loss)
