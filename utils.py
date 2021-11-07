from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


def read_data(path):
    with open(path) as F:
        return json.loads(F)


def whitespace_tokenize(text):
    """
    去掉文本中的空格 并分词
    in: '我     是一个   随机   文本'
    return : ['我', '是一个', '随机', '文本']
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class NerDataset(Dataset):  # 继承Dataset
    def __init__(self, data, label_map, tokenizer, max_seq_length, test=False):  # __init__是初始化该类的一些基础参数
        self.data = data
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        context = self.data[index]["context"]
        query = self.data[index]["query"]
        end_position = self.data[index]["end_position"]

        entity_label = self.data[index]["entity_label"]
        is_impossible = self.data[index]["impossible"]
        span_position = self.data[index]["span_position"]

        start_position = self.data[index]["start_position"]

        query_tokens = self.tokenizer.tokenize(query)
        query_len = len(query_tokens)
        whitespace_doc = whitespace_tokenize(context)
        max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

        if len(start_position) == 0 and len(end_position) == 0:
            all_doc_tokens, doc_start_pos, doc_end_pos, doc_span_pos = \
                process_impossible_data(self.tokenizer, self.max_seq_length, whitespace_doc)
        else:
            all_doc_tokens, doc_start_pos, doc_end_pos, doc_span_pos = \
                process_possible_data(self.tokenizer, self.max_seq_length, whitespace_doc, start_position, end_position,
                                      span_position,
                                      query_len)

        assert len(all_doc_tokens) == len(doc_start_pos)
        assert len(all_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        # 到这里 doc_span_pos 已经完全处理好了 接下来要把query和doc拼接起来进行截长补短操作
        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]

        input_tokens = []
        segment_ids = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        for query_item in query_tokens:
            input_tokens.append(query_item)
            segment_ids.append(0)
            start_pos.append(0)
            end_pos.append(0)

        input_tokens.append("[SEP]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
        input_mask = [1] * len(input_tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            start_pos += padding
            end_pos += padding

        return input_tokens, input_ids, input_mask, segment_ids, start_pos, end_pos, doc_span_pos, is_impossible, \
               self.label_map[entity_label]




def process_impossible_data(tokenizer, max_seq_length, whitespace_doc):
    all_doc_tokens = []
    for token_item in whitespace_doc:
        tmp_subword_lst = tokenizer.tokenize(token_item)
        all_doc_tokens.extend(tmp_subword_lst)

    doc_start_pos = [0] * len(all_doc_tokens)
    doc_end_pos = [0] * len(all_doc_tokens)
    doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

    return all_doc_tokens, doc_start_pos, doc_end_pos, doc_span_pos


def process_possible_data(tokenizer, max_seq_length, whitespace_doc, start_position, end_position, span_position,
                          query_len):
    doc_start_pos = []
    doc_end_pos = []
    doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

    all_doc_tokens = []
    offset_idx_dict = {}

    fake_start_pos = [0] * len(whitespace_doc)
    fake_end_pos = [0] * len(whitespace_doc)

    for start_item in start_position:
        fake_start_pos[start_item] = 1
    for end_item in end_position:
        fake_end_pos[end_item] = 1

    for idx, (token, start_label, end_label) in enumerate(zip(whitespace_doc, fake_start_pos, fake_end_pos)):
        tmp_subword_lst = tokenizer.tokenize(token)
        offset_idx_dict[idx] = len(all_doc_tokens)
        if len(tmp_subword_lst) > 1:

            doc_start_pos.append(start_label)
            doc_start_pos.extend([0] * (len(tmp_subword_lst) - 1))

            doc_end_pos.append(end_label)
            doc_end_pos.extend([0] * (len(tmp_subword_lst) - 1))

            all_doc_tokens.extend(tmp_subword_lst)
        elif len(tmp_subword_lst) == 1:

            doc_start_pos.append(start_label)
            doc_end_pos.append(end_label)
            all_doc_tokens.extend(tmp_subword_lst)
        else:
            raise ValueError("Please check the result of tokenizer !!! !!! ")

    for span_item in span_position:
        s_idx, e_idx = span_item.split(";")
        if query_len + 2 + offset_idx_dict[int(s_idx)] < max_seq_length and \
                query_len + 2 + offset_idx_dict[int(e_idx)] < max_seq_length:
            doc_span_pos[query_len + 2 + offset_idx_dict[int(s_idx)]][
                query_len + 2 + offset_idx_dict[int(e_idx)]] = 1
        else:
            continue

    return all_doc_tokens, doc_start_pos, doc_end_pos, doc_span_pos
