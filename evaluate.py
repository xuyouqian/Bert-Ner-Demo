import torch

from typing import List, Set, Tuple


def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    eval_loss = 0
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    span_gold_lst = []
    end_gold_lst = []
    eval_steps = 0
    ner_cate_lst = []

    for input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos)
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()
        # bach size = 1 时有bug  span_logits 应该是[batch size=1,seq_len,seq_len]
        # 但是这里成了[seq len,seq_len]
        if len(span_logits.shape) != 3:
            span_logits = span_logits.unsqueeze(0)
        span_logits = span_logits.detach().cpu().numpy().tolist()
        span_label = span_logits

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        eval_loss += tmp_eval_loss.mean().item()
        mask_lst += input_mask
        eval_steps += 1

        # 预测值
        start_pred_lst += start_label
        end_pred_lst += end_label
        span_pred_lst += span_label

        # 真实值
        start_gold_lst += start_pos
        end_gold_lst += end_pos
        span_gold_lst += span_pos



    eval_accuracy, eval_precision, eval_recall, eval_f1 = nested_ner_performance(start_pred_lst, end_pred_lst,
                                                                                     span_pred_lst, start_gold_lst,
                                                                                     end_gold_lst, span_gold_lst,
                                                                                     ner_cate_lst, label_list,
                                                                                     threshold=config.entity_threshold,
                                                                                     dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1




def nested_transform_span_triple(start_labels, end_labels, span_labels, ner_cate, print_info=False, threshold=0.5):
    span_triple_lst = []
    # element in span_triple_lst is (ner_cate, start_index, end_index)

    start_labels = [idx for idx, tmp in enumerate(start_labels) if tmp != 0]
    end_labels = [idx for idx, tmp in enumerate(end_labels) if tmp != 0]

    for tmp_start in start_labels:
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        for candidate_end in tmp_end:
            if span_labels[tmp_start][candidate_end] >= threshold:
                tmp_tag = Tag('一', ner_cate, tmp_start, candidate_end)
                span_triple_lst.append(tmp_tag)

    return span_triple_lst


def nested_ner_performance(pred_start, pred_end, pred_span, \
                           gold_start, gold_end, gold_span, ner_cate, label_lst, threshold=0.5, dims=2):
    # transform label index to label tag: "3" -> "per"
    # label_list: ["PER", "ORG", "O", "LOC"]

    if dims == 1:
        # transform (begin, end, span) labels to span list
        pred_span_triple = nested_transform_span_triple(pred_start, pred_end, pred_span, ner_cate, threshold=threshold)
        gold_span_triple = nested_transform_span_triple(gold_start, gold_end, gold_span, ner_cate, threshold=threshold)

        return pred_span_triple, gold_span_triple
    elif dims == 2:
        pred_span_triple_lst = []
        gold_span_triple_lst = []

        acc_lst = []

        for pred_start_item, pred_end_item, pred_span_item, gold_start_item, gold_end_item, gold_span_item, ner_cate_item in zip(
                pred_start, pred_end, pred_span, gold_start, gold_end, gold_span, ner_cate):
            pred_span_triple, gold_span_triple = nested_ner_performance(pred_start_item, pred_end_item, \
                                                                        pred_span_item, gold_start_item, gold_end_item,
                                                                        gold_span_item, ner_cate_item, label_lst,
                                                                        dims=1)

            pred_span_triple_lst.append(pred_span_triple)
            gold_span_triple_lst.append(gold_span_triple)

            tmp_acc_s = compute_acc(pred_start_item, gold_start_item)
            tmp_acc_e = compute_acc(pred_end_item, gold_end_item)
            acc_lst.append((tmp_acc_s + tmp_acc_e) / 2.0)

        span_precision, span_recall, span_f1 = nested_calculate_f1(pred_span_triple_lst,
                                                                                gold_span_triple_lst, dims=2)
        average_acc = sum(acc_lst) / (len(acc_lst) * 1.0)

        return average_acc, span_precision, span_recall, span_f1

    else:
        raise ValueError("Please notice that dims can only be 1 or 2 !")



def compute_acc(pred_label, gold_label):
    dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
    acc = len(dict_match) / float(len(gold_label))
    return acc


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> Tuple[str, List[Tag]]:
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            idx += 1
            continue

    sentence = "".join(term for term, _ in char_label_list)
    return sentence, tags


def nested_calculate_f1(pred_span_tag_lst, gold_span_tag_lst, dims=2):
    if dims == 2:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_span_tags, gold_span_tags in zip(pred_span_tag_lst, gold_span_tag_lst):
            pred_set = set((tag.begin, tag.end, tag.tag) for tag in pred_span_tags)
            gold_set = set((tag.begin, tag.end, tag.tag) for tag in gold_span_tags)

            for pred in pred_set:
                if pred in gold_set:
                    true_positives += 1
                else:
                    false_positives += 1

            for pred in gold_set:
                if pred not in pred_set:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return precision, recall, f1

    else:
        raise ValueError("Can not be other number except 2 !")
