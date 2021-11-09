from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertModel
from torch import nn

import torch


class BertQueryNER(nn.Module):
    def __init__(self, config):
        super(BertQueryNER, self).__init__()

        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.dropout)
        self.hidden_size = config.hidden_size
        self.bert = BertModel.from_pretrained(config.model_path)
        self.loss_wb = config.weight_start
        self.loss_we = config.weight_end
        self.loss_ws = config.weight_span

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, span_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            span_positions: (batch x max_len x max_len)
                span_positions[k][i][j] is one of [0, 1],
                span_positions[k][i][j] represents whether or not from start_pos{i} to end_pos{j} of the K-th sentence in the batch is an entity.
        """

        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)

        sequence_heatmap = sequence_output.last_hidden_state  # batch x seq_len x hidden
        pooler_output = sequence_output.pooler_output
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_heatmap)  # batch x seq_len x 2
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # the shape of start_end_concat[0] is : batch x 1 x seq_len x 2*hidden

        span_matrix = torch.cat([start_extend, end_extend], 3)  # batch x seq_len x seq_len x 2*hidden

        span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        span_logits = torch.squeeze(span_logits)  # batch x seq_len x seq_len



        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            span_loss_fct = Focal_Loss()  # nn.BCEWithLogitsLoss()
            # span_loss = span_loss_fct(span_logits, span_positions.float())
            span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_positions.view(batch_size, -1))
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * span_loss
            return total_loss
        else:
            span_logits = torch.sigmoid(span_logits)  # batch x seq_len x seq_len
            start_logits = torch.argmax(start_logits, dim=-1)
            end_logits = torch.argmax(end_logits, dim=-1)

            return start_logits, end_logits, span_logits


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size / 2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
        super(Focal_Loss, self).__init__()

    def forward(self, pred, target):
        """
        :param pred: batch size,seq_len * seq_len
        :param label: batch size,seq_len * seq_len
        :return:
        """
        pred = F.sigmoid(pred).view(-1)
        target = target.view(-1)

        alpha_factor = torch.where(torch.eq(target, 1), 1 - self.alpha, self.alpha)
        focal_weight = torch.where(torch.eq(target, 1), 1 - pred, pred)
        final_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        loss = -(target * (torch.log(pred)) + (1 - target) * (torch.log(1 - pred)))
        loss = torch.mean(final_weight * loss)
        return loss
