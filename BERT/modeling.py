from data_process import read_data,InputDataSet
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


class BertForSeq(BertPreTrainedModel):

    def __init__(self,config):
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 7
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
            return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )

        output = outputs[1]
        output = self.dropout(output)
        output = self.classifier(output)
        return output

