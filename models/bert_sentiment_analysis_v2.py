import torch
import torch.nn as nn
from models import bert_model

class BertSentimentAnalysis(nn.Module):
    def __init__(self):
        super(BertSentimentAnalysis,self).__init__()
        self.bert_model=bert_model.BertModel()

    def init_positional_enc(self,max_seq_len,vocab_size):
        for pos in max_seq_len:

    def forward(self,input_tensor):
