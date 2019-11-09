import torch.nn as nn
from models.bert_model import *

class Bert_Sentiment_Analysis(nn.Module):
    def __init__(self,config):
        super(Bert_Sentiment_Analysis,self).__init__()
        self.bert=BertModel(config)
        self.final_dense=nn.Linear(config.hidden_size,1)
        self.activation=nn.Sigmoid()

    def compute_loss(self,predictions,labels):
        predictions=predictions.view(-1)
        labels=labels.float().view(-1)
        epsilon=1e-8
        # -q*log(p)-(1-q)*log(1-p)
        loss= -labels*torch.log(predictions+epsilon)-(torch.tensor(1.0)-labels)*torch.log(torch.tensor(1.0)-predictions+epsilon)
        loss=torch.mean(loss)
        return loss

    def forward(self,text_input,positional_enc,labels=None):
        encoded_layers,_=self.bert(text_input,positional_enc,output_all_encoded_layers=True)
        # 第3个BertEncoder的输出，为什么????????
        sequence_output=encoded_layers[2]
        # 取CLS对应的维度
        # 为什么选择第一行进行二分类预测？？？？
        first_token_tensor=sequence_output[:,0,:]

        predictions=self.final_dense(first_token_tensor)
        predictions=self.activation(predictions)

        if labels is not None:
            loss=self.compute_loss(predictions,labels)
            return predictions,loss
        else:
            return predictions
