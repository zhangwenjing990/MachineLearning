from dataset.inference_dataloader import preprocessing
from models.bert_sentiment_analysis import *
import configparser
import os
import json
import warnings
import torch
import math

from models.bert_model import BertConfig

class Sentiment_Analysis:
    def __init__(self,max_seq_len,batch_size,with_cuda=True):
        # 解析配置文件，取得字典大小信息，以及字典文件路径
        config_=configparser.ConfigParser()
        config_.read('./config/sentiment_model_config.ini')
        self.config=config_['DEFAULT']
        self.vocab_size=int(self.config['vocab_size'])
        with open(self.config['word2idx_path'],'r',encoding='utf-8') as f:
            self.word2idx=json.load(f)
        # 初始化传入参数
        self.max_seq_len=max_seq_len
        self.batch_size=batch_size
        # 设置执行设备
        cuda_condition=torch.cuda.is_available() and with_cuda
        self.device=torch.device('cuda:0' if cuda_condition else 'cpu')
        # 初始化模型超参数
        bertconfig=BertConfig(vocab_size=self.vocab_size)
        # 初始化BERT情感分析模型
        self.bert_model=Bert_Sentiment_Analysis(config=bertconfig)
        # # 将模型发送到计算设备
        # self.bert_model.to(self.device)
        # # 转换为评估模式
        # self.bert_model.eval()


        # 初始化文本预处理器
        self.process_batch=preprocessing(hidden_dim=bertconfig.hidden_size,
                                         max_positions=max_seq_len,
                                         word2idx=self.word2idx)
    def __call__(self, text_list,batch_size=1,threshold=.52):
        # 判断输入序列是否为字符串,如果为字符串，则转换为列表
        #异常判断可否全部放到预处理中？？？
        if isinstance(text_list,str):
            text_list=[text_list]
            # print(text_list)
        # 计算原始输入序列长度
        len_=len(text_list)
        # 剔除长度为0的序列
        text_list=[i for i in text_list if len(i)!=0]
        # 判断除0后的序列长度，并给出相应的响应
        if len(text_list)==0:
            raise NotImplementedError("输入文本全部为空")
        if len(text_list)<len_:
            warnings.warn("输入文本中有长度为0的句子")
        # 计算输入序列的最大句子长度
        max_seq_len=max([len(i) for i in text_list])
        # 序列预处理
        text_tokens,positional_enc=self.process_batch(text_list,max_seq_len=max_seq_len)
        # print(positional_enc.size())
        # 扩充维度 [m,n]-->[1,m,n]
        positional_enc=torch.unsqueeze(positional_enc,dim=0).to(self.device) #torch.Size([1, 138, 384])
        # print(positional_enc.size())
        # 计算切片数量，对输入序列按batch_size切片
        n_batches=math.ceil(len(text_tokens)/batch_size)
        #本模型中每次处理一个句子
        for i in range(n_batches):
            start=i*batch_size
            end=start+batch_size
            text_tokens_=text_tokens[start:end].to(self.device) #torch.Size([1, 138])
            predictions=self.bert_model.forward(text_input=text_tokens_,positional_enc=positional_enc)
            nn.ravel(pre)
            print(predictions)







if __name__ == '__main__':
    model=Sentiment_Analysis(300,1)
    #一个字符串代表一个seq
    test_list=[
        "有几次回到酒店房间都没有被整理。两个人入住，只放了一套洗漱用品。",
        "早餐时间询问要咖啡或茶，本来是好事，但每张桌子上没有放“怡口糖”（代糖），又显得没那么周到。房间里卫生间用品补充，有时有点漫不经心个人觉得酒店房间禁烟比较好",
        "南京东路地铁出来就能看到，很方便。酒店大堂和房间布置都有五星级的水准。",
        "服务不及5星，前台非常不专业，入住时会告知你没房要等，不然就加钱升级房间。前台个个冰块脸，对待客人好像仇人一般，带着2岁的小孩前台竟然还要收早餐费。门口穿白衣的大爷是木头人，不会提供任何帮助。入住期间想要多一副牙刷给孩子用，竟然被问为什么。五星设施，一星服务，不会再入住！"
    ]
    # test_list =['zhang','']
    model(test_list)
