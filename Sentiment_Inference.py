from dataset.inference_dataloader import preprocessing
from models.bert_sentiment_analysis import *
import numpy as np
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
        self.bert_model.to(self.device)
        # # 转换为评估模式
        self.bert_model.eval()

        self.hidden_dim=bertconfig.hidden_size
        self.positional_enc=self.init_positional_encoding()


        # 初始化文本预处理器
        self.process_batch=preprocessing(hidden_dim=bertconfig.hidden_size,
                                         max_positions=max_seq_len,
                                         word2idx=self.word2idx)

        self.load_model(self.bert_model,dir_path=self.config['state_dict_dir'])


    def init_positional_encoding(self):
        positional_enc=np.array([[pos/np.power(10000,2*i/self.hidden_dim) for i in range(self.hidden_dim)]
                                 if pos!=0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])
        positional_enc[1:,0::2]=np.sin(positional_enc[1:,0::2])
        positional_enc[1:,1::2]=np.cos(positional_enc[1:,1::2])
        denominator=np.sqrt(np.sum(positional_enc**2,axis=1,keepdims=True))

        positional_enc=positional_enc/(denominator+1e-8)
        positional_enc=torch.from_numpy(positional_enc).type(torch.FloatTensor)
        return positional_enc

    def find_most_recent_state_dict(self,dir_path):
        dic_lis=[i for i in os.listdir(dir_path) if 'model' in i]
        print(dic_lis)
        if len(dic_lis)==0:
            print('在"{}"文件夹中找不到已存在的模型'.format(dir_path))
        dic_lis=sorted(dic_lis,key=lambda k:int(k.split('.')[-1]))
        return dir_path+'/'+dic_lis[-1]





    def load_model(self,model,dir_path="../output"):
        checkpoint_dir=self.find_most_recent_state_dict(dir_path)
        checkpoint=torch.load(checkpoint_dir)
        # print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded'.format(checkpoint_dir))







    def sentiment_print_func(self,text,pred,threshold):
        print(text)
        if pred>=threshold:
            print("正样本，输出值{:.2f}".format(pred))
        else:
            print("负样本，输出值{:.2f}".format(pred))
        # print('----------')

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
            # print(text_tokens_)
            predictions=self.bert_model.forward(text_input=text_tokens_,positional_enc=positional_enc)
            # print(predictions)
            predictions=np.ravel(predictions.detach().cpu().numpy()).tolist()
            # print(predictions)
            for text,pred in zip(text_list[start:end],predictions):
                self.sentiment_print_func(text,pred,threshold)







if __name__ == '__main__':
    model=Sentiment_Analysis(300,1)

    test_list = [
        "有几次回到酒店房间都没有被整理。两个人入住，只放了一套洗漱用品。",
        "早餐时间询问要咖啡或茶，本来是好事，但每张桌子上没有放“怡口糖”（代糖），又显得没那么周到。房间里卫生间用品补充，有时有点漫不经心个人觉得酒店房间禁烟比较好",
        '十六浦酒店有提供港澳码头的SHUTTLE BUS, 但氹仔没有订了普通房, 可能是会员的关系 UPGRADE到了DELUXE房,风景是绿色的河, 感观一般, 但房间还是不错的, 只是装修有点旧了另外品尝了酒店的自助晚餐, 种类不算多, 味道OK, 酒类也免费任饮, 这个不错最后就是在酒店的娱乐场赢了所有费用, 一切都值得了!',
        '地理位置优越，出门就是步行街，也应该是耶路撒冷的中心地带，去老城走约20分钟。房间很实用，虽然不含早餐，但是楼下周边有很多小超市和餐厅、面包店，所以一切都不是问题。',
        '实在失望！如果果晚唔系送朋友去码头翻香港一定会落酒店大堂投诉佢！太离谱了！我地吃个晚饭消费千几蚊 ，买单个黑色衫叫Annie果个唔知系部长定系经理录左我万几蚊！简直系离晒大谱的 ！咁样的管理层咁大间酒店真的都不敢恭维！',
        '酒店服务太棒了, 服务态度非常好, 房间很干净',
        "服务各方面没有不周到而的地方, 各方面没有没想到的细节",
        "房间设施比较旧，虽然是古典风格，但浴室的浴霸比较不好用。很不满意的是大厅坐下得消费，不人性化，而且糕点和沙拉很难吃，贵而且是用塑料盒子装的，5星级？特别是青团，58块钱4个，感觉放了好几天了，超级难吃。。。把外国朋友吓坏了。。。",
        "南京东路地铁出来就能看到，很方便。酒店大堂和房间布置都有五星级的水准。",
        "服务不及5星，前台非常不专业，入住时会告知你没房要等，不然就加钱升级房间。前台个个冰块脸，对待客人好像仇人一般，带着2岁的小孩前台竟然还要收早餐费。门口穿白衣的大爷是木头人，不会提供任何帮助。入住期间想要多一副牙刷给孩子用，竟然被问为什么。五星设施，一星服务，不会再入住！"
    ]
    # test_list =['zhang','']
    model(test_list)
