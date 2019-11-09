import torch
import numpy as np
import configparser
import json

# config=configparser.ConfigParser()
# config.read('../config/sentiment_model_config.ini')
# config_='../corpus/bert_word2idx_extend.json'
# with open(config_,'r',encoding='utf-8') as f:
#     word2idx=json.load(f)

class preprocessing():
    def __init__(self,hidden_dim,max_positions,word2idx):
        # max_positions的作用？是否可改为max_seq_len？？？？

        self.hidden_dim=hidden_dim
        self.max_positions=max_positions+2
        self.word2idx=word2idx
        # 分别初始化填补，未知字，句子开头，结尾，mask,数字索引
        self.padding_index=0
        self.unknown_index=1
        self.cls_index=2
        self.sep_index=3
        self.mask_index=4
        self.num_index=5
        self.positional_encoding=self.init_positional_encoding()

    def init_positional_encoding(self):
        # 初始化 sinosoid,
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_positions)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        return position_enc

    def tokenize(self,text_or_label,dict):
        return [dict.get(word,0) for word in text_or_label]

    def add_cls_sep(self,text_tokens):
        return [self.cls_index]+text_tokens+[self.sep_index]

    def add_cls_sep_padding(self,tokens):
        # 程序中未用到该函数
        return [self.pad_index]+tokens+[self.pad_index]

    def __call__(self, text_list,max_seq_len):
        """
        将输入文本序列化，并pading成一致大小
        max_seq_len参数有些多余，因为可以计算获得？？？？？
        :param text_list: 需要预处理的输入文本序列
        :param max_seq_len: 实际输入文本中句子的最大长度
        :return:返回序列化后的文本，以及相应位置的位置编码
        """
        # 获取输入序列长度
        text_list_len=[len(seq) for seq in text_list]
        # 判断输入序列长度是否符合要求
        if max(text_list_len)>self.max_positions-2:
            raise AssertionError("输入文本最大长度为{}，大于允许长度{}".format(max(text_list_len),self.max_positions-2))
        # 批量最大序列长度，为输入最大句子长度+2
        batch_max_seq_len=max_seq_len+2
        # 根据字典，将输入文本序列化，得到每个字的序列表示，一句话--一个列表
        text_tokens=[self.tokenize(seq,self.word2idx) for seq in text_list]
        # 增加开头和结尾的index
        text_tokens=[self.add_cls_sep(seq) for seq in text_tokens]
        # 把句子转换为torch张量
        text_tokens=[torch.tensor(seq) for seq in text_tokens]
        # padding成相同的句子长度，跟最长的句子保持一致
        text_tokens=torch.nn.utils.rnn.pad_sequence(text_tokens,batch_first=True)
        # 取相应位置的位置编码
        position_enc=torch.from_numpy(self.positional_encoding[:batch_max_seq_len]).type(torch.FloatTensor)
        return text_tokens,position_enc




if __name__ == '__main__':
    test_list=[
        "有几次回到酒店房间都没有被整理。两个人入住，只放了一套洗漱用品。",
        "早餐时间询问要咖啡或茶，本来是好事，但每张桌子上没有放“怡口糖”（代糖），又显得没那么周到。房间里卫生间用品补充，有时有点漫不经心个人觉得酒店房间禁烟比较好",
        "南京东路地铁出来就能看到，很方便。酒店大堂和房间布置都有五星级的水准。",
        "服务不及5星，前台非常不专业，入住时会告知你没房要等，不然就加钱升级房间。前台个个冰块脸，对待客人好像仇人一般，带着2岁的小孩前台竟然还要收早餐费。门口穿白衣的大爷是木头人，不会提供任何帮助。入住期间想要多一副牙刷给孩子用，竟然被问为什么。五星设施，一星服务，不会再入住！"
    ]
    text_preprocess=preprocessing(384,300,word2idx)
    text_preprocess(test_list,136)