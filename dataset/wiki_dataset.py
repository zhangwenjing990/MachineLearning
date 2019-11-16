from torch.utils.data import Dataset
import json
import tqdm
import numpy as np
import random
import torch

corpus_path='../corpus/test_wiki.txt'
word2idx_path='../corpus/bert_word2idx_extend.json'
max_seq_len=300
hidden_dim=384

class BERTDataset(Dataset):
    def __init__(self,corpus_path,word2idx_path,seq_len,hidden_dim=384,on_memory=True):
        self.corpus_path=corpus_path
        self.word2idx_path=word2idx_path
        self.seq_len=seq_len
        self.hidden_dim=hidden_dim
        self.on_memory=on_memory

        self.pad_index=0
        self.unk_index=1
        self.cls_index=2
        self.sep_index=3
        self.mask_index=4
        self.num_index=5

        with open(word2idx_path,'r',encoding='utf-8') as f:
            self.word2idx=json.load(f)
        with open(corpus_path,'r',encoding='utf-8') as f:
            # 计算打开文本的句子长度
            # 如果不全部载入到内存，只计算所有句子长度
            if not on_memory:
                self.corpus_lines=0
                for _ in tqdm.tqdm(f,desc='Loading Dataset'):
                    self.corpus_lines+=1
            # 如果全部载入到内存，将句子转换为列表，并计算长度
            if on_memory:
                self.lines=[eval(line) for line in tqdm.tqdm(f,desc='Loading Dataset')]
                # eval函数将字符串转换为dic/list/tupe/数值
                # print(self.lines[0])
                self.corpus_lines=len(self.lines)

        if not on_memory:
            self.file=open(corpus_path,'r',encoding='utf-8')

            self.random_file=open(corpus_path,'r',encoding='utf-8')
            # 取出一部分数据，以达到随机选择一行的效果
            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines<1000 else 1000 )):
            # for _ in range(np.random.randint(10)):
                self.random_file.__next__()
                # print(self.random_file.__next__())



    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # t1是上句文本，t2是下句文本，is_next_label代表它们是不是从属关系
        t1,t2,is_next_label=self.random_sent(item)

        t1_random,t1_label=self.random_char(t1) #[4, 774, 680, 4],[602, 0, 0, 209]
        t2_random,t2_label=self.random_char(t2)
        # 给序列化句子添加开头和结尾索引
        t1=[self.cls_index]+t1_random+[self.sep_index]
        t2=t2_random+[self.sep_index]
        # 给标签相应位置加上pad_index
        t1_label=[self.pad_index]+t1_label+[self.pad_index]
        t2_label=t2_label+[self.pad_index]
        # 上句和下句的segment标签分别为0和1
        segment_label=([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        # 把句子token截断为bert模型允许的句长
        bert_input=(t1+t2)[:self.seq_len]
        bert_label=(t1_label+t2_label)[:self.seq_len]

        output={'bert_input':torch.tensor(bert_input),
                'bert_label':torch.tensor(bert_label),
                'segment_label':torch.tensor(segment_label),
                'is_next':torch.tensor([is_next_label])}
        return output

    def random_char(self,sentence):
        """

        :param sentence: 文本字符串
        :return:
        """
        char_tokens_=list(sentence)
        char_tokens=self.tokenize_char(char_tokens_)
        # 如果一个字被遮盖掉mask，输出label为真实索引，否则为0
        output_label=[]
        for i,token in enumerate(char_tokens):
            prob=random.random()
            if prob<0.30:
                prob/=0.30
                output_label.append(char_tokens[i])
                if prob<0.8:
                    char_tokens[i]=self.mask_index
                elif prob<0.9:
                    char_tokens[i]=random.randrange(len(self.word2idx))
            else:
                output_label.append(0)
        # print(char_tokens)
        # print(output_label)
        # print(char_tokens_)
        return char_tokens,output_label

    def tokenize_char(self,segments):
        """

        :param segments: 文本字符列表,如['中', '华', '人', '民', '共', '和', '国']
        :return:文本的序列表示，如[602, 774, 680, 209, 3794, 728, 291]
        """
        return [self.word2idx.get(char,self.unk_index) for char in segments]

    def random_sent(self,index):
        """
        以0.5的概率随机替换文本下句内容，并返回替换后的文本和标签
            1-表示为真实样本上下句
            0-表示不是
        :param index:
        :return:
        """
        t1,t2=self.get_corpus_line(index)

        if random.random()>0.5:
            return t1,t2,1
        else:
            return t1,self.get_random_line(),0

    def get_corpus_line(self,item):
        """
        将dic类型的句子转换为含上下句的字符串类型的句子
        给定一个索引，取出该索引对应的文本，上下句分开返回
        :param item:索引
        :return:
        """

        # 如果已经全部载入到内存里，
        if self.on_memory:
            print('原始--文本的句子为:{}'.format(self.lines[item]))
            return self.lines[item]['text1'],self.lines[item]['text2']
        else:
            line=self.file.__next__()
            # 如果为空，重新打开
            if line is None:
                self.file.close()
                self.file=open(self.corpus_path,'r',encoding='utf-8')
                line=self.file.__next__()
            # print('原始-文本的句子为:{}'.format(line))
            line = eval(line)
            t1,t2=line['text1'],line['text2']
            return t1,t2

    def get_random_line(self):
        # 随机取一行
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))]['text2']

        line=self.random_file.__next__()
        if line is None:
            self.random_file.close()
            self.random_file=open(self.corpus_path,'r',encoding='utf-8')
            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines<1000 else 1000)):
                self.random_file.__next__()
            line=self.random_file.__next__()
        return eval(line)['text2']


if __name__ == '__main__':
    data=BERTDataset(corpus_path=corpus_path,word2idx_path=word2idx_path,seq_len=max_seq_len,hidden_dim=hidden_dim,on_memory=False)
    data.__getitem__(0)
    output=data[0]
    print(output)