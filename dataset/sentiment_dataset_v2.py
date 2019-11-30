import torch
import torch.nn as nn
from torch.utils.data import Dataset
import tqdm
from sklearn.utils import shuffle
import re
import json
import random

class CLSDataset(Dataset):
    def __init__(self,corpus_path,word2idx,max_seq_len,data_regularization=False):
        self.corpus_path=corpus_path
        self.word2idx=word2idx
        self.max_seq_len=max_seq_len
        self.data_regularization=data_regularization
        self.pad_index=0
        self.unk_index=1
        self.cls_index=2
        self.sep_index=3
        self.mask_index=4
        self.num_index=5
        with open(self.corpus_path,'r',encoding='utf-8') as f:
            self.lines=[eval(line) for line in tqdm.tqdm(f,desc='Loading Dataset')]
            self.lines=shuffle(self.lines)
            self.corpus_lines=len(self.lines)

    def get_text_and_label(self,item):
        text=self.lines[item]['text']
        label=self.lines[item]['label']
        return text,label

    def tokenize_char(self,segments):
        return [self.word2idx.get(char,self.unk_index) for char in segments]

    def __getitem__(self, item):
        text,label=self.get_text_and_label(item)
        if self.data_regularization:
            if random.random()<1:
                split_spans=[i.span() for i in re.finditer("，|；|。|？|！",text)]
                if len(split_spans)!=0:
                    span_idx=random.randint(0,len(split_spans)-1)
                    cut_position=split_spans[span_idx][1]
                    if random.random()<0.5:
                        if len(text)-cut_positon>2:
                            text=text[cut_position:]
                        else:
                            text=text[:cut_position]
                    else:
                        if cut_position>2:
                            text=text[:cut_position]
                        else:
                            text=text[cut_position:]


        text_input=self.tokenize_char(text)
        text_input=[self.cls_index]+text_input+[self.sep_index]
        text_input=text_input[:self.max_seq_len]
        output={'text_input':torch.tensor(text_input),'label':torch.tensor([label])}
        return output

    def __len__(self):
        return self.corpus_lines

if __name__ == '__main__':
    with open ('../corpus/bert_word2idx_extend.json','r',encoding='utf-8') as f:
        word2idx=json.load(f)
    data=CLSDataset(corpus_path='../corpus/train_sentiment.txt',
                    word2idx=word2idx,
                    max_seq_len=200,
                    data_regularization=True)
    print(data[0])
