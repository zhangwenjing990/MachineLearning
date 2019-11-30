from models.bert_sentiment_analysis import *
import configparser
import json
import torch

class Sentiment_trainer:
    def __init__(self,max_seq_len,batch_size,lr,with_cuda=True):
        config_=configparser.ConfigParser()
        config_.read('./config/sentiment_model_config.ini')
        self.config=config_['DEFAULT']
        self.vocab_size=int(self.config['vocab_size'])
        self.max_seq_len=max_seq_len
        self.batch_size=batch_size
        self.lr=lr

        # 加载字典，模型，训练和测试数据
        with open('./corpus/bert_word2idx_extend.json','r',encoding='utf-8') as f:
            word2idx=json.load(f)
        bertconfig=BertConfig(vocab_size=self.vocab_size)
        self.bert_model=Bert_Sentiment_Analysis(config=bertconfig)
        cuda_condition=torch.cuda.is_available() and with_cuda
        self.device=torch.device('cuda:0' if cuda_condition else 'cpu')
        self.bert_model.to(self.device)
        train_dataset=CLSDataset(corpus_path=)



    def find_most_recent_state_dict(self,dir_path):
        dic_lis=[i for i in os.listdir(dir_path)]
        if len(dic_lis)==0:
            raise FileNotFoundError('在“{}”文件夹中没有文件存在'.format(dir_path))
        dic_lis=[i for i in dic_lis if 'model' in i]
        dic_lis=sorted(dic_lis,key=lambda k:int(k.split('.')[-1]))
        return dir_path+'/'+dic_lis[-1]


    def load_model(self,model,dir_path,load_bert=False):
        checkpoint_dir=self.find_most_recent_state_dict(dir_path)
        checkpoint=torch.load(checkpoint_dir)

        if load_bert:
            checkpoint['model_state_dict']={k[5:]:v for k,v in checkpoint['model_state_dict'].items()
                                            if k[:4]=='bert' and 'pooler' not in k}
        model.load_state_dict(checkpoint['model_state_dict'])
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded!'.format(checkpoint_dir))

if __name__ == '__main__':
    def init_trainer(dynamic_lr,batch_size=24):
        trainer=Sentiment_trainer(max_seq_len=300,
                                  batch_size=batch_size,
                                  lr=dynamic_lr,
                                  with_cuda=True)
        return trainer,dynamic_lr
    start_epoch=0
    train_epoches=9999
    trainer,dynamiic_lr=init_trainer(dynamic_lr=1e-06,batch_size=24)

    for epoch in range(start_epoch,start_epoch+train_epoches):
        if epoch==start_epoch and epoch==0:
            model=trainer.load_model(trainer.bert_model,)
        elif epoch==start_epoch:
            pass
        print('train with learning rate{}'.format(str(dynamiic_lr)))