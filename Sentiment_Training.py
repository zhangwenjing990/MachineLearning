from models.bert_sentiment_analysis import *
import configparser
import json
import torch
from torch.utils.data import DataLoader
from dataset.sentiment_dataset_v2 import CLSDataset
from sklearn import metrics
# from metrics import *
import numpy as np
import os
import pandas as pd
import tqdm

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
            self.word2idx=json.load(f)
        bertconfig=BertConfig(vocab_size=self.vocab_size)
        self.bert_model=Bert_Sentiment_Analysis(config=bertconfig)
        cuda_condition=torch.cuda.is_available() and with_cuda
        self.device=torch.device('cuda:0' if cuda_condition else 'cpu')
        self.bert_model.to(self.device)
        train_dataset=CLSDataset(corpus_path=self.config['train_corpus_path'],
                                 word2idx=self.word2idx,
                                 max_seq_len=self.max_seq_len,
                                 data_regularization=True)
        self.train_dataloader=DataLoader(train_dataset,
                                         batch_size=self.batch_size,
                                         num_workers=0,
                                         collate_fn=lambda x:x)
        test_dataset=CLSDataset(corpus_path=self.config['test_corpus_path'],
                                word2idx=self.word2idx,
                                max_seq_len=self.max_seq_len,
                                data_regularization=False)
        self.test_dataloader=DataLoader(test_dataset,
                                        batch_size=self.batch_size,
                                        num_workers=0,
                                        collate_fn=lambda x:x)
        self.hidden_dim=bertconfig.hidden_size
        self.positional_enc=self.init_positional_encoding()
        self.positional_enc=torch.unsqueeze(self.positional_enc,dim=0)
        self.optim_parameters=list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        if not os.path.exists(self.config['state_dict_dir']):
            os.mkdir(self.config['state_dict_dir'])

    def init_optimizer(self,lr):
        self.optimizer=torch.optim.Adam(self.optim_parameters,lr=lr,weight_decay=1e-3)
    def init_positional_encoding(self):
        position_enc=np.array([
            [pos/np.power(10000,2*i/self.hidden_dim) for i in range(self.hidden_dim)]
             if pos !=0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])
        position_enc[1:,0::2]=np.sin(position_enc[1:,0::2])
        position_enc[1:,1::2]=np.cos(position_enc[1:,1::2])
        denomeinator=np.sqrt(np.sum(position_enc**2,axis=1,keepdims=True))
        position_enc=position_enc/(denomeinator+1e-8)
        position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc


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
        # print([k for k in model.state_dict()])
        # print([k for k in checkpoint['model_state_dict'] if k[:4]=='bert' and 'pooler' not in k])
        if load_bert:
            checkpoint['model_state_dict']={k:v for k,v in checkpoint['model_state_dict'].items()
                                            if k[:4]=='bert' and 'pooler' not in k}
            # print({k for k in checkpoint['model_state_dict']})
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded!'.format(checkpoint_dir))

    def padding(self,output_dic_lis):
        text_input=[i['text_input'] for i in output_dic_lis]
        text_input=torch.nn.utils.rnn.pad_sequence(text_input,batch_first=True)
        label=torch.cat([i['label'] for i in output_dic_lis])
        return {'text_input':text_input,'label':label}

    def iteration(self,epoch,data_loader,train=True,df_name='df_log.pickle'):
        df_path=self.config['state_dict_dir']+'/'+df_name
        if not os.path.isfile(df_path):
            df=pd.DataFrame(columns=['epoch','train_loss','train_auc','test_loss','test_auc'])
            df.to_pickle(df_path)
            print('log DataFrame created!')

        str_code='train' if train else 'test'
        data_iter=tqdm.tqdm(enumerate(data_loader),desc='EPOCH_%s:%d'%(str_code,epoch),
                            total=len(data_loader),bar_format='{l_bar}{r_bar}')
        total_loss=0
        all_predictions,all_labels=[],[]
        for i,data in data_iter:
            # print(len(data))
            # print(data[0]['text_input'])
            data=self.padding(data)
            data={key:value.to(self.device) for key,value in data.items()}
            positional_enc=self.positional_enc[:,:data['text_input'].size()[-1],:].to(self.device)
            predictions,loss=self.bert_model.forward(text_input=data['text_input'],
                                    positional_enc=positional_enc,
                                    labels=data['label'])
            predictions=predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels=data['label'].cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            fpr,tpr,thresholds=metrics.roc_curve(y_true=all_labels,y_score=all_predictions)
            auc=metrics.auc(fpr,tpr)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss+=loss.item()
            if train:
                log_dic={'epoch':epoch,
                         'train_loss':total_loss/(i+1),'train_auc':auc,
                         'test_loss':0,'test_auc':0}
            else:
                log_dic={'epoch':epoch,
                          'train_loss':0,'train_auc':0,
                          'test_loss':total_loss/(i+1),'test_auc':auc}
            # if i%10==0:
            #     data_iter.write(str({k:v for k,v in log_dic.items() if v!=0}))

        # threshold_=find_best_threshold(all_predictions,all_labels)
        # print(str_code+'best threshold:'+str(threshold_))

        if train:
            df=pd.read_pickle(df_path)
            df=df.append([log_dic])
            df.reset_index(inplace=True,drop=True)
            df.to_pickle(df_path)
        else:
            fpr,tpr,thresholds=metrics.roc_curve(y_true=all_labels,y_score=all_predictions)
            auc=metrics.auc(fpr,tpr)

            log_dic={k:v for k,v in log_dic.items() if v!=0 and k!='epoch'}
            df=pd.read_pickle(df_path)
            df.reset_index(inplace=True,drop=True)
            for k,v in log_dic.items():
                df.at[epoch,k]=v
                df.to_pickle(df_path)
            # print(auc)
            return auc


    def train(self,epoch):
        self.bert_model.train()
        self.iteration(epoch,self.train_dataloader,train=True)

    def test(self,epoch):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch,self.test_dataloader,train=False)

    def save_state_dict(self,model,epoch,state_dict_dir='../output',file_path='bert_model'):
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path=state_dict_dir+'/'+file_path+'.epoch.{}'.format(str(epoch))
        model.to('cpu')
        torch.save({'model_state_dict':model.state_dict()},save_path)
        print('{}saved!'.format(save_path))
        model.to(self.device)


if __name__ == '__main__':
    def init_trainer(dynamic_lr,batch_size=1):
        trainer=Sentiment_trainer(max_seq_len=300,
                                  batch_size=batch_size,
                                  lr=dynamic_lr,
                                  with_cuda=True)
        return trainer,dynamic_lr

    start_epoch=1
    train_epoches=1
    trainer,dynamiic_lr=init_trainer(dynamic_lr=1e-06,batch_size=5)
    all_auc=[]
    threshold=999
    patient=10
    best_loss=9999999
    for epoch in range(start_epoch,start_epoch+train_epoches):
        if epoch==start_epoch and epoch==0:
            trainer.load_model(trainer.bert_model,dir_path='./bert_state_dict',load_bert=True)
        elif epoch==start_epoch:
            trainer.load_model(trainer.bert_model,dir_path=trainer.config['state_dict_dir'])
        # print('train with learning rate:{}'.format(str(dynamiic_lr)))
        # trainer.train(epoch)
        # trainer.save_state_dict(trainer.bert_model,epoch,state_dict_dir=trainer.config['state_dict_dir'],
        #                         file_path='sentiment.model')
        # print(next(trainer.bert_model.parameters()))
        auc=trainer.test(epoch)
        print(auc)
        all_auc.append(auc)
        best_auc=max(all_auc)
        print(all_auc)
        if all_auc[-1] < best_auc:
            threshold+=1
            dynamiic_lr*=0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            threshold=0

        if threshold>=patient:
            print('epoch {} has the lowest loss'.format(start_epoch+np.argmax(np.array(all_auc))))
            print('early stop!')
            break

