from torch.utils.data import DataLoader

from dataset.wiki_dataset import BERTDataset
from models.bert_model import *
import numpy as np
import os
import pandas as pd
import tqdm

config={}
config['batch_size']=1
config['max_seq_len']=200
config['vocab_size']=32162
config['lr']=2e-6
config['num_workers']=0
config['test_corpus_path']='./corpus/test_wiki.txt'
config['word2idx_path']='./corpus/bert_word2idx_extend.json'
config['output_path']='./bert_state_dict'

class Pretrainer:
    """
    构建预训练器
    """
    def __init__(self,bert_model,vocab_size,max_seq_len,batch_size,lr,with_cuda=True,):
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.batch_size=batch_size
        self.lr=lr
        # 确定计算设备
        cuda_condition=torch.cuda.is_available() and with_cuda
        self.device=torch.device('cuda:0' if cuda_condition else 'cpu')

        # 初始化模型配置信息
        bertconfig=BertConfig(vocab_size=config['vocab_size'])
        # 初始化bert模型，并发送到计算设备
        self.bert_model=bert_model(config=bertconfig)
        self.bert_model.to(self.device)
        # 准备训练和测试数据集
        train_dataset=BERTDataset(corpus_path='./corpus/test_wiki.txt',
                                  word2idx_path='./corpus/bert_word2idx_extend.json',
                                  seq_len=self.max_seq_len,
                                  hidden_dim=bertconfig.hidden_size,
                                  on_memory=False)

        self.train_dataloader=DataLoader(train_dataset,
                                         batch_size=self.batch_size,
                                         num_workers=config['num_workers'],
                                         collate_fn=lambda x:x)
        # for i in self.train_dataloader:
        #     print(i)
        #     break



        test_dataset=BERTDataset(corpus_path=config['test_corpus_path'],
                                  word2idx_path=config['word2idx_path'],
                                  seq_len=self.max_seq_len,
                                  hidden_dim=bertconfig.hidden_size,
                                  on_memory=False)

        self.test_dataloader=DataLoader(test_dataset,
                                         batch_size=self.batch_size,
                                         num_workers=config['num_workers'],
                                         collate_fn=lambda x:x)

        # 初始化位置编码
        self.positional_enc=self.init_positional_encoding(hidden_dim=bertconfig.hidden_size,
                                                          max_seq_len=self.max_seq_len)

        # 升维
        self.positional_enc=torch.unsqueeze(self.positional_enc,dim=0)
        optim_parameters=list(self.bert_model.parameters())

        # --------------------------------------
        # 计算bert_model中的参数数目
        # num=0
        # for para in optim_parameters:
        #     para_shape=para.detach().numpy().shape
        #     dim_len=len(para_shape)
        #     n=1
        #     for i in range(dim_len):
        #         n=n*para_shape[i]
        #     num+=n
        # bert_model_parameter_num=num
        # print(bert_model_parameter_num)
        # -----------------------------------------

        self.optimizer=torch.optim.Adam(optim_parameters,lr=self.lr)
        print('Total Parameters:',sum([p.nelement() for p in self.bert_model.parameters()]))


        # print(num)



    def init_positional_encoding(self,hidden_dim,max_seq_len):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self,model,dir_path='./output'):
        checkpoint_dir=self.find_most_recent_state_dict(dir_path)
        checkpoint=torch.load(checkpoint_dir)
        # 加载到的checkpoint为一个字典，字典里有每一层的参数值
        # print(checkpint)
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        # 使显存才会在Nvidia - smi中释放
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded for training'.format(checkpoint_dir))

    def find_most_recent_state_dict(self,dir_path):
        dic_lis=[i for i in os.listdir(dir_path)]
        if len(dic_lis)==0:
            raise FileNotFoundError('在{}中找不到任何状态文件'.format(dir_path))
        dic_lis=[i for i in dic_lis if 'model' in i]
        dic_lis=sorted(dic_lis,key=lambda k:int(k.split('.')[-1]))
        return dir_path+'/'+dic_lis[-1]

    def train(self,epoch,df_path='./bert_state_dict/df_log.pickle'):
        #开启训练模式
        self.bert_model.train()

        self.iteration(epoch,self.train_dataloader,train=True,df_path=df_path)

    def iteration(self,epoch,data_loader,train=True,df_path='./bert_state_dict/df_log.pickle'):
        #
        if not os.path.isfile(df_path) and epoch!=0:
            raise RuntimeError('没有找到DataFrame日志路径，因为不是从头开始训练，也不能创建一个新的路径')
        if not os.path.isfile(df_path) and epoch==0:
            df=pd.DataFrame(columns=['epoch','trian_next_seq_loss','train_mlm_loss'
                                     'train_next_sen_acc','train_mlm_acc',
                                     'test_next_sen_loss','test_mlm_loss',
                                     'test_next_sen_acc','test_mlm_acc'])
            # DataFrame文件序列化，以便反序列化为具体对象
            df.to_pickle(df_path)
            print('log DataFrame created!')

        str_code='train' if train else 'test'

        # data_iter数据为[(0,data[0]),...(i,data[i])]
        data_iter=tqdm.tqdm(enumerate(data_loader),desc='EP_%s:%d' % (str_code,epoch),
                            total=len(data_loader),bar_format='{l_bar}{r_bar}')
        # print(data_iter)

        total_next_sen_loss=0
        total_mlm_loss=0
        total_next_sen_acc=0
        total_mlm_acc=0
        total_element=0

        for i,data in data_iter:
            print(type(data))
            data=self.padding(data)
            {key:value.to(self.device) for key,value in data.items()}
            # 位置编码的维度跟padding后的句子长度保持一致
            positional_enc=self.positional_enc[:,:data['bert_input'].size()[-1],:].to(self.device)
            mlm_preds,next_sen_preds=self.bert_model.forward(input_ids=data['bert_input'],
                                                             positional_enc=positional_enc,
                                                             token_type_ids=data['segment_label'])
            print(next_sen_preds.argmax(dim=-1,keepdim=False))
            mlm_acc=self.get_mlm_accuracy(mlm_preds,data['bert_label'])
            next_sen_acc=next_sen_preds.argmax(dim=-1,keepdim=False).eq(data['is_next']).sum().item()
            mlm_loss=self.compute_loss(mlm_preds,data['bert_label'],self.vocab_size,ignore_index=0)
            next_sen_loss=self.compute_loss(next_sen_preds,data['is_next'])
            loss=mlm_loss+next_sen_loss

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_next_sen_loss+=next_sen_loss.item()
            total_mlm_loss+=mlm_loss.item()
            total_next_sen_acc+=next_sen_acc
            total_element+=data['is_next'].nelement()
            total_mlm_acc+=mlm_acc

            if train:
                log_dic={
                    'epoch':epoch,
                    'train_next_sen_loss':total_next_sen_loss/(i+1),
                    'train_mlm_loss':total_mlm_loss/(i+1),
                    'train_next_sen_acc':total_next_sen_acc/total_element,
                    'train_mlm_acc':total_mlm_acc/(i+1),
                    'test_next_sen_loss':0,'test_mlm_loss':0,
                    'test_next_sen_acc':0,'test_mlm_acc':0
                }
            else:
                log_dic={
                    'epoch':epoch,
                    'test_next_sen_loss':total_next_sen_loss/(i+1),
                    'test_mlm_loss':total_mlm_loss/(i+1),
                    'test_next_sen_acc':total_next_sen_acc/total_element,
                    'test_mlm_acc': total_mlm_acc / (i + 1),
                    'train_next_sen_loss':0,'train_mlm_loss':0,
                    'train_next_sen_acc':0,'train_mlm_acc':0}

            if i%10==0:
                data_iter.write(str({k:v for k,v in log_dic.items() if v!=0 and k!='epoch'}))

            # if i==30:
            #     break

            # break
        if train:
            df=pd.read_pickle(df_path)
            df=df.append([log_dic])
            df.reset_index(inplace=True,drop=True)
            df.to_pickle(df_path)
        else:
            log_dic={k:v for k,v in log_dic.items() if v!=0 and k!='epoch'}
            df=pd.read_pickle(df_path)
            df.reset_index(inplace=True,drop=True)
            for k,v in log_dic.items:
                df.at[epoch,k]=v
            df.to_pickle(df_path)
            return float(log_dic['test_next_sen_loss'])+float(log_dic['test_mlm_loss'])

    def compute_loss(self,predictions,labels,num_class=2,ignore_index=None):
        if ignore_index is None:
            loss_func=CrossEntropyLoss()
        else:
            loss_func=CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1,num_class),labels.view(-1))

    def get_mlm_accuracy(self,predictions,labels):
        predictions=torch.argmax(predictions,dim=-1,keepdim=False)
        mask=(labels>0).to(self.device)
        mlm_accuracy=torch.sum((predictions==labels)*mask).float()
        mlm_accuracy/=(torch.sum(mask).float()+ 1e-8)
        # 将tensor转换为python常量输出
        return mlm_accuracy.item()
        # (predictions==labels)
        # print(predictions==labels)
        print(torch.sum(mask))

    def padding(self,output_dic_lis):
        bert_input=[i['bert_input'] for i in output_dic_lis]
        bert_label=[i['bert_label'] for i in output_dic_lis]
        segment_label=[i['segment_label'] for i in output_dic_lis]
        bert_input=torch.nn.utils.rnn.pad_sequence(bert_input,batch_first=True)
        bert_label=torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)
        segment_label=torch.nn.utils.rnn.pad_sequence(segment_label, batch_first=True)
        is_next=torch.cat([i['is_next'] for i in output_dic_lis])

        return {'bert_input':bert_input,
                'bert_label':bert_label,
                'segment_label':segment_label,
                'is_next':is_next}



    def save_state_dict(self,model,epoch,dir_path='./output',file_path='bert.model'):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path=dir_path+'/'+file_path+'.epoch.{}'.format(str(epoch))
        model.to('cpu')
        torch.save({'model_state_dict':model.state_dict()},save_path)
        print('{} saved'.format(save_path))
        model.to(self.device)

    def test(self,epoch,df_path='./bert_state_dict/df_log.pickle'):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch,self.test_dataloader,train=False,df_path=df_path)

if __name__ == '__main__':
    def init_trainer(dynamic_lr,load_model=False):
        """
        定义初始化训练器函数，返回预训练器类的实例
        :param dynamic_lr: 学习率
        :param load_model: 是否加载预训练好的模型
        :return:返回训练器实例
        """
        trainer=Pretrainer(BertForPreTraining,
                           vocab_size=config['vocab_size'],
                           max_seq_len=config['max_seq_len'],
                           batch_size=config['batch_size'],
                           lr=dynamic_lr,
                           with_cuda=True)

        if load_model:
            trainer.load_model(trainer.bert_model,dir_path=config['output_path'])
        return trainer

    start_epoch=0
    train_epoches=1
    trainer=init_trainer(config['lr'],load_model=True)
    all_loss=[]
    threshold=0
    patient=10
    best_f1=0
    dynamic_lr=config['lr']
    for epoch in range(start_epoch,start_epoch+train_epoches):
        print('train with learning rate {}'.format(str(dynamic_lr)))
        # trainer.train(epoch)
        # trainer.save_state_dict(trainer.bert_model,epoch,dir_path=config['output_path'],
        #                         file_path='bert.model')
        trainer.test(epoch)

