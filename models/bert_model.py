import torch
import torch.nn as nn
import copy
import math
from torch.nn import CrossEntropyLoss

def gelu(x):
    return x*0.5*(1.0+torch.erf(x/math.sqrt(2.0)))

ACT2FN={'gelu':gelu,'relu':torch.nn.functional.relu}

class BertConfig(object):
    def __init__(self,vocab_size=32162,
                 hidden_size=384,
                 num_hidden_layers=6,
                 num_attention_heads=12, #12
                 intermediate_size=384*4,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.4,
                 attention_probs_dropout_prob=0.4,

                 type_vocab_size=256,
                 initializer_range=0.02
                 ):
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.hidden_dropout_prob=hidden_dropout_prob

        self.type_vocab_size=type_vocab_size
        self.initializer_range=initializer_range

class BertEmbeddings(nn.Module):
    """"
    与循环网络embedding层不同的是，对embedding参数进行了行归一化，新加入了位置编码和（token_type_embeddings），并且做了LayerNorm和dropout
    """
    def __init__(self,config):
        super(BertEmbeddings,self).__init__()
        self.word_embeddings=nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
        self.token_type_embeddings=nn.Embedding(config.type_vocab_size,config.hidden_size)
        nn.init.orthogonal_(self.word_embeddings.weight)
        nn.init.orthogonal_(self.token_type_embeddings.weight)

        epsilon=1e-8
        #embedding行归一化，div可换成"/"。'.weight'是'parameter'类型。'.weight.data'是Tensor
        self.word_embeddings.weight.data=self.word_embeddings.weight/(torch.norm(self.word_embeddings.weight,p=2,dim=1,keepdim=True)+epsilon)
        # print(self.word_embeddings.weight)
        self.token_type_embeddings.weight.data=self.token_type_embeddings.weight/(torch.norm(self.token_type_embeddings.weight,p=2,dim=1,keepdim=True)+epsilon)
        # print(self.token_type_embeddings.weight)
        self.LayerNorm=BertLayerNorm(config.hidden_size,eps=1e-12)
        # 随机丢掉矩阵中的某一元素值
        self.dropout=nn.Dropout(config.hidden_dropout_prob)

    def forward(self,input_ids,positional_enc,token_type_ids=None):
        """
        [batch_size,sequence_length] --> [batch_size,sequence_length,embedding_dim]
        :param input_ids:
        :param positional_enc:
        :param token_type_ids:
        :return:
        """
        # print('BertEmbeddings输入{}:'.format(input_ids.size()))
        words_embeddings=self.word_embeddings(input_ids)# [1, 138]-->[1, 138, 384]
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)

        token_type_embeddings=self.token_type_embeddings(token_type_ids)
        embeddings=words_embeddings+positional_enc+token_type_embeddings
        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        # print('BertEmbeddings输出{}:'.format(embeddings.size()))
        return embeddings

class BertLayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-12):
        super(BertLayerNorm,self).__init__()
        self.variance_epsilon=eps
        self.weight=nn.Parameter(torch.ones(hidden_size))
        # print(self.weight.size())
        self.bias=nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        #对矩阵行归一化,然后线性加权
        u=x.mean(-1,keepdim=True)
        s=(x-u).pow(2).mean(-1,keepdim=True)
        x=(x-u)/torch.sqrt(s+self.variance_epsilon)
        # print((self.weight*x+self.bias).size())
        # weight的维度为[hidden_dim],x为[1,138,hidden_dim]，相当于给字向量的每一个维度乘上权重
        return self.weight*x+self.bias


class BertPreTrainedModel(nn.Module):
    """
    处理权重初始化的抽象类
    要求config参数必须继承自BertConfig
    """
    def __init__(self,config,*inputs,**kwargs):
        super(BertPreTrainedModel,self).__init__()
        # 判断输入config是否为BertConfig类的实例
        if not isinstance(config,BertConfig):
            raise ValueError("为了从google预训练模型中创建model,输入config应该为'BertConfig'类的实例")
        self.config=config

    def init_bert_weights(self,module):
        if isinstance(module,(nn.Linear)):
            module.weight.data.normal_(mean=0.0,std=self.config.initializer_range)
        elif isinstance(module,BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module,nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertSelfAttention(nn.Module):
    def __init__(self,config):
        super(BertSelfAttention,self).__init__()
        # print(config.num_attention_heads)
        if config.hidden_size%config.num_attention_heads!=0:
            raise ValueError("注意力头的个数({})不是隐藏层维度({})的倍数".format(config.num_attention_heads,config.hidden_size))
        self.num_attention_heads=config.num_attention_heads
        self.attention_head_size=int(config.hidden_size/config.num_attention_heads)
        self.all_head_size=self.num_attention_heads*self.attention_head_size

        self.query=nn.Linear(config.hidden_size,self.all_head_size)
        self.key=nn.Linear(config.hidden_size,self.all_head_size)
        self.value=nn.Linear(config.hidden_size,self.all_head_size)

        self.dropout=nn.Dropout(config.attention_probs_dropout_prob)

        # print(self.value.weight.size())

    def transpose_for_scores(self,x):
        # [batch_size, seq_length, embedding_dim]-->[batch_size, seq_length,num_heads, attention_head_size]
        # -->[batch_size, num_heads, seq_length, attention_head_size]
        new_x_shape=x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
        x=x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,hidden_states,attention_mask,get_attention_matrices=False):
        # print('BertSelfAttention的输入{}:'.format(hidden_states.size()))
        mixed_query_layer=self.query(hidden_states) #[1, 138, 384]
        mixed_key_layer = self.key(hidden_states)
        mixed_key_layer = self.value(hidden_states)

        query_layer=self.transpose_for_scores(mixed_query_layer)
        key_layer=self.transpose_for_scores(mixed_query_layer)
        value_layer=self.transpose_for_scores(mixed_query_layer)

        attention_scores=torch.matmul(query_layer,key_layer.transpose(-1,-2))
        # print(key_layer)
        # print(query_layer.size())
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)
        # attention_mask的作用是使padding的元0素经过softmax后还为0
        attention_scores=attention_scores+attention_mask
        attention_probs_=nn.Softmax(dim=-1)(attention_scores)
        attention_probs=self.dropout(attention_probs_)
        context_layer=torch.matmul(attention_probs,value_layer)
        # print(context_layer)
        context_layer=context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape=context_layer.size()[:-2]+(self.all_head_size,)
        context_layer=context_layer.view(*new_context_layer_shape)
        # print('BertSelfAttention的输出{}:'.format(context_layer.size()))

        if get_attention_matrices:
            return context_layer,attention_probs_
        return context_layer,None


class BertSelfOutput(nn.Module):
    def __init__(self,config):
        super(BertSelfOutput,self).__init__()
        self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm=BertLayerNorm(config.hidden_size,eps=1e-12)


    def forward(self,hidden_states,input_tensor):
        # print('BertSelfOutput的输入{}:'.format(hidden_states.size()))
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        # print('BertSelfOutput的输出{}:'.format(hidden_states.size()))
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self,config):
        super(BertAttention,self).__init__()
        self.self=BertSelfAttention(config)
        self.output=BertSelfOutput(config)

    def forward(self, input_tensor,attention_mask,get_attention_matrices=False):
        # print(input_tensor)
        self_output,attention_matrices=self.self(input_tensor,attention_mask,get_attention_matrices=get_attention_matrices)
        # print(self_output)
        attention_output=self.output(self_output,input_tensor)
        return attention_output,attention_matrices


class BertIntermediate(nn.Module):
    def __init__(self,config):
        super(BertIntermediate,self).__init__()
        # 会改变隐藏层的维度
        self.dense=nn.Linear(config.hidden_size,config.intermediate_size)
        self.intermediate_act_fn=ACT2FN[config.hidden_act]
    def forward(self,hidden_states):
        # print('BertIntermediate的输入{}:'.format(hidden_states.size()))
        hidden_states=self.dense(hidden_states)
        # print(hidden_states)
        hidden_states=self.intermediate_act_fn(hidden_states)
        # print('BertIntermediate的输出{}:'.format(hidden_states.size()))
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self,config):
        super(BertOutput,self).__init__()
        self.dense=nn.Linear(config.intermediate_size,config.hidden_size)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm=BertLayerNorm(config.hidden_size,eps=1e-12)

    def forward(self,hidden_states,input_tensor):
        # print('BertOutput的输入{}:'.format(hidden_states.size()))
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        # print('BertOutput的输出{}:'.format(hidden_states.size()))
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self,config):
        super(BertLayer,self).__init__()
        self.attention=BertAttention(config)
        self.intermediate=BertIntermediate(config)
        self.output=BertOutput(config)

    def forward(self,hidden_states,attention_mask,get_attention_matrices=False):
        attention_output,attention_matrices=self.attention(hidden_states,attention_mask,get_attention_matrices=get_attention_matrices)
        # print("-----------------")
        # print(attention_output)
        intermediate_output=self.intermediate(attention_output)
        # print(intermediate_output)
        layer_output=self.output(intermediate_output,attention_output)
        return layer_output,attention_matrices

class BertEncoder(nn.Module):
    def __init__(self,config):
        super(BertEncoder,self).__init__()
        layer=BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self,hidden_states,attention_mask,output_all_encoded_layers=False,
                get_attention_matrices=False):

        all_attention_matrices=[]
        all_encoder_layers=[]
        # print(hidden_states)
        for layer_module in self.layer:
            hidden_states,attention_matrices=layer_module(hidden_states,attention_mask,get_attention_matrices=get_attention_matrices)
            # print(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)
        return all_encoder_layers,all_attention_matrices

class BertPooler(nn.Module):
    def __init__(self,config):
        super(BertPooler,self).__init__()
        self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.activation=nn.Tanh()
    def forward(self, hidden_states):
        # 取第0行，即CLS的embedding维
        # print('BertPooler的输入{}:'.format(hidden_states.size()))
        first_token_tensor=hidden_states[:,0,:]
        pooled_output=self.dense(first_token_tensor)
        pooled_output=self.activation(pooled_output)
        # print('BertPooler的输出{}:'.format(pooled_output.size()))
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self,config):
        super(BertModel,self).__init__(config)
        self.embeddings=BertEmbeddings(config)
        self.encoder=BertEncoder(config)
        self.pooler=BertPooler(config)
        # self.apply(self.init_bert_weights)

    def forward(self,input_ids,positional_enc,token_type_ids=None,attention_mask=None,
                output_all_encoded_layers=True,get_attention_matrices=False):
        # 输入文本序列相应位置有>0的值时，返回1，否则返回0，张量的size不变--目的是让padding的位置元素足够小
        if attention_mask is None:
            attention_mask=(input_ids>0)
        # 生成与输入文本size相同的全0张量
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)
        # print(attention_mask.size())
        # 扩展attention_mask的维度
        extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)#[1, 138]-->[1, 1, 1, 138]
        # ???????
        # 使mask与实例参数的数据类型一致
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # 将注意力mask矩阵转换为一个很大的负数
        extended_attention_mask=(1.0-extended_attention_mask)*-10000.0
        embedding_output=self.embeddings(input_ids,positional_enc,token_type_ids)
        # print(embedding_output)
        # print('embedding层输出{}'.format(embedding_output.size()))
        # print('extended_attention_mask层输出{}'.format(extended_attention_mask.size()))
        encoded_layers,all_attention_matrices=self.encoder(embedding_output,extended_attention_mask,
                                                           output_all_encoded_layers=output_all_encoded_layers,
                                                           get_attention_matrices=get_attention_matrices)
        # print('encode层输出{}'.format(encoded_layers[-1].size()))
        if get_attention_matrices:
            return all_attention_matrices
        sequence_output=encoded_layers[-1]
        # print(sequence_output.size())
        pooled_output=self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers=encoded_layers[-1]
        return encoded_layers,pooled_output
        # print(input_ids.size())


        # print(extended_attention_mask)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self,config):
        super(BertPredictionHeadTransform,self).__init__()
        self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.transform_act_fn=ACT2FN[config.hidden_act]
        self.LayerNorm=BertLayerNorm(config.hidden_size,eps=1e-12)

    def forward(self, hidden_states):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.transform_act_fn(hidden_states)
        hidden_states=self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self,config,bert_model_embedding_weights):
        super(BertLMPredictionHead,self).__init__()
        self.transform=BertPredictionHeadTransform(config)
        self.decoder=nn.Linear(bert_model_embedding_weights.size(1),
                               bert_model_embedding_weights.size(0),
                               bias=False)
        self.decoder.weight=bert_model_embedding_weights
        self.bias=nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
    def forward(self, hidden_states):
        hidden_states=self.transform(hidden_states)
        hidden_states=self.decoder(hidden_states)+self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self,config,bert_model_embedding_weights):
        super(BertPreTrainingHeads,self).__init__()
        self.predictions=BertLMPredictionHead(config,bert_model_embedding_weights)
        self.seq_relationship=nn.Linear(config.hidden_size,2)

    def forward(self, sequence_output,pooled_output):
        prediction_scores=self.predictions(sequence_output)
        seq_relationship_score=self.seq_relationship(pooled_output)
        return prediction_scores,seq_relationship_score

class BertForPreTraining(BertPreTrainedModel):
    """
    具有预训练头部的BERT模型
    该模块包含具有2个预测头部的BERT模型，分别是：
        --masked的语言模型头部
        --下一个句子的分类头部
    参数：
        config：BertConfig类的实例，里面有建立一个新模型的配置信息
    输入：
        input_ids:
    """
    def __init__(self,config):
        super(BertForPreTraining,self).__init__(config)
        self.bert=BertModel(config)
        self.cls=BertPreTrainingHeads(config,self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.vocab_size=config.vocab_size
        self.next_loss_func=CrossEntropyLoss()
        self.mlm_loss_func=CrossEntropyLoss(ignore_index=0)

    def compute_loss(self,predictions,labels,num_class=2,ignore_index=-100):
        loss_func=CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1,num_class),labels.view(-1))
    def forward(self,input_ids,positional_enc,token_type_ids=None,attention_mask=None,
                masked_lm_labels=None,next_sentence_label=None):
        sequence_output,pooled_output=self.bert(input_ids,positional_enc,
                                                token_type_ids,attention_mask,
                                                output_all_encoded_layers=False)
        # mlm_preds最后一维为vocab_size,next_sen_preds为二分类
        mlm_preds,next_sen_preds=self.cls(sequence_output,pooled_output)
        return mlm_preds,next_sen_preds

if __name__ == '__main__':
    #修改了路径****
    from Sentiment_Inference import Sentiment_Analysis
    model=Sentiment_Analysis(300,1)
    # #一个字符串代表一个seq
    # test_list=[
    #     "有几次回到酒店房间都没有被整理。两个人入住，只放了一套洗漱用品。",
    #     "早餐时间询问要咖啡或茶，本来是好事，但每张桌子上没有放“怡口糖”（代糖），又显得没那么周到。房间里卫生间用品补充，有时有点漫不经心个人觉得酒店房间禁烟比较好",
    #     "南京东路地铁出来就能看到，很方便。酒店大堂和房间布置都有五星级的水准。",
    #     "服务不及5星，前台非常不专业，入住时会告知你没房要等，不然就加钱升级房间。前台个个冰块脸，对待客人好像仇人一般，带着2岁的小孩前台竟然还要收早餐费。门口穿白衣的大爷是木头人，不会提供任何帮助。入住期间想要多一副牙刷给孩子用，竟然被问为什么。五星设施，一星服务，不会再入住！"
    # ]
    # # test_list =['zhang','']
    # text_tokens_,positional_enc=model(test_list)
    #
    # # print(text_tokens_)
    # config=BertConfig()
    # model=BertModel(config)
    # model(text_tokens_,positional_enc)
