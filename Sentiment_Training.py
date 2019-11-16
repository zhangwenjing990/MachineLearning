
import configparser

class Sentiment_trainer:
    def __init__(self,max_seq_len,batch_size,lr,with_cuda=True):
        config_=configparser.ConfigParser()
        config_.read('./config/sentiment_model_config.ini')
        self.config=config_['DEFAULT']
        self.vocab_size=int(self.config['vocab_size'])
        self.batch_size=batch_size
        self.lr=lr


if __name__ == '__main__':
    def init_trainer(dynamic_lr,batch_size=24):
        trainer=Sentiment_trainer(max_seq_len=300,
                                  batch_size=batch_size,
                                  lr=dynamic_lr,
                                  with_cuda=True)
        return trainer,dynamic_lr
    start_epoch=0
    train_epoches=9999
    init_trainer(dynamic_lr=1e-06,batch_size=24)