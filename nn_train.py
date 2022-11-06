import os
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from LSTM_model import PoetryModel
from nueral_network_model import NNLM
from dataProcess import *
from torch.autograd import Variable
from config import Config

class TrainModel(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()
        self.device = torch.device('cuda') if self.config.use_gpu else torch.device('cpu')

    def train(self, data_loader, model, optimizer, criterion):
        for epoch in range(self.config.epoch_num):
            for step, (x,y)in enumerate(data_loader):
                # 处理数据
                x = x.to(self.device)
                y = y.to(self.device)
                input_ = x
                target = y
                target = target.view(-1)
                output = model(input_)
                # 前向计算
                loss = criterion(output, target) # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
                # 计算语言模型的困惑度
                perplexity=torch.exp(loss)
                optimizer.zero_grad()
                # 反向计算梯度
                loss.backward()
                # 权重更新
                optimizer.step()
                if step % 200 == 0:
                    print('epoch: %d,loss: %f,perplexity: %d' % (epoch, loss.data,perplexity))

            if epoch % 1 == 0:
                # 保存模型
                dir = config.model_save_path
                torch.save(model.state_dict(), dir)

    def run(self):
        # 1 获取数据
        data_x, data_y,char_to_ix, ix_to_chars = get_data(self.config)
        vocab_size = len(char_to_ix)
        print('样本数：%d' % len(data_x))
        print('标签数：%d' % len(data_y))
        print('词典大小： %d' % vocab_size)
        data_x = Variable(torch.LongTensor(data_x))
        data_y = Variable(torch.LongTensor(data_y))
        # 2 设置dataloader
        data = Data.TensorDataset(data_x,data_y)  
        data_loader = Data.DataLoader(data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=2)
        # 3 创建模型
        model = NNLM(n_class=vocab_size,
                            embedding_dim = config.embedding_dim_nn,
                            input_size = config.embedding_dim_nn*config.n_gram,
                            hidden_size = config.hidden_dim)
        model.to(self.device)
        # 4 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        # 5 创建损失函数,使用与logsoftmax的输出
        criterion = nn.CrossEntropyLoss()
        # 6.训练
        self.train(data_loader, model, optimizer, criterion)

if __name__ == '__main__':
    obj = TrainModel()
    obj.run()





