#   coding:utf-8
#   This file is part of potentialmind.
#
#   potentialmind is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2019/11/08 15:36:27'

import os
import torch
import torch.nn as nn
from emml.train.read_data import load_pmdata
from matfleet.utilities import now_time


class DNN(nn.Module):
    DP_RADIO = 0.5
    B_ININT = -0.0

    def __init__(self, n_feature, n_hidden, n_output, batch_normalize=True, dropout=True, activation=nn.ReLU()):
        super(DNN, self).__init__()
        assert isinstance(n_hidden, (list, tuple))

        self.ACTIVATION = activation
        self.do_bn = batch_normalize
        self.do_dp = dropout
        self.fcs, self.bns, self.dps = [], [], []
        
        self.bn_input = nn.BatchNorm1d(n_feature, momentum=0.5)
        self.n_hidden = [n_feature] + n_hidden
        
        for hid_index in range(1, len(self.n_hidden)):
            fc = torch.nn.Linear(self.n_hidden[hid_index - 1], self.n_hidden[hid_index])
            setattr(self, 'fc%d' % hid_index, fc)
            self._set_init(fc)
            self.fcs.append(fc)
            
            if self.do_bn:
                bn = nn.BatchNorm1d(self.n_hidden[hid_index], momentum=0.5)
                setattr(self, 'bn%d' % hid_index, bn)
                self.bns.append(bn)
            
            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'dp%s' % hid_index, dp)
                self.dps.append(dp)
        
        self.predict = torch.nn.Linear(self.n_hidden[-1], n_output)
        self._set_init(self.predict)
    
    def _set_init(self, fc):
        nn.init.normal_(fc.weight, mean=0, std=0.1)
        nn.init.constant_(fc.bias, self.B_ININT)
    
    def forward(self, x):
        # if self.do_bn: x = self.bn_input(x)
        for i in range(len(self.n_hidden) - 1):
            x = self.fcs[i](x)
            if self.do_bn:
                x = self.bns[i](x)
            if self.do_dp:
                x = self.dps[i](x)
            x = self.ACTIVATION(x)
        
        x = self.predict(x)
        return x


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def do_time():
    return now_time().replace(' ', '_').replace('-', '_').replace(':', '_')


def train(train_csv_fn, test_csv_fn, valid_csv_fn, col_index=-5, output_all=False,
          restore=False, module_params_fn=None, lr=0.01, epoch=10000, cuda=True,
          save_dir='', label='1', head_dir=r"../data",
          n_feature=28, hidden_nodes=None, activation=nn.ReLU(), optimizer='Adam',
          save_module=True, save_step=100):
    rember_loss = 0
    _go = True

    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    
    if hidden_nodes is None:
        hidden_nodes = [100, 50, 50, 20]
    
    train_pmdata_loader = load_pmdata(filename=train_csv_fn, shuffle=True, output_all=output_all,
                                      output_index=col_index, batch_size=82)
    test_pmdata_loader = load_pmdata(filename=test_csv_fn, shuffle=True, output_all=output_all,
                                     output_index=col_index, batch_size=20)
    valid_pmdata_loader = load_pmdata(filename=valid_csv_fn, shuffle=True, output_all=output_all,
                                     output_index=col_index, batch_size=26)

    if cuda:
        dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=1,
                  batch_normalize=True, dropout=True, activation=activation).cuda()
    else:
        dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=1,
                  batch_normalize=True, dropout=True, activation=activation)
    
    if restore:
        dnn.load_state_dict(torch.load(module_params_fn))
    print(dnn)
    # dnn = DNN(660, 1000, 4)
    # optimizer = torch.optim.Adam(dnn.parameters(), lr)  # weight_decay=0.01
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(dnn.parameters(), lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(dnn.parameters(), lr)
    else:
        raise ValueError("Only support Adam and SGD")
    loss_func = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in dnn.parameters():
    #     reg_loss += l1_crit(param, 100)
    tfn = os.path.join(save_dir, 'running_%s.log' % label)

    for ep in range(epoch):
        epoch = ep + 1
        # if epoch % 1500 == 0:
        #     lr = lr * 0.5
        #     adjust_learning_rate(optimizer, lr)

        for step, (b_x, b_y) in enumerate(train_pmdata_loader):
            # input_data = torch.DoubleTensor(b_x)
            # print(input_data)
            # print(input_data.shape)
            # print(dnn.fc1.weight.grad)
            if cuda:
                b_x, b_y = b_x.cuda(), b_y.cuda()
            else:
                b_x, b_y = b_x, b_y
            output = dnn(b_x.float())
            label_y = b_y.reshape(-1, 1)
            loss = loss_func(output, label_y.float())  # + 0.005 * reg_loss

            dnn.eval()
            for _, (test_x, test_y) in enumerate(test_pmdata_loader):
                
                if cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()

                test_output = dnn(test_x.float())
                testlabel_y = test_y.reshape(-1, 1)
                test_loss = loss_func(test_output, testlabel_y.float())  # + 0.005 * reg_loss
            # print(output.cpu().data.numpy().shape, label_y.cpu().data.numpy().shape)
            # if rember_loss != 0:
            #     # if ((test_loss.cpu().data.numpy() - rember_loss) > 0.1) or ((test_loss.cpu().data.numpy() - loss.cpu().data.numpy()) > 0.02):
            #     if (test_loss.cpu().data.numpy() - rember_loss) > 0.1:
            #         _go = False
            # else:
            #     _go = True
            for _, (valid_x, valid_y) in enumerate(test_pmdata_loader):
    
                if cuda:
                    valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
    
                valid_output = dnn(valid_x.float())
                validlabel_y = valid_y.reshape(-1, 1)
                valid_loss = loss_func(valid_output, validlabel_y.float())  # + 0.005 * reg_loss

            txt_temple = 'Epoch: {0} | Step: {1} | ' \
                         'train loss: {2:.6f} | ' \
                         'test loss: {3:.6f} |' \
                         'valid loss: {4: .6f}'.format(epoch, step,
                                                       loss.cpu().data.numpy(),
                                                       test_loss.cpu().data.numpy(),
                                                       valid_loss.cpu().data.numpy())
            if epoch % 10 == 0:
                print(txt_temple)
            # now_step = step + epoch * math.ceil(TOTAL_LINE / BATCH_SIZE)
            if epoch == 0:
                write(tfn, txt_temple, 'w')
            else:
                # if now_step % SAVE_STEP == 0:
                write(tfn, txt_temple, 'a')
                
            if save_module:
                if epoch % save_step == 0:
                    torch.save(dnn, os.path.join(save_dir, 'dnn_%d_%s.pkl' % (epoch, label)))
                    torch.save(dnn.state_dict(), os.path.join(save_dir, 'dnn_params_%d_%s.pkl' % (epoch, label)))

            if _go:
                rember_loss = test_loss.cpu().data.numpy()
                dnn.train()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                return None


def write(fn, content, mode='w'):
    with open(fn, mode) as f:
        f.write(content + '\n')

sele = lambda x: x

def run_train(nfeature, head_dir, save_dir):
    labels = ["3layer_100_relu", "3layer_100_sigmoid",
              "3layer_100_tanh", "3layer_100_relu_sgd",
              "4layer_100", "4layer_500"]
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                   nn.ReLU(), nn.ReLU(), nn.ReLU()]
    hidden_layers = [[100, 50, 20], [100, 50, 20],
                     [100, 50, 20], [100, 50, 20],
                     [100, 100, 50, 20], [500, 100, 50, 20]]
    optimizers = ['Adam', 'Adam', 'Adam', 'SGD', 'SGD', 'SGD']
    
    labels = sele(labels)
    activations = sele(activations)
    hidden_layers = sele(hidden_layers)
    optimizers = sele(optimizers)

    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        if i >= 3:
            epoch = 12000
            save_step = 100
        else:
            epoch = 3000
            save_step = 50
        
        train(cuda=True,
              epoch=epoch,
              save_dir=save_dir,
              label=label,
              head_dir=head_dir,
              n_feature=nfeature,
              hidden_nodes=hidden_layer,
              activation=activation,
              optimizer=optimizer,
              save_module=True,
              save_step=save_step)


def run_test(nfeature, save_dir, head_dir=r"..\\rdata\\"):
    # labels = ["3layer_100_adam", "3layer_100_sgd",
    #           "3layer_100_sgd_Sigmod", "3layer_100_sgd_Tanh",
    #           "4layer_100_sgd", "4layer_500_sgd",
    #           "5layer_100", "6layer_100"]
    # activations = [nn.ReLU(), nn.ReLU(),
    #                nn.Sigmoid(), nn.Tanh(),
    #                nn.ReLU(), nn.ReLU(),
    #                nn.ReLU(), nn.ReLU()]
    labels = ["3layer_100_relu", "3layer_100_sigmoid",
              "3layer_100_tanh", "3layer_100_relu_sgd",
              "4layer_100", "4layer_500"]
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                   nn.ReLU(), nn.ReLU(), nn.ReLU()]
    hidden_layers = [[100, 50, 20], [100, 50, 20],
                     [100, 50, 20], [100, 50, 20],
                     [100, 100, 50, 20], [500, 100, 50, 20]]
    
    nums = [200, 50, 50, 12000, 12000, 11600]
    
    for m in ['train_30_train.csv', 'train_30_test.csv']:
        for i in range(len(labels)):
            nlabel = labels[i]
            nactivation = activations[i]
            nhidden_layer = hidden_layers[i]
            num = nums[i]
                
            ttest(test_csv_fn=os.path.join(head_dir, m),
                  mp_fn=os.path.join(save_dir, 'dnn_params_%d_%s.pkl' % (num, nlabel)),
                  output_fn='result_%s_%s.out' % (m, nlabel), activation=nactivation,
                  save_dir=save_dir,
                  n_feature=nfeature, hidden_nodes=nhidden_layer)
            
            
if __name__ == '__main__':
    # hdir = r'G:\ztml\ztml\rdata\all_rmcoref_data'
    # save_dirs = 'final_training_module'
    # nssfeature = 11
    # # run_train(nssfeature, hdir, save_dirs)
    # run_test(nssfeature, save_dirs, hdir)

    train_csv_fn = '../data/5.train_82_train.xlsx'
    test_csv_fn = '../data/5.train_20_test.xlsx'
    valid_csv_fn= '../data/5.26_for_check.xlsx'
    fn_index = {-5: '∆GO*', -4: '∆GOH*', -3: '∆GOOH*', -2: 'ηORR', -1: 'ηOER'}
    col_index = -1
    # hidden_nodes = [2000, 500, 200, 100, 50, 20],

    train(train_csv_fn=train_csv_fn,
          test_csv_fn=test_csv_fn,
          valid_csv_fn=valid_csv_fn,
          cuda=False,
          epoch=5000,
          save_dir='./%s' % fn_index[col_index],
          label=fn_index[col_index],
          head_dir='../data',
          n_feature=28,
          hidden_nodes=[100, 50, 20],
          optimizer='Adam',
          activation=nn.ReLU(),
          save_module=True,
          save_step=500,
          col_index=col_index)

