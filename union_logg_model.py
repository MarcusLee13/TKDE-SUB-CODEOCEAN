import time
import json
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from pandas.core.frame import DataFrame
import pickle
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import os
import math
import load_data
#from  batchifyadd import Batch
from batchify11 import Batch
import argparse
import random
import torch.nn.functional as F
import word_embed_dataloader
f_logger = open("/content/drive/MyDrive/GTY3PLG5.txt","w")


def cal_mean(data_fname):
    mean = 0
    data = []
    for line in open(data_fname,"rb"):
        data.append(json.loads(line))
    for i in data:
        mean += i["overall"]
    mean = mean/len(data)
    return mean


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--seed', default=0, type=int)

    return arg.parse_args()

'''
class Log_Bert(nn.Module):

    def __init__(self,word_weight_matrix):
        super(Log_Bert, self).__init__()

        self.review_embeds = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.review_embeds.weight = nn.Parameter(word_weight_matrix, requires_grad=False)

        self.conv20 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3,300), padding=(1,0)),)
            #nn.MaxPool1d(kernel_size=u_max // 3, padding=1))

        self.conv21 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3,300), padding=(1,0)),)
            #nn.MaxPool1d(kernel_size=i_max // 3, padding=1))

        self.linear = nn.Linear(600,100,bias=True)
        self.linear2 = nn.Linear(100, 5, bias=False)


    def forward(self, x,y, batch_size,user_list,item_list):

        x = self.review_embeds(x)
        y = self.review_embeds(y)
        x = x.view(len(x),1,-1,300)
        y = y.view(len(y), 1, -1, 300)
        x = self.conv20(x)
        y = self.conv21(y)
        x = F.max_pool2d(x,kernel_size=(x.size()[2]//3,1), padding=(1,0)).squeeze() #(32,100,3)
        y = F.max_pool2d(y, kernel_size=(y.size()[2] // 3, 1), padding=(1, 0)).squeeze()

        x = x.view(len(x),-1)
        y = y.view(len(y),-1)
        out = torch.cat((x*y,x-y),dim=-1)
        out = self.linear(out)
        out = self.linear2(out)

        return out

        out = self.linear7(out).unsqueeze(1) 
        #out = self.linear8(out).unsqueeze(1)



        return out.squeeze()
'''

class Log_Bert(nn.Module):

    def __init__(self,u_max,):
        super(Log_Bert, self).__init__()

        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(768)

        #user
        self.lstm = nn.LSTM(768, hidden_size = 768, num_layers=1, batch_first=True, bidirectional=True)  # (batch, seq, feature)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=768, kernel_size=(1)),nn.MaxPool1d(kernel_size=(u_max//2)+3))  #(batch,feature,1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=768, kernel_size=(3), #dilation=2
                      ),nn.MaxPool1d(kernel_size=(u_max//2)+3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=768, kernel_size=(4), #dilation=3
                      ),nn.MaxPool1d(kernel_size=(u_max//2)+3))

        self.linear1 = nn.Linear(768,384)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        #item
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=768, kernel_size=(5)),nn.MaxPool1d(kernel_size=k_size)) #(batch,5,768)

        #union
        self.dropout = nn.Dropout(0.2, inplace=False)
        #self.linear2 = nn.Linear(15360,768)
        self.linear3 = nn.Linear(768,384)
        self.linear4 = nn.Linear(3840,768)
        self.linear5 = nn.Linear(3840,768)
        self.linear6 = nn.Linear(8448,400)
        #self.linear7 = nn.Linear(2000,384)
        self.classifier = nn.Linear(400, 5)

    def forward(self, x,y, batch_size):

        x = self.bn1(x)
        y = self.bn2(y)

        x = x.permute(0,2,1) #[32,768,seq]
        x1, (h_n, h_c) = self.lstm(x, None)
        x2 = x1.view(x1.size(0),x1.size(1), 2, -1)
        x3 = torch.sum(x2[:, :, :, :] , dim=1)
        x4 = x3[:,0,:].view(-1,768,1)
        x5 =  x3[:,1,:].view(-1,768,1)
        x_lstm = torch.cat((x4, x5), dim=2)  #([32, 768, 2])

        x = x.permute(0,2,1)
        x1 = self.conv1(x)
        #print("x1", x1.size())
        x2 = self.conv2(x)
        #print("x2", x2.size())
        x3 = self.conv3(x)
        #print("x3", x3.size())

        x_cnn = torch.cat((x1,x2,x3), dim=2) #[32, 768, 3]

        x = torch.cat((x_lstm,x_cnn), dim =2) #(batch,768,5)
        print(x.shape)
        f_logger.writelines(x.shape+'\n')
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.tanh(x)  #[32,5,384]
        #print("x",x.size())
        y = self.conv4(y)
        y = y.permute(0, 2, 1) #[32,5,384]
        #print(y.size())
        y = self.linear3(y)
        y = y.permute(0, 2, 1) #[32,384,5]
        #print(y.size())

        out = torch.bmm(x,y) #(batch,5,5)
        #print("out",out.size())
        out1 =torch.nn.functional.softmax(out,dim=1)
        #print("out1", out1.size())
        out1 = torch.bmm(y,out1)  #[batch, 5, 384]
        out2 =torch.nn.functional.softmax(out,dim=2)
        out2 = torch.bmm(out2,x) #[batch, 384, 5]
        #print(out2.size())

        out2 = out2.view(-1, 1920)
        out1 = out1.reshape(-1, 1920)
        x = x.view(-1, 1920)
        y = y.reshape(-1, 1920)
        out2 = torch.cat((y, out2), dim=1)
        #print("out_x",out_x.size())
        out1= torch.cat((x, out1), dim=1)

        out_x = self.linear4(out1)
        #print(out_x.size())
        out_y =self.linear5(out2)
        out = torch.cat((out2,out1,out_x*out_y),dim=1)
        #out = out_x*out_y
        #print(out.size())
        out = self.dropout(out)
        out = self.linear6(out)
        out = self.classifier(out) #[32,5]
        #print("out", out.size())

        return out


class Optimer_Weight(nn.Module):

    def __init__(self,):
        super(Optimer_Weight, self).__init__()

        #self.conv1 = nn.Sequential(
        #    nn.Conv1d(in_channels=5, out_channels=1, kernel_size=(1), bias=False))
        #self.conv1 = nn.Linear(5, 1, bias=False)
        a = torch.randn(1,5)
        b = torch.tensor([[-2.8, -1.5, -0.8, 1., 2.5]])
        self.conv1 = nn.Parameter(b, requires_grad=False)
        # self.c = nn.Linear(5, 1)
        #self.p = nn.Parameter(torch.tensor([[0.2]], requires_grad=True))
        self.linear = nn.Linear(5,1)

    def cuda(self, device=None):
        super(Optimer_Weight, self).cuda(device=device)
        self.conv1 = self.conv1.cuda(device)
        return self

    def forward(self,classified ):
        #classified = classified.permute(0, 2, 1)
        classified = classified.squeeze(-1)
        classified = nn.functional.softmax(classified, dim=-1)
        #pre = self.conv1(classified)
        #pre = torch.sum(classified * self.conv1, dim=-1)
        pre = self.linear(classified)
        #pre = nn.functional.sigmoid(pre) * 4 + 1   #+self.p
        return pre


def train_model(user_review_lists,item_review_lists,overalls,log_lr,weight_lr, batch_size,log_bert,optimer_weight):

    for p in log_bert.parameters():
        p.requires_grad = True

    for p in optimer_weight.parameters():
        p.requires_grad = True

    optimizer_log = torch.optim.Adam(log_bert.parameters(), lr=log_lr, weight_decay=0)
    optimer_weight = torch.optim.Adam(optimer_weight.parameters(), lr=log_lr, weight_decay=0)
    optimizer_log.zero_grad()
    optimer_weight.zero_grad()
    loss_func = nn.MSELoss()

    user_review_lists = user_review_lists.cuda()
    item_review_lists = item_review_lists.cuda()
    overalls = overalls.cuda()

    pre = log_bert(user_review_lists, item_review_lists, batch_size)
    pre = torch.unsqueeze(pre, -1)
    pre = optimer_weight(pre)
    loss1 = loss_func(pre, overalls)
    loss1.backward(retain_graph = True)
    optimizer_log.step()
    optimer_weight.step()

    return loss1

def _test_model(user_review_lists,item_review_lists,overalls, batch_size,log_bert,optimer_weight,user_id,item_id):

    for p in log_bert.parameters():
        p.requires_grad = False

    for p in optimer_weight.parameters():
        p.requires_grad = False

    loss_func = nn.MSELoss()

    user_review_lists = user_review_lists.cuda()
    item_review_lists = item_review_lists.cuda()
    overalls = overalls.cuda()
    user_id = user_id.cuda()
    item_id = item_id.cuda()

    pre = log_bert(user_review_lists, item_review_lists, batch_size)
    #pre = torch.unsqueeze(pre, -1)
    pre = optimer_weight(pre)


    score_loss = []
    loss = loss_func(pre.squeeze(), overalls)
    score_loss.append(loss)

    pre = pre.squeeze(-1)
    for i in range(5):
        pre1 = (overalls == i+1).float() * pre
        a = torch.sum((overalls == i+1).float())
        if a != 0:
            loss = torch.sum(torch.abs(pre1-(overalls == i+1).float()*overalls))/a
        else:
            loss =0
        score_loss.append(loss)

    return score_loss[0],score_loss[1],score_loss[2],score_loss[3],score_loss[4],score_loss[5]




if __name__ == '__main__':
    
    data_list = ['Grocery_and_Gourmet_Food_5']
    '''
    ['Video_Games_5',"Toys_and_Games_5","Sports_and_Outdoors_5","Beauty_5","Baby_5"]
    ["Health_and_Personal_Care_5",'Clothing_Shoes_and_Jewelry_5','Books_5']
        ["Clothing_Shoes_and_Jewelry_5",'Electronics_5',
     "Baby_5", "Beauty_5", 'Grocery_and_Gourmet_Food_5', 'Books_5', "Health_and_Personal_Care_5", 'Movies_and_TV_5',
     "Office_Products_5", "Patio_Lawn_and_Garden_5", "Pet_Supplies_5", "Home_and_Kitchen_5",
     "Sports_and_Outdoors_5", "Toys_and_Games_5", "Video_Games_5", "Apps_for_Android_5"]
     
    '''

    batch_size = 32
    log_lr = 4e-4
    weight_lr = 0.001
    weight_lr1 = 0.0005

    for data_name in data_list:
        print(f'Dataset is: {data_name}')
        f_logger.writelines(f'Dataset is: {data_name}'+'\n')
        vocab_dict_path = '/content/pythonProject/archive/glove.6B.300d.txt'
        file_address = '/content/pythonProject/dual_loss/bert-unbase/TADO_final_results.txt'
        save_path_base = data_name
        train_data, test_data, user_dict, item_dict, \
        u_max, i_max, user_num, item_num = \
            load_data.prepare_data(data_path='/content/pythonProject/dual_loss/' + data_name + '.json',
                                   bert_dir='/content/pythonProject/dual_loss/bert-unbase',
                                   save_encode_result_pickle_path=f'./{save_path_base}_encode_result.pkl', )
        print(f'{data_name} finished!')
        f_logger.writelines(f'{data_name} finished!'+'\n')

        k_size = math.floor((i_max - 4) / 5)
        print(k_size)
        f_logger.writelines(k_size+'\n')
        batch = Batch(train_data, test_data,user_dict,item_dict, u_max,i_max, batch_size=32, train=True)

        log_bert = Log_Bert(u_max)
        log_bert = log_bert.cuda()
        optimer_weight = Optimer_Weight()
        optimer_weight = optimer_weight.cuda()
        optimizer_log = torch.optim.Adam(log_bert.parameters(), lr=log_lr, weight_decay=0.01)
        optimizer_weight = torch.optim.Adam(optimer_weight.parameters(), lr=weight_lr, weight_decay=0)
        loss_func_weight = nn.MSELoss()
        loss_func_log = nn.CrossEntropyLoss()

        args = get_args()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        for epoch in range(15):
            batch.set_train(True)
            num = 0
            loss = 0
            f = open(file_address, 'a')
            print(f'file address: {file_address}')
            f_logger.writelines(f'file address: {file_address}'+'\n')
            for _, user_review_lists, _, item_review_lists, _, overalls,_,_ in tqdm(batch):

                optimizer_log.zero_grad()
                optimer_weight.zero_grad()

                user_review_lists = user_review_lists.cuda()
                item_review_lists = item_review_lists.cuda()
                overalls = overalls.cuda()


                for p in log_bert.parameters():
                    p.requires_grad = True

                for p in optimer_weight.parameters():
                    p.requires_grad = False

                pre = log_bert(user_review_lists, item_review_lists, batch_size)

                loss1 = loss_func_log(pre, (overalls - 1).long())
                loss1.backward()
                optimizer_log.step()

                for p in log_bert.parameters():
                    p.requires_grad = False

                for p in optimer_weight.parameters():
                    p.requires_grad = True

                pre = log_bert(user_review_lists, item_review_lists, batch_size)
                pre = torch.unsqueeze(pre, -1)
                pre = optimer_weight(pre)

                loss2 = loss_func_weight(pre.squeeze(), overalls)
                loss2.backward()
                optimizer_weight.step()

                del user_review_lists, item_review_lists, overalls

                '''
           
                for p in log_bert.parameters():
                    p.requires_grad = True

                for p in optimer_weight.parameters():
                    p.requires_grad = True

                pre = log_bert(user_review_lists, item_review_lists, batch_size)
                pre = torch.unsqueeze(pre, -1)
                pre = optimer_weight(pre)
                loss1 = loss_func_weight(pre,overalls)
                loss1.backward()
                optimizer_weight.step()
                optimizer_log.step()


                loss += loss1
                num += 1
                del user_review_lists, item_review_lists, overalls
            loss /= num
            print(f'{epoch} is %.6f' % loss)

            for p in log_bert.parameters():
                p.requires_grad = False

            for p in optimer_weight.parameters():
                p.requires_grad = False
            '''

            batch.set_train(False)
            step = 0
            loss1_test = 0
            loss2_test = 0
            loss3_test = 0
            loss4_test = 0
            loss5_test = 0
            loss_test = 0
            for _, user_review_lists, _, item_review_lists, _, overalls,user_id, item_id in tqdm(batch):
                loss6, loss1, loss2, loss3, loss4, loss5 = _test_model(user_review_lists, item_review_lists,
                                                                       overalls, batch_size, log_bert,
                                                                       optimer_weight,user_id, item_id)


                loss1_test += loss1
                loss2_test += loss2
                loss3_test += loss3
                loss4_test += loss4
                loss5_test += loss5

                loss_test += loss6
                step += 1
                del user_review_lists, item_review_lists, overalls

            loss1_test /= step
            loss2_test /= step
            loss3_test /= step
            loss4_test /= step
            loss5_test /= step

            loss_test /= step

            print('test loss in %.6f' % loss_test)
            print(f'loss1:{loss1_test}, loss2:{loss2_test}, loss3:{loss3_test}, loss4:{loss4_test}, loss5:{loss5_test} ')
            f_logger.writelines('test loss in %.6f' % loss_test+'\n')
            f_logger.writelines(f'loss1:{loss1_test}, loss2:{loss2_test}, loss3:{loss3_test}, loss4:{loss4_test}, loss5:{loss5_test} '+'\n')
            f.writelines([data_name, ': ', 'Epoch: ', str(epoch), ', overall loss: ', str(loss_test), ', test 1 loss: ', str(loss1_test),
                          ', test 2 loss: ', str(loss2_test), ', test 3 loss: ', str(loss3_test), ', test 4 loss: ', str(loss4_test),
                          'test 5 loss: ', str(loss5_test), '\n'])
            f.flush()
        f.close()
        f_logger.close()

        '''
        for i in range(2):

            num = 0
            loss = 0
            for _, user_review_lists, _, item_review_lists, _, overalls in tqdm(batch):

                for p in optimer_weight.parameters():
                    p.requires_grad = True

                optimer_weight.zero_grad()
                loss_func_weight = nn.MSELoss()

                user_review_lists = user_review_lists.cuda()
                item_review_lists = item_review_lists.cuda()
                overalls = overalls.cuda()

                pre = log_bert(user_review_lists, item_review_lists, batch_size)
                pre = torch.unsqueeze(pre, -1)
                pre = optimer_weight(pre)

                loss2 = loss_func_weight(pre, overalls)
                loss2.backward()
                optimizer_weight.step()
                loss += loss2
                num += 1
            loss /= num
            print('train weight loss is %.6f' % loss)


        for p in log_bert.parameters():
            p.requires_grad = False

        for p in optimer_weight.parameters():
            p.requires_grad = False


        loss_test = 0
        batch.set_train(False)
        for _, user_review_lists, _, item_review_lists, _, overalls in tqdm(batch):
            batch.set_train(False)
            loss2 = _test_model(user_review_lists, item_review_lists, overalls, batch_size, log_bert,
                                optimer_weight)
            loss_test += loss2
            num += 1

        loss_test /= num
        print('test loss in %.6f' % loss_test)


        optimizer_weight = torch.optim.Adam(optimer_weight.parameters(), lr=weight_lr1, weight_decay=0)
        for epoch in range(2):
            batch.set_train(True)
            num = 0
            loss = 0
            for _, user_review_lists, _, item_review_lists, _, overalls in tqdm(batch):

                optimer_weight.zero_grad()
                loss_func_weight = nn.MSELoss()                                    


                user_review_lists = user_review_lists.cuda()
                item_review_lists = item_review_lists.cuda()
                overalls = overalls.cuda()

                for p in log_bert.parameters():
                    p.requires_grad = False

                for p in optimer_weight.parameters():
                    p.requires_grad = True

                pre = log_bert(user_review_lists, item_review_lists, batch_size)
                pre = torch.unsqueeze(pre, -1)
                pre = optimer_weight(pre)

                loss2 = loss_func_weight(pre, overalls)
                loss2.backward()
                optimizer_weight.step()

                loss += loss2
                num +=1
            loss /= num
            print(f'{epoch} is %.6f' % loss)


        batch.set_train(False)
        step = 0
        loss_test = 0
        num = 0
        for _, user_review_lists, _, item_review_lists, _, overalls in tqdm(batch):
            loss2 = _test_model(user_review_lists,item_review_lists,overalls, batch_size,log_bert,optimer_weight)
            loss_test += loss2
            num += 1

        loss_test /=num
        print('test loss in %.6f' % loss_test)
        '''



'''
for data_name in data_list:

    vocab_dict_path = '/tmp/pycharm_project_410/glove.6B.300d.txt'
    file_path = '/tmp/pycharm_project_410/' + data_name + '.json'
    # f'./{data_name}.json'
    glove_data = '/tmp/pycharm_project_410/data' + data_name+ '_.glove_data.pkl'
    glove_matrix = '/tmp/pycharm_project_410/data' + data_name +'_glove_matrix.pkl'

    glove_data, matrix, review_len = word_embed_dataloader.word_to_id(glove_data, glove_matrix, vocab_dict_path, file_path)
    train_data, test_data, user_dict, item_dict, u_max, i_max, num_users, num_items = word_embed_dataloader.prepare_data(
        glove_data)
    batch = word_embed_dataloader.Batch(train_data, test_data, user_dict, item_dict, u_max, i_max, batch_size, review_len,
                             train=True)

    #batch = word_embed_dataloader.Batch(train_data, test_data,  user_dict, item_dict,u_max, i_max, train=True)

    log_bert = Log_Bert(matrix,u_max,i_max,num_users, num_items)
    log_bert = log_bert.cuda()
    optimer_weight = Optimer_Weight()
    optimer_weight = optimer_weight.cuda()
    optimizer_log = torch.optim.Adam(log_bert.parameters(), lr=log_lr, weight_decay=0)
    optimizer_weight = torch.optim.Adam(optimer_weight.parameters(), lr=weight_lr, weight_decay=0)
    loss_func_weight = nn.MSELoss()
    loss_func_log = nn.CrossEntropyLoss()

    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for epoch in range(10):
        batch.set_train(True)
        num = 0
        loss = 0
        #for _, user_review_lists, mask_user, item_review_lists,mask_item, overalls,user_list,item_list in tqdm(batch):
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, \
            overalls, user_id, item_id, rating_ratio, mean, var in tqdm(batch):

            optimizer_log.zero_grad()
            optimer_weight.zero_grad()

            user_review_lists = user_review_lists.cuda()
            item_review_lists = item_review_lists.cuda()
            overalls = overalls.cuda()

            user_id = user_id.cuda()
            item_id = item_id.cuda()


            for p in log_bert.parameters():
                p.requires_grad = True

            for p in optimer_weight.parameters():
                p.requires_grad = False

            pre = log_bert(user_review_lists, item_review_lists, batch_size,user_id,item_id)

            loss1 = loss_func_log(pre, (overalls - 1).long())
            loss1.backward()
            optimizer_log.step()

            for p in log_bert.parameters():
                p.requires_grad = False

            for p in optimer_weight.parameters():
                p.requires_grad = True

            pre = log_bert(user_review_lists, item_review_lists, batch_size,user_id,item_id)
            pre = torch.unsqueeze(pre, -1)
            pre = optimer_weight(pre)

            loss2 = loss_func_weight(pre, overalls)
            loss2.backward()
            optimizer_weight.step()

            del user_review_lists, item_review_lists,overalls

        batch.set_train(False)
        step = 0
        loss1_test = 0
        loss2_test = 0
        loss3_test = 0
        loss4_test = 0
        loss5_test = 0
        loss_test =  0
        #for _, user_review_lists, _, item_review_lists, _, overalls,user_list,item_list in tqdm(batch):
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, \
            overalls, user_id, item_id, rating_ratio, mean, var in tqdm(batch):
            loss6,loss1,loss2,loss3,loss4,loss5= _test_model(user_review_lists, item_review_lists, overalls, batch_size, log_bert, optimer_weight,user_id,item_id)

            loss1_test += loss1
            loss2_test += loss2
            loss3_test += loss3
            loss4_test += loss4
            loss5_test += loss5

            loss_test += loss6
            step += 1
            del user_review_lists, item_review_lists, overalls

        loss1_test /= step
        loss2_test /= step
        loss3_test /= step
        loss4_test /= step
        loss5_test /= step

        loss_test /= step

        print('test loss in %.6f' % loss_test)
        print('test 1loss:%.6f, 2loss:%.6f, 3loss:%.6f, 4loss:%.6f, 5loss:%.6f' \
              % (loss1_test, loss2_test, loss3_test, loss4_test, loss5_test))
        '''