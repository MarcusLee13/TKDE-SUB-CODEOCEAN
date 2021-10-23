import os
import json
import pickle
import time
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
#from batchify11 import Batch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np



def load_json(file_path):

    raw_data = []
    data = []
    count = 1
    count2 = 1
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            #count += 1
            line = json.loads(line)
            raw_data.append(line)

    def take_time(elem):
        return int(elem['unixReviewTime'])
    raw_data.sort(key=take_time)

    #print(type(raw_data)) # raw_data 是一个dict组成的list

    for d in raw_data:
        if 'reviewText' not in d.keys():   # 验错
            #count2 += 1
            continue

        data.append({'reviewerID': d['reviewerID'],
                           'asin': d['asin'],
                           'reviewText': d['reviewText'],
                           'overall': d['overall'],
                     "unixReviewTime": d["unixReviewTime"]})
    #print(count2)
    #print(count)

    print('Finish loading the data')
    return data


def get_encode_result(data, bert_dir, batch_size=128, use_cuda=True):
    bert = BertModel.from_pretrained(bert_dir)
    print('successfully load pre-trained bert')
    for p in bert.parameters():
        p.requires_grad = False
    if torch.cuda.is_available() and use_cuda:
        bert = bert.cuda()
        print('transfer bert model to gpu')

    encode_vector = torch.zeros(len(data), 768)
    item_id = []
    user_id = []
    score = []

    for i in tqdm(range(len(data) // batch_size + 1)):
        if i * batch_size >= len(data):
            break
        d = data[i * batch_size: min(len(data), (i + 1) * batch_size)]
        segment_ids = []
        mask = []
        for dd in d:
            segment_ids.append(dd['reviewText'])
            mask.append(dd['masks'])
        segment_ids = torch.tensor(segment_ids)
        mask = torch.tensor(mask)
        if torch.cuda.is_available() and use_cuda:
            segment_ids = segment_ids.cuda()
            mask = mask.cuda()

        encode_result, _ = bert(segment_ids, attention_mask=mask, output_all_encoded_layers=False)
        del segment_ids, mask
        encode_result = encode_result[:, 0]
        encode_result = encode_result.cpu()

        encode_vector[i * batch_size: min(len(data), (i + 1) * batch_size)] = encode_result
        for j in range(encode_result.size(0)):
            item_id.append(d[j]['asin'])
            user_id.append(d[j]['reviewerID'])
            score.append(d[j]['overall'])

    result = encode_vector, item_id, user_id, score

    item_list = list(set(item_id))
    user_list = list(set(user_id))

    for i in range(len(result[1])):
        result[1][i] = item_list.index(result[1][i])
        result[2][i] = user_list.index(result[2][i])

    print('successfully get bert encode result')

    return result


def build_user_item_dict(encode_result_dict):
    user_dict = {}
    item_dict = {}
    dd, ii, uu, ss = encode_result_dict   # encode_vector, item_id, user_id, score
    i_max = 0
    u_max = 0

    umax_list = []
    imax_list = []

    for iss in range(len(ii)):
        d, i, u, s = dd[iss], ii[iss], uu[iss], ss[iss]
        if u not in user_dict:
            user_dict[u] = [d], [i]
            u_max = max(u_max, 1)
            continue

        user_dict[u][0].append(d)
        user_dict[u][1].append(i)
        umax_list.append(len(user_dict[u][1]))
        u_max = max(u_max, len(user_dict[u][1]))


    for iss in range(len(ii)):
        d, i, u, s = dd[iss], ii[iss], uu[iss], ss[iss]
        if i not in item_dict:
            item_dict[i] = [d], [u]
            i_max = max(i_max, 1)
            continue

        item_dict[i][0].append(d)
        item_dict[i][1].append(u)
        imax_list.append(len(item_dict[i][1]))
        i_max = max(i_max,len(item_dict[i][1]))

    '''
    for u in user_dict:
        print(len(user_dict[u][1]))
    print(len(user_dict))
    '''

    '''
    umax_list = np.array(umax_list)
    u_max = np.percentile(umax_list, 90)
    u_max = int(u_max)

    imax_list = np.array(imax_list)
    i_max = np.percentile(imax_list, 90)
    i_max = int(i_max)
    '''

    '''
    del_user_list = []
    for u in user_dict:
        d,i = user_dict[u]
        if len(i) < 6:
            del_user_list.append(u)

    user_dict1={}
    for i in user_dict:
        if i not in del_user_list:
            user_dict1[i] = user_dict[i]
    user_dict=user_dict1


    del_item_list = []
    for u in item_dict:
        d, i = item_dict[u]
        if len(i) < 6:
            del_item_list.append(u)

    item_dict1 = {}
    for i in item_dict:
        if i not in del_item_list:
            item_dict1[i] = item_dict[i]
    item_dict = item_dict1
    '''


    return user_dict, item_dict, u_max, i_max ,len(user_dict) ,len(item_dict)


def prepare_data(data_path, bert_dir, save_encode_result_pickle_path, max_seq_len=512,):
    if save_encode_result_pickle_path.endswith('.pkl'):
        save_encode_result_pickle_path = save_encode_result_pickle_path[:-4]
    if os.path.exists(f'{save_encode_result_pickle_path}.pkl'):
        print('pickle file has been created, jump the prepare data function')
        with open(f'{save_encode_result_pickle_path}.pkl', 'rb') as f:
            data_set = pickle.load(f)

    else:
        data = load_json(data_path)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_dir, 'vocab.txt'))
        print('successfully load bert tokenizer')

        for d in tqdm(data):
            if isinstance(d['reviewText'], str):
                d['reviewText'] = tokenizer.tokenize(d['reviewText'])
            while len(d['reviewText']) > max_seq_len - 2:
                d['reviewText'] = d['reviewText'][:max_seq_len - 2]
                '''
                ins = {}
                for k in d.keys():
                    ins[k] = d[k]
                ins['reviewText'] = d['reviewText'][:max_seq_len - 2]
                data.append(ins)
                d['reviewText'] = d['reviewText'][max_seq_len - 2:]
                '''
            d['reviewText'] = ['[CLS]'] + d['reviewText'] + ['[SEP]']
            d['masks'] = [1] * len(d['reviewText']) + [0] * (max_seq_len - len(d['reviewText']))
            d['reviewText'] = d['reviewText'] + (max_seq_len - len(d['reviewText'])) * ['[PAD]']
            d['reviewText'] = tokenizer.convert_tokens_to_ids(d['reviewText'])

        def take_time(elem):
            return int(elem['unixReviewTime'])

        data.sort(key=take_time)

        datas = get_encode_result(data, bert_dir, batch_size=16)
        with open(f'{save_encode_result_pickle_path}.pkl', 'wb') as f:
            pickle.dump(datas, f)
            print(
                f'successfully save encode test result to pickle file \'{save_encode_result_pickle_path}_test.pkl\'')
        with open(f'{save_encode_result_pickle_path}.pkl', 'rb') as f:
            data_set = pickle.load(f)

    splits = int(len(data_set[0]) * 0.8)

    #train_data = data_set[0][:splits], data_set[1][:splits], data_set[2][:splits], data_set[3][:splits] #iid,uid,rating
    #test_data = data_set[0][:splits],data_set[1][splits:], data_set[2][splits:], data_set[3][splits:]

    user_list = list(set(data_set[2]))
    item_list = list(set(data_set[1]))

    train_data = []
    test_data = []
    for i in range(len(data_set[1])):
        a = data_set[0][i],data_set[1][i],data_set[2][i],data_set[3][i]
        if data_set[1][i] in item_list or data_set[2][i] in user_list:
            train_data.append(a)
            try:
                user_list.remove(data_set[2][i])
            except:
                item_list.remove(data_set[1][i])
            try:
                item_list.remove(data_set[1][i])
            except:
                continue

            if len(user_list) ==0 and len(item_list) ==0:
                break
        else:
            test_data.append(a)

    data = train_data + test_data
    data = [i[0] for i in data], [i[1] for i in data],[i[2] for i in data], [i[3] for i in data]
    train_data = data[0][:splits], data[1][:splits],data[2][:splits],data[3][:splits]
    test_data = data[0][splits:], data[1][splits:],data[2][splits:],data[3][splits:]
    del data


    start_time = time.time()
    user_dict, item_dict, u_max, i_max ,user_num,item_num= build_user_item_dict(data_set)
    end_time = time.time()
    print(f'successfully get user dict and item dict in time {end_time - start_time}')


    return train_data, test_data, user_dict, item_dict,\
          u_max,i_max,user_num,item_num




if __name__ == '__main__':

    data_list =["Musical_Instruments_5"]

    '''
    ["Musical_Instruments_5","Cell_Phones_and_Accessories_5", "Clothing_Shoes_and_Jewelry_5",
    "Amazon_Instant_Video_5","Baby_5","Beauty_5",Grocery_and_Gourmet_Food_5,'Books_5'
    "Home_and__Music_5","ElectroKitchen_5","Musical_Instruments_5","Clothing_Shoes_and_Jewelry_5",'Automotive_5'.
     "Digitalnics_5","Health_and_Personal_Care_5",'Movies_and_TV_5',
     "Office_Products_5","Patio_Lawn_and_Garden_5","Pet_Supplies_5","Home_and_Kitchen_5",
     "Sports_and_Outdoors_5","Toys_and_Games_5","Video_Games_5" ,"Apps_for_Android_5"]

    #data_list = {'Grocery_and_Gourmet_Food_5':13,'Tools_and_Home_Improvement_5':4}
  
    data_list = {"Automotive_5":3,"Clothing_Shoes_and_Jewelry_5":4,"Home_and_Kitchen_5":10,
                 "Cell_Phones_and_Accessories_5":10,"Amazon_Instant_Video_5":13,"Baby_5":12,"Beauty_5":7,
     "Musical_Instruments_5":3,"Digital_Music_5":8,"Health_and_Personal_Care_5":9,"Office_Products_5":9,
     "Patio_Lawn_and_Garden_5":5,"Pet_Supplies_5":9,"Sports_and_Outdoors_5":7,"Video_Games_5":14,"Apps_for_Android_5":89,
    "Kindle_Store_5":6,'Automotive_5':3}
    '''

    for data_name in data_list:

        #try:
        save_path_base = data_name
        train_data, test_data, u_max, i_max,user_num,item_num  = \
            prepare_data(data_path='/home/wuyuexin/文档/bert_cnn_prune/data/' + data_name + '.json',
                         bert_dir='/home/wuyuexin/文档/bert_cnn_prune/coding/bert-unbase',
                         save_encode_result_pickle_path=f'./{save_path_base}_encode_result.pkl', )
        print(f'{data_name} finished!')
        # except RuntimeError:
        # print(f'{data_name} failed!')
        print(u_max)
        print(i_max)

        batch = Batch(train_data, test_data, u_max, i_max, batch_size=32, train=True)

        #i_max = data_list[data_name]
        # step = 0
        # 超参数
        batch_size = 32
        learning_rate = 4e-4
        mask_rating = 0.1
        mask_iter = 2
        num_epoch = 1  # 预训练两轮
        data_fname = '/home/wuyuexin/文档/bert_cnn_prune/data/' + data_name + '.json'
        #mean = logg.cal_mean(data_fname)


        '''
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls in tqdm(batch): #预训练模型
            # the training code is here
            del encode_vectors, mask_users, mask_items
            deepconn_bert = prune.Deepconn_Bert(u_max,i_max)
            deepconn_bert = deepconn_bert.cuda()
            del user_review_lists, item_review_lists, overalls
        '''
        k_size = 3
        #k_size1 = 2

        batch.set_train(True)
        log_bert = logg.Log_Bert(u_max,k_size)
        log_bert =log_bert.cuda()

        for epoch in range(3):
            num = 0
            loss = 0
            for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls in tqdm(batch):
                # the training code is here
                del encode_vectors, mask_users, mask_items
                for p in log_bert.parameters():
                    p.requires_grad = True

                optimizer = torch.optim.Adam(log_bert.parameters(), lr=learning_rate, weight_decay=0.01)
                optimizer.zero_grad()

                #loss_func = nn.MSELoss()
                loss_func = nn.CrossEntropyLoss()
                # Train the model
                user_review_lists = user_review_lists.cuda()
                item_review_lists = item_review_lists.cuda()
                overalls = overalls.cuda()
                # overalls = overalls.long()

                pre_z = log_bert(user_review_lists, item_review_lists, batch_size)

                loss1 = loss_func(pre_z, (overalls - 1).long())
                # Backward and optimize
                loss1.backward()
                optimizer.step()
                del user_review_lists, item_review_lists, overalls
                loss += loss1
                num += 1
            loss /= num

            print("epoch %d : train CE = %.6f" % (epoch + 1, loss))


        file_path = '/home/wuyuexin/文档/bert_cnn_prune/data/coding' + data_name + 'model.pth'
        log_bert.load_state_dict(torch.load('/home/wuyuexin/文档/bert_cnn_prune/data/coding' + data_name + 'model.pth'),
                        strict=True)
        loss_test = 0
        loss_test2 = 0
        step = 0
        up = 0
        down =0
        batch.set_train(True)
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls in batch:
            del encode_vectors, mask_users, mask_items
            loss1,loss2,up1,down1 = logg.test_model(user_review_lists, item_review_lists, overalls, log_bert, batch_size)
            loss_test+=loss1
            loss_test2 += loss2
            step+=1
            up += up1
            down +=down1
            del user_review_lists, item_review_lists, overalls


        print("%s test MSE: %.6f/%.6f" % (data_name, loss_test / step, loss_test2 / step))
        print('TRAIN up: %d down: %d '% (up,down))

        torch.save(log_bert.state_dict(),'/home/wuyuexin/文档/bert_cnn_prune/data/coding' + data_name +'model.pth')
        '''
        loss_test = 0
        loss_test2 = 0
        step = 0
        up = 0
        down = 0
        batch.set_train(True)
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls in batch:
            del encode_vectors, mask_users, mask_items
            loss1, loss2, up1, down1 = logg.test_model(user_review_lists, item_review_lists, overalls, log_bert,
                                                       batch_size)
            loss_test += loss1
            loss_test2 += loss2
            step += 1
            up += up1
            down += down1
            del user_review_lists, item_review_lists, overalls

        print("%s train MSE: %.6f/%.6f" % (data_name, loss_test / step, loss_test2 / step))
        print('TRAIN up: %d down: %d ' % (up, down))

        loss_test = 0
        loss_test2 = 0
        step = 0
        up = 0
        down = 0
        batch.set_train(False)
        for encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls in batch:
            del encode_vectors, mask_users, mask_items
            loss1, loss2, up1, down1 = logg.test_model(user_review_lists, item_review_lists, overalls, log_bert,
                                                       batch_size)
            loss_test += loss1
            loss_test2 += loss2
            step += 1
            up += up1
            down += down1
            del user_review_lists, item_review_lists, overalls

        print("%s test MSE: %.6f/%.6f" % (data_name, loss_test / step, loss_test2 / step))
        print('TRAIN up: %d down: %d ' % (up, down))

    #except:
       #continue
        '''


