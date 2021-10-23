import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from fastNLP import Vocabulary
from fastNLP.io.embed_loader import EmbedLoader
import torch
import pickle
import os
import random
from collections import Counter

def load_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    raw_data = []
    data = []
    for line in lines:
        raw_data.append(json.loads(line))

    user_id = []
    item_id = []
    for d in raw_data:
        data.append({'reviewerID': d['reviewerID'],
                     'asin': d['asin'],
                     'reviewText': d['reviewText'],
                     'overall': d['overall'],
                     "unixReviewTime": d["unixReviewTime"]})
        user_id.append(d['reviewerID'])
        item_id.append(d['asin'])

    user_id = list(set(user_id))
    item_id = list(set(item_id))

    for d in data:

        d['reviewerID'] = user_id.index(d['reviewerID'])
        d['asin'] = item_id.index(d['asin'])

    return data,len(user_id),len(item_id)


def feature_word(data_name):

    data,user_num,item_num, = load_json(data_name)
    corpus = []
    for d in data:
        corpus.append(d['reviewText'])

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=0.01, max_df=0.8)
    vectorizer.fit_transform(corpus)
    feature_words = vectorizer.get_feature_names()
    # weight = vectorizer.idf_
    for i in range(len(feature_words)):  
        try:
            int(feature_words[i])
        except:
            feature_words = feature_words[i:]
            # weight = weight[i:]
            break

    feature_words = list(set(feature_words))

    review_len = []
    for i in range(len(data)):
        data[i]['reviewText'] = data[i]['reviewText'].split(' ')
        data[i]['reviewText'] = [j for j in data[i]['reviewText'] if j in feature_words]
        review_len.append(len(data[i]['reviewText']))
    review_len = np.array(review_len)
    review_len = np.percentile(review_len, 80)
    review_len = int(review_len)

    # feature_words_dict = sorted(feature_words_dict.items(), key=lambda x: x[1])
    for i in range(len(data)):
        if (len(data[i]['reviewText'])) > review_len:
            data[i]['reviewText'] = data[i]['reviewText'][:review_len]
        else:
            data[i]['reviewText'] = data[i]['reviewText'] + (review_len - len(data[i]['reviewText'])) * ['PAD']

    return data, feature_words,user_num,item_num

def word_to_id(glove_data,glove_matrix,vocab_dict_path,file_path):

    if os.path.exists(glove_data) == False or os.path.exists(glove_matrix) == False:
        data, feature_words ,user_num,item_num, = feature_word(file_path)
        vocab = Vocabulary(max_size=len(feature_words) + 1, unknown='unk', padding='PAD')
        vocab.add_word_lst(feature_words)
        vocab.build_vocab()
        matrix = EmbedLoader.load_with_vocab(vocab_dict_path, vocab)
        matrix = torch.tensor(matrix)

        for d in range(len(data)):
            review = []
            for word in data[d]['reviewText']:
                review.append(vocab.to_index(word))
            data[d]['reviewText'] = review

        with open(glove_data, 'wb') as f:
            pickle.dump(data, f)

        with open(glove_matrix, 'wb') as f:
            pickle.dump(matrix, f)

    with open(glove_data, 'rb') as f:
        glove_data = pickle.load(f)
    with open(glove_matrix, 'rb') as f:
        matrix = pickle.load(f)

    return glove_data, matrix, len(glove_data[0]['reviewText'])

def build_user_item_dict(data_list):
    user_dict = {}
    item_dict = {}
    data = [d['reviewText'] for d in data_list], [d['asin'] for d in data_list], \
           [d['reviewerID'] for d in data_list], [d['overall'] for d in data_list]
    dd, ii, uu, ss = data   # encode_vector, item_id, user_id, score
    i_max = 0
    u_max = 0

    #umax_list = []
    for iss in range(len(dd)):
        d, i, u, s = dd[iss], ii[iss], uu[iss], ss[iss]
        if u not in user_dict:
            user_dict[u] = [d],[i], [s]
            u_max = max(u_max, 1)
            continue

        user_dict[u][0].append(d)
        user_dict[u][1].append(i)
        user_dict[u][2].append(s)
        #umax_list.append(len(user_dict[u][1]))
        u_max = max(u_max, len(user_dict[u][1]))

    '''
    umax_list = np.array(umax_list)
    u_max = np.percentile(umax_list, 80)
    u_max = int(u_max)
    '''

    #imax_list = []
    for iss in range(len(dd)):
        d, i, u, s = dd[iss], ii[iss], uu[iss], ss[iss]
        if i not in item_dict:
            item_dict[i] = [d], [u] ,
            i_max = max(i_max, 1)
            continue

        item_dict[i][0].append(d)
        item_dict[i][1].append(u)
        #imax_list.append(len(item_dict[i][1]))
        i_max = max(i_max, len(item_dict[i][1])) 

    '''
    num_user = 0
    for i in user_dict:
        user_dict[i][2].append(num_user)
        num_user +=1

    num_item = 0
    for i in item_dict:
        item_dict[i][2].append(num_item)
        num_item += 1
    '''

    '''
    imax_list = np.array(imax_list)
    i_max = np.percentile(imax_list,80)
    i_max = int(i_max)
    '''
    return user_dict, item_dict, u_max, i_max


def prepare_data(data,):

    '''
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data.sort(key=lambda x: int(x['unixReviewTime']))
    with open(matrix_path, 'rb') as f:
        matrix = pickle.load(f)
    '''
    # user_dict, item_dict, u_max, i_max = build_user_item_dict(data)
    user_list = []
    item_list = []
    for d in data:
        user_list.append(d['reviewerID'])
        item_list.append(d['asin'])

    user_list = list(set(user_list))
    item_list = list(set(item_list))

    user_num =len(user_list)
    item_num = len(item_list)

    splits = int(len(data) * 0.8)

    train_data = []
    test_data = []
    for i in data:
        if i['reviewerID'] in user_list or i['asin'] in item_list:
            train_data.append(i)
            try:
                user_list.remove(i['reviewerID'])
            except:
                item_list.remove(i['asin'])

            try:
                item_list.remove(i['asin'])
            except:
                continue

            if len(user_list) == 0 and len(item_list) == 0:
                break

        else:
            test_data.append(i)

    '''
    for i in data:
        if len(train_data)<splits and (i not in train_data):
            train_data.append(i)
    '''

    data = train_data + test_data
    train_data = data[:splits]
    test_data = data[splits:]

    user_dict, item_dict, u_max, i_max, = build_user_item_dict(data)

    '''
    start_time = time.time()
    train_user_dict, train_item_dict, t_u_max, t_i_max,= build_user_item_dict(train_data)
    end_time = time.time()
    print(f'successfully get train user dict and item dict in time {end_time - start_time}')
    start_time = time.time()
    test_user_dict, test_item_dict, te_u_max, te_i_max, = build_user_item_dict(test_data)
    end_time = time.time()
    print(f'successfully get test user dict and item dict in time {end_time - start_time}')
    return (train_data, train_user_dict, train_item_dict), (test_data, test_user_dict, test_item_dict), \
           max(t_u_max, te_u_max), max(t_i_max, te_i_max),len(test_user_dict),len(test_item_dict)
    '''

    return train_data,test_data,user_dict,item_dict,u_max,i_max , user_num,item_num


def get_x(user_dict, item_dict, user_id, item_id,PADDING_LENGTH,train):

    #if user_id in user_dict:
    user_review_list, item_list,result = user_dict[user_id]

   
    a = torch.tensor(result)
    var = torch.var(a)
    mean = torch.mean(a)

    rating_percent = [10,10,10,10,10] #[1,1,1,1,1] #[15,15,15,15,15]
    result1 = Counter(result)
    for i in result1:
        a = int(i)
        #rating_percent[a - 1] = len(result) / (result1[i])
        rating_percent[a - 1] = (result1[i])/len(result)

    user_review_list = torch.tensor(user_review_list)
    #if train == True :
    mask_user = [1 if i != item_id else 0 for i in item_list]
    user_review_list = user_review_list * (torch.tensor(mask_user)).unsqueeze(-1).long()
    '''
        else:
            mask_user = [1]*len(item_list)
            user_review_list = user_review_list * (torch.tensor(mask_user)).unsqueeze(-1).long()
       
    
    else:
        mask_user = [0]
        user_review_list = torch.zeros(1, PADDING_LENGTH).long()
    '''
    #if item_id in item_dict:
    item_review_list, user_list= item_dict[item_id]
    item_review_list = torch.tensor(item_review_list)

    #if train == True:
    mask_item = [1 if i != user_id else 0 for i in user_list]
    item_review_list = item_review_list * (torch.tensor(mask_item)).unsqueeze(-1).long()
    '''
        else:
            mask_item = [1] * len(user_list)
            item_review_list = item_review_list * (torch.tensor(mask_item)).unsqueeze(-1).long()
    else:
        mask_item = [0]
        item_review_list = torch.zeros(1, PADDING_LENGTH).long()
    '''
    return user_review_list, mask_user, item_review_list, mask_item, user_id, item_id,rating_percent,var,mean


class Batch:

    def __init__(self, train_data, test_data,user_dict,item_dict, u_max, i_max, batch_size, pad_length,train=True):
        self.train_data = train_data
        self.test_data = test_data
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.train = train
        self.batch_size = batch_size
        self.u_max = u_max
        self.i_max = i_max
        self.pad_length = pad_length
        self.init_iter()


    def fetch_one(self, batch_size=None,):
        if self.idx >= (len(self.train_data) if self.train else len(self.test_data)):
            return None
        if batch_size is None:
            batch_size = self.batch_size
        data = [d['reviewText'] for d in (self.train_data if self.train else self.test_data)], \
               [d['asin'] for d in (self.train_data if self.train else self.test_data)], \
               [d['reviewerID'] for d in (self.train_data if self.train else self.test_data)], \
               [d['overall'] for d in (self.train_data if self.train else self.test_data)]
        vector, item, user, score = data
        encode_vectors = []
        user_review_lists = []
        mask_users = []
        item_review_lists = []
        mask_items = []
        overalls = []
        user_id_list = []
        item_id_list = []
        rating_ratio = []
        mean_list = []
        var_list = []

        u_max = self.u_max
        i_max = self.i_max

        for j in range(batch_size):
            if self.idx + j >= len(item):
                break
            encode_vector, item_id, user_id, overall,= \
                vector[self.idx_list[self.idx + j]], item[self.idx_list[self.idx + j]],user[self.idx_list[self.idx + j]], \
                score[self.idx_list[self.idx + j]]
            encode_vector = torch.tensor(encode_vector)
            encode_vector = encode_vector.unsqueeze(0)  # [1, 768]

            user_review_list, mask_user, item_review_list, mask_item,u_id,i_id,rating_pecent,var,mean = get_x(
                self.user_dict ,self.item_dict,user_id, item_id,self.pad_length,self.train)

            user_review_list = user_review_list[:u_max]
            #mask_user = mask_user[:u_max]
            item_review_list = item_review_list [:i_max]
            #mask_item = mask_item[:i_max]

            user_review = torch.zeros(1, u_max, user_review_list.size(1))
            mask_user = [0] * u_max


            for i in range(user_review_list.size(0)):
                idx = min(int(i * (u_max-1) / user_review_list.size(0)), u_max)
                user_review[0, idx] = user_review_list[i]
                mask_user[idx] = 1

            #pad_user_review = torch.zeros(u_max - user_review_list.size(0), user_review_list.size(1))
            #mask_user = mask_user + [0] * (u_max - len(mask_user))
            #user_review_list = torch.cat((user_review_list, pad_user_review), dim=0)  # [u_max, 768]
            user_review_list = user_review
            mask_user = torch.tensor(mask_user).float().unsqueeze(0)  # [1, u_max]
            mask_user_list = (user_review_list != 0).float()
            mask_user = mask_user.expand(self.pad_length, -1).permute(1, 0) * mask_user_list.squeeze(0)
            mask_user = mask_user.unsqueeze(0)

            item_review = torch.zeros(1, i_max, item_review_list.size(1))
            mask_item = [0] * i_max
            for i in range(item_review_list.size(0)):
                idx = min(int(i * (i_max - 1) / item_review_list.size(0)), i_max)
                item_review[0, idx] = item_review_list[i]
                mask_item[idx] = 1
            # pad_item_review = torch.zeros(i_max - item_review_list.size(0), item_review_list.size(1))
            # mask_item = mask_item + [0] * (i_max - len(mask_item))
            # item_review_list = torch.cat((item_review_list, pad_item_review), dim=0)  # [i_max, 768]
            item_review_list = item_review
            mask_item = torch.tensor(mask_item).float().unsqueeze(0)  # [1, i_max]
            mask_item_list = (item_review_list != 0).float()
            mask_item = mask_item.expand(self.pad_length, -1).permute(1, 0) * mask_item_list.squeeze(0)
            mask_item = mask_item.unsqueeze(0)

            encode_vectors.append(encode_vector)
            user_review_lists.append(user_review_list.unsqueeze(0))
            mask_users.append(mask_user)
            item_review_lists.append(item_review_list.unsqueeze(0))
            mask_items.append(mask_item)
            overalls.append(overall)
            user_id_list.append(u_id)
            item_id_list.append(i_id)
            rating_ratio.append(rating_pecent)
            mean_list.append(mean)
            var_list.append(var)

        encode_vectors = torch.cat(encode_vectors, dim=0)
        user_review_lists = torch.cat(user_review_lists, dim=0).permute(0, 1, 3, 2)
        mask_users = torch.cat(mask_users, dim=0)  # [batch_size, u_max]
        item_review_lists = torch.cat(item_review_lists, dim=0).permute(0, 1, 3, 2)
        mask_items = torch.cat(mask_items, dim=0)  # [batch_size, i_max]
        overalls = torch.tensor(overalls).float()
        rating_ratio = torch.tensor(rating_ratio)
        user_id_list = torch.tensor(user_id_list).unsqueeze(1)
        item_id_list = torch.tensor(item_id_list).unsqueeze(1)
        mean_list = torch.tensor(mean_list)
        var_list = torch.tensor(var_list)

        self.idx += batch_size

        user_review_lists = user_review_lists.squeeze(1).long()
        item_review_lists = item_review_lists.squeeze(1).long()

        # user_review_lists: [batch, PADDING_LENGTH, u_max]
        # item_review_lists: [batch, PADDING_LENGTH, i_max]
        # mask_users: [batch, u_max, PADDING_LENGTH]
        # item_users: [batch, i_max, PADDING_LENGTH]

        return encode_vectors, user_review_lists, mask_users, item_review_lists,\
               mask_items, overalls, user_id_list,item_id_list,rating_ratio,mean_list,var_list

    def __iter__(self):
        def batch_iter():
            self.init_iter()
            while 1:
                res = self.fetch_one()
                if res is None:
                    break
                yield res
        return batch_iter()

    def __len__(self):
        data_length = len(self.train_data) if self.train else \
            len(self.test_data)
        length = data_length // self.batch_size + 1 if data_length % self.batch_size != 0 else 0
        return length

    def init_iter(self):
        self.idx = 0 #if self.train else len(self.train_data)
        self.idx_list = [i for i in
                         range(len(self.train_data) if self.train else
                               len(self.test_data) )]
        random.shuffle(self.idx_list)
        self.idx_list = [0] * self.idx + self.idx_list
        self.idx_list = [i + self.idx for i in self.idx_list]

    def set_train(self, flag=True):
        self.train = flag
        self.init_iter()
