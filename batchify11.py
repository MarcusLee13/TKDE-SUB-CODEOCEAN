
import torch
import random

def get_x(user_dict, item_dict, user_id, item_id):

    '''
    if user_id in user_dict:
        user_review_list, item_list = user_dict[user_id]
        mask_user = [1 if i != item_id else 0 for i in item_list]
        user_review_list = (torch.cat(user_review_list, dim=0)).view(len(item_list), -1) * \
                           (torch.tensor(mask_user)).unsqueeze(-1).float()
    else:
        mask_user = [0]
        user_review_list = torch.zeros(1, 768)

    if item_id in item_dict:
        item_review_list, user_list = item_dict[item_id]
        mask_item = [1 if i != user_id else 0 for i in user_list]
        item_review_list = (torch.cat(item_review_list, dim=0)).view(len(user_list), -1) * \
                           (torch.tensor(mask_item)).unsqueeze(-1).float()
    else:
        mask_item = [0]
        item_review_list = torch.zeros(1, 768)

    return user_review_list, mask_user, item_review_list, mask_item
    '''

    user_review_list, item_list = user_dict[user_id]
    mask_user = [1 if i != item_id else 0 for i in item_list]
    user_review_list = (torch.cat(user_review_list, dim=0)).view(len(item_list), -1) * \
                       (torch.tensor(mask_user)).unsqueeze(-1).float()

    item_review_list, user_list = item_dict[item_id]
    mask_item = [1 if i != user_id else 0 for i in user_list]
    item_review_list = (torch.cat(item_review_list, dim=0)).view(len(user_list), -1) * \
                       (torch.tensor(mask_item)).unsqueeze(-1).float()

    return user_review_list, mask_user, item_review_list, mask_item



class Batch:

    def __init__(self, train_data, test_data,user_dict,item_dict, u_max,i_max,batch_size, train=True):
        self.train_data = train_data
        self.test_data = test_data
        self.train = train
        self.batch_size = batch_size
        self.u_max = u_max
        self.i_max = i_max
        self.user_dict = user_dict
        self.item_dict = item_dict

        self.idx = 0


    def fetch_one(self, batch_size=None):
        if self.idx >= (len(self.train_data[1]) if self.train else len(self.test_data[1])):
            return None
        if batch_size is None:
            batch_size = self.batch_size

        vector, item, user, score = self.train_data if self.train else self.test_data
        encode_vectors = []
        user_review_lists = []
        mask_users = []
        item_review_lists = []
        mask_items = []
        overalls = []
        item_list =[]
        user_list = []

        u_max = self.u_max
        i_max = self.i_max

        for j in range(batch_size):
            if self.idx + j >= len(item):
                break
            encode_vector, item_id, user_id, overall = \
                vector[self.idx_list[self.idx + j]], item[self.idx_list[self.idx + j]], \
                user[self.idx_list[self.idx + j]], score[self.idx_list[self.idx + j]]
            encode_vector = encode_vector.unsqueeze(0)  # [1, 768]

            user_review_list, mask_user, item_review_list, mask_item = get_x(
                self.user_dict, self.item_dict, user_id, item_id)

            '''
            user_review_list = user_review_list[:u_max]
            mask_user = mask_user[:u_max]
            item_review_list = item_review_list [:i_max]
            mask_item = mask_item[:i_max]

            user_review = torch.zeros(u_max, user_review_list.size(1))
            mask_user = [0] * u_max
            for i in range(user_review_list.size(0)):
                idx = min(int(i * (u_max-1) / user_review_list.size(0)), u_max)
                user_review[idx] = user_review_list[i]
                mask_user[idx] = 1
            '''
            pad_user_review = torch.zeros(u_max - user_review_list.size(0), user_review_list.size(1))
            mask_user = mask_user + [0] * (u_max - len(mask_user))
            mask_user = torch.tensor(mask_user).float().unsqueeze(0)
            user_review_list = torch.cat((user_review_list, pad_user_review), dim=0)  # [u_max, 768]

            #user_review_list = user_review
            mask_user = torch.tensor(mask_user).float().unsqueeze(0)  # [1, u_max]


            '''
            item_review = torch.zeros(i_max, item_review_list.size(1))
            mask_item = [0] * i_max
            for i in range(item_review_list.size(0)):
                idx = min(int(i * (i_max-1) / item_review_list.size(0)), i_max)
                item_review[idx] = item_review_list[i]
                mask_item[idx] = 1
            '''
            pad_item_review = torch.zeros(i_max - item_review_list.size(0), item_review_list.size(1))
            mask_item = mask_item + [0] * (i_max - len(mask_item))
            mask_item = torch.tensor(mask_item).float().unsqueeze(0)
            item_review_list = torch.cat((item_review_list, pad_item_review), dim=0)  # [i_max, 768]

            #item_review_list = item_review
            mask_item = torch.tensor(mask_item).float().unsqueeze(0)  # [1, i_max]

            encode_vectors.append(encode_vector)
            user_review_lists.append(user_review_list.unsqueeze(0))
            mask_users.append(mask_user)
            item_review_lists.append(item_review_list.unsqueeze(0))
            mask_items.append(mask_item)
            overalls.append(overall)
            user_list.append(user_id)
            item_list.append(item_id)

        encode_vectors = torch.cat(encode_vectors, dim=0)  # [batch_size, 768]
        user_review_lists = torch.cat(user_review_lists, dim=0).permute(0, 2, 1)  # [batch_size, 768, u_max]
        mask_users = torch.cat(mask_users, dim=0)  # [batch_size, u_max]
        item_review_lists = torch.cat(item_review_lists, dim=0).permute(0, 2, 1)  # [batch_size, 768, i_max]
        mask_items = torch.cat(mask_items, dim=0)  # [batch_size, i_max]
        overalls = torch.tensor(overalls).float()
        user_list = torch.tensor(user_list)
        item_list = torch.tensor(item_list)

        self.idx += batch_size

        return encode_vectors, user_review_lists, mask_users, item_review_lists, mask_items, overalls,user_list,item_list,

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
        data_length = len(self.train_data[1]) if self.train else len(self.test_data[1])
        length = data_length // self.batch_size + 1 if data_length % self.batch_size != 0 else 0
        return length

    def init_iter(self):
        self.idx = 0 #if self.train else len(self.train_data[1])
        self.idx_list = [i for i in range(len(self.train_data[1]) if self.train else \
        len(self.test_data[1]))]
        random.shuffle(self.idx_list)
        self.idx_list = [0] * self.idx + self.idx_list
        self.idx_list = [i + self.idx for i in self.idx_list]

    def set_train(self, flag=True):
        self.train = flag
        self.init_iter()



