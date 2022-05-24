'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader_handle_inference_icl import BasicDataset
from time import time
from model_CI_LightGCN import PairWiseModel
import random
import os
import multiprocessing
        
class BPRLoss:
    def __init__(self, 
                 recmodel : PairWiseModel, 
                 config : dict):
        self.model = recmodel
        self.weight_decay = world.lgcn_weight_dency
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # print('####################',reg_loss,reg_loss.size())
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()


# def UniformSample_ST( dataset):#follow LGCN tf的正常版本，可以跑通
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # total_start = time()
    # dataset : BasicDataset
    # users=np.random.randint(0,dataset.n_user,dataset.trainDataSize)
    # random.shuffle(users)#shuffle后的当前stage trainuser
    # hisPos = dataset.hisPos#所有历史交互
    # nowPos = dataset.nowPos#所有当前交互
    # S = []
    # for i, user in enumerate(users):
    #     start = time()
    #     posForUser = nowPos[user]
    #     if len(posForUser) == 0:
    #         continue
    #     posindex = np.random.randint(0, len(posForUser))
    #     positem = posForUser[posindex]
    #     while True:
    #         negitem = np.random.randint(0, dataset.m_items)
    #         if negitem in hisPos[user]:
    #             continue
    #         else:
    #             break
    #     S.append([user, positem, negitem])
    # total = time() - total_start
    # return np.array(S), total

# def UniformSample_FT( dataset):#follow LGCN tf的正常版本，可以跑通
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # total_start = time()
    # dataset : BasicDataset
    # users=np.random.randint(0,dataset.n_user,dataset.trainDataSize)
    # random.shuffle(users)#shuffle后的当前stage trainuser
    # hisPos = dataset.hisPos#所有历史交互
    # nowPos = dataset.nowPos#所有当前交互
    # S = []
    # for i, user in enumerate(users):
    #     start = time()
    #     posForUser = nowPos[user]
    #     if len(posForUser) == 0:
    #         continue
    #     posindex = np.random.randint(0, len(posForUser))
    #     positem = posForUser[posindex]
    #     while True:
    #         negitem = np.random.randint(0, dataset.m_items)
    #         if negitem in hisPos[user]:
    #             continue
    #         else:
    #             break
    #     S.append([user, positem, negitem])
    # total = time() - total_start
    # return np.array(S), total


def UniformSample_handle( dataset, sample_mode):
    total_start = time()
    dataset : BasicDataset
    hisPos = dataset.hisPos
    nowPos = dataset.nowPos
    S = []
    if sample_mode=='all':
        users=np.random.randint(0,dataset.n_user,dataset.trainDataSize)
        random.shuffle(users)
        for i, user in enumerate(users):
            posForUser = nowPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in hisPos[user]:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        total = time() - total_start

    elif sample_mode=='new':
        users_condidate=dataset.trainUser_unique_only
        for i in range(dataset.trainDataSize):
            user = random.sample(users_condidate, 1)[0]
            posForUser = nowPos[int(user)]
            if len(posForUser) == 0:
                raise  ValueError(f'Sample function wrong ,there is a trainUser ,{int(user)} have no pos item')
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = random.sample(dataset.trainItem_unique_only, 1)[0]
                if negitem in hisPos[int(user)]:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        total = time() - total_start
    else:
        raise TypeError(f'no this mode {sample_mode} is not in [all ,new]')
    return np.array(S), total


def set_seed(seed):
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName(stage):
    if world.model_name == 'fullretrain_lgcn':
        file = f"fullretrain_lgcn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'finetune_lgcn':
        file = f"finetune_lgcn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'static_lgcn':
        file = f"static_lgcn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'sml':
        file = f"sml-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'sml_x':
        file = f"sml_x-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'Transfer_LGCN':
        file = f"Transfer_LGCN-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'metah':
        file = f"metah-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'metahs':
        file = f"metahs-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
    elif world.model_name == 'our_model':
        file = f"our_model-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-{world.exptag}.pth.tar"
    elif world.model_name == 'laipi_model':
        file = f"laipi_model-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-{world.exptag}.pth.tar"
    elif world.model_name == 'handle_lgcn':
        file = f"handle_lgcn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-{world.exptag}.pth.tar"
    elif world.model_name == 'CILightGCN':
        file = f"CILightGCN-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy.pth.tar"
        return os.path.join(f'../start_from_zero/{world.dataset}/',file)
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

# ====================Metrics==============================
# =========================================================
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def Recall_onepos_999neg(ratinglist,groundTrue,k):#k=[5,10,20]
    Recall=[]
    Precision=[]
    for tk in k:
        hit_sum=0
        for inter in range(len(groundTrue)):
            if groundTrue[inter] in ratinglist[inter][0:tk]:
                hit_sum+=1
        Recall_k=float(hit_sum/len(groundTrue))
        Precision_k=float(hit_sum/len(groundTrue)/tk)

        Recall.append(Recall_k)
        Precision.append(Precision_k)
    return (Recall,Precision)


def NDCG_onepos_999neg(ratinglist,groundTrue,k):
    NDCG=[]
    for tk in k:
        NDCG_sum=0.
        for inter in range(len(groundTrue)):
            for r,iid in enumerate(ratinglist[inter][0:tk]):
                if iid==groundTrue[inter]:
                    NDCG_oneinter=(1./(np.log2(r+1.+1.)))
                    break
                else:
                    NDCG_oneinter=0
            NDCG_sum+=NDCG_oneinter
        NDCG_k=float(NDCG_sum/len(groundTrue))

        NDCG.append(NDCG_k)
    return (NDCG)
    
def topk(matrix, K, axis=1):
    column_index = np.arange(matrix.shape[1 - axis])[:, None]
    topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
    topk_data = matrix[column_index, topk_index]
    topk_index_sort = np.argsort(-topk_data, axis=axis)
    topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return  topk_index_sort

# ====================end Metrics=============================
# =========================================================