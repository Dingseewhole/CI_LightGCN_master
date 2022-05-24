import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
import time
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph_only(self):
        """
        because user(item) never interc with user(item)
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        with no self loop
        A = 
            |0,   R|
            |R^T, 0|
        """
        raise NotImplementedError

    def getSparseGraph_all(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    def GraphSize(self):
        """
        to show how much data use for build Adj
        """
        raise NotImplementedError

class Loader_hat(BasicDataset):
    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.m_item
    @property
    def trainDataSize(self):
        return self._traindataSize
    @property
    def testDataSize(self):
        return self._testdataSize
    @property
    def testDict(self):
        return self.__testDict
    @property
    def onlyGraphSize(self):
        return self._onlyGraphSize
    @property
    def allGraphSize(self):
        return self._allGraphSize
    def __init__(self,stage,config = world.config,path="../data/finetune_yelp"):#stage=30
        print(f'===========================load data start , at stage {stage}===================================')
        dataset_starttime=time.time()
        information=np.load(path+'/information.npy')
        self.n_user = information[1]
        self.m_item = information[2]
        self.path = path
        self.datasetStage=stage
        self._traindataSize = 0
        self._testdataSize = 0
        self.nowPos =[]
        self.hisPos =[]
        self.Graph_only = None
        self.Graph_all = None
        self.Graph_handle = None
        self.Rscale = None
        self.Graph_pure = None
        self.now_user_degree = None
        self.now_item_degree = None
        self.elder_user_degree = None
        self.elder_item_degree = None
        self.old_user_degree = None
        self.old_item_degree = None
        self.active_user_now = []
        self.active_item_now = []
        self.trained_user = []
        self.trained_item = []

        #=============================build incremental graph==============================
        onlytime=time.time()
        trainAlldata_only , trainItem_only, trainUser_only=[], [], []
        for s in range(stage-1,stage):
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_only.append(np.load(train_file))
        trainAlldata_only=np.concatenate(trainAlldata_only, axis=0)
        trainUserlist_only=list(set(trainAlldata_only[:,0]))
        trainItemlist_only=list(set(trainAlldata_only[:,1]))
        
        trainDict_only={}
        for ui in range(len(trainUserlist_only)):
            trainDict_only[int(trainUserlist_only[ui])]=[]
        for inter in range(trainAlldata_only.shape[0]):
            try:
                trainDict_only[int(trainAlldata_only[inter,0])].append(int(trainAlldata_only[inter,1]))
            except:
                trainDict_only[int(trainAlldata_only[inter,0])]=[int(trainAlldata_only[inter,1])]
        
        for k in trainDict_only.keys():
            trainUser_only.extend([k] * len(trainDict_only[k]))
            trainItem_only.extend(trainDict_only[k])

        self.active_user_now = list(set(trainUser_only))
        self.active_item_now = list(set(trainItem_only))
        self.trainUser_unique_only=trainUserlist_only
        self.trainItem_unique_only=trainItemlist_only
        self.trainUser_only = np.array(trainUser_only)
        self.trainItem_only = np.array(trainItem_only)
        self._onlyGraphSize=len(self.trainUser_only)
        self.UserItemNet_only = csr_matrix((np.ones(len(self.trainUser_only)), (self.trainUser_only, self.trainItem_only)),shape=(self.n_user, self.m_item))
        del trainUserlist_only,trainUser_only,trainItem_only,trainDict_only, trainAlldata_only#在生成A矩阵后就删除全量结果
        # print(f'Loader_hat only graph at stage {self.datasetStage-1} reday size: {self._onlyGraphSize} ,used time {time.time()-onlytime}')

        
        trainAlldata_all , trainItem_all, trainUser_all=[], [], []
        alltime=time.time()
        for s in range(0,stage):
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_all.append(np.load(train_file))
        trainAlldata_all=np.concatenate(trainAlldata_all, axis=0)
        trainUserlist_all=list(set(trainAlldata_all[:,0]))
        trainDict_all={}
        for ui in range(len(trainUserlist_all)):
            trainDict_all[int(trainUserlist_all[ui])]=[]
        for inter in range(trainAlldata_all.shape[0]):
            try:
                trainDict_all[int(trainAlldata_all[inter,0])].append(int(trainAlldata_all[inter,1]))
            except:
                trainDict_all[int(trainAlldata_all[inter,0])]=[int(trainAlldata_all[inter,1])]
        for k in trainDict_all.keys():
            trainUser_all.extend([k] * len(trainDict_all[k]))
            trainItem_all.extend(trainDict_all[k])

        self.trained_user = list(set(trainUser_all))
        self.trained_item = list(set(trainItem_all))
        self.trainUser_all = np.array(trainUser_all)
        self.trainItem_all = np.array(trainItem_all)
        self._allGraphSize=len(self.trainUser_all)
        self.UserItemNet_all = csr_matrix((np.ones(len(self.trainUser_all)), (self.trainUser_all, self.trainItem_all)),shape=(self.n_user, self.m_item))
        del trainUserlist_all, trainUser_all,trainItem_all, trainDict_all, trainAlldata_all#在生成A矩阵后就删除全量结果
        # print(f'Loader_hat all graph reday at stage: 0-{self.datasetStage-1} size: {self._allGraphSize} ,used time {time.time()-alltime}')

        posItems_his, posItems_now = [], []
        for user in list(range(self.n_user)):
            posItems_his.append(self.UserItemNet_all[user].nonzero()[1])
            posItems_now.append(self.UserItemNet_only[user].nonzero()[1])
            self._traindataSize+=len(posItems_now[user])
        self.hisPos = posItems_his
        self.nowPos = posItems_now

        #=============================build testing data==============================
        testUniqueUsers, testdata_all= [], []
        for s in range(stage,stage+1):
            test_file = path + '/test/'+str(s)+'.npy'
            testdata_all.append(np.load(test_file))
        testdata_all=np.concatenate(testdata_all, axis=0)
        test_item_dict={}
        test_user=list(set(testdata_all[:,0]))
        for ut in range(len(test_user)):
            test_item_dict[int(test_user[ut])]=[]
        for inter in range(testdata_all.shape[0]):
            try:
                test_item_dict[int(testdata_all[inter,0])][int(testdata_all[inter,1])]=[int(x) for x in testdata_all[inter,1:]]
            except:
                key_temp=int(testdata_all[inter,1])
                test_item_dict[int(testdata_all[inter,0])]={int(key_temp):[int(x) for x in testdata_all[inter,1:]]}
        for key in test_item_dict.keys():
            testUniqueUsers.append(key)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self._testdataSize=testdata_all.shape[0]
        self.__testDict = test_item_dict
        # print(f"Loader_hat at stage : {self.datasetStage} , {self._traindataSize} interactions for train , {self._testdataSize} interactions for test")
        print(f'===========================load data is ready ,used time : {time.time()-dataset_starttime} s ===================================')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_only(self):
        adj_starttime=time.time()
        if self.Graph_only is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                # print("successfully loaded only graph...")
                norm_adj = pre_adj_mat
            except :
                print('load incremental graph faulse at: '+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved norm_mat at "+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_only = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_only = self.Graph_only.coalesce().to(world.device)
            print(f"incremental adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_only

    def getSparseGraph_all(self):
        adj_starttime=time.time()
        if self.Graph_all is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                # print("successfully loaded all graph...")
                norm_adj = pre_adj_mat
            except :
                print('load full graph faulse at: '+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz  '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved full graph norm_mat at "+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_all = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_all = self.Graph_all.coalesce().to(world.device)
            print(f"full graph adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_all

    def getSparseGraph_handle(self):
        adj_starttime=time.time()
        if self.Graph_handle or self.Rscale is None:
            try:
                pre_adj_mat_all = sp.load_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                # print("successfully loaded handle graph ")
                rscale_vec = np.load(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-.npy')
                # print("successfully loaded handle rescale")
                norm_adj = pre_adj_mat_all
            except :
                print('load incremental graph faulse at: '+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
                s = time.time()
                adj_mat_all = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_all = adj_mat_all.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat_all[:self.n_users, self.n_users:] = R
                adj_mat_all[self.n_users:, :self.n_users] = R.T
                adj_mat_all = adj_mat_all.todok()

                adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only = adj_mat_only.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat_only[:self.n_users, self.n_users:] = R
                adj_mat_only[self.n_users:, :self.n_users] = R.T
                adj_mat_only = adj_mat_only.todok()

                rowsum_all = np.array(adj_mat_all.sum(axis=1))
                rowsum_only = np.array(adj_mat_only.sum(axis=1))

                degree_new=rowsum_only
                degree_old=rowsum_all-rowsum_only
                degree_all=rowsum_all
                d_inv = np.power(degree_all, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat_only)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                old_scale=np.power(degree_old, 0.5)
                new_scale=np.power(degree_all, 0.5)
                rscale_vec=np.true_divide(old_scale , new_scale)# n*1 array
                np.save(self.path + '/fenzi-'+'-', old_scale)
                np.save(self.path + '/fenmu'+'-', new_scale)
                rscale_vec[np.isinf(rscale_vec)] = 0.
                rscale_vec[np.isnan(rscale_vec)] = 0.
                end = time.time()
                print(f"costing {end-s}s, saved incremental graph norm_mat at "+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
                np.save(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-', rscale_vec)

            if sum(np.isinf(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isinf(rscale_vec))}')
            elif sum(np.isnan(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isnan(rscale_vec))}')
            self.Graph_handle = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_handle = self.Graph_handle.coalesce().to(world.device)
            self.Rscale=torch.from_numpy(rscale_vec).to(world.device)
            print(f"incremental adjacency matrix & Rescale vector done use time :{time.time()-adj_starttime}")
        return self.Graph_handle, self.Rscale

    def getSparseGraph_pure(self):
        adj_starttime=time.time()
        if self.Graph_pure is None:
            try:
                pre_adj_mat_pure = sp.load_npz(self.path + '/pure_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                # print("successfully loaded pure graph ")
                norm_adj = pre_adj_mat_pure
            except :
                print('load pure_graph faulse at: '+self.path + '/pure_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
                s = time.time()

                #生成adj_only
                adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only = adj_mat_only.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat_only[:self.n_users, self.n_users:] = R
                adj_mat_only[self.n_users:, :self.n_users] = R.T
                adj_mat_only = adj_mat_only.todok()

                norm_adj = adj_mat_only.tocsr()

                print(f"costing {time.time()-adj_starttime}s, saved pure graph norm_mat at "+self.path + '/pure_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/pure_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)

            norm_adj = norm_adj.todok()
            norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_pure = norm_adj.coalesce().to(world.device)
            # print(f"pure adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_pure

    def get_degree(self):
        if self.now_user_degree==None or self.now_item_degree==None or self.elder_user_degree==None or self.elder_item_degree==None or self.old_user_degree==None or self.old_item_degree==None:
            try:
                now_item_degree=np.load(self.path + '/item_degree_only-'+str(self.datasetStage-1)+'-.npy')
                all_item_degree=np.load(self.path + '/item_degree_all-'+str(self.datasetStage-1)+'-.npy')
                now_user_degree=np.load(self.path + '/user_degree_only-'+str(self.datasetStage-1)+'-.npy')
                all_user_degree=np.load(self.path + '/user_degree_all-'+str(self.datasetStage-1)+'-.npy')
                elder_item_degree=np.load(self.path + '/item_degree_only-'+str(self.datasetStage-2)+'-.npy')
                elder_user_degree=np.load(self.path + '/user_degree_only-'+str(self.datasetStage-2)+'-.npy')
                self.now_user_degree = torch.from_numpy(now_user_degree).type(torch.FloatTensor).to(world.device)
                self.now_item_degree = torch.from_numpy(now_item_degree).type(torch.FloatTensor).to(world.device)
                self.elder_user_degree = torch.from_numpy(elder_user_degree).type(torch.FloatTensor).to(world.device)
                self.elder_item_degree = torch.from_numpy(elder_item_degree).type(torch.FloatTensor).to(world.device)
                self.old_user_degree = torch.from_numpy(all_user_degree-now_user_degree).type(torch.FloatTensor).to(world.device)
                self.old_item_degree = torch.from_numpy(all_item_degree-now_item_degree).type(torch.FloatTensor).to(world.device)
                print('============================load degree success=================================')
            except:
                print('============================load degree faulse=================================')
                trainAlldata_only_elder , trainItem_only_elder, trainUser_only_elder=[], [], []
                for s in range(self.datasetStage-2,self.datasetStage-1):
                    train_file_elder = self.path + '/train/'+str(s)+'.npy'
                    trainAlldata_only_elder.append(np.load(train_file_elder))
                trainAlldata_only_elder=np.concatenate(trainAlldata_only_elder, axis=0)
                trainUserlist_only_elder=list(set(trainAlldata_only_elder[:,0]))
                trainItemlist_only_elder=list(set(trainAlldata_only_elder[:,1]))
                
                trainDict_only_elder={}
                for ui in range(len(trainUserlist_only_elder)):
                    trainDict_only_elder[int(trainUserlist_only_elder[ui])]=[]
                for inter in range(trainAlldata_only_elder.shape[0]):
                    if trainDict_only_elder.get(int(trainAlldata_only_elder[inter,0])):
                        trainDict_only_elder[int(trainAlldata_only_elder[inter,0])].append(int(trainAlldata_only_elder[inter,1]))
                    else:
                        trainDict_only_elder[int(trainAlldata_only_elder[inter,0])]=[int(trainAlldata_only_elder[inter,1])]
                for k in trainDict_only_elder.keys():
                    trainUser_only_elder.extend([k] * len(trainDict_only_elder[k]))
                    trainItem_only_elder.extend(trainDict_only_elder[k])
                trainUser_only_elder = np.array(trainUser_only_elder)
                trainItem_only_elder = np.array(trainItem_only_elder)
                UserItemNet_only_elder = csr_matrix((np.ones(len(trainUser_only_elder)), (trainUser_only_elder, trainItem_only_elder)),shape=(self.n_user, self.m_item))
                del trainUserlist_only_elder,trainUser_only_elder,trainItem_only_elder,trainDict_only_elder, trainAlldata_only_elder#在生成A矩阵后就删除全量结果

                now_item_degree : numpy.matrix
                now_item_ = self.UserItemNet_only.tolil()
                elder_item = UserItemNet_only_elder.tolil()
                all_item_ = self.UserItemNet_all.tolil()
                now_item_degree = np.array(now_item_.sum(axis=0))[0]
                elder_item_degree = np.array(elder_item.sum(axis=0))[0]
                all_item_degree = np.array(all_item_.sum(axis=0))[0]
                now_item_degree = now_item_degree.reshape(-1,1)
                elder_item_degree = elder_item_degree.reshape(-1,1)
                all_item_degree = all_item_degree.reshape(-1,1)
                np.save(self.path + '/item_degree_only-'+str(self.datasetStage-1)+'-', now_item_degree)
                np.save(self.path + '/item_degree_only-'+str(self.datasetStage-2)+'-', elder_item_degree)
                np.save(self.path + '/item_degree_all-'+str(self.datasetStage-1)+'-', all_item_degree)

                now_user_degree : numpy.matrix
                now_user_ = self.UserItemNet_only.tolil()
                elder_user_ = UserItemNet_only_elder.tolil()
                all_user_ = self.UserItemNet_all.tolil()
                now_user_degree = np.array(now_user_.sum(axis=1))
                elder_user_degree = np.array(elder_user_.sum(axis=1))
                all_user_degree = np.array(all_user_.sum(axis=1))
                now_user_degree = now_user_degree.reshape(-1,1)
                elder_user_degree = elder_user_degree.reshape(-1,1)
                all_user_degree = all_user_degree.reshape(-1,1)
                np.save(self.path + '/user_degree_only-'+str(self.datasetStage-1)+'-', now_user_degree)
                np.save(self.path + '/user_degree_only-'+str(self.datasetStage-2)+'-', elder_user_degree)
                np.save(self.path + '/user_degree_all-'+str(self.datasetStage-1)+'-', all_user_degree)

                self.now_user_degree = torch.from_numpy(now_user_degree).type(torch.FloatTensor).to(world.device)
                self.now_item_degree = torch.from_numpy(now_item_degree).type(torch.FloatTensor).to(world.device)
                self.elder_user_degree = torch.from_numpy(elder_user_degree).type(torch.FloatTensor).to(world.device)
                self.elder_item_degree = torch.from_numpy(elder_item_degree).type(torch.FloatTensor).to(world.device)
                self.old_user_degree = torch.from_numpy(all_user_degree-now_user_degree).type(torch.FloatTensor).to(world.device)
                self.old_item_degree = torch.from_numpy(all_item_degree-now_item_degree).type(torch.FloatTensor).to(world.device)

            if (now_user_degree.shape[0] != self.n_user) or (all_user_degree.shape[0] != self.n_user) or (elder_user_degree.shape[0] != self.n_user):
                raise ValueError(f'{now_user_degree.shape[1]}!={self.n_user}')
            if (now_item_degree.shape[0] != self.m_item) or (all_item_degree.shape[0] != self.m_item) or (elder_item_degree.shape[0] != self.m_item):
                raise ValueError(f'{now_item_degree.shape[1]}!={self.n_user}')

        return self.now_user_degree, self.now_item_degree, self.elder_user_degree, self.elder_item_degree, self.old_user_degree, self.old_item_degree

# class Loader_future(BasicDataset)
#     @property
#     def n_users(self):
#         return self.n_user
#     @property
#     def m_items(self):
#         return self.m_item
#     @property
#     def trainDataSize(self):
#         return self._traindataSize
#     @property
#     def testDataSize(self):
#         return self._testdataSize
#     @property
#     def testDict(self):
#         return self.__testDict
#     @property
#     def onlyGraphSize(self):
#         return self._onlyGraphSize
#     @property
#     def allGraphSize(self):
#         return self._allGraphSize
#     def __init__(self,stage,config = world.config,path="../data/finetune_yelp"):#stage=30
#         print(f'===========================load future start , at stage {stage}===================================')
#         dataset_starttime=time.time()
#         information=np.load(path+'/information.npy')
#         self.n_user = information[1]
#         self.m_item = information[2]
#         self.path = path
#         self.datasetStage=stage
#         self._traindataSize = 0
#         self._testdataSize = 0
#         self.nowPos =[]
#         self.hisPos =[]
#         self.Graph_only = None
#         self.Graph_all = None
#         self.Graph_handle = None
#         self.Rscale = None

#         #=============================制作one stage graph==============================
#         onlytime=time.time()
#         trainAlldata_only , trainItem_only, trainUser_only=[], [], []
#         for s in range(stage,stage+1):#sml版本赋值20，stage的结果会是19.npy的的数据用来生成graph
#             train_file = path + '/train/'+str(s)+'.npy'
#             trainAlldata_only.append(np.load(train_file))
#         trainAlldata_only=np.concatenate(trainAlldata_only, axis=0)
#         trainUserlist_only=list(set(trainAlldata_only[:,0]))#0到这一stage（stage-1.npy）的所有user数量
#         trainItemlist_only=list(set(trainAlldata_only[:,1]))#0到这一stage（stage-1.npy）的所有user数量
#         #制作train dict
#         trainDict_only={}
#         for ui in range(len(trainUserlist_only)):
#             trainDict_only[int(trainUserlist_only[ui])]=[]
#         for inter in range(trainAlldata_only.shape[0]):
#             if trainDict_only.get(int(trainAlldata_only[inter,0])):
#                 trainDict_only[int(trainAlldata_only[inter,0])].append(int(trainAlldata_only[inter,1]))
#             else:
#                 trainDict_only[int(trainAlldata_only[inter,0])]=[int(trainAlldata_only[inter,1])]
#         #制作trainUser_only、trainItem_only用来生成UserItemNet_only
#         for k in trainDict_only.keys():
#             trainUser_only.extend([k] * len(trainDict_only[k]))
#             trainItem_only.extend(trainDict_only[k])
#         self.trainUser_unique_only=trainUserlist_only
#         self.trainItem_unique_only=trainItemlist_only
#         self.trainUser_only = np.array(trainUser_only)
#         self.trainItem_only = np.array(trainItem_only)
#         self._onlyGraphSize=len(self.trainUser_only)
#         self.UserItemNet_only = csr_matrix((np.ones(len(self.trainUser_only)), (self.trainUser_only, self.trainItem_only)),shape=(self.n_user, self.m_item))
#         del trainUserlist_only,trainUser_only,trainItem_only,trainDict_only, trainAlldata_only#在生成A矩阵后就删除全量结果
#         print(f'Loader_future only graph at stage {self.datasetStage} reday size: {self._onlyGraphSize} ,used time {time.time()-onlytime}')

#         #制作all stage graph
#         trainAlldata_all , trainItem_all, trainUser_all=[], [], []
#         alltime=time.time()
#         for s in range(0,stage+1):#pre版本赋值19，stage的结果会是0-18.npy的的数据用来生成graph
#             train_file = path + '/train/'+str(s)+'.npy'
#             trainAlldata_all.append(np.load(train_file))
#         trainAlldata_all=np.concatenate(trainAlldata_all, axis=0)
#         trainUserlist_all=list(set(trainAlldata_all[:,0]))#0到这一stage（stage-1.npy）的所有user数量
#         trainDict_all={}
#         for ui in range(len(trainUserlist_all)):
#             trainDict_all[int(trainUserlist_all[ui])]=[]
#         for inter in range(trainAlldata_all.shape[0]):
#             if trainDict_all.get(int(trainAlldata_all[inter,0])):
#                 trainDict_all[int(trainAlldata_all[inter,0])].append(int(trainAlldata_all[inter,1]))
#             else:
#                 trainDict_all[int(trainAlldata_all[inter,0])]=[int(trainAlldata_all[inter,1])]
#         for k in trainDict_all.keys():
#             trainUser_all.extend([k] * len(trainDict_all[k]))
#             trainItem_all.extend(trainDict_all[k])
#         self.trainUser_all = np.array(trainUser_all)
#         self.trainItem_all = np.array(trainItem_all)
#         self._allGraphSize=len(self.trainUser_all)
#         self.UserItemNet_all = csr_matrix((np.ones(len(self.trainUser_all)), (self.trainUser_all, self.trainItem_all)),shape=(self.n_user, self.m_item))
#         del trainUserlist_all, trainUser_all,trainItem_all, trainDict_all, trainAlldata_all#在生成A矩阵后就删除全量结果
#         print(f'Loader_future all graph reday at stage: 0-{self.datasetStage} size: {self._allGraphSize} ,used time {time.time()-alltime}')

#         #生成当前stage正样本train数据集hisPos,nowPos
#         posItems_his, posItems_now = [], []
#         for user in list(range(self.n_user)):
#             posItems_his.append(self.UserItemNet_all[user].nonzero()[1])
#             posItems_now.append(self.UserItemNet_only[user].nonzero()[1])
#             self._traindataSize+=len(posItems_now[user])
#         self.hisPos = posItems_his
#         self.nowPos = posItems_now


#         #制作testset
#         testUniqueUsers, testdata_all= [], []
#         for s in range(stage,stage+1):#s赋值19开始，stage的结果会是19.npy
#             test_file = path + '/test/'+str(s)+'.npy'
#             testdata_all.append(np.load(test_file))
#         testdata_all=np.concatenate(testdata_all, axis=0)
#         test_item_dict={}#user是第一层key正样本是第二层key，value是这条iter的正+负样本，是testDict
#         test_user=list(set(testdata_all[:,0]))
#         for ut in range(len(test_user)):#创建一个以test user id为key的字典
#             test_item_dict[int(test_user[ut])]=[]
#         for inter in range(testdata_all.shape[0]):#嵌套字典，第二层循环的key是test正样本value是所有交互数据
#             if test_item_dict.get(int(testdata_all[inter,0])):
#                 test_item_dict[int(testdata_all[inter,0])][int(testdata_all[inter,1])]=[int(x) for x in testdata_all[inter,1:]]
#             else:
#                 key_temp=int(testdata_all[inter,1])
#                 test_item_dict[int(testdata_all[inter,0])]={int(key_temp):[int(x) for x in testdata_all[inter,1:]]}
#         for key in test_item_dict.keys():
#             testUniqueUsers.append(key)
#         self.testUniqueUsers = np.array(testUniqueUsers)
#         self._testdataSize=testdata_all.shape[0]
#         self.__testDict = test_item_dict#双层嵌套字典
#         print(f"Loader_future at stage : {self.datasetStage} , {self._traindataSize} interactions for train , {self._testdataSize} interactions for test")
#         print(f'===========================load data is ready ,used time : {time.time()-dataset_starttime} s ===================================')

#     def _convert_sp_mat_to_sp_tensor(self, X):
#         coo = X.tocoo().astype(np.float32)
#         row = torch.Tensor(coo.row).long()
#         col = torch.Tensor(coo.col).long()
#         index = torch.stack([row, col])
#         data = torch.FloatTensor(coo.data)
#         return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

#     def getSparseGraph_only(self):
#         # print("try loading adjacency matrix")
#         adj_starttime=time.time()
#         if self.Graph_only is None:
#             try:
#                 pre_adj_mat = sp.load_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 print("successfully loaded only graph...")
#                 norm_adj = pre_adj_mat
#             except :
#                 print('load only graph faulse at: '+self.path + '/__ONLY__no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
#                 s = time.time()
#                 adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat = adj_mat.tolil()
#                 R = self.UserItemNet_only.tolil()
#                 adj_mat[:self.n_users, self.n_users:] = R
#                 adj_mat[self.n_users:, :self.n_users] = R.T
#                 adj_mat = adj_mat.todok()
#                 # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

#                 rowsum = np.array(adj_mat.sum(axis=1))
#                 d_inv = np.power(rowsum, -0.5).flatten()
#                 d_inv[np.isinf(d_inv)] = 0.
#                 d_mat = sp.diags(d_inv)

#                 norm_adj = d_mat.dot(adj_mat)
#                 norm_adj = norm_adj.dot(d_mat)
#                 norm_adj = norm_adj.tocsr()
#                 end = time.time()
#                 print(f"costing {end-s}s, saved norm_mat at "+self.path + '/__ONLY__no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 sp.save_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz', norm_adj)
#             self.Graph_only = self._convert_sp_mat_to_sp_tensor(norm_adj)
#             self.Graph_only = self.Graph_only.coalesce().to(world.device)
#             print(f"only adjacency matrix done use time :{time.time()-adj_starttime}")
#         return self.Graph_only

#     def getSparseGraph_all(self):
#         # print("try loading adjacency matrix")
#         adj_starttime=time.time()
#         if self.Graph_all is None:
#             try:
#                 pre_adj_mat = sp.load_npz(self.path + '/no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 print("successfully loaded all graph...")
#                 norm_adj = pre_adj_mat
#             except :
#                 print('load all graph faulse at: '+self.path + '/no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
#                 s = time.time()
#                 adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat = adj_mat.tolil()
#                 R = self.UserItemNet_all.tolil()
#                 adj_mat[:self.n_users, self.n_users:] = R
#                 adj_mat[self.n_users:, :self.n_users] = R.T
#                 adj_mat = adj_mat.todok()
#                 # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

#                 rowsum = np.array(adj_mat.sum(axis=1))
#                 d_inv = np.power(rowsum, -0.5).flatten()
#                 d_inv[np.isinf(d_inv)] = 0.
#                 d_mat = sp.diags(d_inv)

#                 norm_adj = d_mat.dot(adj_mat)
#                 norm_adj = norm_adj.dot(d_mat)
#                 norm_adj = norm_adj.tocsr()
#                 end = time.time()
#                 print(f"costing {end-s}s, saved all graph norm_mat at "+self.path + '/no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 sp.save_npz(self.path + '/no self loop'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz', norm_adj)
#             self.Graph_all = self._convert_sp_mat_to_sp_tensor(norm_adj)
#             self.Graph_all = self.Graph_all.coalesce().to(world.device)
#             print(f"all adjacency matrix done use time :{time.time()-adj_starttime}")
#         return self.Graph_all

#     def getSparseGraph_handle(self):
#         adj_starttime=time.time()
#         if self.Graph_handle or self.Rscale is None:
#             try:
#                 pre_adj_mat_all = sp.load_npz(self.path + '/handle_graph'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 rscale_vec = np.load(self.path + '/rscale_vec-at'+str(self.datasetStage)+'-.npy')
#                 print("successfully loaded handle graph and rescale")
#                 norm_adj = pre_adj_mat_all
#             except :
#                 print('load handle_graph faulse at: '+self.path + '/handle_graph'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
#                 s = time.time()
#                 adj_mat_all = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat_all = adj_mat_all.tolil()
#                 R = self.UserItemNet_all.tolil()
#                 adj_mat_all[:self.n_users, self.n_users:] = R
#                 adj_mat_all[self.n_users:, :self.n_users] = R.T
#                 adj_mat_all = adj_mat_all.todok()
#                 # adj_mat_all = adj_mat_all + sp.eye(adj_mat_all.shape[0])

#                 adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat_only = adj_mat_only.tolil()
#                 R = self.UserItemNet_only.tolil()
#                 adj_mat_only[:self.n_users, self.n_users:] = R
#                 adj_mat_only[self.n_users:, :self.n_users] = R.T
#                 adj_mat_only = adj_mat_only.todok()
#                 # adj_mat_only = adj_mat_only + sp.eye(adj_mat_only.shape[0])

#                 rowsum_all = np.array(adj_mat_all.sum(axis=1))
#                 rowsum_only = np.array(adj_mat_only.sum(axis=1))

#                 degree_new=rowsum_only
#                 degree_old=rowsum_all-rowsum_only
#                 degree_all=rowsum_all
#                 d_inv = np.power(degree_all, -0.5).flatten()#flatten直接变成一行了
#                 d_inv[np.isinf(d_inv)] = 0.
#                 d_mat = sp.diags(d_inv)

#                 norm_adj = d_mat.dot(adj_mat_only)
#                 norm_adj = norm_adj.dot(d_mat)
#                 norm_adj = norm_adj.tocsr()

#                 old_scale=np.power(degree_old, 0.5)
#                 # old_scale[np.isinf(old_scale)]=0.
#                 new_scale=np.power(degree_all, 0.5)
#                 # new_scale[np.isinf(degree_all)]=0.
#                 rscale_vec=np.true_divide(old_scale , new_scale)# n*1 array
#                 rscale_vec[np.isinf(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
#                 rscale_vec[np.isnan(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
#                 end = time.time()
#                 print(f"costing {end-s}s, saved handle graph norm_mat at "+self.path + '/handle_graph'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz')
#                 sp.save_npz(self.path + '/handle_graph'+str(self.datasetStage)+'.npy s_pre_adj_mat.npz', norm_adj)
#                 np.save(self.path + '/rscale_vec-at'+str(self.datasetStage)+'-', rscale_vec)

#             self.Graph_handle = self._convert_sp_mat_to_sp_tensor(norm_adj)
#             self.Graph_handle = self.Graph_handle.coalesce().to(world.device)
#             self.Rscale=torch.from_numpy(rscale_vec).to(world.device)
#             print(f"handle adjacency matrix & Rescale vector done use time :{time.time()-adj_starttime}")
#         return self.Graph_handle, self.Rscale

#     def get_future_degree(self):
#         adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#         adj_mat = adj_mat.tolil()
#         R = self.UserItemNet_only.tolil()
#         R = R.todok()
#         # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

#         rowsum = np.array(R.sum(axis=0))
#         # Degree_dict={k:v for k,v in zip(range(rowsum.shape[0]), rowsum)}
#         # return Degree_dict
#         return rowsum

class Loader_pre(BasicDataset):#stage=19 18 onan_graph,0-18 allgraph , 0-18train 0-18his 19test
    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.m_item
    @property
    def trainDataSize(self):
        return self._traindataSize
    @property
    def testDataSize(self):
        return self._testdataSize
    @property
    def testDict(self):
        return self.__testDict
    @property
    def onlyGraphSize(self):
        return self._onlyGraphSize
    @property
    def allGraphSize(self):
        return self._allGraphSize
    def __init__(self,stage,config = world.config,path="../data/finetune_yelp"):#stage=30
        print(f'===========================load pre start , at stage {stage}===================================')
        dataset_starttime=time.time()
        information=np.load(path+'/information.npy')
        self.n_user = information[1]
        self.m_item = information[2]
        self.path = path
        self.datasetStage=stage
        self._traindataSize = 0
        self._testdataSize = 0
        self.nowPos =[]
        self.hisPos =[]
        self.Graph_only = None
        self.Graph_all = None
        self.Graph_handle = None
        self.Rscale = None

        #=============================制作one stage graph==============================
        onlytime=time.time()
        trainAlldata_only , trainItem_only, trainUser_only=[], [], []
        for s in range(stage-1,stage):#sml版本赋值20，stage的结果会是19.npy的的数据用来生成graph
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_only.append(np.load(train_file))
        trainAlldata_only=np.concatenate(trainAlldata_only, axis=0)
        trainUserlist_only=list(set(trainAlldata_only[:,0]))#0到这一stage（stage-1.npy）的所有user数量
        trainItemlist_only=list(set(trainAlldata_only[:,1]))
        #制作train dict
        trainDict_only={}
        for ui in range(len(trainUserlist_only)):
            trainDict_only[int(trainUserlist_only[ui])]=[]
        for inter in range(trainAlldata_only.shape[0]):
            if trainDict_only.get(int(trainAlldata_only[inter,0])):
                trainDict_only[int(trainAlldata_only[inter,0])].append(int(trainAlldata_only[inter,1]))
            else:
                trainDict_only[int(trainAlldata_only[inter,0])]=[int(trainAlldata_only[inter,1])]
        #制作trainUser_only、trainItem_only用来生成UserItemNet_only
        for k in trainDict_only.keys():
            trainUser_only.extend([k] * len(trainDict_only[k]))
            trainItem_only.extend(trainDict_only[k])
        self.trainUser_unique_only=trainUserlist_only
        self.trainItem_unique_only=trainItemlist_only
        self.trainUser_only = np.array(trainUser_only)
        self.trainItem_only = np.array(trainItem_only)
        self._onlyGraphSize=len(self.trainUser_only)
        self.UserItemNet_only = csr_matrix((np.ones(len(self.trainUser_only)), (self.trainUser_only, self.trainItem_only)),shape=(self.n_user, self.m_item))
        del trainUserlist_only,trainUser_only,trainItem_only,trainDict_only, trainAlldata_only#在生成A矩阵后就删除全量结果
        # print(f'Loader_pre only graph at stage {self.datasetStage} reday size: {self._onlyGraphSize} ,used time {time.time()-onlytime}')

        #制作all stage graph
        trainAlldata_all , trainItem_all, trainUser_all=[], [], []
        alltime=time.time()
        for s in range(0,stage):#pre版本赋值19，stage的结果会是0-18.npy的的数据用来生成graph
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_all.append(np.load(train_file))
        trainAlldata_all=np.concatenate(trainAlldata_all, axis=0)
        trainUserlist_all=list(set(trainAlldata_all[:,0]))#0到这一stage（stage-1.npy）的所有user数量
        trainDict_all={}
        for ui in range(len(trainUserlist_all)):
            trainDict_all[int(trainUserlist_all[ui])]=[]
        for inter in range(trainAlldata_all.shape[0]):
            if trainDict_all.get(int(trainAlldata_all[inter,0])):
                trainDict_all[int(trainAlldata_all[inter,0])].append(int(trainAlldata_all[inter,1]))
            else:
                trainDict_all[int(trainAlldata_all[inter,0])]=[int(trainAlldata_all[inter,1])]
        for k in trainDict_all.keys():
            trainUser_all.extend([k] * len(trainDict_all[k]))
            trainItem_all.extend(trainDict_all[k])
        self.trainUser_all = np.array(trainUser_all)
        self.trainItem_all = np.array(trainItem_all)
        self._allGraphSize=len(self.trainUser_all)
        self.UserItemNet_all = csr_matrix((np.ones(len(self.trainUser_all)), (self.trainUser_all, self.trainItem_all)),shape=(self.n_user, self.m_item))
        del trainUserlist_all, trainUser_all,trainItem_all, trainDict_all, trainAlldata_all#在生成A矩阵后就删除全量结果
        print(f'Loader_pre all graph reday at stage: 0-{self.datasetStage} size: {self._allGraphSize} ,used time {time.time()-alltime}')

        #生成当前stage正样本train数据集hisPos,nowPos
        posItems_now = []
        for user in list(range(self.n_user)):
            posItems_now.append(self.UserItemNet_all[user].nonzero()[1])
            self._traindataSize+=len(posItems_now[user])
        self.hisPos = posItems_now
        self.nowPos = posItems_now

        #制作testset
        testUniqueUsers, testdata_all= [], []
        for s in range(stage,stage+1):#s赋值19开始，stage的结果会是19.npy
            test_file = path + '/test/'+str(s)+'.npy'
            testdata_all.append(np.load(test_file))
        testdata_all=np.concatenate(testdata_all, axis=0)
        test_item_dict={}#user是第一层key正样本是第二层key，value是这条iter的正+负样本，是testDict
        test_user=list(set(testdata_all[:,0]))
        for ut in range(len(test_user)):#创建一个以test user id为key的字典
            test_item_dict[int(test_user[ut])]=[]
        for inter in range(testdata_all.shape[0]):#嵌套字典，第二层循环的key是test正样本value是所有交互数据
            if test_item_dict.get(int(testdata_all[inter,0])):
                test_item_dict[int(testdata_all[inter,0])][int(testdata_all[inter,1])]=[int(x) for x in testdata_all[inter,1:]]
            else:
                key_temp=int(testdata_all[inter,1])
                test_item_dict[int(testdata_all[inter,0])]={int(key_temp):[int(x) for x in testdata_all[inter,1:]]}
        for key in test_item_dict.keys():
            testUniqueUsers.append(key)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self._testdataSize=testdata_all.shape[0]
        self.__testDict = test_item_dict#双层嵌套字典
        print(f"Loader_hat at stage : {self.datasetStage} , {self._traindataSize} interactions for train , {self._testdataSize} interactions for test")
        print(f'===========================load data is ready ,used time : {time.time()-dataset_starttime} s ===================================')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_only(self):
        # print("try loading adjacency matrix")
        adj_starttime=time.time()
        if self.Graph_only is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded only graph...")
                norm_adj = pre_adj_mat
            except :
                print('load only graph faulse at: '+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved norm_mat at "+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_only = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_only = self.Graph_only.coalesce().to(world.device)
            print(f"only adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_only

    def getSparseGraph_all(self):
        # print("try loading adjacency matrix")
        adj_starttime=time.time()
        if self.Graph_all is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded all graph...")
                norm_adj = pre_adj_mat
            except :
                print('load all graph faulse at: '+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved all graph norm_mat at "+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_all = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_all = self.Graph_all.coalesce().to(world.device)
            print(f"all adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_all

    def getSparseGraph_handle(self):
        adj_starttime=time.time()
        if self.Graph_handle or self.Rscale is None:
            try:
                pre_adj_mat_all = sp.load_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                rscale_vec = np.load(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-.npy')
                print("successfully loaded handle graph and rescale")
                norm_adj = pre_adj_mat_all
            except :
                print('load handle_graph faulse at: '+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
                s = time.time()
                adj_mat_all = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_all = adj_mat_all.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat_all[:self.n_users, self.n_users:] = R
                adj_mat_all[self.n_users:, :self.n_users] = R.T
                adj_mat_all = adj_mat_all.todok()
                # adj_mat_all = adj_mat_all + sp.eye(adj_mat_all.shape[0])

                adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only = adj_mat_only.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat_only[:self.n_users, self.n_users:] = R
                adj_mat_only[self.n_users:, :self.n_users] = R.T
                adj_mat_only = adj_mat_only.todok()
                # adj_mat_only = adj_mat_only + sp.eye(adj_mat_only.shape[0])

                rowsum_all = np.array(adj_mat_all.sum(axis=1))
                rowsum_only = np.array(adj_mat_only.sum(axis=1))

                degree_new=rowsum_only
                degree_old=rowsum_all-rowsum_only
                degree_all=rowsum_all
                d_inv = np.power(degree_all, -0.5).flatten()#flatten直接变成一行了
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat_only)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                old_scale=np.power(degree_old, 0.5)
                # old_scale[np.isinf(old_scale)]=0.
                new_scale=np.power(degree_all, 0.5)
                # new_scale[np.isinf(degree_all)]=0.
                rscale_vec=np.true_divide(old_scale , new_scale)# n*1 array
                rscale_vec[np.isinf(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                rscale_vec[np.isnan(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                end = time.time()
                print(f"costing {end-s}s, saved handle graph norm_mat at "+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
                np.save(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-', rscale_vec)

            self.Graph_handle = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_handle = self.Graph_handle.coalesce().to(world.device)
            self.Rscale=torch.from_numpy(rscale_vec).to(world.device)
            print(f"handle adjacency matrix & Rescale vector done use time :{time.time()-adj_starttime}")
        return self.Graph_handle, self.Rscale

class test_cold_start(BasicDataset):
    def __init__(self,stage,config = world.config,path="../data/finetune_yelp"):#stage=30
        trainAlldata_all , trainItem_all, trainUser_all=[], [], []
        self.datasetStage = stage
        self.trainUserlist_all=[]
        self.trainItemlist_all=[]
        self.path = path
        alltime=time.time()
        for s in range(0,world.FR_start):
            train_file = self.path + '/train/'+str(s)+'.npy'
            trainAlldata_all.append(np.load(train_file))
        trainAlldata_all=np.concatenate(trainAlldata_all, axis=0)
        self.trainUserlist_all=list(set(trainAlldata_all[:,0]))
        self.trainItemlist_all=list(set(trainAlldata_all[:,1]))
    def get_test_dict(self,ori_testDict):
        testUniqueUsers, testdata_all= [], []
        for s in range(self.datasetStage,self.datasetStage+1):
            test_file = self.path + '/test/'+str(s)+'.npy'
            testdata_all.append(np.load(test_file))
        testdata_all=np.concatenate(testdata_all, axis=0)
        test_dictOO={}
        test_dictON={}
        test_dictNO={}
        test_dictNN={}
        OO_sum=0
        ON_sum=0
        NO_sum=0
        NN_sum=0

        test_user=list(set(testdata_all[:,0]))
        test_itme=list(set(testdata_all[:,1]))
        test_user_O=[i for i in test_user if i in self.trainUserlist_all]
        test_user_N=[i for i in test_user if i not in self.trainUserlist_all]
        test_item_O=[i for i in test_itme if i in self.trainItemlist_all]
        if len(test_user_O)+len(test_user_N) != len(test_user):
            raise ValueError(f'{len(test_user_O)} + {len(test_user_N)} != {len(test_user)}')
        for userO in test_user_O:
            for posit in ori_testDict[int(userO)].keys():
                if posit in test_item_O:
                    if test_dictOO.get(int(userO)):
                        test_dictOO[int(userO)][posit] = ori_testDict[int(userO)][posit]
                    else:
                        test_dictOO[int(userO)] = {int(posit):ori_testDict[int(userO)][posit]}
                    OO_sum = OO_sum+1
                else :
                    if test_dictON.get(int(userO)):
                        test_dictON[int(userO)][posit] = ori_testDict[int(userO)][posit]
                    else:
                        test_dictON[int(userO)] = {int(posit):ori_testDict[int(userO)][posit]}
                    ON_sum = ON_sum+1

        for userN in test_user_N:
            for posit in ori_testDict[int(userN)].keys():
                if posit in test_item_O:
                    if test_dictNO.get(int(userN)):
                        test_dictNO[int(userN)][posit] = ori_testDict[int(userN)][posit]
                    else:
                        test_dictNO[int(userN)] = {int(posit):ori_testDict[int(userN)][posit]}
                    NO_sum = NO_sum+1
                else :
                    if test_dictNN.get(int(userN)):
                        test_dictNN[int(userN)][posit] = ori_testDict[int(userN)][posit]
                    else:
                        test_dictNN[int(userN)] = {int(posit):ori_testDict[int(userN)][posit]}
                    NN_sum = NN_sum+1

        return test_dictOO,test_dictON,test_dictNO,test_dictNN,OO_sum,ON_sum,NO_sum,NN_sum

# class Loader_hat_future_degree(BasicDataset):
    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.m_item
    @property
    def trainDataSize(self):
        return self._traindataSize
    @property
    def testDataSize(self):
        return self._testdataSize
    @property
    def testDict(self):
        return self.__testDict
    @property
    def onlyGraphSize(self):
        return self._onlyGraphSize
    @property
    def allGraphSize(self):
        return self._allGraphSize
    def __init__(self,stage,config = world.config,path="../data/finetune_yelp"):#stage=30
        print(f'===========================load data start , at stage {stage}===================================')
        dataset_starttime=time.time()
        information=np.load(path+'/information.npy')
        self.n_user = information[1]
        self.m_item = information[2]
        self.path = path
        self.datasetStage=stage
        self._traindataSize = 0
        self._testdataSize = 0
        self.nowPos =[]
        self.hisPos =[]
        self.Graph_only = None
        self.Graph_all = None
        self.Graph_handle = None
        self.Rscale = None
        self.Graph_future_degree = None
        self.Rscale_future_degree = None

        #=============================制作one stage graph==============================
        onlytime=time.time()
        trainAlldata_only , trainItem_only, trainUser_only=[], [], []
        for s in range(stage-1,stage):#sml版本赋值20，stage的结果会是19.npy的的数据用来生成graph
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_only.append(np.load(train_file))
        trainAlldata_only=np.concatenate(trainAlldata_only, axis=0)
        trainUserlist_only=list(set(trainAlldata_only[:,0]))#这一stage（stage-1.npy）的所有user数量
        trainItemlist_only=list(set(trainAlldata_only[:,1]))#这一stage（stage-1.npy）的所有item数量
        #制作train dict
        trainDict_only={}
        for ui in range(len(trainUserlist_only)):
            trainDict_only[int(trainUserlist_only[ui])]=[]
        for inter in range(trainAlldata_only.shape[0]):
            if trainDict_only.get(int(trainAlldata_only[inter,0])):
                trainDict_only[int(trainAlldata_only[inter,0])].append(int(trainAlldata_only[inter,1]))
            else:
                trainDict_only[int(trainAlldata_only[inter,0])]=[int(trainAlldata_only[inter,1])]
        #制作trainUser_only、trainItem_only用来生成UserItemNet_only
        for k in trainDict_only.keys():
            trainUser_only.extend([k] * len(trainDict_only[k]))
            trainItem_only.extend(trainDict_only[k])
        self.trainUser_unique_only=trainUserlist_only
        self.trainItem_unique_only=trainItemlist_only
        self.trainUser_only = np.array(trainUser_only)
        self.trainItem_only = np.array(trainItem_only)
        self._onlyGraphSize=len(self.trainUser_only)
        self.UserItemNet_only = csr_matrix((np.ones(len(self.trainUser_only)), (self.trainUser_only, self.trainItem_only)),shape=(self.n_user, self.m_item))
        del trainUserlist_only,trainUser_only,trainItem_only,trainDict_only, trainAlldata_only#在生成A矩阵后就删除全量结果
        print(f'Loader_hat only graph at stage {self.datasetStage-1} reday size: {self._onlyGraphSize} ,used time {time.time()-onlytime}')

        #=============================制作future stage graph==============================
        onlytime=time.time()
        trainAlldata_only_future , trainItem_only_future, trainUser_only_future=[], [], []
        for s in range(stage,stage+1):#sml版本赋值20，stage的结果会是19.npy的的数据用来生成graph
            train_file_future = path + '/train/'+str(s)+'.npy'
            trainAlldata_only_future.append(np.load(train_file_future))
        trainAlldata_only_future=np.concatenate(trainAlldata_only_future, axis=0)
        trainUserlist_only_future=list(set(trainAlldata_only_future[:,0]))#这一stage（stage-1.npy）的所有user数量
        trainItemlist_only_future=list(set(trainAlldata_only_future[:,1]))#这一stage（stage-1.npy）的所有item数量
        #制作train dict
        trainDict_only_future={}
        for ui in range(len(trainUserlist_only_future)):
            trainDict_only_future[int(trainUserlist_only_future[ui])]=[]
        for inter in range(trainAlldata_only_future.shape[0]):
            if trainDict_only_future.get(int(trainAlldata_only_future[inter,0])):
                trainDict_only_future[int(trainAlldata_only_future[inter,0])].append(int(trainAlldata_only_future[inter,1]))
            else:
                trainDict_only_future[int(trainAlldata_only_future[inter,0])]=[int(trainAlldata_only_future[inter,1])]
        #制作trainUser_only_future、trainItem_only_future用来生成UserItemNet_only
        for k in trainDict_only_future.keys():
            trainUser_only_future.extend([k] * len(trainDict_only_future[k]))
            trainItem_only_future.extend(trainDict_only_future[k])
        self.trainUser_unique_only=trainUserlist_only_future
        self.trainItem_unique_only=trainItemlist_only_future
        self.trainUser_only_future = np.array(trainUser_only_future)
        self.trainItem_only_future = np.array(trainItem_only_future)
        self._onlyGraphSize_future=len(self.trainUser_only_future)
        self.UserItemNet_only_future = csr_matrix((np.ones(len(self.trainUser_only_future)), (self.trainUser_only_future, self.trainItem_only_future)),shape=(self.n_user, self.m_item))
        del trainUserlist_only_future,trainUser_only_future,trainItem_only_future,trainDict_only_future, trainAlldata_only_future#在生成A矩阵后就删除全量结果
        print(f'Loader_hat only_future graph at stage {self.datasetStage-1} reday size: {self._onlyGraphSize_future} ,used time {time.time()-onlytime}')

        #制作all stage graph
        trainAlldata_all , trainItem_all, trainUser_all=[], [], []
        alltime=time.time()
        for s in range(0,stage+1):#sml版本赋值20，stage的结果会是0-19.npy的的数据用来生成graph
            train_file = path + '/train/'+str(s)+'.npy'
            trainAlldata_all.append(np.load(train_file))
        trainAlldata_all=np.concatenate(trainAlldata_all, axis=0)
        trainUserlist_all=list(set(trainAlldata_all[:,0]))#0到这一stage（stage-1.npy）的所有user数量
        trainDict_all={}
        for ui in range(len(trainUserlist_all)):
            trainDict_all[int(trainUserlist_all[ui])]=[]
        for inter in range(trainAlldata_all.shape[0]):
            if trainDict_all.get(int(trainAlldata_all[inter,0])):
                trainDict_all[int(trainAlldata_all[inter,0])].append(int(trainAlldata_all[inter,1]))
            else:
                trainDict_all[int(trainAlldata_all[inter,0])]=[int(trainAlldata_all[inter,1])]
        for k in trainDict_all.keys():
            trainUser_all.extend([k] * len(trainDict_all[k]))
            trainItem_all.extend(trainDict_all[k])
        self.trainUser_all = np.array(trainUser_all)
        self.trainItem_all = np.array(trainItem_all)
        self._allGraphSize=len(self.trainUser_all)
        self.UserItemNet_all = csr_matrix((np.ones(len(self.trainUser_all)), (self.trainUser_all, self.trainItem_all)),shape=(self.n_user, self.m_item))
        del trainUserlist_all, trainUser_all,trainItem_all, trainDict_all, trainAlldata_all#在生成A矩阵后就删除全量结果
        print(f'Loader_hat all graph reday at stage: 0-{self.datasetStage-1} size: {self._allGraphSize} ,used time {time.time()-alltime}')

        #生成当前stage正样本train数据集hisPos,nowPos
        posItems_his, posItems_now = [], []
        for user in list(range(self.n_user)):
            posItems_his.append(self.UserItemNet_all[user].nonzero()[1])
            posItems_now.append(self.UserItemNet_only[user].nonzero()[1])
            self._traindataSize+=len(posItems_now[user])
        self.hisPos = posItems_his
        self.nowPos = posItems_now

        #制作testset
        testUniqueUsers, testdata_all= [], []
        for s in range(stage,stage+1):#s赋值20开始，stage的结果会是20.npy
            test_file = path + '/test/'+str(s)+'.npy'
            testdata_all.append(np.load(test_file))
        testdata_all=np.concatenate(testdata_all, axis=0)
        test_item_dict={}#user是第一层key正样本是第二层key，value是这条iter的正+负样本，是testDict
        test_user=list(set(testdata_all[:,0]))
        for ut in range(len(test_user)):#创建一个以test user id为key的字典
            test_item_dict[int(test_user[ut])]=[]
        for inter in range(testdata_all.shape[0]):#嵌套字典，第二层循环的key是test正样本value是所有交互数据
            if test_item_dict.get(int(testdata_all[inter,0])):
                test_item_dict[int(testdata_all[inter,0])][int(testdata_all[inter,1])]=[int(x) for x in testdata_all[inter,1:]]
            else:
                key_temp=int(testdata_all[inter,1])
                test_item_dict[int(testdata_all[inter,0])]={int(key_temp):[int(x) for x in testdata_all[inter,1:]]}
        for key in test_item_dict.keys():
            testUniqueUsers.append(key)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self._testdataSize=testdata_all.shape[0]
        self.__testDict = test_item_dict#双层嵌套字典
        # print(f"Loader_hat at stage : {self.datasetStage} , {self._traindataSize} interactions for train , {self._testdataSize} interactions for test")
        print(f'===========================load data is ready ,used time : {time.time()-dataset_starttime} s ===================================')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_only(self):
        # print("try loading adjacency matrix")
        adj_starttime=time.time()
        if self.Graph_only is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded only graph...")
                norm_adj = pre_adj_mat
            except :
                print('load only graph faulse at: '+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz    '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved norm_mat at "+self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/__ONLY__no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_only = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_only = self.Graph_only.coalesce().to(world.device)
            print(f"only adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_only

    def getSparseGraph_all(self):
        # print("try loading adjacency matrix")
        adj_starttime=time.time()
        if self.Graph_all is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded all graph...")
                norm_adj = pre_adj_mat
            except :
                print('load all graph faulse at: '+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz  '+', generating adjacency matrix')
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved all graph norm_mat at "+self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/no self loop'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
            self.Graph_all = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_all = self.Graph_all.coalesce().to(world.device)
            print(f"all adjacency matrix done use time :{time.time()-adj_starttime}")
        return self.Graph_all

    def getSparseGraph_handle(self):
        adj_starttime=time.time()
        if self.Graph_handle or self.Rscale is None:
            try:
                pre_adj_mat_all = sp.load_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded handle graph ")
                rscale_vec = np.load(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-.npy')
                print("successfully loaded handle rescale")
                norm_adj = pre_adj_mat_all
            except :
                print('load handle_graph faulse at: '+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
                s = time.time()
                #生成adj_all
                adj_mat_all = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_all = adj_mat_all.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat_all[:self.n_users, self.n_users:] = R
                adj_mat_all[self.n_users:, :self.n_users] = R.T
                adj_mat_all = adj_mat_all.todok()
                # adj_mat_all = adj_mat_all + sp.eye(adj_mat_all.shape[0])

                #生成adj_only
                adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only = adj_mat_only.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat_only[:self.n_users, self.n_users:] = R
                adj_mat_only[self.n_users:, :self.n_users] = R.T
                adj_mat_only = adj_mat_only.todok()
                # adj_mat_only = adj_mat_only + sp.eye(adj_mat_only.shape[0])

                rowsum_all = np.array(adj_mat_all.sum(axis=1))
                rowsum_only = np.array(adj_mat_only.sum(axis=1))

                degree_new=rowsum_only
                degree_old=rowsum_all-rowsum_only
                degree_all=rowsum_all
                d_inv = np.power(degree_all, -0.5).flatten()#flatten直接变成一行了
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat_only)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                old_scale=np.power(degree_old, 0.5)
                # old_scale[np.isinf(old_scale)]=0.
                new_scale=np.power(degree_all, 0.5)
                # new_scale[np.isinf(degree_all)]=0.
                rscale_vec=np.true_divide(old_scale , new_scale)# n*1 array
                np.save(self.path + '/fenzi-'+'-', old_scale)
                np.save(self.path + '/fenmu'+'-', new_scale)
                rscale_vec[np.isinf(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                rscale_vec[np.isnan(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                end = time.time()
                print(f"costing {end-s}s, saved handle graph norm_mat at "+self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                sp.save_npz(self.path + '/handle_graph'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
                np.save(self.path + '/rscale_vec-at'+str(self.datasetStage-1)+'-', rscale_vec)

            if sum(np.isinf(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isinf(rscale_vec))}')
            elif sum(np.isnan(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isnan(rscale_vec))}')
            self.Graph_handle = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_handle = self.Graph_handle.coalesce().to(world.device)
            self.Rscale=torch.from_numpy(rscale_vec).to(world.device)
            print(f"handle adjacency matrix & Rescale vector done use time :{time.time()-adj_starttime}")
        return self.Graph_handle, self.Rscale

    def getSparseGraph_future_degree_adj(self):
        adj_starttime=time.time()
        if self.Graph_future_degree or self.Rscale_future_degree is None:
            try:
                pre_adj_mat_all = sp.load_npz(self.path + '/future_degree_adj'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz')
                print("successfully loaded handle graph ")
                rscale_vec = np.load(self.path + '/future_degree rscale_vec-at'+str(self.datasetStage-1)+'-.npy')
                print("successfully loaded handle rescale")
                norm_adj = pre_adj_mat_all
            except :
                print('load future_degree faulse at: '+self.path + '/future_degree'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz '+', generating adjacency matrix')
                s = time.time()
                #生成adj_all
                adj_mat_all = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_all = adj_mat_all.tolil()
                R = self.UserItemNet_all.tolil()
                adj_mat_all[:self.n_users, self.n_users:] = R
                adj_mat_all[self.n_users:, :self.n_users] = R.T
                adj_mat_all = adj_mat_all.todok()
                # adj_mat_all = adj_mat_all + sp.eye(adj_mat_all.shape[0])

                #生成adj_only
                adj_mat_only = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only = adj_mat_only.tolil()
                R = self.UserItemNet_only.tolil()
                adj_mat_only[:self.n_users, self.n_users:] = R
                adj_mat_only[self.n_users:, :self.n_users] = R.T
                adj_mat_only = adj_mat_only.todok()
                # adj_mat_only = adj_mat_only + sp.eye(adj_mat_only.shape[0])

                adj_mat_only_future = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat_only_future = adj_mat_only_future.tolil()
                R = self.UserItemNet_only_future.tolil()
                adj_mat_only_future[:self.n_users, self.n_users:] = R
                adj_mat_only_future[self.n_users:, :self.n_users] = R.T
                adj_mat_only_future = adj_mat_only_future.todok()
                # adj_mat_only_future = adj_mat_only_future + sp.eye(adj_mat_only_future.shape[0])

                rowsum_all = np.array(adj_mat_all.sum(axis=1))
                rowsum_only = np.array(adj_mat_only.sum(axis=1))
                rowsum_future = np.array(adj_mat_only_future.sum(axis=1))

                degree_new=rowsum_only
                degree_old=rowsum_all-rowsum_only
                degree_all=rowsum_all
                d_inv = np.power(rowsum_future, -0.5).flatten()#flatten直接变成一行了
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat_only)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                old_scale=np.power(degree_old, 0.5)
                # old_scale[np.isinf(old_scale)]=0.
                new_scale=np.power(degree_all, 0.5)
                # new_scale[np.isinf(degree_all)]=0.
                rscale_vec=np.true_divide(old_scale , new_scale)# n*1 array
                rscale_vec[np.isinf(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                rscale_vec[np.isnan(rscale_vec)] = 0.#分母是0分子也一定是0，这种情况是inf，这时应该inf=0
                end = time.time()
                sp.save_npz(self.path + '/future_degree_adj'+str(self.datasetStage-1)+'.npy s_pre_adj_mat.npz', norm_adj)
                np.save(self.path + '/future_degree rscale_vec-at'+str(self.datasetStage-1)+'-', rscale_vec)

            if sum(np.isinf(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isinf(rscale_vec))}')
            elif sum(np.isnan(rscale_vec))!=0:
                raise TypeError(f'recale_vec have {sum(np.isnan(rscale_vec))}')
            self.Graph_future_degree = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_future_degree = self.Graph_future_degree.coalesce().to(world.device)
            self.Rscale_future_degree=torch.from_numpy(rscale_vec).to(world.device)
            print(f"handle adjacency matrix & Rescale vector done use time :{time.time()-adj_starttime}")
        return self.Graph_future_degree, self.Rscale_future_degree

    def get_degree(self):
        try:
            now_item_degree=np.load(self.path + '/item_degree_only-'+str(self.datasetStage-1)+'-.npy')
            all_item_degree=np.load(self.path + '/item_degree_all-'+str(self.datasetStage-1)+'-.npy')
            now_user_degree=np.load(self.path + '/user_degree_only-'+str(self.datasetStage-1)+'-.npy')
            all_user_degree=np.load(self.path + '/user_degree_all-'+str(self.datasetStage-1)+'-.npy')
        except:
            now_item_degree : numpy.matrix
            now_item_ = self.UserItemNet_only.tolil()
            all_item_ = self.UserItemNet_all.tolil()
            now_item_degree = np.array(now_item_.sum(axis=0))[0]
            all_item_degree = np.array(all_item_.sum(axis=0))[0]
            now_item_degree = now_item_degree.reshape(-1,1)
            all_item_degree = all_item_degree.reshape(-1,1)
            np.save(self.path + '/item_degree_only-'+str(self.datasetStage-1)+'-', now_item_degree)
            np.save(self.path + '/item_degree_all-'+str(self.datasetStage-1)+'-', all_item_degree)

            now_user_degree : numpy.matrix
            now_user_ = self.UserItemNet_only.tolil()
            all_user_ = self.UserItemNet_all.tolil()
            now_user_degree = np.array(now_user_.sum(axis=1))
            all_user_degree = np.array(all_user_.sum(axis=1))
            now_user_degree = now_user_degree.reshape(-1,1)
            all_user_degree = all_user_degree.reshape(-1,1)
            np.save(self.path + '/user_degree_only-'+str(self.datasetStage-1)+'-', now_user_degree)
            np.save(self.path + '/user_degree_all-'+str(self.datasetStage-1)+'-', all_user_degree)
        if (now_user_degree.shape[0] != self.n_user) or (all_user_degree.shape[0] != self.n_user):
            raise ValueError(f'{now_user_degree.shape[1]}!={self.n_user}')
        if (now_item_degree.shape[0] != self.m_item) or (all_item_degree.shape[0] != self.m_item):
            raise ValueError(f'{now_item_degree.shape[1]}!={self.n_user}')
        return torch.from_numpy(now_item_degree).type(torch.FloatTensor), torch.from_numpy(all_item_degree-now_item_degree).type(torch.FloatTensor),torch.from_numpy(now_user_degree).type(torch.FloatTensor), torch.from_numpy(all_user_degree-now_user_degree).type(torch.FloatTensor)
