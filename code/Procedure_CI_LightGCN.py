import world
import numpy as np
import torch
import utils
from pprint import pprint
from time import time
from tqdm import tqdm
import multiprocessing
import pickle
from torch import optim
import collections

# def MultiProcess_batchusers_test(IN):
    # batch_usersusers=IN[0]
    # UsersRatings_cpu=IN[1]
    # TestDict=IN[2]
    # posit_list=[]
    # rating_K_list=[]
    # for userIndex in range(batch_usersusers.shape[0]):
    #     userID=int(batch_usersusers[userIndex])
    #     for posit in list(TestDict[userID].keys()):
    #         OneIter_Itemlist=TestDict[userID][posit]
    #         OneItreation_Rating=UsersRatings_cpu[userIndex,OneIter_Itemlist]
    #         OneItreation_Rating=np.expand_dims(OneItreation_Rating, axis=0)
    #         rating_K = utils.topk(OneItreation_Rating, K=max(world.topks))
    #         rating_index=np.array(rating_K[0])
    #         OneIter_Itemlist_array=np.array(OneIter_Itemlist)
    #         rating_K_list.append(OneIter_Itemlist_array[rating_index].tolist())
    #         posit_list.append(posit)

    # return (posit_list,rating_K_list )

def train_joint(model, dataset, old_embeddings, opt_lgcn, old_knowledge, epoch, ww=None):
    Meta_model=model
    Meta_model.train()

    S, sam_time = utils.UniformSample_handle( dataset, world.sample_mode)
    print(f"BPR[lgcn handle sample time][{sam_time:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    aver_loss1 = 0.
    aver_loss2 = 0.
    aver_icl_regloss = 0.

    (old_User, old_Item) = old_knowledge
    del old_User

    for (batch_i,(batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):

        knn_0 = old_Item[batch_pos.long()] @ old_Item.t() 
        knn_1 = old_Item[batch_pos.long()].pow(2).sum(dim=1).unsqueeze(1)
        knn_2 = old_Item.pow(2).sum(dim=1).unsqueeze(0)
        knn = knn_1 + knn_2 -2 * knn_0
        knn = knn.pow(2)
        knn[:,dataset.active_item_now]=np.inf
        batch_mtach_items_itself = torch.topk(knn, world.icl_k+1, largest=False, dim=1)[1]
        batch_mtach_items = batch_mtach_items_itself[:,1:world.icl_k+1]

        del knn_0, knn_1, knn_2, knn 

        loss, loss1, loss2, icl_regloss = Meta_model.get_our_loss(old_embeddings, batch_users, batch_pos, batch_neg, batch_mtach_items)
        opt_lgcn.zero_grad()
        loss.backward()
        opt_lgcn.step()
        aver_loss+=loss.cpu().item()
        aver_loss1+=loss1.cpu().item()
        aver_loss2+=loss2.cpu().item()
        aver_icl_regloss+=icl_regloss.cpu().item()

    aver_loss = aver_loss / total_batch
    aver_loss1 = aver_loss1 / total_batch
    aver_loss2 = aver_loss2 / total_batch
    aver_icl_regloss = aver_icl_regloss / total_batch
    return f"[Train aver loss{aver_loss:.4e} = train_loss {aver_loss1:.4e} + icl_loss  {aver_loss2:.4e} + icl_regloss  {aver_icl_regloss:.4e} + reg]"
def test_joint(model, dataset, old_embeddings, epoch, ww=None):
    Meta_model=model
    Meta_model.eval()

    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)), 'recall': np.zeros(len(world.topks)), 'ndcg': np.zeros(len(world.topks))}
    testDict= dataset.testDict

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            u_batch_size = world.config['test_u_batch_size']
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        if len(users)%u_batch_size==0:
            total_batch = len(users) // u_batch_size
        else:
            total_batch = len(users) // u_batch_size + 1

        if world.multi==1:
            INlist=[]
            for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
                batch_users_tensor = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_tensor.to(world.device)
                users_list.append(batch_users)

                rating = Meta_model.get_finalprediction(old_embeddings, batch_users_gpu)
                Rating_cpu=rating.cpu()
                test_Dict_onebatch={}
                for i in range(b ):
                    test_Dict_onebatch[int(batch_users_tensor[i])]=testDict[int(batch_users_tensor[i])]
                INlist.append((batch_users_tensor.numpy(),Rating_cpu.numpy(),test_Dict_onebatch))
                
            with multiprocessing.Pool(world.CORES) as pool:
                PollResults=pool.map(MultiProcess_batchusers_test,INlist)
            for MultiProcess_result in PollResults:
                rating_list.extend(MultiProcess_result[1])
                groundTrue_list.extend(MultiProcess_result[0])
        else:
            for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
                batch_users_tensor = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_tensor.to(world.device)
                users_list.append(batch_users)

                rating = Meta_model.get_finalprediction(old_embeddings, batch_users_gpu)
                Rating_cpu=rating.cpu()
                test_Dict_onebatch={}
                for i in range(batch_users_tensor.size()[0]):
                    test_Dict_onebatch[int(batch_users_tensor[i])]=testDict[int(batch_users_tensor[i])]
                batch_usersusers=batch_users_tensor.numpy()
                UsersRatings_cpu=Rating_cpu.numpy()
                TestDict=test_Dict_onebatch
                posit_list=[]
                rating_K_list=[]
                for userIndex in range(batch_usersusers.shape[0]):
                    userID=int(batch_usersusers[userIndex])
                    for posit in list(TestDict[userID].keys()):
                        OneIter_Itemlist=TestDict[userID][posit]
                        OneItreation_Rating=UsersRatings_cpu[userIndex,OneIter_Itemlist]
                        OneItreation_Rating=np.expand_dims(OneItreation_Rating, axis=0)
                        rating_K = utils.topk(OneItreation_Rating, K=max(world.topks))
                        rating_index=np.array(rating_K[0])
                        OneIter_Itemlist_array=np.array(OneIter_Itemlist)
                        rating_K_list.append(OneIter_Itemlist_array[rating_index].tolist())
                        posit_list.append(posit)
                rating_list.extend(rating_K_list)
                groundTrue_list.extend(posit_list)

        (Recall,Precision)=utils.Recall_onepos_999neg(rating_list,groundTrue_list,world.topks)
        (Ndcg)=utils.NDCG_onepos_999neg(rating_list,groundTrue_list,world.topks)
        assert total_batch == len(users_list)

        results['recall'] =Recall
        results['precision']=Precision
        results['ndcg']= Ndcg
        print(results)
        return results

def test_joint_icl_Mount(model, dataset, old_embeddings, old_User, epoch, w=None):
    Meta_model=model
    Meta_model.eval()

    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)), 'recall': np.zeros(len(world.topks)), 'ndcg': np.zeros(len(world.topks))}
    results_ac = {'precision': np.zeros(len(world.topks)), 'recall': np.zeros(len(world.topks)), 'ndcg': np.zeros(len(world.topks))}
    results_inac = {'precision': np.zeros(len(world.topks)), 'recall': np.zeros(len(world.topks)), 'ndcg': np.zeros(len(world.topks))}
    testDict= dataset.testDict
    activeUser_all = dataset.active_user_now
    activeItem_all = dataset.active_item_now

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            u_batch_size = world.config['test_u_batch_size']
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []

        rating_list = []
        groundTrue_list = []

        rating_list_ac = []
        groundTrue_list_ac = []

        rating_list_inac = []
        groundTrue_list_inac = []
        if len(users)%u_batch_size==0:
            total_batch = len(users) // u_batch_size
        else:
            total_batch = len(users) // u_batch_size + 1
        
        user_output, item_output, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree=Meta_model.get_layer_weights()

        for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
            batch_users_tensor = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_tensor.to(world.device)
            users_list.append(batch_users)

            knn_0 = old_User[batch_users_gpu.long()] @ old_User.t() 
            knn_1 = old_User[batch_users_gpu.long()].pow(2).sum(dim=1).unsqueeze(1)
            knn_2 = old_User.pow(2).sum(dim=1).unsqueeze(0)
            knn = knn_1 + knn_2 -2 * knn_0
            knn = knn.pow(2)
            batch_mtach_users_itself = torch.topk(knn, world.inference_k+1, largest=False, dim=1)[1]
            batch_mtach_users = batch_mtach_users_itself[:,1:world.inference_k+1]


            rating = Meta_model.get_finalprediction(old_embeddings, batch_users_gpu, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree, activeUser_all, activeItem_all, dataset.trained_user, dataset.trained_item, batch_mtach_users)
            Rating_cpu=rating.cpu()
            test_Dict_onebatch={}
            for i in range(batch_users_tensor.size()[0]):
                test_Dict_onebatch[int(batch_users_tensor[i])]=testDict[int(batch_users_tensor[i])]
            batch_usersusers=batch_users_tensor.numpy()
            UsersRatings_cpu=Rating_cpu.numpy()
            TestDict=test_Dict_onebatch
            posit_list=[]
            rating_K_list=[]

            posit_list_ac=[]
            rating_K_list_ac=[]

            posit_list_inac=[]
            rating_K_list_inac=[]
            for userIndex in range(batch_usersusers.shape[0]):
                userID=int(batch_usersusers[userIndex])
                for posit in list(TestDict[userID].keys()):
                    OneIter_Itemlist=TestDict[userID][posit]
                    OneItreation_Rating=UsersRatings_cpu[userIndex,OneIter_Itemlist]
                    OneItreation_Rating=np.expand_dims(OneItreation_Rating, axis=0)
                    rating_K = utils.topk(OneItreation_Rating, K=max(world.topks))
                    rating_index=np.array(rating_K[0])
                    OneIter_Itemlist_array=np.array(OneIter_Itemlist)
                    rating_K_list.append(OneIter_Itemlist_array[rating_index].tolist())
                    posit_list.append(posit)
                    if userID in activeUser_all:
                        rating_K_list_ac.append(OneIter_Itemlist_array[rating_index].tolist())
                        posit_list_ac.append(posit)    
                    elif userID not in activeUser_all:
                        rating_K_list_inac.append(OneIter_Itemlist_array[rating_index].tolist())
                        posit_list_inac.append(posit)

            rating_list_ac.extend(rating_K_list_ac)
            groundTrue_list_ac.extend(posit_list_ac)

            rating_list_inac.extend(rating_K_list_inac)
            groundTrue_list_inac.extend(posit_list_inac)

            rating_list.extend(rating_K_list)
            groundTrue_list.extend(posit_list)

        (Recall,Precision)=utils.Recall_onepos_999neg(rating_list,groundTrue_list,world.topks)
        (Ndcg)=utils.NDCG_onepos_999neg(rating_list,groundTrue_list,world.topks)

        (Recall_ac,Precision_ac)=utils.Recall_onepos_999neg(rating_list_ac, groundTrue_list_ac, world.topks)
        (Ndcg_ac)=utils.NDCG_onepos_999neg(rating_list_ac, groundTrue_list_ac, world.topks)

        (Recall_inac,Precision_inac)=utils.Recall_onepos_999neg(rating_list_inac, groundTrue_list_inac, world.topks)
        (Ndcg_inac)=utils.NDCG_onepos_999neg(rating_list_inac, groundTrue_list_inac, world.topks)

        assert total_batch == len(users_list)
        # print(len(groundTrue_list_ac),len(groundTrue_list_inac),len(groundTrue_list))
        assert len(groundTrue_list_ac)+len(groundTrue_list_inac) == len(groundTrue_list), f'user num wrong error {len(groundTrue_list_ac)}+{len(groundTrue_list_inac)}!={len(groundTrue_list)}'

        results['recall'] =Recall
        results['precision']=Precision
        results['ndcg']= Ndcg

        results_ac['recall'] =Recall_ac
        results_ac['precision']=Precision_ac
        results_ac['ndcg']= Ndcg_ac

        results_inac['recall'] =Recall_inac
        results_inac['precision']=Precision_inac
        results_inac['ndcg']= Ndcg_inac

        # print(results)
        # print(f"Activate users num :{len(groundTrue_list_ac)} " ,results_ac)
        # print(f"Inactivate users num : {len(groundTrue_list_inac)} " ,results_inac)

        return (results, results_ac, len(groundTrue_list_ac), results_inac, len(groundTrue_list_inac))

# def train_transfer(model, dataset, old_embeddings, opt_Trans, epoch, w):
#     Meta_model=model
#     Meta_model.train()

#     S, sam_time = utils.UniformSample_handle( dataset, 'new')
#     print(f"BPR[lgcn handle sample time][{sam_time:.2f}]")
#     users = torch.Tensor(S[:, 0]).long()
#     posItems = torch.Tensor(S[:, 1]).long()
#     negItems = torch.Tensor(S[:, 2]).long()
#     users = users.to(world.device)
#     posItems = posItems.to(world.device)
#     negItems = negItems.to(world.device)
#     users, posItems, negItems = utils.shuffle(users, posItems, negItems)
#     total_batch = len(users) // world.config['bpr_batch_size'] + 1
#     aver_loss = 0.

#     for (batch_i,(batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
#         loss=Meta_model.get_our_loss(old_embeddings, batch_users, batch_pos, batch_neg)
#         opt_Trans.zero_grad()
#         loss.backward()
#         opt_Trans.step()
#         aver_loss+=loss.cpu().item()

#     aver_loss = aver_loss / total_batch
#     w.add_scalar(f'Train Transfer at stage{dataset.datasetStage}', aver_loss, epoch)
#     return f"[BPR[Train Transfer aver loss{aver_loss:.4e}]"

# def test_future(model, dataset, old_embeddings, epoch, w):
#     Meta_model=model
#     Meta_model.eval()

#     max_K = max(world.topks)
#     results = {'precision': np.zeros(len(world.topks)), 'recall': np.zeros(len(world.topks)), 'ndcg': np.zeros(len(world.topks))}
#     testDict= dataset.testDict

#     with torch.no_grad():
#         users = list(testDict.keys())
#         try:
#             u_batch_size = world.config['test_u_batch_size']
#             assert u_batch_size <= len(users) / 10
#         except AssertionError:
#             print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#         users_list = []
#         rating_list = []
#         groundTrue_list = []
#         if len(users)%u_batch_size==0:
#             total_batch = len(users) // u_batch_size
#         else:
#             total_batch = len(users) // u_batch_size + 1

#         if world.multi==1:
#             INlist=[]
#             for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
#                 batch_users_tensor = torch.Tensor(batch_users).long()
#                 batch_users_gpu = batch_users_tensor.to(world.device)
#                 users_list.append(batch_users)

#                 rating = Meta_model.get_finalprediction(old_embeddings, batch_users_gpu)
#                 Rating_cpu=rating.cpu()
#                 test_Dict_onebatch={}
#                 for i in range(b ):
#                     test_Dict_onebatch[int(batch_users_tensor[i])]=testDict[int(batch_users_tensor[i])]
#                 INlist.append((batch_users_tensor.numpy(),Rating_cpu.numpy(),test_Dict_onebatch))
                
#             # multiprocessing.set_start_method('spawn',True)
#             with multiprocessing.Pool(world.CORES) as pool:
#                 PollResults=pool.map(MultiProcess_batchusers_test,INlist)
#                 # PollResults_list.append(PollResults.get(timeout=1000))
#             for MultiProcess_result in PollResults:
#                 rating_list.extend(MultiProcess_result[1])
#                 groundTrue_list.extend(MultiProcess_result[0])#是int不是tensor不需要,cpu()
#         else:
#             for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
#                 batch_users_tensor = torch.Tensor(batch_users).long()
#                 batch_users_gpu = batch_users_tensor.to(world.device)
#                 users_list.append(batch_users)

#                 rating = Meta_model.get_finalprediction(old_embeddings, batch_users_gpu)
#                 Rating_cpu=rating.cpu()
#                 test_Dict_onebatch={}
#                 for i in range(batch_users_tensor.size()[0]):
#                     test_Dict_onebatch[int(batch_users_tensor[i])]=testDict[int(batch_users_tensor[i])]
#                 batch_usersusers=batch_users_tensor.numpy()
#                 UsersRatings_cpu=Rating_cpu.numpy()
#                 TestDict=test_Dict_onebatch
#                 posit_list=[]
#                 rating_K_list=[]
#                 for userIndex in range(batch_usersusers.shape[0]):
#                     userID=int(batch_usersusers[userIndex])
#                     for posit in list(TestDict[userID].keys()):
#                         OneIter_Itemlist=TestDict[userID][posit]
#                         OneItreation_Rating=UsersRatings_cpu[userIndex,OneIter_Itemlist]
#                         OneItreation_Rating=np.expand_dims(OneItreation_Rating, axis=0)
#                         rating_K = utils.topk(OneItreation_Rating, K=max(world.topks))
#                         rating_index=np.array(rating_K[0])
#                         OneIter_Itemlist_array=np.array(OneIter_Itemlist)
#                         rating_K_list.append(OneIter_Itemlist_array[rating_index].tolist())
#                         posit_list.append(posit)
#                 rating_list.extend(rating_K_list)
#                 groundTrue_list.extend(posit_list)

#         (Recall,Precision)=utils.Recall_onepos_999neg(rating_list,groundTrue_list,world.topks)
#         (Ndcg)=utils.NDCG_onepos_999neg(rating_list,groundTrue_list,world.topks)
#         assert total_batch == len(users_list)

#         results['recall'] =Recall
#         results['precision']=Precision
#         results['ndcg']= Ndcg
#         w.add_scalars(f'Test future/Recall@{world.topks} at stage{dataset.datasetStage}',
#                       {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
#         w.add_scalars(f'Test future/Precision@{world.topks} at stage{dataset.datasetStage}',
#                       {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
#         w.add_scalars(f'Test future/NDCG@{world.topks} at stage{dataset.datasetStage}',
#                       {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
#         print(results)
#         return results
