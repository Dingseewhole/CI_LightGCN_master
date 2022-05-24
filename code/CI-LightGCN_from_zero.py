import world
import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import logging
import datetime
import copy
import os
from os.path import join
import collections
import model_CI_LightGCN_zero as model
import dataloader_handle_inference_icl as dataloader
import Procedure_CI_LightGCN_zero
from torch import optim
import shutil

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
model_name=world.model_name
# ===================================
results_handle_LGCN_stack=[]
for stage in range(world.Meta_start-1,world.FR_end):
    if stage==world.Meta_start-1:
        pritrainFile = f"static_lgcn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-{stage-1}.npy-.pth.tar"
        pretrain_oldweights_load=join(world.FILE_PATH,pritrainFile)
        dataset_pre=dataloader.Loader_pre(stage,path="../data/"+world.dataset)
        Pretrain_model = model.LightGCN_handle(world.config, dataset_pre, 'origin_all')
        Pretrain_model.load_state_dict(torch.load(f'./checkpoints/static_base_LightGCN_{world.dataset}-3-64-28.npy-.pth.tar',map_location=torch.device('cpu')))
        Pretrain_model=Pretrain_model.to(world.device)
        print(f'#################### Baisc LightGCN has already trained at 0-{stage-1} and saved at ./checkpoints/static_base_LightGCN_{world.dataset}-3-64-28.npy-.pth.tar #######################')

        _0, _1, old_allLayerEmbs=Pretrain_model.get_layer_weights()
        old_dict=collections.OrderedDict({'embedding_user':[copy.deepcopy(old_allLayerEmbs[0][0].detach()), copy.deepcopy(old_allLayerEmbs[1][0].detach()), copy.deepcopy(old_allLayerEmbs[2][0].detach()), copy.deepcopy(old_allLayerEmbs[3][0].detach())],'embedding_item':[copy.deepcopy(old_allLayerEmbs[0][1].detach()), copy.deepcopy(old_allLayerEmbs[1][1].detach()), copy.deepcopy(old_allLayerEmbs[2][1].detach()), copy.deepcopy(old_allLayerEmbs[3][1].detach())]})
        torch.save(Pretrain_model.state_dict(), utils.getFileName(stage))#old LGCN权重
        torch.save(old_dict, os.path.join(f'../start_from_zero/{world.dataset}', f"Embeddings_at_stage_{stage}.pth.tar"))
        del Pretrain_model, old_dict, _0, _1, old_allLayerEmbs

    else:
        stagestart=time.time()
        weight_file_load = utils.getFileName(stage-1)
        weight_file_save =utils.getFileName(stage)
        embeddings_load=os.path.join(f'../start_from_zero/{world.dataset}', f"Embeddings_at_stage_{stage-1}.pth.tar")
        embeddings_save=os.path.join(f'../start_from_zero/{world.dataset}', f"Embeddings_at_stage_{stage}.pth.tar")

        dataset_LGCN=dataloader.Loader_hat(stage,path="../data/"+world.dataset)
        LGCN_joint = model.LightGCN_joint(world.config, dataset_LGCN, 'degree')

        LGCN_joint.load_state_dict(torch.load(weight_file_load,map_location=torch.device('cpu')),strict=False)
        LGCN_joint=LGCN_joint.to(world.device)
        degree_params, conv2d_params, LGCN_params = [], [], []
        for pname, p in LGCN_joint.named_parameters():
            print(pname)
            if (pname in ['Denominator.weight', 'old_scale.weight']):
                degree_params += [p]
            elif (pname in ['conv1.weight', 'conv2.weight', 'conv3.weight']):
                conv2d_params += [p] 
            else :
                LGCN_params += [p]
        if world.dataset == 'news':
            opt_LGCN=optim.Adam([{'params': LGCN_params, 'lr': world.lgcn_lr}, {'params': conv2d_params, 'lr': 0.1*world.lgcn_lr},{'params': degree_params, 'lr': 0.00001*world.lgcn_lr}])
        elif world.dataset == 'finetune_yelp':
            opt_LGCN=optim.Adam([{'params': LGCN_params, 'lr': world.lgcn_lr}, {'params': conv2d_params, 'lr': 0.001*world.lgcn_lr},{'params': degree_params, 'lr': 0.001*world.lgcn_lr}])
        elif world.dataset == 'gowalla':
            opt_LGCN=optim.Adam([{'params': LGCN_params, 'lr': world.lgcn_lr}, {'params': conv2d_params, 'lr': world.lgcn_lr},{'params': degree_params, 'lr': 0.0001*world.lgcn_lr}])

        oldknn_UserEmb, oldknn_ItemEmb, _1, _2, _3, _4, _5 = LGCN_joint.get_layer_weights()
        del _1, _2, _3, _4, _5
        old_knowledge = (copy.deepcopy(oldknn_UserEmb.detach()), copy.deepcopy(oldknn_ItemEmb.detach()))
        LastStage_embeddings=torch.load(embeddings_load, map_location=torch.device('cpu'))#这里面有四层内容，每层都是【user，item】
        Old_User = copy.deepcopy(oldknn_UserEmb.detach())

        #======================training phase=============================
        for epoch in range(world.finetune_epochs):
            epoch_start1=time.time()
            print(f'======= CI-LightGCN  Train  EPOCH[{epoch}/{world.finetune_epochs}] ==========')
            train_LGCN_loss=Procedure_CI_LightGCN_zero.train_joint(LGCN_joint, dataset_LGCN, LastStage_embeddings, opt_LGCN, old_knowledge, 0)
            print('Train in stage: '+str(stage) +' at epoch: '+str(epoch)+ ' used time : '+str(time.time() - epoch_start1)+' LOSS: '+str(train_LGCN_loss))

            if epoch%100==0:
                teststart1=time.time()
                # (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN_zero.test_joint_icl(LGCN_joint, dataset_LGCN, LastStage_embeddings, 0)
                (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN_zero.test_joint_icl_Mount(LGCN_joint, dataset_LGCN, LastStage_embeddings, Old_User, 0)
                print('CI-LightGCN result at stage: ' + str(stage) +' result: '+str(test_result))


        #=================================按stage统计结果===================================
        # (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN_zero.test_joint_icl(LGCN_joint, dataset_LGCN, LastStage_embeddings, 0)
        (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN_zero.test_joint_icl_Mount(LGCN_joint, dataset_LGCN, LastStage_embeddings, Old_User, 0)
        results_handle_LGCN_stack.append([copy.deepcopy(np.array(test_result['precision'])),copy.deepcopy(np.array(test_result['recall'])),copy.deepcopy(np.array(test_result['ndcg']))])

        for s_len in range(len(results_handle_LGCN_stack)):
            print('###############  One stage result : {}##############'.format(str(results_handle_LGCN_stack[s_len])))
            print('\n')
        #=========================save model===================
        weight_dict={}
        for k in LGCN_joint.state_dict().keys():
            if ('Denominator.weight' not in k) or ('old_scale.weight' not in k):
                weight_dict[k]=LGCN_joint.state_dict()[k]
        torch.save(weight_dict, weight_file_save)
        new_dict=LGCN_joint.get_embeddings(LastStage_embeddings)
        torch.save(new_dict,embeddings_save)
        # os.remove(embeddings_load)
        # os.remove(weight_file_load)

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% All training phases complted %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

results_handle_LGCN_performance={}

P,N,R=np.array([0.0 for i in world.topks]),np.array([0.0 for i in world.topks]),np.array([0.0 for i in world.topks])
if world.dataset=='finetune_yelp' or world.dataset=='gowalla':
    performance_num=7
    for s_len in range(len(results_handle_LGCN_stack)-performance_num,len(results_handle_LGCN_stack)):
        P+=results_handle_LGCN_stack[s_len][0]
        R+=results_handle_LGCN_stack[s_len][1]
        N+=results_handle_LGCN_stack[s_len][2]
    P=P/int(performance_num)
    R=R/int(performance_num)
    N=N/int(performance_num)

    results_handle_LGCN_performance['precision']=P
    results_handle_LGCN_performance['recall']=R
    results_handle_LGCN_performance['ndcg']=N
elif world.dataset=='news':
    total_test_num = 0
    performance_num=10
    test_num =[[45679],[70063], [81868], [55591], [67086], [86421], [53877], [66358], [67974],[51038], [70061], [62112], [34779],[50657], [48357]]
    test_id=0
    for s_len in range(len(results_handle_LGCN_stack)-performance_num,len(results_handle_LGCN_stack)):
        P+=results_handle_LGCN_stack[s_len][0]*test_num[test_id][0]
        R+=results_handle_LGCN_stack[s_len][1]*test_num[test_id][0]
        N+=results_handle_LGCN_stack[s_len][2]*test_num[test_id][0]
        test_id += 1
        total_test_num += test_num[test_id][0]
    P=P/int(total_test_num)
    R=R/int(total_test_num)
    N=N/int(total_test_num)

    results_handle_LGCN_performance['precision']=P
    results_handle_LGCN_performance['recall']=R
    results_handle_LGCN_performance['ndcg']=N
print('###############  Results of all testing stages: {} #############'.format(str(results_handle_LGCN_stack)))
print('###############  Over all results of CI-LightGCN is: {} #############'.format(str(results_handle_LGCN_performance)))
