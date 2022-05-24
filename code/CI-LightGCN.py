import world
import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import datetime
import copy
import os
from os.path import join
import collections
import model_CI_LightGCN as model
import dataloader_handle_inference_icl as dataloader
import Procedure_CI_LightGCN
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
        print(f'####################LGCN_base has already trained at 0-{stage-1} and saved at ./checkpoints/static_base_LightGCN_{world.dataset}-3-64-28.npy-.pth.tar ##########################')

        _0, _1, old_allLayerEmbs=Pretrain_model.get_layer_weights()
        old_dict=collections.OrderedDict({'embedding_user':[copy.deepcopy(old_allLayerEmbs[0][0].detach()), copy.deepcopy(old_allLayerEmbs[1][0].detach()), copy.deepcopy(old_allLayerEmbs[2][0].detach()), copy.deepcopy(old_allLayerEmbs[3][0].detach())],'embedding_item':[copy.deepcopy(old_allLayerEmbs[0][1].detach()), copy.deepcopy(old_allLayerEmbs[1][1].detach()), copy.deepcopy(old_allLayerEmbs[2][1].detach()), copy.deepcopy(old_allLayerEmbs[3][1].detach())]})
        del Pretrain_model, old_dict, _0, _1, old_allLayerEmbs

    else:
        stagestart=time.time()
        weight_file_load = os.path.join(f'../save_for_inference/{world.dataset}', f'Weights-3-64-{stage-1}-npy.pth.tar')
        embeddings_load=os.path.join(f'../save_for_inference/{world.dataset}', f"Embeddings_at_stage_{stage}.pth.tar")

        dataset_LGCN=dataloader.Loader_hat(stage,path="../data/"+world.dataset)
        LGCN_joint = model.LightGCN_joint(world.config, dataset_LGCN, 'degree')

        LGCN_joint.load_state_dict(torch.load(weight_file_load,map_location=torch.device('cpu')),strict=False)
        LGCN_joint=LGCN_joint.to(world.device)
        degree_params, conv2d_params, LGCN_params = [], [], []
        for pname, p in LGCN_joint.named_parameters():
            # print(pname)
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
        LastStage_embeddings=torch.load(embeddings_load, map_location=torch.device('cpu'))
        Old_User = copy.deepcopy(oldknn_UserEmb.detach())
            

        (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN.test_joint_icl_Mount(LGCN_joint, dataset_LGCN, LastStage_embeddings, Old_User, 0)
        print('CI-LightGCN result at stage: ' +str(stage) +' result: '+str(test_result))


        #=================================Result generate===================================
        (test_result, test_result_ac, user_ac, test_result_inac, user_inac)=Procedure_CI_LightGCN.test_joint_icl_Mount(LGCN_joint, dataset_LGCN, LastStage_embeddings, Old_User, 0)
        results_handle_LGCN_stack.append([copy.deepcopy(np.array(test_result['precision'])),copy.deepcopy(np.array(test_result['recall'])),copy.deepcopy(np.array(test_result['ndcg']))])


        for s_len in range(len(results_handle_LGCN_stack)):
            print('############### All previous results of CI-LightGCN until stage: {} , is: {}##############'.format(str(world.Meta_start+s_len),str(results_handle_LGCN_stack[s_len])))
            print('\n')

        print(f'Last train Performance is {test_result}')
        print(f'==========the training phase at stage {stage} complted, use time: {(time.time()-stagestart)/3600} , final performance is {test_result}==========')

print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ALL STAGES TRAINING COMPLTED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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
print('#####################################################')
print('Results of all test stages:')
print(str(results_handle_LGCN_stack))
print('###############  Result of over all stages : {}#############'.format(str(results_handle_LGCN_performance)))
