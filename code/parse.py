'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64, help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=128,help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='finetune_yelp',help="available datasets: [ news, finetune_yelp]")
    parser.add_argument('--path', type=str,default="../checkpoints", help="path to save weights")
    parser.add_argument('--topks', type=str,default="[5,10,20]",help="@k test list")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='handle_lgcn', help='rec-model, support [mf, lgn,finetune_lgcn,fullretrain_lgcn,static_lgcn,sml,sml_x,Transfer_LGCN,metah,handle_lgcn]')
    parser.add_argument('--finetune_epochs', type=int, default=400, help='epochs for train our model')
    parser.add_argument('--rescale_zoom', type=float, default=1, help='to fix the bais from rescale')
    parser.add_argument('--sample_mode', type=str, default='new', help='which user(item) can be negative sampled')
    parser.add_argument('--inference_sample_mode', type=str, default='new', help='which user(item) can be negative sampled')
    parser.add_argument('--setstart', type=str,default= ' ')
    parser.add_argument('--start_end', type=str,default='[]')
    parser.add_argument('--ST_stage', type=int,default=10,help='yelp[10,20,30] news[22,48]')
    parser.add_argument('--FT_epochs', type=int,default=400)
    parser.add_argument('--FR_epochs', type=int,default=700)
    parser.add_argument('--ST_epochs', type=int,default=600,help='yelp10、30 600 is best')
    parser.add_argument('--handle_epochs', type=int,default=400, help='epochs for train our model')
    parser.add_argument('--conv2d_reg', type=float,default=1e-4,help="conv2d's weight reg")
    parser.add_argument('--inference_lr', type=float,default=1e-3,help="the learning rate")
    parser.add_argument('--conv2d_channel', type=int,default=1)
    parser.add_argument('--conv2d_channel2', type=int,default=0)
    parser.add_argument('--notactive', type=int,default=1, help="training item in knn sample pool")
    parser.add_argument('--radio_loss', type=float,default=0.5, help="ICL radio")
    parser.add_argument('--icl_k', type=int,default=0, help="ICL KNN's k in training")
    parser.add_argument('--icl_reg', type=float,default=1.0, help="ICL radio")
    parser.add_argument('--A', type=float,default=0.0, help="Inference_radio (1-A)*old+A*new")
    parser.add_argument('--inference_k', type=int,default=0, help="ICL KNN's k in inference")
    
    return parser.parse_args()