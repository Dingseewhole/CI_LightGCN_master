import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "/home/dingsh/2020_lgcn_finetune"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')


config = {}
all_dataset = ['news','finetune_yelp','gowalla']
all_models  = ['mf', 'lgn','finetune_lgcn','fullretrain_lgcn','static_lgcn','sml','sml_x','Transfer_LGCN','metah','metahs','our_model','handle_model']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['test_u_batch_size'] = args.testbatch
config['lr'] = args.lr

setstart=args.setstart
if setstart!=' ':
    Meta_start=eval(args.start_end)[0]
    FT_end=eval(args.start_end)[1]
    FR_end=40
elif args.dataset=='finetune_yelp':
    FT_start=30
    FT_end=40
    FR_start=30
    FR_end=40
    Meta_start=30#10
elif args.dataset=='news':
    Meta_start=48
    FT_start=48
    FT_end=63
    FR_start=48
    FR_end=63
elif args.dataset=='gowalla':
    FT_start=30
    FT_end=40
    FR_start=30
    FR_end=40
    Meta_start=30

ST_stage=args.ST_stage
FR_epochs=args.FR_epochs
FT_epochs=args.FT_epochs
ST_epochs=args.ST_epochs

device = torch.device('cuda')
GPU_availiable = torch.cuda.is_available()
# CORES = multiprocessing.cpu_count() // 2 - 1
seed = args.seed

dataset = args.dataset
model_name = args.model
# exptag=args.exptag

lgcn_lr=args.lr
lgcn_weight_dency=args.decay

PATH = args.path
topks = eval(args.topks)
recdim=args.recdim
multi=0
handle_epochs=args.handle_epochs
rescale_zoom=args.rescale_zoom
sample_mode=args.sample_mode
# inference_sample_mode = args.inference_sample_mode
# conv2d_channel = args.conv2d_channel
# conv2d_channel2 = args.conv2d_channel2

# transfer_lr=args.transfer_lr
# trans_epochs=args.trans_epochs
conv2d_reg=args.conv2d_reg
# transfer_decay=args.transfer_decay
finetune_epochs=args.finetune_epochs
# infer_epochs = args.infer_epochs

icl_k=args.icl_k
dis='l2'
notactive=args.notactive
inference_lr=args.inference_lr
# M = args.M
# M_cold = args.M_cold
radio_loss = args.radio_loss
icl_reg = args.icl_reg

A=args.A
inference_k=args.inference_k
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
