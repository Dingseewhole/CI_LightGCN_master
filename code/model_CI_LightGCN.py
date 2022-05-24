import world
import torch
from dataloader_handle_inference_icl import BasicDataset
from torch import nn
import numpy as np
from torch import optim
import collections
import copy
import torch.nn.functional as F
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
class LightGCN_handle(BasicModel):
    def __init__(self, config:dict, dataset, graph_mode):
        super(LightGCN_handle, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.stage=dataset.datasetStage
        self.graph_mode=graph_mode
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print(f'use xavier initilizer')
        self.f = nn.Sigmoid()
        if self.graph_mode=='handle':
            self.Graph, self.Rescale = self.dataset.getSparseGraph_handle()
        elif self.graph_mode=='origin_all':
            self.Graph = self.dataset.getSparseGraph_all()
        elif self.graph_mode=='origin_only':
            self.Graph = self.dataset.getSparseGraph_only()
        else:
            raise AssertionError('not set graph mode')

    def get_layer_weights(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        allembs_list=[[users_emb,items_emb]]

        g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            all_emb_nan20 = torch.where(torch.isnan(all_emb), torch.full_like(all_emb, 0), all_emb)
            embs.append(all_emb)
            allembs_list.append([torch.split(all_emb_nan20, [self.num_users, self.num_items])[0],torch.split(all_emb_nan20, [self.num_users, self.num_items])[1]])
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, allembs_list

    def getUsersRating(self, users):
        all_users, all_items, _= self.get_layer_weights()
        del _
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items ,_= self.get_layer_weights()
        del _
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
       

    def get_our_loss(self, old_weights, users, pos, neg):
        _0, _1, allLayerEmbs=self.get_layer_weights()
        del _0, _1

        user_rescale, item_rescale =torch.split(self.Rescale, [self.num_users, self.num_items])
        
        #=============================0 layer=================================
        user_layer0=allLayerEmbs[0][0][users.long()]
        item_pos_layer0=allLayerEmbs[0][1][pos.long()]
        item_neg_layer0=allLayerEmbs[0][1][neg.long()]
        #=============================1 layer=================================
        user_layer1_1=old_weights['embedding_user'][1][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer1_1=old_weights['embedding_item'][1][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer1_1=old_weights['embedding_item'][1][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom

        user_layer1_2 = allLayerEmbs[1][0][users.long()]
        item_pos_layer1_2 = allLayerEmbs[1][1][pos.long()]
        item_neg_layer1_2 = allLayerEmbs[1][1][neg.long()]

        user_layer1 = user_layer1_1 + user_layer1_2
        item_pos_layer1 = item_pos_layer1_1 + item_pos_layer1_2
        item_neg_layer1 = item_neg_layer1_1 + item_neg_layer1_2
        #=============================2 layer=================================
        user_layer2_1=old_weights['embedding_user'][2][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer2_1=old_weights['embedding_item'][2][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer2_1=old_weights['embedding_item'][2][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom

        user_layer2_2 = allLayerEmbs[2][0][users.long()]
        item_pos_layer2_2 = allLayerEmbs[2][1][pos.long()]
        item_neg_layer2_2 = allLayerEmbs[2][1][neg.long()]

        user_layer2 = user_layer2_1 + user_layer2_2
        item_pos_layer2 = item_pos_layer2_1 + item_pos_layer2_2
        item_neg_layer2 = item_neg_layer2_1 + item_neg_layer2_2
        #=============================3 layer=================================
        user_layer3_1=old_weights['embedding_user'][3][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer3_1=old_weights['embedding_item'][3][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer3_1=old_weights['embedding_item'][3][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom

        user_layer3_2 = allLayerEmbs[3][0][users.long()]
        item_pos_layer3_2 = allLayerEmbs[3][1][pos.long()]
        item_neg_layer3_2 = allLayerEmbs[3][1][neg.long()]

        user_layer3 = user_layer3_1 + user_layer3_2
        item_pos_layer3 = item_pos_layer3_1 + item_pos_layer3_2
        item_neg_layer3 = item_neg_layer3_1 + item_neg_layer3_2
        #=============================computer======
        users_emb_stack = torch.stack([user_layer0,user_layer1,user_layer2,user_layer3], dim=1)
        users_emb=torch.mean(users_emb_stack, dim=1)
        pos_emb_stack = torch.stack([item_pos_layer0,item_pos_layer1,item_pos_layer2,item_pos_layer3], dim=1)
        pos_emb=torch.mean(pos_emb_stack, dim=1)
        neg_emb_stack = torch.stack([item_neg_layer0,item_neg_layer1,item_neg_layer2,item_neg_layer3], dim=1)
        neg_emb=torch.mean(neg_emb_stack, dim=1)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        reg_loss = (1/2)*(user_layer0.norm(2).pow(2) + item_pos_layer0.norm(2).pow(2) + item_neg_layer0.norm(2).pow(2))/float(len(users))

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss*world.lgcn_weight_dency
        loss = loss + reg_loss

        return loss

    def get_finalprediction(self, old_weights, users):
        with torch.no_grad():
            _0, _1, allLayerEmbs=self.get_layer_weights()
            del _0, _1
            user_rescale, item_rescale =torch.split(self.Rescale, [self.num_users, self.num_items])

            #=============================0 layer=================================
            user_layer0=allLayerEmbs[0][0][users.long()]
            item_layer0=allLayerEmbs[0][1]
            #=============================1 layer=================================
            user_layer1_1=old_weights['embedding_user'][1][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer1_1=old_weights['embedding_item'][1].to(world.device)*item_rescale*world.rescale_zoom

            user_layer1_2 = allLayerEmbs[1][0][users.long()]
            item_layer1_2 = allLayerEmbs[1][1]

            user_layer1 = user_layer1_1 + user_layer1_2
            item_layer1 = item_layer1_1 + item_layer1_2
            #=============================2 layer=================================
            user_layer2_1=old_weights['embedding_user'][2][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer2_1=old_weights['embedding_item'][2].to(world.device)*item_rescale*world.rescale_zoom

            user_layer2_2 = allLayerEmbs[2][0][users.long()]
            item_layer2_2 = allLayerEmbs[2][1]

            user_layer2 = user_layer2_1 + user_layer2_2
            item_layer2 = item_layer2_1 + item_layer2_2
            #=============================3 layer=================================
            user_layer3_1=old_weights['embedding_user'][3][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer3_1=old_weights['embedding_item'][3].to(world.device)*item_rescale*world.rescale_zoom

            user_layer3_2 = allLayerEmbs[3][0][users.long()]
            item_layer3_2 = allLayerEmbs[3][1]

            user_layer3 = user_layer3_1 + user_layer3_2
            item_layer3 = item_layer3_1 + item_layer3_2
            #=============================computer=================================
            users_emb_stack = torch.stack([user_layer0,user_layer1,user_layer2,user_layer3], dim=1)
            users_emb=torch.mean(users_emb_stack, dim=1)
            items_emb_stack = torch.stack([item_layer0,item_layer1,item_layer2,item_layer3], dim=1)
            items_emb=torch.mean(items_emb_stack, dim=1)

            rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def get_embeddings(self, old_weights):
        with torch.no_grad():
            _0, _1, allLayerEmbs=self.get_layer_weights()
            del _0, _1
            user_rescale, item_rescale =torch.split(self.Rescale, [self.num_users, self.num_items])

            #=============================0 layer=================================
            user_layer0=allLayerEmbs[0][0]
            item_layer0=allLayerEmbs[0][1]
            #=============================1 layer=================================
            user_layer1_1=old_weights['embedding_user'][1].to(world.device)*user_rescale*world.rescale_zoom
            item_layer1_1=old_weights['embedding_item'][1].to(world.device)*item_rescale*world.rescale_zoom

            user_layer1_2 = allLayerEmbs[1][0]
            item_layer1_2 = allLayerEmbs[1][1]

            user_layer1 = user_layer1_1 + user_layer1_2
            item_layer1 = item_layer1_1 + item_layer1_2
            #=============================2 layer=================================
            user_layer2_1=old_weights['embedding_user'][2].to(world.device)*user_rescale*world.rescale_zoom
            item_layer2_1=old_weights['embedding_item'][2].to(world.device)*item_rescale*world.rescale_zoom

            user_layer2_2 = allLayerEmbs[2][0]
            item_layer2_2 = allLayerEmbs[2][1]

            user_layer2 = user_layer2_1 + user_layer2_2
            item_layer2 = item_layer2_1 + item_layer2_2
            #=============================3 layer=================================
            user_layer3_1=old_weights['embedding_user'][3].to(world.device)*user_rescale*world.rescale_zoom
            item_layer3_1=old_weights['embedding_item'][3].to(world.device)*item_rescale*world.rescale_zoom

            user_layer3_2 = allLayerEmbs[3][0]
            item_layer3_2 = allLayerEmbs[3][1]

            user_layer3 = user_layer3_1 + user_layer3_2
            item_layer3 = item_layer3_1 + item_layer3_2

            new_dict=collections.OrderedDict({'embedding_user':[copy.deepcopy(user_layer0.detach()), copy.deepcopy(user_layer1.detach()), copy.deepcopy(user_layer2.detach()), copy.deepcopy(user_layer3.detach())],'embedding_item':[copy.deepcopy(item_layer0.detach()), copy.deepcopy(item_layer1.detach()), copy.deepcopy(item_layer2.detach()), copy.deepcopy(item_layer3.detach())]})
        return new_dict
class LightGCN_joint(BasicModel):
    def __init__(self, config:dict, dataset, graph_mode):
        super(LightGCN_joint, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.stage=dataset.datasetStage
        self.graph_mode=graph_mode
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print(f'use xavier initilizer')
        self.conv1=nn.Conv2d(1,1,(2,1),stride=1,bias=False)
        self.conv2=nn.Conv2d(1,1,(2,1),stride=1,bias=False)
        self.conv3=nn.Conv2d(1,1,(2,1),stride=1,bias=False)
        if world.dataset == 'news':
            nn.init.constant_(self.conv1.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv2.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv3.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv1.weight[0,0,0,0],0)
            nn.init.constant_(self.conv2.weight[0,0,0,0],0)
            nn.init.constant_(self.conv3.weight[0,0,0,0],0)
            self.Denominator = nn.Linear(1, 1,bias= False)
            self.old_scale = nn.Linear(1, 1,bias= False)
            nn.init.constant_(self.Denominator.weight,0)
            nn.init.constant_(self.old_scale.weight,0)
        elif world.dataset == 'finetune_yelp':
            nn.init.constant_(self.conv1.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv2.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv3.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv1.weight[0,0,0,0],1.0)
            nn.init.constant_(self.conv2.weight[0,0,0,0],1.0)
            nn.init.constant_(self.conv3.weight[0,0,0,0],1.0)
            self.Denominator = nn.Linear(1, 1,bias= False)
            self.old_scale = nn.Linear(1, 1,bias= False)
            nn.init.constant_(self.Denominator.weight,1.0)
            nn.init.constant_(self.old_scale.weight,1.0)
        elif world.dataset == 'gowalla':
            nn.init.constant_(self.conv1.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv2.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv3.weight[0,0,1,0],1.0)
            nn.init.constant_(self.conv1.weight[0,0,0,0],1.0)
            nn.init.constant_(self.conv2.weight[0,0,0,0],1.0)
            nn.init.constant_(self.conv3.weight[0,0,0,0],1.0)
            self.Denominator = nn.Linear(1, 1,bias= False)
            self.old_scale = nn.Linear(1, 1,bias= False)
            nn.init.constant_(self.Denominator.weight,1.0)
            nn.init.constant_(self.old_scale.weight,1.0)
        self.f = nn.Sigmoid()
        if self.graph_mode=='handle':
            self.Graph, self.Rescale = self.dataset.getSparseGraph_handle()
        elif self.graph_mode=='origin_all':
            self.Graph = self.dataset.getSparseGraph_all()
        elif self.graph_mode=='origin_only':
            self.Graph = self.dataset.getSparseGraph_only()
        elif self.graph_mode=='degree':
            self.Graph = self.dataset.getSparseGraph_pure()
        else:
            raise AssertionError('not set graph mode')

    def transfer1_forward(self,x_old, x_new):
        x = torch.cat((x_old,x_new),dim=-1)
        x = x.view(-1,1,2,x_new.shape[-1])
        x = self.conv1(x)
        x=x.view(-1,x_new.shape[-1])
        return x

    def transfer2_forward(self,x_old, x_new):
        x = torch.cat((x_old,x_new),dim=-1)
        x = x.view(-1,1,2,x_new.shape[-1])
        x = self.conv2(x)
        x=x.view(-1,x_new.shape[-1])
        return x

    def transfer3_forward(self,x_old, x_new):
        x = torch.cat((x_old,x_new),dim=-1)
        x = x.view(-1,1,2,x_new.shape[-1])
        x = self.conv3(x)
        x=x.view(-1,x_new.shape[-1])
        return x

    def icl_transfer(self,operation,x_old, x_new):
        x = torch.cat((x_old,x_new),dim=-1)
        x = x.view(-1,world.icl_k,2,x_new.shape[-1])
        weight=copy.deepcopy(operation.weight.data)
        x = torch.sum(torch.mul(x,weight),dim = 2)
        x=x.view(-1, world.icl_k, x_new.shape[-1])
        return x

    def denominator_forward(self, Degree_new, Degree_old):
        x_Denominator = self.Denominator(Degree_old)
        x_Denominator = torch.nn.functional.relu(x_Denominator, inplace=True) + Degree_new
        x_molecular = torch.ones_like(x_Denominator)
        return x_molecular, x_Denominator

    def oldscale_forward(self, old_scale):
        old_scale = self.old_scale(old_scale)
        old_scale = torch.nn.functional.relu(old_scale, inplace=True)
        return old_scale

    def get_layer_weights(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        allembs_list=[[users_emb,items_emb]]

        if self.graph_mode != 'degree':
            g_droped = self.Graph
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
                allembs_list.append([torch.split(all_emb, [self.num_users, self.num_items])[0],torch.split(all_emb, [self.num_users, self.num_items])[1]])
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items, allembs_list
        else:
            g_droped = self.Graph
            now_user_degree, now_item_degree, elder_user_degree, elder_item_degree, old_user_degree, old_item_degree = self.dataset.get_degree()
            degree_molecular,degree_Denominator = self.denominator_forward(torch.cat((now_user_degree, now_item_degree),dim=0), torch.cat((old_user_degree, old_item_degree),dim=0))
            degree_Denominator = degree_Denominator.pow(0.5)
            norm_degree =  torch.div(degree_molecular , (degree_Denominator+1e-9))
            norm_degree = norm_degree.flatten()
            for layer in range(self.n_layers):
                all_emb = torch.mul(norm_degree.view(-1,1), all_emb)
                all_emb = torch.sparse.mm(g_droped, all_emb)
                all_emb = torch.mul(norm_degree.view(-1,1), all_emb)
                embs.append(all_emb)
                allembs_list.append([torch.split(all_emb, [self.num_users, self.num_items])[0],torch.split(all_emb, [self.num_users, self.num_items])[1]])
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items, allembs_list, degree_molecular, degree_Denominator, old_user_degree, old_item_degree

    def get_our_loss(self, old_weights, users, pos, neg, mtach_items):
        _0, _1, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree=self.get_layer_weights()
        del _0, _1

        old_degree = torch.cat([old_user_degree, old_item_degree], dim=0)
        old_scale = self.oldscale_forward(old_degree)
        old_scale = old_scale.pow(0.5)
        old_scale = torch.mul(degree_molecular, old_scale)
        new_scale = degree_Denominator
        rscale_vec = torch.div(old_scale , new_scale+1e-9)
        user_rescale, item_rescale =torch.split(rscale_vec, [self.num_users, self.num_items])
        
        #=============================0 layer=================================
        user_layer0=allLayerEmbs[0][0][users.long()]
        item_pos_layer0=allLayerEmbs[0][1][pos.long()]
        item_neg_layer0=allLayerEmbs[0][1][neg.long()]
        Kmtachitem_layer0=allLayerEmbs[0][1][mtach_items.long()]
        matchitem_layer0=Kmtachitem_layer0
        #=============================1 layer=================================
        user_layer1_1=old_weights['embedding_user'][1][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer1_1=old_weights['embedding_item'][1][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer1_1=old_weights['embedding_item'][1][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom
        Kmatchitem_layer1_1=old_weights['embedding_item'][1][mtach_items.long()].to(world.device)*item_rescale[mtach_items.long()]*world.rescale_zoom

        user_layer1_2 = allLayerEmbs[1][0][users.long()]
        item_pos_layer1_2 = allLayerEmbs[1][1][pos.long()]
        item_neg_layer1_2 = allLayerEmbs[1][1][neg.long()]
        Kmatchitem_layer1_2 = allLayerEmbs[1][1][mtach_items.long()]

        user_layer1 = self.transfer1_forward(user_layer1_1 , user_layer1_2)
        item_pos_layer1 = self.transfer1_forward( item_pos_layer1_1 , item_pos_layer1_2)
        item_neg_layer1 = self.transfer1_forward(item_neg_layer1_1 , item_neg_layer1_2)
        matchitem_layer1 = self.icl_transfer(self.conv1, Kmatchitem_layer1_1 , Kmatchitem_layer1_2)
        #=============================2 layer=================================
        user_layer2_1=old_weights['embedding_user'][2][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer2_1=old_weights['embedding_item'][2][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer2_1=old_weights['embedding_item'][2][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom
        Kmatchitem_layer2_1=old_weights['embedding_item'][2][mtach_items.long()].to(world.device)*item_rescale[mtach_items.long()]*world.rescale_zoom

        user_layer2_2 = allLayerEmbs[2][0][users.long()]
        item_pos_layer2_2 = allLayerEmbs[2][1][pos.long()]
        item_neg_layer2_2 = allLayerEmbs[2][1][neg.long()]
        Kmatchitem_layer2_2 = allLayerEmbs[2][1][mtach_items.long()]

        user_layer2 = self.transfer2_forward(user_layer2_1 , user_layer2_2)
        item_pos_layer2 = self.transfer2_forward(item_pos_layer2_1 , item_pos_layer2_2)
        item_neg_layer2 = self.transfer2_forward(item_neg_layer2_1 , item_neg_layer2_2)
        matchitem_layer2 = self.icl_transfer(self.conv2, Kmatchitem_layer2_1 , Kmatchitem_layer2_2)
        #=============================3 layer=================================
        user_layer3_1=old_weights['embedding_user'][3][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
        item_pos_layer3_1=old_weights['embedding_item'][3][pos.long()].to(world.device)*item_rescale[pos.long()]*world.rescale_zoom
        item_neg_layer3_1=old_weights['embedding_item'][3][neg.long()].to(world.device)*item_rescale[neg.long()]*world.rescale_zoom
        Kmatchitem_layer3_1=old_weights['embedding_item'][3][mtach_items.long()].to(world.device)*item_rescale[mtach_items.long()]*world.rescale_zoom

        user_layer3_2 = allLayerEmbs[3][0][users.long()]
        item_pos_layer3_2 = allLayerEmbs[3][1][pos.long()]
        item_neg_layer3_2 = allLayerEmbs[3][1][neg.long()]
        Kmatchitem_layer3_2 = allLayerEmbs[3][1][mtach_items.long()]

        user_layer3 = self.transfer3_forward(user_layer3_1 , user_layer3_2)
        item_pos_layer3 = self.transfer3_forward(item_pos_layer3_1 , item_pos_layer3_2)
        item_neg_layer3 = self.transfer3_forward(item_neg_layer3_1 , item_neg_layer3_2)
        matchitem_layer3 = self.icl_transfer(self.conv3, Kmatchitem_layer3_1 , Kmatchitem_layer3_2)
        #=============================computer======
        users_emb_stack = torch.stack([user_layer0,user_layer1,user_layer2,user_layer3], dim=1)
        users_emb=torch.mean(users_emb_stack, dim=1)
        pos_emb_stack = torch.stack([item_pos_layer0,item_pos_layer1,item_pos_layer2,item_pos_layer3], dim=1)
        pos_emb=torch.mean(pos_emb_stack, dim=1)
        neg_emb_stack = torch.stack([item_neg_layer0,item_neg_layer1,item_neg_layer2,item_neg_layer3], dim=1)
        neg_emb=torch.mean(neg_emb_stack, dim=1)

        matchitem_emb_stack = torch.stack([matchitem_layer0, matchitem_layer1, matchitem_layer2, matchitem_layer3], dim=1)
        matchitem_emb = torch.mean(matchitem_emb_stack, dim=1)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        users_emb_icl = users_emb.unsqueeze(1)
        icl_pos_scores = torch.mul(users_emb_icl, matchitem_emb)
        icl_pos_scores = torch.sum(icl_pos_scores, dim=-1)
        icl_neg_scores = torch.mul(users_emb_icl, neg_emb)
        icl_neg_scores = torch.sum(icl_neg_scores, dim=-1)

        reg_loss1 = (1/2)*(user_layer0.norm(2).pow(2) + item_pos_layer0.norm(2).pow(2) + item_neg_layer0.norm(2).pow(2)) / float(len(users))
        reg_loss2 = (1/2)*(self.conv1.weight.norm(2).pow(2) + self.conv2.weight.norm(2).pow(2) + self.conv3.weight.norm(2).pow(2))
        reg_loss3 = (1/2)*(self.Denominator.weight.norm(2).pow(2) + self.old_scale.weight.norm(2).pow(2))
        reg_loss_icl = (1/2)*(matchitem_layer0.norm(2).pow(2)) / float(world.icl_k) / float(len(users))


        loss1 = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss2 = torch.mean(torch.nn.functional.softplus(icl_neg_scores - icl_pos_scores))

        if world.dataset == 'news':
            reg_loss = world.lgcn_weight_dency * reg_loss1 + world.conv2d_reg * reg_loss2 + 1e-3*reg_loss3
        elif world.dataset =='finetune_yelp':
            reg_loss = world.lgcn_weight_dency * reg_loss1+ world.conv2d_reg * reg_loss2 + 0*reg_loss3 + world.lgcn_weight_dency * reg_loss_icl
        elif world.dataset =='gowalla':
            reg_loss = world.lgcn_weight_dency * reg_loss1 + world.conv2d_reg * reg_loss2 + 0*reg_loss3
        loss = (1-world.ratio_loss)*loss1 + world.ratio_loss*loss2 + reg_loss

        return loss, (1-world.ratio_loss)*loss1, world.ratio_loss*loss2, world.lgcn_weight_dency * reg_loss_icl

    def get_finalprediction(self, old_weights, users, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree, active_user, active_item, trained_user, trained_item, match_users):
        with torch.no_grad():
            old_degree = torch.cat([old_user_degree, old_item_degree], dim=0)
            old_scale = self.oldscale_forward(old_degree)
            old_scale = old_degree.pow(0.5)
            old_scale = torch.mul(degree_molecular, old_scale)
            new_scale = degree_Denominator
            rscale_vec = torch.div(old_scale , new_scale+1e-9)
            user_rescale, item_rescale =torch.split(rscale_vec, [self.num_users, self.num_items])
            #=============================0 layer=================================
            user_layer0=allLayerEmbs[0][0][users.long()]
            item_layer0=allLayerEmbs[0][1]
            match_user_layer0=allLayerEmbs[0][0][match_users.long()]
            match_user_layer0 = torch.mean(match_user_layer0, dim=1)
            #=============================1 layer=================================
            user_layer1_1=old_weights['embedding_user'][1][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer1_1=old_weights['embedding_item'][1].to(world.device)*item_rescale*world.rescale_zoom
            match_user_layer1_1=old_weights['embedding_user'][1][match_users.long()].to(world.device)*user_rescale[match_users.long()]*world.rescale_zoom

            user_layer1_2 = allLayerEmbs[1][0][users.long()]
            item_layer1_2 = allLayerEmbs[1][1]
            match_user_layer1_2 = allLayerEmbs[1][0][match_users.long()]
            match_user_layer1_2 = torch.mean(match_user_layer1_2, dim=1)

            user_layer1 = self.transfer1_forward(user_layer1_1 , user_layer1_2)
            item_layer1 = self.transfer1_forward(item_layer1_1 , item_layer1_2)
            match_user_layer1 = match_user_layer1_2
            #=============================2 layer=================================
            user_layer2_1=old_weights['embedding_user'][2][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer2_1=old_weights['embedding_item'][2].to(world.device)*item_rescale*world.rescale_zoom
            match_user_layer2_1=old_weights['embedding_user'][2][match_users.long()].to(world.device)*user_rescale[match_users.long()]*world.rescale_zoom

            user_layer2_2 = allLayerEmbs[2][0][users.long()]
            item_layer2_2 = allLayerEmbs[2][1]
            match_user_layer2_2 = allLayerEmbs[2][0][match_users.long()]
            match_user_layer2_2 = torch.mean(match_user_layer2_2, dim=1)

            user_layer2 = self.transfer2_forward(user_layer2_1 , user_layer2_2)
            item_layer2 = self.transfer2_forward(item_layer2_1 , item_layer2_2)
            match_user_layer2 = match_user_layer2_2
            #=============================3 layer=================================
            user_layer3_1=old_weights['embedding_user'][3][users.long()].to(world.device)*user_rescale[users.long()]*world.rescale_zoom
            item_layer3_1=old_weights['embedding_item'][3].to(world.device)*item_rescale*world.rescale_zoom
            match_user_layer3_1=old_weights['embedding_user'][3][match_users.long()].to(world.device)*user_rescale[match_users.long()]*world.rescale_zoom

            user_layer3_2 = allLayerEmbs[3][0][users.long()]
            item_layer3_2 = allLayerEmbs[3][1]
            match_user_layer3_2 = allLayerEmbs[3][0][match_users.long()]
            match_user_layer3_2 = torch.mean(match_user_layer3_2, dim=1)

            user_layer3 = self.transfer3_forward(user_layer3_1 , user_layer3_2)
            item_layer3 = self.transfer3_forward(item_layer3_1 , item_layer3_2)
            match_user_layer3 = match_user_layer3_2
            #=============================computer=================================
            users_emb_stack = torch.stack([user_layer0,user_layer1,user_layer2,user_layer3], dim=1)
            users_emb=torch.mean(users_emb_stack, dim=1)
            items_emb_stack = torch.stack([item_layer0,item_layer1,item_layer2,item_layer3], dim=1)
            items_emb=torch.mean(items_emb_stack, dim=1)
            icl_users_emb_stack = torch.stack([match_user_layer0,match_user_layer1,match_user_layer2,match_user_layer3], dim=1)
            icl_users_emb=torch.mean(icl_users_emb_stack, dim=1)
            #==================enhance who?===========
            inactive_user = [1 if i not in active_user else 0 for i in users.tolist()]
            inactive_user_mask = torch.tensor(inactive_user).view(-1,1).to(world.device)


            users_emb = users_emb * (torch.ones_like(inactive_user_mask)-inactive_user_mask) + (1-world.A) * users_emb * inactive_user_mask +  world.A * icl_users_emb * inactive_user_mask 

            rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def get_embeddings(self, old_weights):
        with torch.no_grad():
            _0, _1, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree=self.get_layer_weights()
            del _0, _1

            old_degree = torch.cat([old_user_degree, old_item_degree], dim=0)
            old_scale = self.oldscale_forward(old_degree)
            old_scale = old_degree.pow(0.5)
            old_scale = torch.mul(degree_molecular, old_scale)
            new_scale = degree_Denominator
            rscale_vec = torch.div(old_scale , new_scale+1e-9)
            user_rescale, item_rescale =torch.split(rscale_vec, [self.num_users, self.num_items])

            #=============================0 layer=================================
            user_layer0=allLayerEmbs[0][0]
            item_layer0=allLayerEmbs[0][1]
            #=============================1 layer=================================
            user_layer1_1=old_weights['embedding_user'][1].to(world.device)*user_rescale*world.rescale_zoom
            item_layer1_1=old_weights['embedding_item'][1].to(world.device)*item_rescale*world.rescale_zoom

            user_layer1_2 = allLayerEmbs[1][0]
            item_layer1_2 = allLayerEmbs[1][1]

            user_layer1 = self.transfer1_forward(user_layer1_1 , user_layer1_2)
            item_layer1 = self.transfer1_forward(item_layer1_1 , item_layer1_2)
            #=============================2 layer=================================
            user_layer2_1=old_weights['embedding_user'][2].to(world.device)*user_rescale*world.rescale_zoom
            item_layer2_1=old_weights['embedding_item'][2].to(world.device)*item_rescale*world.rescale_zoom

            user_layer2_2 = allLayerEmbs[2][0]
            item_layer2_2 = allLayerEmbs[2][1]

            user_layer2 = self.transfer2_forward(user_layer2_1 , user_layer2_2)
            item_layer2 = self.transfer2_forward(item_layer2_1 , item_layer2_2)
            #=============================3 layer=================================
            user_layer3_1=old_weights['embedding_user'][3].to(world.device)*user_rescale*world.rescale_zoom
            item_layer3_1=old_weights['embedding_item'][3].to(world.device)*item_rescale*world.rescale_zoom

            user_layer3_2 = allLayerEmbs[3][0]
            item_layer3_2 = allLayerEmbs[3][1]

            user_layer3 = self.transfer3_forward(user_layer3_1 , user_layer3_2)
            item_layer3 = self.transfer3_forward(item_layer3_1 , item_layer3_2)

            new_dict=collections.OrderedDict({'embedding_user':[copy.deepcopy(user_layer0.detach()), copy.deepcopy(user_layer1.detach()), copy.deepcopy(user_layer2.detach()), copy.deepcopy(user_layer3.detach())],'embedding_item':[copy.deepcopy(item_layer0.detach()), copy.deepcopy(item_layer1.detach()), copy.deepcopy(item_layer2.detach()), copy.deepcopy(item_layer3.detach())]})
        return new_dict
