'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import random
import os
import torch
import numpy as np

SEED = 20198
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return 
seed_everything(SEED)

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')

from time import time
from tqdm import tqdm

from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    
    print("model loaded...")

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("training started...")

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, auc_loger, ap_loger = [], [], [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in (range(n_batch)):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
        

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False, batch_test_flag = True)
        
        with torch.no_grad():
            
            n_batch_test = data_generator.n_test // args.batch_size + 1
            
            model.eval()
            for idx in (range(n_batch)):
                users, pos_items, neg_items = data_generator.sampleTest()
                
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)
                
                pos_out = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), axis=1)
                neg_out = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), axis=1)
                
                y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().detach().cpu()
                y_true = torch.cat([torch.ones(pos_out.size(0)),torch.zeros(neg_out.size(0))], dim=0)
                
                ret['auc'] = roc_auc_score(y_true, y_pred)
                ret['ap'] = average_precision_score(y_true, y_pred)
                
            model.train()

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        auc_loger.append(ret['auc'])
        ap_loger.append(ret['ap'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],  ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f, %.5f], ap=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, 
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['auc'], ret['auc'], ret['ap'], ret['ap'])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['precision'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['precision'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    auc = np.array(auc_loger)
    ap = np.array(ap_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(pres[:, 0]).index(cur_best_pre_0)

    final_perf = "Best Iter=[%d]@[%.1f]\tprecision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0,
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    print(f'AP: {ap[idx]}, AUC: {auc[idx]}')