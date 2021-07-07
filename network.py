from __future__ import print_function
import os, sys
sys.path.append('/data/msyuan/scMSDA/gcn')

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from preprocess import*
from torchdata import*
from loss import*
from gcn.models import GCN

len_baronME_close = 7487
len_E_close = 2229
len_ME_close = 1917

len_baronEM_close = 8012
len_E_close = 2229
len_M_close = 2042

len_baronEM_partial = 8569
len_E_partial = 2229
len_M_partial = 2042

len_baronM_close = 8012
len_muraro_close = 2042

len_baronM_partial = 8569
len_muraro_partial = 2042

len_baronE_close = 7487
len_enge_close = 2229

len_baronE_partial = 8569
len_enge_partial = 2229

len_baron = 8296
len_enge = 2282
len_muraro = 2122

len_10x = 1866
len_dropseq = 1509
len_fluidigm = 685

len_spleen = 9552
len_spleen_ = 1697

len_tongue = 7538
len_tongue_ = 1416

len_trachea = 11269
len_trachea_ = 1350

len_bh_c = 5452
len_e_c = 1429
len_m_c = 1453
len_s_c = 673
len_x_c = 576

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], activation="relu"):
        super(Autoencoder, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
        
    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * 1.5)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi
    
class Solver(object):
    def __init__(self, z_dim, encodeLayer, decodeLayer,
                 nfeat, nclasses, sigma, entropy_thr, Lambda_global, Lambda_local, beta, alpha,
                 use_target=1, ndomain=2, highly_variable=2000, batch_size=256, activation='relu', 
                 count_thr=0, unas_thr=2.0, use_filter=False, filter_index=False, 
                 learning_rate=0.002, interval=2, optimizer='adam', checkpoint_dir=None, save_epoch=10):
        
        self.batch_size = batch_size
        self.highly_variable = highly_variable
        self.nfeat = nfeat 
        self.nclasses= nclasses 
        self.sigma = sigma 
        self.entropy_thr = entropy_thr 
        self.Lambda_global = Lambda_global
        self.Lambda_local = Lambda_local
        self.beta = beta 
        self.alpha = alpha 
        self.use_target = use_target
        self.count_thr = count_thr
        self.unas_thr = unas_thr 
        self.use_filter=use_filter
        self.filter_index = filter_index
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.interval = interval
        self.lr = learning_rate
        self.best_correct = 0
        if use_target == 1:
            self.ndomain = ndomain
        else:
            self.ndomain = ndomain - 1

        # load source and target domains
        if self.use_filter:
            print(self.filter_index)
            self.datasets, self.dataset_size, input_size, self.target_size = dataset_read(self.highly_variable, self.batch_size, use_filter=True, filter_index=self.filter_index)
            self.niter = self.dataset_size / self.batch_size
            print('Filtered dataset loaded!')
        else:
            self.datasets, self.dataset_size, input_size, self.target_size = dataset_read(self.highly_variable, self.batch_size)
            self.niter = self.dataset_size / self.batch_size
            print('Dataset loaded!')

        # define the feature extractor and GCN-based classifier  
        self.Autoencoder = Autoencoder(input_size, z_dim, nclasses, encodeLayer, decodeLayer, activation)
        self.GCN = GCN(nfeat=nfeat, nclasses=nclasses)
        self.Autoencoder.cuda()
        self.GCN.cuda()
        print('Model initialized!')

        if self.checkpoint_dir is not None:
            self.state = torch.load(self.checkpoint_dir)
            self.Autoencoder.load_state_dict(self.state['Autoencoder'])
            self.GCN.load_state_dict(self.state['GCN'])
            print('Model load from: ', self.checkpoint_dir)

        # initialize statistics (prototypes and adjacency matrix)
        if self.checkpoint_dir is None:
            self.mean = torch.zeros(self.nclasses * self.ndomain, self.nfeat).cuda()
            self.count = torch.zeros(self.nclasses * self.ndomain, 1).cuda()
            self.adj = torch.zeros(self.nclasses * self.ndomain, self.nclasses * self.ndomain).cuda()
            print('Statistics initialized!')
        else:
            self.mean = self.state['mean'].cuda()
            self.adj = self.state['adj'].cuda()
            print('Statistics loaded!')

        # define the optimizer
        self.set_optimizer(which_opt=optimizer, lr=self.lr)
        print('Optimizer defined!')

    # optimizer definition
    def set_optimizer(self, which_opt='adam', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_autoencoder = optim.SGD(self.Autoencoder.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_gcn = optim.SGD(self.GCN.parameters(),
                                     lr=lr, weight_decay=0.0005,
                                     momentum=momentum)
        elif which_opt == 'adam':
            self.opt_autoencoder = optim.Adam(self.Autoencoder.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_gcn = optim.Adam(self.GCN.parameters(),
                                      lr=lr, weight_decay=0.0005)

    # empty gradients
    def reset_grad(self):
        self.opt_autoencoder.zero_grad()
        self.opt_gcn.zero_grad()

    # compute the discrepancy between two probabilities
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist

    # construct the extended adjacency matrix
    def construct_adj(self, feats):
        dist = self.euclid_dist(self.mean, feats)
        sim = torch.exp(-dist / (2 * self.sigma ** 2))
        E = torch.eye(feats.shape[0]).float().cuda()

        A = torch.cat([self.adj, sim], dim = 1)
        B = torch.cat([sim.t(), E], dim = 1)
        gcn_adj = torch.cat([A, B], dim = 0)

        return gcn_adj

    # assign pseudo labels to target samples
    def pseudo_label(self, logit, feat):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        #print(entropy)
        mask = (entropy < self.entropy_thr).float()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)

        return feat_, label_

    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        curr_count = list()
        num_labels = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels = num_labels + tmp_label.shape[0]
            if domain_idx == self.ndomain-1 and self.use_target==1:
                tmp_label = tmp_label.unsqueeze(-1)

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.nclasses, self.nfeat)).cuda())
                curr_count.append(torch.zeros((self.nclasses,1)).cuda())
            else:
                #print(tmp_label.shape, tmp_label.unsqueeze(-1).shape)
                onehot_label = torch.zeros([tmp_label.shape[0], self.nclasses]).scatter_(1, tmp_label.cpu(), 1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                curr_mean.append(tmp_mean)
                curr_count.append(onehot_label.unsqueeze(-1).sum(0))

        curr_mean = torch.cat(curr_mean, dim = 0)
        curr_count = torch.cat(curr_count, dim = 0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)
        self.mean = self.mean.detach() * (1 - curr_mask) + (
                    self.mean.detach() * self.beta + curr_mean.detach() * (1 - self.beta)) * curr_mask
        self.count = self.count.detach() + curr_count.detach()
        
        curr_dist = self.euclid_dist(self.mean, self.mean)
        self.adj = torch.exp(-curr_dist / (2 * self.sigma ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local

    # compute global relation alignment loss
    def adj_loss(self):
        adj_loss = 0

        for i in range(self.ndomain):
            for j in range(self.ndomain):
                index = []
                for k in range(self.nclasses):
                    if self.mean[k+i*self.nclasses].sum()!=0 and self.count[k+i*self.nclasses].sum()>self.count_thr:
                        if self.mean[k+j*self.nclasses].sum()!=0 and self.count[k+j*self.nclasses].sum()>self.count_thr:
                            index.append(k)
                #print(index)
                adj_ii = self.adj[i * self.nclasses:(i + 1) * self.nclasses,
                         i * self.nclasses:(i + 1) * self.nclasses]
                adj_jj = self.adj[j * self.nclasses:(j + 1) * self.nclasses,
                         j * self.nclasses:(j + 1) * self.nclasses]
                adj_ij = self.adj[i * self.nclasses:(i + 1) * self.nclasses,
                         j * self.nclasses:(j + 1) * self.nclasses]

                adj_loss = adj_loss + ((adj_ii[index,index] - adj_jj[index,index]) ** 2).mean()
                adj_loss = adj_loss + ((adj_ij[index,index] - adj_ii[index,index]) ** 2).mean()
                adj_loss = adj_loss + ((adj_ij[index,index] - adj_jj[index,index]) ** 2).mean()

        adj_loss = adj_loss / (self.ndomain * (self.ndomain - 1) / 2 * 3)

        return adj_loss
    
    # per epoch training in a Domain Generalization setting
    def train_gcn_baseline(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.Autoencoder.train()
        self.GCN.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            cell_s = list()
            raw_list = list()
            sf_list = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain):
                tmp_cell = data['S' + str(domain_idx + 1)].cuda()
                tmp_raw = data['S' + str(domain_idx + 1) + '_raw'].cuda()
                tmp_sf = data['S' + str(domain_idx + 1) + '_sf'].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                cell_s.append(tmp_cell)
                raw_list.append(tmp_raw)
                sf_list.append(tmp_sf)
                label_s.append(tmp_label)

                if tmp_cell.size()[0] < self.batch_size:
                    stop_iter = True
            
            if stop_iter:
                break

            self.reset_grad()

            
            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain):
                tmp_cell = cell_s[domain_idx]
                tmp_raw = raw_list[domain_idx]
                tmp_sf = sf_list[domain_idx]
                tmp_feat, tmp_mean, tmp_disp, tmp_pi = self.Autoencoder(tmp_cell)
                if domain_idx == 0:
                    loss_zinb = self.Autoencoder.zinb_loss(tmp_raw,tmp_mean,tmp_disp,tmp_pi,tmp_sf)
                else:
                    loss_zinb = loss_zinb + self.Autoencoder.zinb_loss(tmp_raw,tmp_mean,tmp_disp,tmp_pi,tmp_sf)
                feat_list.append(tmp_feat)


            # Update the global mean and adjacency matrix
            loss_local = self.update_statistics(feat_list, label_s)
            feats = torch.cat(feat_list, dim=0)
            labels = torch.cat(label_s, dim=0)

            # add query samples to the domain graph
            gcn_feats = torch.cat([self.mean, feats], dim=0)
            gcn_adj = self.construct_adj(feats)

            # output classification logit with GCN
            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            # define GCN classification losses
            domain_logit = gcn_logit[:self.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(self.nclasses)] * self.ndomain, dim=0)
            domain_label = domain_label.long().cuda()
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[self.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit, labels.squeeze())

            loss_cls = loss_cls_src + loss_cls_dom

            # define relation alignment losses
            loss_global = self.adj_loss() * self.Lambda_global
            loss_local = loss_local * self.Lambda_local
            loss_relation = loss_local + loss_global

            loss = self.alpha*loss_zinb + loss_cls + loss_relation

            # back-propagation
            loss.backward()
            self.opt_gcn.step()
            self.opt_autoencoder.step()

            '''
            # record training information
            if epoch == 0 and batch_idx == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()
            '''

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_zinb: {:.5f}\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_zinb.item(), loss_cls_dom.item(), loss_cls_src.item(), loss_global.item(), loss_local.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_zinb.item(), loss_cls_dom.item(), loss_cls_src.item(), loss_global.item(), loss_local.item()))
                    record.close()

        return batch_idx

    # per epoch training in a Multi-Source Domain Adaptation setting
    def train_gcn_adapt(self, epoch, pretrain=False, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.Autoencoder.train()
        self.GCN.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            cell_s = list()
            raw_list = list()
            sf_list = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain-1):
                tmp_cell = data['S' + str(domain_idx + 1)].cuda()
                tmp_raw = data['S' + str(domain_idx + 1) + '_raw'].cuda()
                tmp_sf = data['S' + str(domain_idx + 1) + '_sf'].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                cell_s.append(tmp_cell)
                raw_list.append(tmp_raw)
                sf_list.append(tmp_sf)
                label_s.append(tmp_label)

                if tmp_cell.size()[0] < self.batch_size:
                    stop_iter = True


            if stop_iter:
                break

            # get the target batch
            cell_t = data['T'].cuda()
            raw_t = data['T' + '_raw'].cuda()
            sf_t = data['T' + '_sf'].cuda()
            raw_list.append(raw_t)
            sf_list.append(sf_t)
            if cell_t.size()[0] < self.batch_size:
                break
                
            self.reset_grad()
            

            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_cell = cell_s[domain_idx]
                tmp_raw = raw_list[domain_idx]
                tmp_sf = sf_list[domain_idx]
                tmp_feat, tmp_mean, tmp_disp, tmp_pi = self.Autoencoder(tmp_cell)
                if domain_idx == 0:
                    loss_zinb = self.Autoencoder.zinb_loss(tmp_raw,tmp_mean,tmp_disp,tmp_pi,tmp_sf)
                else:
                    loss_zinb = loss_zinb + self.Autoencoder.zinb_loss(tmp_raw,tmp_mean,tmp_disp,tmp_pi,tmp_sf)
                feat_list.append(tmp_feat)
                

            tmp_raw = raw_list[self.ndomain - 1]
            tmp_sf = sf_list[self.ndomain - 1]
            feat_t, mean_t, disp_t, pi_t = self.Autoencoder(cell_t)
            loss_zinb = loss_zinb + self.Autoencoder.zinb_loss(tmp_raw,mean_t,disp_t,pi_t,tmp_sf)
            feat_list.append(feat_t)
            
            feats = torch.cat(feat_list, dim=0)
            labels = torch.cat(label_s, dim=0)
            
            # add query samples to the domain graph

            gcn_feats = torch.cat([self.mean, feats], dim=0)
            gcn_adj = self.construct_adj(feats)


            # output classification logit with GCN
            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            
            # predict the psuedo labels for target domain
            feat_t_, label_t_ = self.pseudo_label(gcn_logit[-feat_t.shape[0]:, :], feat_t)
            feat_list.pop()
            feat_list.append(feat_t_)
            label_s.append(label_t_)

            # update the statistics for source and target domains
            loss_local = self.update_statistics(feat_list, label_s)

            # define GCN classification losses
            domain_logit = gcn_logit[:self.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(self.nclasses)] * self.ndomain, dim=0)
            domain_label = domain_label.long().cuda()
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[self.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit[:-feat_t.shape[0]], labels.squeeze())

            target_logit = query_logit[-feat_t.shape[0]:]
            target_prob = F.softmax(target_logit, dim=1)
            loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean()

            loss_cls = loss_cls_dom + loss_cls_src + loss_cls_tgt
            
            if loss_cls_dom.item() > 1000:
                break

            # define relation alignment losses
            loss_global = self.adj_loss() * self.Lambda_global
            loss_local = loss_local * self.Lambda_local
            loss_relation = loss_local + loss_global

            if pretrain == False:
                loss = self.alpha*loss_zinb + loss_cls + loss_relation
            else:
                loss = loss_zinb

            # back-propagation
            loss.backward(retain_graph = True)
            self.opt_gcn.step()
            self.opt_autoencoder.step()

            # record training information
            #if epoch ==0 and batch_idx==0:
                #record = open(record_file, 'a')
                #record.write(str(self.args)+'\n')
                #record.close()
            

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_zinb: {:.5f}\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_target: {:.5f}\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_zinb.item(), loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt.item(),
                        loss_global.item(), loss_local.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_zinb: {:.5f}\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_target: {:.5f}\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_zinb.item(), loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt.item(),
                            loss_global.item(), loss_local.item()))
                    record.close()
        return batch_idx

    # per epoch test on target domain
    def test(self, epoch, record_file=None, save_model=False):
        self.Autoencoder.eval()
        self.GCN.eval()

        test_loss = 0
        correct = 0
        size = 0
        
        pred_list = list()
        label_list = list()
        
        cluster_false = torch.zeros(self.nclasses).cuda()
        cluster_size = torch.zeros(self.nclasses).cuda()
        

        for batch_idx, data in enumerate(self.datasets):
            if batch_idx > self.target_size/self.batch_size:
                break
            cell = data['T']
            label = data['T_label']
            cell, label = cell.cuda(), label.long().cuda()
            labels = label.squeeze()

            feat,_,_,_ = self.Autoencoder(cell)

            gcn_feats = torch.cat([self.mean, feat], dim=0)
            gcn_adj = self.construct_adj(feat)
            gcn_logit = self.GCN(gcn_feats, gcn_adj)
            output = gcn_logit[self.mean.shape[0]:, :]

            #test_loss += -F.nll_loss(output, labels).item()
            pred = output.max(1)[1]
            pred_list.append(pred)
            label_list.append(labels)
            k = labels.size()[0]
            correct += pred.eq(labels).cpu().sum()
            size += k
            

        test_loss = test_loss / size

        if correct > self.best_correct:
            self.best_correct = correct
            self.best_pred = torch.cat(pred_list, dim=0)
            if save_model:
                best_state = {'Autoencoder': self.Autoencoder.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.mean.cpu(),
                              'adj': self.adj.cpu(), 'epoch': epoch}
                torch.save(best_state, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        # save checkpoint
        if save_model and epoch % self.save_epoch == 0:
            state = {'Autoencoder': self.Autoencoder.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.mean.cpu(),
                     'adj': self.adj.cpu()}
            torch.save(state, os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))

        # record test information
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
                test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
                                          100. * float(self.best_correct) / size))
        #print(cluster_false, cluster_size, cluster_false/cluster_size)

        if record_file:
            if epoch == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write(
                '\nEpoch {:>3} Average loss: {:.5f}, Accuracy: {:.5f}, Best Accuracy: {:.5f}'.format(
                    epoch, test_loss, 100. * float(correct) / size, 100. * float(self.best_correct) / size))
            record.close()
        
        pred_label = torch.cat(pred_list, dim=0)
        test_label = torch.cat(label_list, dim=0)
        
        return pred_label.cpu(), test_label.cpu(), 100. * float(self.best_correct) / size
    
    def open_filter(self):
        self.Autoencoder.eval()
        self.GCN.eval()

        t,t_raw,t_sf,t_label = dataset_read(self.highly_variable, self.batch_size, open_case=True)
        feat_t,_,_,_ = self.Autoencoder(t)
        
        gcn_feats_t = torch.cat([self.mean, feat_t], dim=0)
        gcn_adj_t = self.construct_adj(feat_t)
        gcn_logit_t = self.GCN(gcn_feats_t, gcn_adj_t)
        output_t = gcn_logit_t[self.mean.shape[0]:, :]
        
        pred_t = F.softmax(output_t, dim=1)
        entropy_t = (-pred_t * torch.log(pred_t)).sum(-1)
        print(entropy_t.detach().cpu().numpy())
        mask_t = (entropy_t < self.unas_thr).float()
        index_t = torch.nonzero(mask_t).squeeze(-1)
        
        self.filter_index = index_t.cpu().numpy()
        
        return index_t.cpu().numpy()  
        
        
    def test_feature(self, file_path):
        self.Autoencoder.eval()
        self.GCN.eval()

        feature_list = []
        S0, S1, S2, T = dataset_read(self.highly_variable, self.batch_size, use_latent=True)
        T, labels = dataset_read(self.highly_variable, self.batch_size, use_cluster=True)

        feat_S0,_,_,_ = self.Autoencoder(S0)
        feat_S1,_,_,_ = self.Autoencoder(S1)
        feat_S2,_,_,_ = self.Autoencoder(S2)
        feat_T,_,_,_ = self.Autoencoder(T)
        
        df_s0 = pd.DataFrame(feat_S0.detach().cpu().numpy())
        df_s0.to_csv(os.path.join(file_path+"source_0_feature.csv"))
        df_s1 = pd.DataFrame(feat_S1.detach().cpu().numpy())
        df_s1.to_csv(os.path.join(file_path+"source_1_feature.csv"))
        df_s2 = pd.DataFrame(feat_S2.detach().cpu().numpy())
        df_s2.to_csv(os.path.join(file_path+"source_2_feature.csv"))
        df_t = pd.DataFrame(feat_T.detach().cpu().numpy())
        df_t.to_csv(os.path.join(file_path+"target_feature.csv"))
        df_l = pd.DataFrame(labels)
        df_l.to_csv(os.path.join(file_path+"target_labels.csv"))