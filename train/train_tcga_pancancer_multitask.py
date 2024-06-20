import os
import sys
from setproctitle import setproctitle
setproctitle("CLUE")
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import CancerDataset, MetaBric_Dataset, MetastaticDataset, CPTAC_BRCA_Dataset
from model.clue_model import Clue_model, DownStream_predictor, dfs_freeze, un_dfs_freeze
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from util.loss_function import cox_loss, c_index
from lifelines.utils import concordance_index
import pickle
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from LogME import LogME
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
logme = LogME(regression=False)
from typing import Optional, Sequence
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 3.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        # log_p = F.softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(66)
### ===================================================================================
### TCGA
omics_files = [
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]
clinical_file = '../data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv'
train_index_path = '../data/TCGA_PanCancer_Data_cleaned/train_data.csv'
test_index_path = '../data/TCGA_PanCancer_Data_cleaned/test_data.csv'
### ===================================================================================
### Metastatic
split_path = "../data/metastasis prediction/data/five_split.pkl"
label_path = "../data/metastasis prediction/data/label.pkl"
prim_omics_files = [
    '../data/metastasis prediction/data/pancancer_primary_mrna_new_norm.csv',
    '../data/metastasis prediction/data/pancancer_primary_dna_new.csv']
meta_omics_files = [
    '../data/metastasis prediction/data/pancancer_metastatic_mrna_new_norm.csv',
    '../data/metastasis prediction/data/pancancer_metastatic_dna_new.csv'
]
### ===================================================================================
### CPTAC 
CPTAC_omics_files = [
    '../data/CPTAC/BRCA/RNA.csv',
    '../data/CPTAC/BRCA/MUT.csv',
]
CPTAC_label_path = "../data/CPTAC/BRCA/CLI.csv"
CPTAC_split_path = "../data/CPTAC/BRCA/five_split.pkl"
### ===================================================================================
### Metabric
train_omics_files = [
    '../data/moanna_data/tcga_matched_data/training/training_moanna_exp_normalized.csv',
    '../data/moanna_data/tcga_matched_data/training/training_moanna_mut.csv',
    '../data/moanna_data/tcga_matched_data/training/training_moanna_cnv.csv',
]
val_omics_files = [
    '../data/moanna_data/tcga_matched_data/validation/validation_moanna_exp_normalized.csv',
    '../data/moanna_data/tcga_matched_data/validation/validation_moanna_mut.csv',
    '../data/moanna_data/tcga_matched_data/validation/validation_moanna_cnv.csv',
]
test_omics_files = [
    '../data/moanna_data/tcga_matched_data/testing/testing_moanna_exp_normalized.csv',
    '../data/moanna_data/tcga_matched_data/testing/testing_moanna_mut.csv',
    '../data/moanna_data/tcga_matched_data/testing/testing_moanna_cnv.csv',
]
train_label_path = '../data/moanna_data/tcga_matched_data/training/moanna_training_label.tsv'
val_label_path = '../data/moanna_data/tcga_matched_data/validation/moanna_validation_label.tsv'
test_label_path = '../data/moanna_data/tcga_matched_data/testing/moanna_testing_label.tsv'
### ===================================================================================
omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']
omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}
incomplete_omics = {'gex': 0, 'methy': 1}

#   pretrain
def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer, fold, pretrain_omics):
    model.train()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} training-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_ad_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()

    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time,  omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()
            all_label = torch.concat((all_label, cancer_label), dim=0)

            input_x = []
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic)
                input_x.append(omic)
            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)
            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0), pretrain_omics)

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()
            multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)

            pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
            contrastive_loss = model.contrastive_loss(multi_embedding, cancer_label)
            loss += contrastive_loss
            # loss = ce_loss
            total_dsc_loss += dsc_loss.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                               cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

        print('total loss: ', total_loss / len(train_dataloader))
        Loss.append(total_loss / len(train_dataloader))
        print('self elbo loss: ', total_self_elbo / len(train_dataloader))
        Loss.append(total_self_elbo / len(train_dataloader))
        print('cross elbo loss: ', total_cross_elbo / len(train_dataloader))
        Loss.append(total_cross_elbo / len(train_dataloader))
        print('cross infer loss: ', total_cross_infer_loss / len(train_dataloader))
        Loss.append(total_cross_infer_loss / len(train_dataloader))
        print('ad loss', total_ad_loss / len(train_dataloader))
        print('dsc loss', total_dsc_loss / len(train_dataloader))
        Loss.append(total_dsc_loss / len(train_dataloader))

        torch.save(pancancer_embedding, f'../model/model_dict/TCGA_pancancer_multi_train_embedding_fold{fold}_epoch{epoch}_v2.pt')
        torch.save(all_label, f'../model/model_dict/TCGA_pancancer_train_fold{fold}_epoch{epoch}_all_label_v2.pt')

        pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
        print('pretrain score:', pretrain_score)
        return Loss, pretrain_score


def val_pretrain(test_dataloader, model, epoch, cancer, fold, pretrain_omics):
    model.eval()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} val-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()
                all_label = torch.concat((all_label, cancer_label), dim=0)
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    # print(omic.size())
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0), pretrain_omics)
                multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)

                pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
                total_self_elbo += self_elbo.item()
                total_cross_elbo += cross_elbo.item()
                total_cross_infer_loss += cross_infer_loss.item()

                total_dsc_loss += dsc_loss.item()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                                   cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

            print('test total loss: ', total_loss / len(test_dataloader))
            Loss.append(total_loss / len(test_dataloader))
            print('test self elbo loss: ', total_self_elbo / len(test_dataloader))
            Loss.append(total_self_elbo / len(test_dataloader))
            print('test cross elbo loss: ', total_cross_elbo / len(test_dataloader))
            Loss.append(total_cross_elbo / len(test_dataloader))
            print('test cross infer loss: ', total_cross_infer_loss / len(test_dataloader))
            Loss.append(total_cross_infer_loss / len(test_dataloader))
            print('test ad loss', total_cross_infer_dsc_loss / len(test_dataloader))
            print('test dsc loss', total_dsc_loss / len(test_dataloader))
            Loss.append(total_dsc_loss / len(test_dataloader))
            torch.save(pancancer_embedding,
                       f'../model/model_dict/TCGA_pancancer_multi_test_embedding_fold{fold}_epoch{epoch}_v2.pt')
            torch.save(all_label, f'../model/model_dict/TCGA_pancancer_test_fold{fold}_epoch{epoch}_all_label_v2.pt')
    return Loss


def TCGA_Dataset_pretrain(fold, epochs, device_id, cancer_types=None):
    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                  fold + 1)
    test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path, fold + 1)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = Clue_model(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512], omics_data_type)
    torch.cuda.set_device(device_id)
    model.cuda()
    print(len(train_dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)

    Loss_list = []
    test_Loss_list = []
    pretrain_score_list = []
    for epoch in range(epochs):

        start_time = time.time()
        # if epoch > 7:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.000001
        loss, pretrain_score = train_pretrain(train_dataloader, model, epoch, 'PanCancer', optimizer, dsc_optimizer, fold, omics)
        Loss_list.append(loss)
        pretrain_score_list.append(pretrain_score)

        test_Loss_list.append(val_pretrain(test_dataloader, model, epoch, 'PanCancer', fold, omics))
        print(f'fold{fold} time used: ', time.time() - start_time)

    model_dict = model.state_dict()
    Loss_list = torch.Tensor(Loss_list)
    test_Loss_list = torch.Tensor(test_Loss_list)
    pretrain_score_list = pd.DataFrame(pretrain_score_list, columns=['pretrain_score'])
    pretrain_score_list.to_csv(f'pretrain_score_list_fold{fold}.csv')
    torch.save(test_Loss_list, f'../model/model_dict/TCGA_pancancer_pretrain_test_loss_fold{fold}.pt')
    torch.save(Loss_list, f'../model/model_dict/TCGA_pancancer_pretrain_train_loss_fold{fold}.pt')
    torch.save(model_dict, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{fold}_dim64.pt')


#   survival prediction
def train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics):
    model.train()
    print(f'-----start {cancer} epoch {epoch} training-----')
    total_loss = 0
    train_risk_score = torch.Tensor([]).cuda()
    train_censors = torch.Tensor([]).cuda()
    train_event_times = torch.Tensor([]).cuda()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, _ = data
            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            risk_score = model(input_x, os_event.size(0), omics)
            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0), omics)
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            loss = CoxLoss + 0.6 * pretrain_loss
            total_loss += CoxLoss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=CoxLoss.item())

        print('cox loss: ', total_loss / len(train_dataloader))
        train_c_index = concordance_index(train_event_times.detach().cpu().numpy(),
                                          -train_risk_score.detach().cpu().numpy(),
                                          train_censors.detach().cpu().numpy())

        print(f'{cancer} train survival c-index: ', train_c_index)


def cal_c_index(dataloader, model, best_c_index, best_c_indices, omics):
    model.eval()
    best_c_index_list = best_c_indices
    device = next(model.parameters()).device

    with torch.no_grad():
        risk_scores = {
            'all': torch.zeros((len(dataloader)), device=device),
            'gex': torch.zeros((len(dataloader)), device=device),
            'mut': torch.zeros((len(dataloader)), device=device),
            'cnv': torch.zeros((len(dataloader)), device=device),
            'gex_cnv': torch.zeros((len(dataloader)), device=device),
            'gex_mut': torch.zeros((len(dataloader)), device=device),
            'cnv_mut': torch.zeros((len(dataloader)), device=device),
        }

        censors = torch.zeros((len(dataloader)))
        event_times = torch.zeros((len(dataloader)))

        for i, data in enumerate(dataloader):
            os_event, os_time, omics_data, cancer_label = data
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            os_event = os_event.cuda()
            os_time = os_time.cuda()

            survival_risk = model(input_x, os_event.size(0), omics)

            gex_survival_risk = model(input_x, os_event.size(0), {'gex': 0})
            mut_survival_risk = model([input_x[2]], os_event.size(0), {'mut': 2})
            cnv_survival_risk = model([input_x[3]], os_event.size(0), {'cnv': 3})

            gex_cnv_survival_risk = model([input_x[0], input_x[3]], os_event.size(0), {'gex': 0, 'cnv': 3})
            gex_mut_survival_risk = model([input_x[0], input_x[2]], os_event.size(0), {'gex': 0, 'mut': 2})
            cnv_mut_survival_risk = model([input_x[2], input_x[3]], os_event.size(0), {'mut': 2, 'cnv': 3})

            risk_scores['gex'][i] = gex_survival_risk
            risk_scores['mut'][i] = mut_survival_risk
            risk_scores['cnv'][i] = cnv_survival_risk
            risk_scores['gex_mut'][i] = gex_mut_survival_risk
            risk_scores['gex_cnv'][i] = gex_cnv_survival_risk
            risk_scores['cnv_mut'][i] = cnv_mut_survival_risk
            risk_scores['all'][i] = survival_risk

            censors[i] = os_event
            event_times[i] = os_time

        c_indices = {}
        for key in risk_scores.keys():
            c_indices[key] = concordance_index(event_times.cpu().numpy(), -risk_scores[key].cpu().numpy(),
                                               censors.cpu().numpy())

        if c_indices['all'] > best_c_index:
            best_c_index = c_indices['all']
            best_c_index_list = [c_indices[key] for key in c_indices.keys()]

        print(f'test survival c-index: ', c_indices)

    return best_c_index_list, best_c_index


def TCGA_Dataset_survival_prediction(fold, epochs, cancer_types, pretrain_model_path, fixed, device_id):
    pancancer_c_index = []
    for cancer in cancer_types:
        print(cancer)
        train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                      fold + 1, [cancer])
        test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path,
                                     fold + 1, [cancer])

        train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        task = {'output_dim': 1}
        model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                     pretrain_model_path, task, omics_data_type, fixed)
        torch.cuda.set_device(device_id)
        model.cuda()
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 0.0001},
            {'params': model.downstream_predictor.parameters(), 'lr': 0.00001},

        ]
        optimizer = torch.optim.Adam(param_groups)
        best_c_index = 0
        best_c_indices = []
        for epoch in range(epochs):
            # adjust_learning_rate(optimizer, epoch, 0.0001)
            start_time = time.time()
            train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
            best_c_indices, best_c_index = cal_c_index(test_dataloader, model, best_c_index, best_c_indices, omics)
            print(f'{fold} time used: ', time.time() - start_time)
        best_c_indices.insert(0, cancer)
        pancancer_c_index.append(best_c_indices)

        #   clean memory of gpu cuda
        del model
        del optimizer
        del train_dataloader
        del test_dataloader
        torch.cuda.empty_cache()

    pancancer_c_index = pd.DataFrame(pancancer_c_index,
                                     columns=['cancer', 'multiomics', 'gex', 'mut',
                                              'cnv', 'gex_cnv', 'gex_mut', 'cnv_mut'])

    pancancer_c_index.to_csv(f'all_pancancer_pretrain_cross_encoders_c_index_fold{fold}_all_omics.csv',
                             encoding='utf-8')


#   cancer classification
def train_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.train()
    pancancer_embedding = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            classification_pred = model([input_x[0], input_x[1], input_x[3]], os_event.size(0), {'gex': 0, 'methy': 1, 'cna': 3})
            _, labels_pred = torch.max(classification_pred, 1)

            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss([input_x[0], input_x[1], input_x[3]], os_event.size(0), {'gex': 0, 'methy': 1, 'cna': 3})
            pred_loss = criterion(classification_pred, cancer_label)
            total_loss += pred_loss.item()
            loss = pred_loss

            embedding_tensor = model.cross_encoders.get_embedding(input_x, os_event.size(0), omics)
            pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())

            tepoch.set_postfix(loss=pred_loss.item())

        # Calculate accuracy, precision, recall and F1 score
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print('fold {} train:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'
              .format(fold, total_loss / len(dataloader), acc, precision, recall, f1))
        all_labels = torch.Tensor(all_labels)
        torch.save(all_labels, f'train_pancancer_fold{fold}_label.pt')
    torch.save(pancancer_embedding, f'train_pancancer_fold{fold}_embedding.pt')


def test_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.eval()
    pancancer_embedding = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tepoch:
            total_samples = 0
            all_labels = []
            all_predictions = []

            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()

                input_x = [omics_data[key].cuda() for key in omics_data.keys()]

                embedding_tensor = model.cross_encoders.get_embedding(input_x[:2], os_event.size(0), {'gex': 0, 'methy': 1})
                pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

                classification_pred = model([input_x[0], input_x[1], input_x[3]], os_event.size(0), {'gex': 0, 'methy': 1, 'cna': 3})
                _, labels_pred = torch.max(classification_pred, 1)

                pred_loss = criterion(classification_pred, cancer_label)
                total_loss += pred_loss.item()

                total_samples += cancer_label.size(0)
                all_labels.extend(cancer_label.tolist())
                # print('test pred label', labels_pred)
                all_predictions.extend(labels_pred.tolist())

                tepoch.set_postfix(loss=pred_loss.item())

            # Calculate accuracy, precision, recall and F1 score
            acc = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

            print('fold {} test:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.
                  format(fold, total_loss / len(dataloader), acc, precision, recall, f1))
            all_labels = torch.Tensor(all_labels)

            torch.save(all_labels, f'test_pancancer_fold{fold}_label.pt')
            torch.save(pancancer_embedding, f'test_pancancer_fold{fold}_embedding.pt')
            return acc, precision, recall, f1


def TCGA_Dataset_classification(fold, epochs, pretrain_model_path, fixed, device_id):
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                  fold + 1, )
    test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path,
                                 fold + 1)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    task = {'output_dim': 32}

    model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                 pretrain_model_path, task, omics_data_type, fixed)

    torch.cuda.set_device(device_id)
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.00001},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},
    ]

    optimizer = torch.optim.Adam(param_groups)
    best_acc = 0
    classification_score = []
    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, 0.0001)

        start_time = time.time()
        train_classification(train_dataloader, model, epoch, 'pancancer', fold, optimizer, omics, criterion)

        acc, precision, recall, f1 = test_classification(test_dataloader, model, epoch, 'pancancer', fold, optimizer, omics, criterion)
        if acc > best_acc:
            classification_score = [[fold, acc, precision, recall, f1]]
        classification_score = pd.DataFrame(classification_score, columns=['fold', 'acc', 'precision', 'recall', 'f1'])
        classification_score.to_csv(f'classification_score_fold{fold}.csv')
        print(f'fold {fold} time used: ', time.time() - start_time)

    #   clean memory of gpu cuda
    del model
    del optimizer
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()


def metastatic_test_classification(dataloader, model, epoch, cancer, fold, optimizer, omics_, criterion):
    total_loss = 0
    model.eval()
    pancancer_embedding = torch.Tensor([]).cuda()
    prob_logits = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()

            input_x = []
            for key in omics_data.keys():
                if key in omics_.keys():
                    input_x.append(omics_data[key].cuda())
            
            classification_pred = model(input_x, cancer_label.size(0), omics_)
            embedding_tensor = model.cross_encoders.get_embedding(input_x, cancer_label.size(0), omics_)
            pancancer_embedding = torch.cat((pancancer_embedding, embedding_tensor), dim=0)
            prob_logits = torch.cat((prob_logits,classification_pred.detach()))
            _, labels_pred = torch.max(classification_pred, 1)
            # print(classification_pred)
            pred_loss = criterion(classification_pred, cancer_label)
            total_loss += pred_loss.item()

            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())
            
            tepoch.set_postfix(loss=pred_loss.item())
        

        # Calculate accuracy, precision, recall and F1 score
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print('fold {:d}: test:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(fold,total_loss, acc,
                                                                                                   precision,
                                                                                                   recall, f1))


def metastatic_train_classification(dataloader, model, epoch, cancer, fold, optimizer, omics_, criterion):
    total_loss = 0
    model.train()
    pancancer_embedding = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()

            input_x = []
            for key in omics_data.keys():
                if key in omics_.keys():
                    input_x.append(omics_data[key].cuda())
            
            classification_pred = model(input_x, cancer_label.size(0), omics_)
            _, labels_pred = torch.max(classification_pred, 1)

            pred_loss = criterion(classification_pred, cancer_label)
            loss = pred_loss

            embedding_tensor = model.cross_encoders.get_embedding(input_x, cancer_label.size(0), omics_)
            pancancer_embedding = torch.cat((pancancer_embedding, embedding_tensor), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += pred_loss.item()
            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())

            tepoch.set_postfix(loss=pred_loss.item())

        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print('fold {:d}: train:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(fold, total_loss, acc,
                                                                                                    precision,
                                                                                                    recall, f1))
        all_labels = torch.Tensor(all_labels)
        torch.save(all_labels, f'pancancer_label_{fold}.pt')
    torch.save(pancancer_embedding, f'pancancer_embedding_{fold}.pt')


def Metastatic_Dataset_classification(fold, epochs, pretrain_model_path, fixed, device_id):
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = MetastaticDataset(prim_omics_files,meta_omics_files, ['gex', 'methy'], split_path, fold, label_path)
    test_dataset = MetastaticDataset(prim_omics_files,meta_omics_files, ['gex', 'methy'], split_path, fold, label_path, is_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    task = {'output_dim': 2}

    model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                 pretrain_model_path, task, omics_data_type, fixed, omics)
                                 
    print('load done.')
    torch.cuda.set_device(device_id)
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.0001},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},
    ]

    optimizer = torch.optim.Adam(param_groups)
    start_time = time.time()
    for epoch in range(epochs):
    
        metastatic_train_classification(train_dataloader, model, epoch, 'pancancer', fold, optimizer, incomplete_omics, criterion)
        
        with torch.no_grad():
            metastatic_test_classification(test_dataloader, model, epoch, 'pancancer', fold, optimizer, incomplete_omics, criterion)
    torch.save(model.state_dict(),os.path.join('../model/model_dict',f'two_cls_metastatic_fold_{fold}.pt'))
    
    print(f'{fold} time used: ', time.time() - start_time)
    #   clean memory of gpu cuda
    del model
    del optimizer
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()


def metabric_test_classification(taski, dataloader, model, epoch, cancer, fold, optimizer, omics, criterion, is_test=True):
    total_loss = 0
    model.eval()
    pancancer_embedding = torch.Tensor([]).cuda()
    prob_logits = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            omics_data, cancer_label0, cancer_label1, cancer_label2 = data
            
            cancer_labels = [cancer_label0, cancer_label1, cancer_label2]
            cancer_label = cancer_labels[taski].cuda()
            cancer_label = cancer_label.squeeze()

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            classification_pred = model(input_x, cancer_label.size(0), incomplete_omics)
            embedding_tensor = model.cross_encoders.get_embedding(input_x, cancer_label.size(0), incomplete_omics)
            pancancer_embedding = torch.cat((pancancer_embedding, embedding_tensor), dim=0)
            prob_logits = torch.cat((prob_logits,classification_pred.detach()))
            _, labels_pred = torch.max(classification_pred, 1)

            pred_loss = criterion(classification_pred, cancer_label)
            total_loss += pred_loss.item()

            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())

            tepoch.set_postfix(loss=pred_loss.item())

        # Calculate accuracy, precision, recall and F1 score
        print(sum(all_labels)/len(all_labels), all_predictions)
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        if is_test:
            print('fold {:d}: test:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(fold,total_loss, acc,
                                                                                                   precision,
                                                                                                   recall, f1))
        else:
            print('fold {:d}: val:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(fold,total_loss, acc,
                                                                                                   precision,
                                                                                                   recall, f1))
        
        os.makedirs('../train/metabric',exist_ok=True)
        if is_test:
            torch.save(all_labels, f'./metabric/test_metabric_label_task_{taski}_{fold}.pt')
            torch.save(prob_logits, f'./metabric/test_metabric_prob_task_{taski}_{fold}.pt')
            torch.save(pancancer_embedding, f'./metabric/test_metabric_embedding_task_{taski}_{fold}.pt')
        else:
            torch.save(all_labels, f'./metabric/val_metabric_label_task_{taski}_{fold}.pt')
            torch.save(prob_logits, f'./metabric/val_metabric_prob_task_{taski}_{fold}.pt')
            torch.save(pancancer_embedding, f'./metabric/val_metabric_embedding_task_{taski}_{fold}.pt')


def metabric_train_classification(taski, dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.train()
    pancancer_embedding = torch.Tensor([]).cuda()
    prob_logits = torch.Tensor([]).cuda()
    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            omics_data, cancer_label0, cancer_label1, cancer_label2 = data
            
            cancer_labels = [cancer_label0, cancer_label1, cancer_label2]
            cancer_label = cancer_labels[taski].cuda()
            cancer_label = cancer_label.squeeze()

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            classification_pred = model(input_x, cancer_label.size(0), incomplete_omics)
            _, labels_pred = torch.max(classification_pred, 1)

            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, cancer_label.size(0),incomplete_omics)
            embedding_tensor = model.cross_encoders.get_embedding(input_x, cancer_label.size(0), incomplete_omics)

            if len(cancer_label.unique())==2 or taski==2:
                pred_loss = criterion(classification_pred, cancer_label)
                
                loss = pred_loss + 0.1 * pretrain_loss
                total_loss += pred_loss.item()
                loss = pred_loss 

                embedding_tensor = model.cross_encoders.get_embedding(input_x, cancer_label.size(0), incomplete_omics)
                pancancer_embedding = torch.cat((pancancer_embedding, embedding_tensor), dim=0)
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prob_logits = torch.cat((prob_logits,classification_pred.detach()))
                total_samples += cancer_label.size(0)
                all_labels.extend(cancer_label.tolist())
                all_predictions.extend(labels_pred.tolist())

                tepoch.set_postfix(loss=pred_loss.item())

        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print('fold {:d}: train:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(fold, total_loss, acc,
                                                                                                    precision,
                                                                                                    recall, f1))
        all_labels = torch.Tensor(all_labels)


def MetaBric_classification(fold, epochs, pretrain_model_path, fixed, device_id):
    torch.cuda.set_device(device_id)
    criterion = {
                 'ERStatus': FocalLoss(alpha=torch.Tensor([0.4,0.6])).cuda(),
                 'HER2Status': FocalLoss(alpha=torch.Tensor([0.5,0.5])).cuda(),
                 'Pam50Subtype': FocalLoss(alpha=torch.Tensor([0.1918, 0.2308, 0.3319, 0.4032])).cuda(),
                 'BasalNonBasal': FocalLoss(alpha=torch.Tensor([0.80,0.20])).cuda(),
                }
    print(criterion)
    # cna 7460
    # gex 6016 
    # mut 4539
    train_dataset = MetaBric_Dataset(train_omics_files,['gex','mut','cna'],train_label_path) 
    val_dataset = MetaBric_Dataset(val_omics_files,['gex','mut','cna'],val_label_path)
    test_dataset = MetaBric_Dataset(test_omics_files,['gex','mut','cna'],test_label_path)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    tasks = {'ERStatus':{'output_dim': 2},
            'HER2Status':{'output_dim': 2},
            'Pam50Subtype':{'output_dim': 4},
            'BasalNonBasal':{'output_dim': 2}
            }
    
    cnt = 0 
    for ti,task in tasks.items():
        print(ti)
        model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                 pretrain_model_path, task, omics_data_type, fixed)
        
        model.cuda()
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 8e-5, 'weight_decay':3e-4},
            {'params': model.downstream_predictor.parameters(), 'lr': 8e-5, 'weight_decay':3e-4},
        ]
        optimizer = torch.optim.Adam(param_groups)

        for epoch in range(epochs):
            start_time = time.time()
            metabric_train_classification(cnt, train_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti])
            with torch.no_grad():
                metabric_test_classification(cnt, val_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti], is_test=False)
                metabric_test_classification(cnt, test_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti], is_test=True)
            print(f'{fold} time used: ', time.time() - start_time)

        torch.save(model.state_dict(),f'../model/model_dict/metabric_task_{ti}_fold_{fold}.pt')
        del model
        del optimizer
        torch.cuda.empty_cache()
        cnt += 1


def MetaBric_CPTAC_classification(fold,epochs,pretrain_model_path, fixed, device_id):
    torch.cuda.set_device(device_id)
    criterion = {'ERStatus': FocalLoss(alpha=torch.Tensor([0.5,0.5])).cuda(),
                 'HER2Status': FocalLoss(alpha=torch.Tensor([0.9,0.15,0.76])).cuda(),
                 'Pam50Subtype': FocalLoss(alpha=torch.Tensor([0.1, 0.1657, 0.3319, 0.4032])).cuda(),
                }
    train_dataset = CPTAC_BRCA_Dataset(CPTAC_omics_files, ['gex','mut'], CPTAC_label_path, CPTAC_split_path, fold, mode='train')  
    val_dataset = CPTAC_BRCA_Dataset(CPTAC_omics_files, ['gex','mut'], CPTAC_label_path, CPTAC_split_path, fold, mode='val')  
    test_dataset = CPTAC_BRCA_Dataset(CPTAC_omics_files, ['gex','mut'], CPTAC_label_path, CPTAC_split_path, fold, mode='test') 

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    tasks = {
                'ERStatus':{'output_dim': 2},
                'HER2Status':{'output_dim': 3},
                'Pam50Subtype':{'output_dim': 4}
            }
    cnt = 0
    for ti,task in tasks.items():
        print(ti," Start!!---------------------")
        model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                 pretrain_model_path, task, omics_data_type, fixed)
        
        model.cuda()
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 8e-5, 'weight_decay':3e-4},
            {'params': model.downstream_predictor.parameters(), 'lr': 8e-5, 'weight_decay':3e-4},
        ]
        optimizer = torch.optim.Adam(param_groups)

        for epoch in range(epochs):
            start_time = time.time()
            metabric_train_classification(cnt, train_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti])
            with torch.no_grad():
                metabric_test_classification(cnt, val_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti], is_test=False)
                metabric_test_classification(cnt, test_dataloader, model, epoch, 'metabric', fold, optimizer, incomplete_omics, criterion[ti], is_test=True)
            print(f'{fold} time used: ', time.time() - start_time)


        torch.save(model.state_dict(),f'../model/model_dict/CPTAC_metabric_task_{ti}_fold_{fold}.pt')
        del model
        del optimizer
        torch.cuda.empty_cache()
        cnt += 1


def metabric_test_classification_Shap(taski, dataloader, model, cancer, fold, omics_, task, is_test=True):

    # ==================== IG =====================
    model.eval()
    ig = IntegratedGradients(model)
    omic_mut = None
    omic_gex = None
    omic_cna = None
    labels = []
    for label_i in range(task['output_dim']):
        a0 = []
        a1 = []
        a2 = []
        with tqdm(dataloader, unit='batch') as tepoch:

            for batch, data in enumerate(tepoch):
                omics_data, cancer_label0, cancer_label1, cancer_label2, cancer_label3, cancer_label4, cancer_label5 = data

                if omic_mut is None:
                    omic_mut = omics_data['mut']
                else:
                    omic_mut = torch.cat((omic_mut,omics_data['mut']),dim=0)
                if omic_gex is None:
                    omic_gex = omics_data['gex']
                else:
                    omic_gex = torch.cat((omic_gex,omics_data['gex']),dim=0)
                if omic_cna is None:
                    omic_cna = omics_data['cna']
                else:
                    omic_cna = torch.cat((omic_cna,omics_data['cna']),dim=0)

                cancer_labels = [cancer_label0, cancer_label1, cancer_label2, cancer_label3, cancer_label4, cancer_label5]
                cancer_labels_ = [cancer_label0.item(), cancer_label1.item(), cancer_label2.item(), cancer_label3.item(), cancer_label4.item(), cancer_label5.item()]
                labels.append(cancer_labels_)
                cancer_label = cancer_labels[taski].to(device)
                cancer_label = cancer_label.squeeze()
                cancer_label_taski = torch.ones_like(cancer_label)*label_i
                input_x = [omics_data[key].to(device) for key in omics_data.keys()]
                
                input_x = (*input_x,)

                base_x = [torch.zeros_like(omics_data[key]).to(device) for key in omics_data.keys()]

                base_x = (*base_x,)

                attributions = ig.attribute(input_x,target=cancer_label_taski)

                a0.append(attributions[0].detach().cpu())
                a1.append(attributions[1].detach().cpu())
                a2.append(attributions[2].detach().cpu())
                
        attributions_ = (torch.cat(a0,dim=0),torch.cat(a1,dim=0),torch.cat(a2,dim=0))
        if is_test:
            torch.save(omic_mut,f'../data/metabric_data/test_mut_task_{task}_fold_{fold}.pt')
            torch.save(omic_gex,f'../data/metabric_data/test_gex_task_{taski}_fold_{fold}.pt')
            torch.save(omic_cna,f'../data/metabric_data/test_cna_task_{taski}_fold_{fold}.pt')
            joblib.dump(labels,f'../data/metabric_data/test_task_all_fold_{fold}.pkl')
        if is_test:
            torch.save(attributions_,os.path.join('../train/pretrain','ig_gs', f'test_metabric_ig_gex_mut_cna_task_{taski}_fold_{fold}_label_{label_i}.pt'))
        else:
            torch.save(attributions_,os.path.join('../train/pretrain','ig_gs', f'val_metabric_ig_gex_mut_cna_task_{taski}_fold_{fold}_label_{label_i}.pt'))
        # ======================== GS ====================
        model.eval()
        gs = GradientShap(model)
        a0 = []
        a1 = []
        a2 = []
        with tqdm(dataloader, unit='batch') as tepoch:

            for batch, data in enumerate(tepoch):
                omics_data, cancer_label0, cancer_label1, cancer_label2, cancer_label3, cancer_label4, cancer_label5 = data
                
                cancer_labels = [cancer_label0, cancer_label1, cancer_label2, cancer_label3, cancer_label4, cancer_label5]
                cancer_label = cancer_labels[taski].to(device)
                cancer_label = cancer_label.squeeze()
                input_x = [omics_data[key].to(device) for key in omics_data.keys()]
                input_x = (*input_x,)
                base_x = [torch.zeros_like(omics_data[key]).to(device) for key in omics_data.keys()]
                base_x = (*base_x,)

                attributions = gs.attribute(input_x, baselines=base_x, target=cancer_label)
                a0.append(attributions[0].detach().cpu())
                a1.append(attributions[1].detach().cpu())
                a2.append(attributions[2].detach().cpu())
    
        attributions_ = (torch.cat(a0,dim=0),torch.cat(a1,dim=0),torch.cat(a2,dim=0))
        if is_test:
            torch.save(attributions_,os.path.join('../train/pretrain','ig_gs',f'test_metabric_gs_gex_mut_cna_task_{taski}_fold_{fold}_label_{label_i}.pt'))
        else:
            torch.save(attributions_,os.path.join('../train/pretrain','ig_gs',f'val_metabric_gs_gex_mut_cna_task_{taski}_fold_{fold}_label_{label_i}.pt'))


def MetaBric_classification_Shap(fold, pretrain_model_path, fixed, device_id):

    # torch.cuda.set_device(device_id)
    
    val_dataset = MetaBric_Dataset(val_omics_files,['gex','mut','cna'],val_label_path)
    test_dataset = MetaBric_Dataset(test_omics_files,['gex','mut','cna'],test_label_path)

    # print(len(val_dataset))
    # print(len(test_dataset))


    val_dataloader = DataLoader(val_dataset, batch_size=507, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tasks = {'Pam50Subtype':{'output_dim': 4}
            }
    cnt = 0 

    for ti,task in tasks.items():
        print(ti)
        model = DownStream_predictor(4, [6016, 6617, 4539, 7460], 64, [2048, 1024, 512],
                                    pretrain_model_path, task, omics_data_type, fixed).to(device)
        
        print('loading')
        model.load_state_dict(torch.load(f'../model/model_dict/metabric_task_{ti}_fold_{fold}.pt',map_location=device))
        print('load done')
        model.to(device)
        # metabric_test_classification_Shap(cnt, val_dataloader, model, 'metabric', fold, incomplete_omics, is_test=False)
        metabric_test_classification_Shap(cnt, test_dataloader, model, 'metabric', fold, incomplete_omics, task, is_test=True)
        
        cnt += 1
        del model

device_ids = [0, 4, 5, 6, 7]
folds = 5
all_epochs = 100

#   multiprocessing pretrain_fold
def multiprocessing_train_fold(function, func_args_list):
    processes = []
    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


pretrain_func_args = [(i, all_epochs, device_ids[i]) for i in range(folds)]
class_func_args = [(i, all_epochs, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(folds)]
cancer_type_list = ['LGG', 'BLCA', 'BRCA', 'COAD', 'HNSC', 'KIRC', 'LIHC', 'LUAD', 'STAD']
survival_func_args = [(i, all_epochs, cancer_type_list, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(folds)]
Metastatic_class_func_args = [(i,all_epochs, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(folds)]
MetaBric_class_func_args = [(i, all_epochs, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(folds)]
MetaBric_CPTAC_class_func_args = [(i, class_epochs, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(fs,fe)]
Metabric_Shap_func_args = [(i, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold{i}_dim64.pt', False, device_ids[i]) for i in range(folds)]

multiprocessing_train_fold(TCGA_Dataset_classification, class_func_args)
