import os
import sys
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import TCGA_Sur_Board
from model.clue_model import Clue_model, DownStream_predictor
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from util.loss_function import cox_loss, c_index
from lifelines.utils import concordance_index


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


def get_fold_ids(cancer, fold):
    file_path = '/home/wfa/project/clue/data/SurvBoard/Splits/TCGA'
    train_ids = pd.read_csv(os.path.join(file_path, f'{cancer}/train/train_fold{fold}.csv'))['train_ids'].tolist()
    test_ids = pd.read_csv(os.path.join(file_path, f'{cancer}/test/test_fold{fold}.csv'))['test_ids'].tolist()
    return np.array(train_ids), np.array(test_ids)


# Pan_Cancer = ['BLCA', 'GBM', 'CESC', 'LUSC', 'KIRP', 'LIHC', 'SARC', 'PAAD', 'LGG', 'LUAD', 'KIRC', 'BRCA', 'HNSC']

omics_type = ['gex', 'mut', 'cnv']

Pan_Cancer = ['PAAD', 'BLCA', 'GBM', 'CESC', 'LUSC', 'KIRP', 'LIHC', 'SARC', 'LGG', 'LUAD', 'KIRC', 'BRCA', 'HNSC']
test_cancer = ['BLCA', 'GBM', 'CESC', 'LUSC', 'KIRP', 'LIHC', 'SARC', 'LGG', 'LUAD', 'KIRC', 'BRCA', 'HNSC']


def train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics):
    model.train()
    # model = model.state_dict()
    # for epoch in range(epochs):
    print(f'-----start epoch {epoch} training-----')
    total_loss = 0
    train_risk_score = torch.Tensor([]).cuda()
    train_censors = torch.Tensor([]).cuda()
    train_event_times = torch.Tensor([]).cuda()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_time, os_event, omics_data = data
            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = []
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic.size())
                input_x.append(omic)
            risk_score, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model(input_x, os_event.size(0), omics)
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            # CELoss = self_elbo + 0.1 * (cross_elbo + cross_infer_loss * cross_infer_loss - dsc_loss * 0.2)
            loss = CoxLoss
            total_loss += CoxLoss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=CoxLoss.item())

        print('cox loss: ', total_loss / len(train_dataloader))
        # train_c_index = c_index(train_risk_score, train_event_times, train_censors).item()
        train_c_index = concordance_index(train_event_times.detach().cpu().numpy(), -train_risk_score.detach().cpu().numpy(), train_censors.detach().cpu().numpy())
        print(f'{cancer} train survival c-index: ', train_c_index)


def cal_c_index(dataloader, model, omics, mode):
    model.eval()
    with torch.no_grad():
        gex_risk_score = torch.zeros((len(dataloader)))
        mut_risk_score = torch.zeros((len(dataloader)))
        cnv_risk_score = torch.zeros((len(dataloader)))
        gex_cnv_risk_score = torch.zeros((len(dataloader)))
        gex_mut_risk_score = torch.zeros((len(dataloader)))
        cnv_mut_risk_score = torch.zeros((len(dataloader)))
        risk_score = torch.zeros((len(dataloader)))

        censors = torch.zeros((len(dataloader)))
        event_times = torch.zeros((len(dataloader)))

    for i, data in enumerate(dataloader):
        os_time, os_event, omics_data = data
        input_x = []
        for key in omics_data.keys():
            omic = omics_data[key]
            omic = omic.cuda()
            # print(omic.size())
            input_x.append(omic)
        os_event = os_event.cuda()
        os_time = os_time.cuda()
        survival_risk, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model(input_x, os_event.size(0), omics)
        gex_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'gex': 0})
        mut_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'mut': 1})
        cnv_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'cnv': 2})

        gex_cnv_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'gex': 0, 'cnv': 2})
        gex_mut_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'gex': 0, 'mut': 1})
        cnv_mut_survival_risk, _, _, _, _ = model(input_x, os_event.size(0), {'cnv': 2, 'mut': 1})

        gex_risk_score[i] = gex_survival_risk
        mut_risk_score[i] = mut_survival_risk
        cnv_risk_score[i] = cnv_survival_risk
        gex_mut_risk_score[i] = gex_mut_survival_risk
        gex_cnv_risk_score[i] = gex_cnv_survival_risk
        cnv_mut_risk_score[i] = cnv_mut_survival_risk
        risk_score[i] = survival_risk

        censors[i] = os_event
        event_times[i] = os_time

    test_c_index = concordance_index(event_times.detach().cpu().numpy(), -risk_score.detach().cpu().numpy(),  censors.detach().cpu().numpy())
    gex_c_index = concordance_index(event_times.detach().cpu().numpy(), -gex_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())
    mut_c_index = concordance_index(event_times.detach().cpu().numpy(), -mut_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())
    cnv_c_index = concordance_index(event_times.detach().cpu().numpy(), -cnv_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())
    gex_cnv_c_index = concordance_index(event_times.detach().cpu().numpy(), -gex_cnv_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())
    gex_mut_c_index = concordance_index(event_times.detach().cpu().numpy(), -gex_mut_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())
    cnv_mut_c_index = concordance_index(event_times.detach().cpu().numpy(), -cnv_mut_risk_score.detach().cpu().numpy(), censors.detach().cpu().numpy())

    c_index_list = [test_c_index, gex_c_index, mut_c_index, cnv_c_index, gex_cnv_c_index, gex_mut_c_index, cnv_mut_c_index]
    print(f'{mode}  survival c-index: ', test_c_index, gex_c_index, mut_c_index, cnv_c_index, gex_cnv_c_index, gex_mut_c_index, cnv_mut_c_index)

    return c_index_list, test_c_index


cancer_c_index = []
fold = 0
for cancer in test_cancer:
    print(f'{cancer} start training')
    TCGA_file_path = '/home/wfa/project/clue/data/SurvBoard/TCGA/'
    cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type)
    # model = Clue_model(3, [20531, 19687, 24776], 64, [1024, 512, 128])
    epochs = 20
    best_c_indices = [cancer, 0]
    # for fold in range(folds):
    torch.cuda.set_device(fold)
    print(f'-----start fold {fold} training-----')
    task = {'output_dim': 1}
    fixed = False
    model = DownStream_predictor(3, [20531, 19687, 24776], 64, [1024, 512, 128],
                                 '/home/wfa/project/clue/model/model_dict/all_pancancer_pretrain_model_fold0_dim64.pt', task, fixed)
    omics = {'gex': 0, 'mut': 1, 'cnv': 2}
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.0000001},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.001}
    ]
    optimizer = torch.optim.Adam(param_groups)
    best_c_index = 0
    best_test_c_indices = []

    train_ids, test_ids = get_fold_ids(cancer, fold)
    train_dataloader = DataLoader(cancer_dataset, sampler=train_ids, batch_size=16)
    test_dataloader = DataLoader(cancer_dataset, sampler=test_ids, batch_size=1)
    for epoch in range(epochs):
        start_time = time.time()
        train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
        test_c_indices, test_c_index = cal_c_index(test_dataloader, model,  omics, 'test')
        best_c_index = test_c_index if test_c_index > best_c_index else best_c_index
        print(f'{fold} time used: ', time.time() - start_time)

    best_c_indices[1] = best_c_index
    cancer_c_index.append(best_c_indices)

cancer_c_index = pd.DataFrame(cancer_c_index, columns=['cancer', f'{fold}'])
cancer_c_index.to_csv(f'all_pancancer_cross_encoders_multiomics_c_index_fold{fold}.csv', encoding='utf-8')


