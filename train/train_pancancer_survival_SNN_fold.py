import os
import sys
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import TCGA_Sur_Board
from model.clue_model import Clue_model, DownStream_predictor, PanCancer_SNN_predictor
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from util.loss_function import cox_loss, c_index


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



#
Pan_Cancer = ['LGG', 'BLCA', 'GBM', 'CESC', 'LUSC', 'KIRP', 'LIHC', 'SARC', 'PAAD',  'LUAD', 'KIRC', 'BRCA', 'HNSC']
# Pan_Cancer = ['BLCA']

omics_type = ['gex', 'mut', 'cnv']


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
            omics_input = torch.Tensor([]).cuda()
            for key in omics:
                omic = omics_data[key]
                omic = omic.cuda()
                omics_input = torch.concat((omics_input, omic), dim=1)
                # print(omic.size())
                # input_x.append(omic)
            input_x.append(omics_input)
            risk_score = model(input_x)
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            # loss = ce_loss
            total_loss += CoxLoss.item()
            optimizer.zero_grad()
            CoxLoss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=CoxLoss.item())

        print(omics, ' cox loss: ', total_loss / len(train_dataloader))
        train_c_index = c_index(train_risk_score, train_event_times, train_censors).item()
        print(f'{cancer} train survival c-index: ', train_c_index)
        # model_dict = model.state_dict()
        # torch.save(model_dict, f'../model/model_dict/{cancer}_pretrain_model_fold{fold}.pt')


def cal_c_index(dataloader, model, best_c_index, omics):
    model.eval()
    with torch.no_grad():
        risk_score = torch.zeros((len(dataloader)))
        censors = torch.zeros((len(dataloader)))
        event_times = torch.zeros((len(dataloader)))

    for i, data in enumerate(test_dataloader):
        os_time, os_event, omics_data = data
        input_x = []
        omics_input = torch.Tensor([]).cuda()
        for key in omics:
            omic = omics_data[key]
            omic = omic.cuda()
            # print(omic.size())
            omics_input = torch.concat((omics_input, omic), dim=1)
        input_x.append(omics_input)
        os_event = os_event.cuda()
        os_time = os_time.cuda()
        survival_risk = model(input_x)

        risk_score[i] = survival_risk
        censors[i] = os_event
        event_times[i] = os_time

    test_c_index = c_index(risk_score, event_times, censors).item()
    best_c_index = test_c_index if test_c_index > best_c_index else best_c_index
    print(omics, f'{cancer} survival c-index', test_c_index)
    return best_c_index


fold = 0
cancer_c_index = []
for cancer in Pan_Cancer:
    print(f'{cancer} start training')
    TCGA_file_path = '/home/wfa/project/clue/data/SurvBoard/TCGA/'
    cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type)
    # model = Clue_model(3, [20531, 19687, 24776], 64, [1024, 512, 128])
    task = {'output_dim': 1}
    omics = ['gex', 'mut', 'cnv']
    # omics = ['gex']
    torch.cuda.set_device(5)
    # model = PanCancer_SNN_predictor(3, [20531, 19687, 24776], [1024, 512, 128], 64, task)
    model = PanCancer_SNN_predictor(1, [64994, 19687, 24776], [1024, 512, 128], 64, task)
    model.cuda()
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_c_indices = [cancer, 0]
    print(f'-----start fold {fold} training-----')
    train_ids, test_ids = get_fold_ids(cancer, fold)
    train_dataloader = DataLoader(cancer_dataset, sampler=train_ids, batch_size=16, drop_last=True)
    test_dataloader = DataLoader(cancer_dataset, sampler=test_ids, batch_size=1, drop_last=True)
    best_c_index = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
        best_c_index = cal_c_index(test_dataloader, model, best_c_index=best_c_index, omics=omics)
        print(fold, 'time used: ', time.time() - start_time)
    best_c_indices[1] = best_c_index
    cancer_c_index.append(best_c_indices)
cancer_c_index = pd.DataFrame(cancer_c_index, columns=['cancer', f'{fold}'])
cancer_c_index.to_csv(f'snn_multiomics_c_index_fold{fold}.csv', encoding='utf-8')






