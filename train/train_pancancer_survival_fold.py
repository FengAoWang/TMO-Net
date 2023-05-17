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

# test_cancer = ['PAAD']


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            os_time, os_event, omics_data = data
            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            risk_score = model(input_x, os_event.size(0), omics)
            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0))
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            loss = CoxLoss + 0.1 * pretrain_loss
            total_loss += CoxLoss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=CoxLoss.item())

        print('cox loss: ', total_loss / len(train_dataloader))
        train_c_index = concordance_index(train_event_times.detach().cpu().numpy(), -train_risk_score.detach().cpu().numpy(), train_censors.detach().cpu().numpy())
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

        for i, data in enumerate(test_dataloader):
            os_time, os_event, omics_data = data
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            os_event = os_event.cuda()
            os_time = os_time.cuda()

            survival_risk = model(input_x, os_event.size(0), omics)

            gex_survival_risk = model(input_x, os_event.size(0), {'gex': 0})
            mut_survival_risk = model(input_x, os_event.size(0), {'mut': 1})
            cnv_survival_risk = model(input_x, os_event.size(0), {'cnv': 2})

            gex_cnv_survival_risk = model(input_x, os_event.size(0), {'gex': 0, 'cnv': 2})
            gex_mut_survival_risk = model(input_x, os_event.size(0), {'gex': 0, 'mut': 1})
            cnv_mut_survival_risk = model(input_x, os_event.size(0), {'cnv': 2, 'mut': 1})

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

        print(f'{cancer} test survival c-index: ', c_indices)

    return best_c_index_list, best_c_index


omics_type = ['gex', 'mut', 'cnv']
test_cancer = ['BRCA', 'LGG', 'LIHC', 'PAAD', 'LUSC', 'LUAD',  'BLCA',  'HNSC', 'KIRP',  'GBM', 'CESC',  'SARC', 'KIRC']

cancer_c_index = []
fold = 1
TCGA_file_path = '/home/wfa/project/clue/data/SurvBoard/TCGA/'
model_path = '/home/wfa/project/clue/model/model_dict/all_pancancer_pretrain_model_fold0_dim64_latent_z.pt'
task = {'output_dim': 1}
fixed = False
epochs = 2

torch.cuda.set_device(0)
omics = {'gex': 0, 'mut': 1, 'cnv': 2}
omics_data_type = ['gaussian', 'gaussian', 'gaussian']

for cancer in test_cancer:
    print(f'{cancer} start training')

    torch.cuda.empty_cache()
    cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type)
    torch.cuda.set_device(0)

    print(f'-----start fold {fold} training-----')

    model = DownStream_predictor(3, [20531, 19687, 24776], 64, [4096, 1024, 512], model_path, task, omics_data_type, fixed)
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.00001},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.00001}
    ]

    optimizer = torch.optim.Adam(param_groups)
    train_ids, test_ids = get_fold_ids(cancer, fold)
    train_dataloader = DataLoader(cancer_dataset, sampler=train_ids, batch_size=32, drop_last=True)
    test_dataloader = DataLoader(cancer_dataset, sampler=test_ids, batch_size=1)

    best_c_index = 0
    best_c_indices = []

    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, 0.0001)
        start_time = time.time()
        train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
        best_c_indices, best_c_index = cal_c_index(test_dataloader, model, best_c_index, best_c_indices, omics)
        print(f'{fold} time used: ', time.time() - start_time)
    best_c_indices.insert(0, cancer)
    cancer_c_index.append(best_c_indices)

    torch.cuda.empty_cache()


print(cancer_c_index)
cancer_c_index = pd.DataFrame(cancer_c_index, columns=['cancer', f'{fold} multiomics', f'{fold} gex', f'{fold} mut', f'{fold} cnv', f'{fold} gex_cnv', f'{fold} gex_mut', f'{fold} cnv_mut'])
cancer_c_index.to_csv(f'all_pancancer_pretrain_cross_encoders_c_index_fold{fold}_all_omics.csv', encoding='utf-8')
