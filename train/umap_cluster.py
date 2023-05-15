import os
import sys
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import TCGA_Sur_Board, Pancancer_TCGA_Sur_Board
from model.clue_model import Clue_model, DownStream_predictor
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from util.loss_function import cox_loss, c_index
from lifelines.utils import concordance_index
import joblib


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
device_ids = [0, 1, 2, 3] #
omics_type = ['gex', 'mut', 'cnv']
test_cancer = ['LGG', 'BRCA', 'KIRP', 'PAAD', 'BLCA',  'GBM', 'CESC', 'LUSC',  'LIHC', 'SARC',  'LUAD', 'KIRC',  'HNSC']


task = {'output_dim': 1}
fixed = False

model = Clue_model(3, [20531, 19687, 24776], 128, [4096, 1024, 512])
model_pretrain_dict = torch.load('/home/wfa/project/clue/model/model_dict/all_pancancer_pretrain_model_fold0_dim128.pt', map_location='cpu')
model.load_state_dict(model_pretrain_dict)
torch.cuda.set_device(3)
model.cuda()
omics = {'gex': 0, 'mut': 1, 'cnv': 2}
for cancer in test_cancer:
    model.eval()
    TCGA_file_path = '/home/wfa/project/clue/data/SurvBoard/TCGA/'
    cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type)
    # model = Clue_model(3, [20531, 19687, 24776], 64, [1024, 512, 128])
    best_c_indices = []
    # for fold in range(folds):
    # model = torch.nn.DataParallel(model, device_ids=device_ids)  # 指定要用到的设备

    train_dataloader = DataLoader(cancer_dataset, batch_size=128)
    embedding_tensor = torch.Tensor([]).cuda()
    for i, data in enumerate(train_dataloader):
        os_time, os_event, omics_data = data
        os_event = os_event.cuda()
        os_time = os_time.cuda()
        input_x = []
        for key in omics_data.keys():
            omic = omics_data[key]
            omic = omic.cuda()
            # print(omic.size())
            input_x.append(omic)
        output = model.latent_z(input_x, omics)
        embedding_tensor = torch.concat((embedding_tensor, output), dim=0)

        print(output)
    torch.save(embedding_tensor, f'{cancer}_embedding.pt')
