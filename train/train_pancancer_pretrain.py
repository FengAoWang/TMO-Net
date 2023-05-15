import os
import sys
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import TCGA_Sur_Board
from model.clue_model import Clue_model, DownStream_predictor, dfs_freeze, un_dfs_freeze
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
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
    torch.backends.cudnn.benchmark = False


set_seed(66)


def get_fold_ids(cancer, fold):
    file_path = '/home/wfa/project/clue/data/SurvBoard/Splits/TCGA'
    train_ids = pd.read_csv(os.path.join(file_path, f'{cancer}/train/train_fold{fold}.csv'))['train_ids'].tolist()
    test_ids = pd.read_csv(os.path.join(file_path, f'{cancer}/test/test_fold{fold}.csv'))['test_ids'].tolist()
    return np.array(train_ids), np.array(test_ids)


def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer):
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
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_time, os_event, omics_data = data
            input_x = []
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic.size())
                input_x.append(omic)
            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)
            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0))

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()

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
        return Loss


def val_pretrain(test_dataloader, model, epoch, cancer):
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
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_time, os_event, omics_data = data
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    # print(omic.size())
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x, os_event.size(0))
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
    return Loss


Pan_Cancer = ['PAAD', 'BLCA', 'GBM', 'CESC', 'LUSC', 'KIRP', 'LIHC', 'SARC', 'LGG', 'LUAD', 'KIRC', 'BRCA', 'HNSC']

# Pan_Cancer = ['BLCA']
omics_type = ['gex', 'mut', 'cnv']
omics_data = ['gaussian', 'gaussian', 'gaussian']
fold = 0
torch.cuda.set_device(6)
model = Clue_model(3, [20531, 19687, 24776], 64, [4096, 1024, 512], omics_data)
model.cuda()

pretrain_dataset = []
test_dataset = []
for cancer in Pan_Cancer:
    print(f'{cancer} data processing')
    train_ids, test_ids = get_fold_ids(cancer, fold)
    TCGA_file_path = '/home/wfa/project/clue/data/SurvBoard/TCGA/'
    cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type, dataset_ids=train_ids)
    test_cancer_dataset = TCGA_Sur_Board(TCGA_file_path, cancer, omics_type, dataset_ids=test_ids)

    # print(cancer_dataset[1])
    pretrain_dataset.append(cancer_dataset)
    test_dataset.append(test_cancer_dataset)

PanCancer_dataset = ConcatDataset(pretrain_dataset)
test_PanCancer_dataset = ConcatDataset(test_dataset)

print(len(PanCancer_dataset))
epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)


train_dataloader = DataLoader(PanCancer_dataset, batch_size=32)
test_dataloader = DataLoader(test_PanCancer_dataset, batch_size=32)

best_c_index = 0
Loss_list = []
test_Loss_list = []
for epoch in range(epochs):
    start_time = time.time()
    Loss_list.append(train_pretrain(train_dataloader, model, epoch, 'PanCancer', optimizer, dsc_optimizer))
    test_Loss_list.append(val_pretrain(test_dataloader, model, epoch, 'PanCancer'))
    print(f'fold{fold} time used: ', time.time() - start_time)

model_dict = model.state_dict()
Loss_list = torch.Tensor(Loss_list)
test_Loss_list = torch.Tensor(test_Loss_list)

torch.save(test_Loss_list, f'../model/model_dict/all_pancancer_pretrain_test_loss_fold{fold}.pt')
torch.save(Loss_list, f'../model/model_dict/all_pancancer_pretrain_train_loss_fold{fold}.pt')
torch.save(model_dict, f'../model/model_dict/all_pancancer_pretrain_model_fold{fold}_dim64_latent_z.pt')
