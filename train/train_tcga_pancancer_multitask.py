import os
import sys
sys.path.append('/home/wfa/project/clue')
import torch
from dataset.dataset import CancerDataset
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from LogME import LogME
logme = LogME(regression=False)



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

omics_files = [
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]

clinical_file = '../data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv'

train_index_path = '../data/TCGA_PanCancer_Data_cleaned/train_data.csv'
test_index_path = '../data/TCGA_PanCancer_Data_cleaned/test_data.csv'

# omics_data_type = ['gaussian', 'gaussian']
omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']

# omics_data_type = ['bernoulli', 'bernoulli', 'bernoulli', 'bernoulli']

omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}
# omics = {'gex': 0, 'methy': 1}

#   pretrain
def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer, fold):
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
            os_time, os_event, omics_data, cancer_label = data
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
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)
            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                  os_event.size(0))

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()
            multi_embedding = model.get_embedding(input_x, os_event.size(0), omics)

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

        torch.save(pancancer_embedding, f'../model/model_dict/TCGA_pancancer_multi_train_embedding_fold{fold}_epoch{epoch}.pt')
        torch.save(all_label, f'../model/model_dict/TCGA_pancancer_train_fold{fold}_epoch{epoch}_all_label.pt')

        pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
        print('pretrain score:', pretrain_score)
        return Loss


def val_pretrain(test_dataloader, model, epoch, cancer, fold):
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
                os_time, os_event, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()
                all_label = torch.concat((all_label, cancer_label), dim=0)
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    # print(omic.size())
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                      os_event.size(0))
                multi_embedding = model.get_embedding(input_x, os_event.size(0), omics)

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
                       f'../model/model_dict/TCGA_pancancer_multi_test_embedding_fold{fold}_epoch{epoch}.pt')
            torch.save(all_label, f'../model/model_dict/TCGA_pancancer_test_fold{fold}_epoch{epoch}_all_label.pt')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)

    Loss_list = []
    test_Loss_list = []
    for epoch in range(epochs):

        start_time = time.time()
        # if epoch > 7:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.000001
        Loss_list.append(train_pretrain(train_dataloader, model, epoch, 'PanCancer', optimizer, dsc_optimizer, fold))

        test_Loss_list.append(val_pretrain(test_dataloader, model, epoch, 'PanCancer', fold))
        print(f'fold{fold} time used: ', time.time() - start_time)

    model_dict = model.state_dict()
    Loss_list = torch.Tensor(Loss_list)
    test_Loss_list = torch.Tensor(test_Loss_list)

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
            os_time, os_event, omics_data, _ = data
            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            risk_score = model(input_x, os_event.size(0), omics)
            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0))
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
            os_time, os_event, omics_data, _ = data
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            os_event = os_event.cuda()
            os_time = os_time.cuda()

            survival_risk = model(input_x, os_event.size(0), omics)

            gex_survival_risk = model(input_x, os_event.size(0), {'gex': 0})
            mut_survival_risk = model(input_x, os_event.size(0), {'mut': 2})
            cnv_survival_risk = model(input_x, os_event.size(0), {'cnv': 3})

            gex_cnv_survival_risk = model(input_x, os_event.size(0), {'gex': 0, 'cnv': 3})
            gex_mut_survival_risk = model(input_x, os_event.size(0), {'gex': 0, 'mut': 2})
            cnv_mut_survival_risk = model(input_x, os_event.size(0), {'cnv': 3, 'mut': 2})

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
        train_dataset = CancerDataset(omics_files, ['cna', 'gex', 'methy', 'mut'], clinical_file, train_index_path,
                                      fold + 1, [cancer])
        test_dataset = CancerDataset(omics_files, ['cna', 'gex', 'methy', 'mut'], clinical_file, test_index_path,
                                     fold + 1, [cancer])

        train_dataloader = DataLoader(train_dataset, batch_size=16, drop_last=True)
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
            os_time, os_event, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            classification_pred = model(input_x, os_event.size(0), omics)
            _, labels_pred = torch.max(classification_pred, 1)

            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0))
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
        torch.save(all_labels, 'pancancer_label.pt')
    torch.save(pancancer_embedding, 'pancancer_embedding.pt')


def test_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tepoch:
            total_samples = 0
            all_labels = []
            all_predictions = []

            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_time, os_event, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()

                input_x = [omics_data[key].cuda() for key in omics_data.keys()]

                classification_pred = model(input_x, os_event.size(0), omics)
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

    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch, 0.0001)
        start_time = time.time()
        train_classification(train_dataloader, model, epoch, 'pancancer', fold, optimizer, omics, criterion)
        with torch.no_grad():
            test_classification(test_dataloader, model, epoch, 'pancancer', fold, optimizer, omics, criterion)

        print(f'fold {fold} time used: ', time.time() - start_time)

    #   clean memory of gpu cuda
    del model
    del optimizer
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()


device_ids = [0, 3, 5, 6, 7]
folds = 5
all_epochs = 10


#   multiprocessing pretrain_fold
def multiprocessing_train_fold(function, func_args_list):
    processes = []
    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


class_func_args = [(i, all_epochs, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold0_dim64.pt', False, device_ids[i]) for i in range(folds)]
cancer_type_list = ['LGG', 'BLCA', 'COAD']
survival_func_args = [(i, all_epochs, cancer_type_list, f'../model/model_dict/TCGA_pancancer_pretrain_model_fold0_dim64.pt', False, device_ids[i]) for i in range(folds)]
pretrain_func_args = [(i, all_epochs, device_ids[i]) for i in range(folds)]

multiprocessing_train_fold(TCGA_Dataset_pretrain, pretrain_func_args)
