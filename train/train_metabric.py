import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import CustomDataset
import random
import numpy as np
from model.clue_model import DownStream_predictor
import torchmetrics
import torch.nn.functional as F

train_file = ['../data/moanna_data/moanna_data/tcga_matched_data/training/training_moanna_exp.csv',
              '../data/moanna_data/moanna_data/tcga_matched_data/training/training_moanna_mut.csv']
train_label = '../data/moanna_data/moanna_data/tcga_matched_data/training/moanna_training_label.tsv'

train_dataset = CustomDataset(train_file, train_label, 'BasalNonBasal')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_file = ['../data/moanna_data/moanna_data/tcga_matched_data/validation/validation_moanna_exp.csv',
              '../data/moanna_data/moanna_data/tcga_matched_data/validation/validation_moanna_mut.csv']
val_label = '../data/moanna_data/moanna_data/tcga_matched_data/validation/moanna_validation_label.tsv'

val_dataset = CustomDataset(val_file, val_label, 'BasalNonBasal')
val_dataloader = DataLoader(val_dataset, batch_size=32)


test_file = ['../data/moanna_data/moanna_data/tcga_matched_data/testing/testing_moanna_exp.csv',
              '../data/moanna_data/moanna_data/tcga_matched_data/testing/testing_moanna_mut.csv']
test_label = '../data/moanna_data/moanna_data/tcga_matched_data/testing/moanna_testing_label.tsv'

test_dataset = CustomDataset(test_file, test_label, 'BasalNonBasal')
test_dataloader = DataLoader(test_dataset, batch_size=32)


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

task = {'output_dim': 2}
model = DownStream_predictor(2, [6016, 4539], 64, [1024, 512, 128], 'tcga_pretrained_model_2_exp_mut_modal.pt', task)
model.cuda()
criterion = nn.BCEWithLogitsLoss()
multi_class_criterion = nn.CrossEntropyLoss()


epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).cuda()
test_f1 = torchmetrics.F1Score(num_classes=2).cuda()
val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).cuda()
test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).cuda()
train_recall = torchmetrics.Recall(average='none', num_classes=2)
train_precision = torchmetrics.Precision(average='none', num_classes=2)
train_auc = torchmetrics.AUROC(average="macro", num_classes=32)


def test_model(model, dataloader, acc, mode):
    model.eval()
    for batch, data in enumerate(dataloader):
        exp_tensor, mut_tensor, label = data
        label = label.view(-1, 1)
        label = label.cuda()
        input_x = []
        input_x.append(exp_tensor.cuda())
        input_x.append(mut_tensor.cuda())
        output = model(input_x, label.size(0))
        pred_label = F.softmax(output)
        # pred_label = torch.where(pred_label > 0.5, 1, 0)
        pred_label = pred_label.argmax(1)
        # print(pred_label, pred_label.view(-1, 1), label)
        acc(pred_label.view(-1, 1), label)
    print(f'{mode} acc: {acc.compute()}')
    return acc.compute()


best_val_acc = 0
best_test_acc = 0

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch, data in enumerate(train_dataloader):
        exp_tensor, mut_tensor,  label = data
        # label = label.view(-1, 1)
        label = label.cuda()
        input_x = []
        input_x.append(exp_tensor.cuda())
        input_x.append(mut_tensor.cuda())
        output = model(input_x, label.size(0))
        pred_label = F.softmax(output)
        # pred_label = torch.where(pred_label > 0.5, 1, 0)
        pred_label = pred_label.argmax(1)

        loss = multi_class_criterion(output, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pred_label, pred_label.view(-1, 1), label)
        train_acc(pred_label.view(-1, 1), label.view(-1, 1))
    print(f'{epoch} epoch, train loss: {total_loss / len(train_dataloader)}')
    print(f'train acc: {train_acc.compute()}')

    with torch.no_grad():
        model.eval()
        val_acc_value = test_model(model, val_dataloader, val_acc, 'val')
        best_val_acc = val_acc_value if val_acc_value > best_val_acc else best_val_acc
        test_acc_value = test_model(model, test_dataloader, test_acc, 'test')
        best_test_acc = test_acc_value if test_acc_value > best_test_acc else best_test_acc

print(best_val_acc, best_test_acc)
