import os

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd


PanCancer = {
    'ACC': 0, 'BLCA': 1, 'CESC': 2, 'CHOL': 3,
    'COAD': 4, 'DLBC': 5, 'ESCA': 6, 'GBM': 7,
    'HNSC': 8, 'KICH': 9, 'KIRC': 10, 'KIRP': 11,
    'LGG': 12, 'LIHC': 13, 'LUAD': 14, 'LUSC': 15,
    'MESO': 16, 'OV': 17, 'PAAD': 18, 'PCPG': 19,
    'PRAD': 20, 'READ': 21, 'SARC': 22, 'SKCM': 23,
    'STAD': 24, 'TGCT': 25, 'THCA': 26, 'THYM': 27,
    'UCEC': 28, 'UCS': 29, 'UVM': 30, 'BRCA': 31
    }


class argo_dataset(Dataset):
    def __init__(self, label_path, exp_path, mut_path):
        super(argo_dataset, self).__init__()
        self.sample_ids = pd.read_csv(label_path)
        self.exp_data = pd.read_csv(exp_path)
        self.mut_data = pd.read_csv(mut_path)

    def __len__(self):
        return self.sample_ids.shape[0]

    def __getitem__(self, item):
        sample_id = self.sample_ids.iloc[item, 1]
        os_event = torch.Tensor([self.sample_ids.loc[item, 'os.event']])
        os_delay = torch.Tensor([self.sample_ids.loc[item, 'os.delay']])

        exp_tensor = self.exp_data.loc[self.exp_data['ID'] == sample_id, ].values.tolist()
        exp_tensor = exp_tensor[0][2:]
        exp_tensor = torch.FloatTensor(exp_tensor)

        mut_tensor = self.mut_data.loc[self.mut_data['ID'] == sample_id].values.tolist()

        mut_tensor = mut_tensor[0][2:]
        mut_tensor = torch.FloatTensor(mut_tensor)

        return exp_tensor, mut_tensor, os_event, os_delay


class Pancancer_TCGA_Sur_Board(Dataset):
    def __init__(self, TCGA_file_path, cancer_type, omics_type, dataset_ids=None):

        super(Pancancer_TCGA_Sur_Board, self).__init__()
        self.cancer_type = cancer_type
        self.TCGA_file_path = TCGA_file_path
        self.data = self.get_pancancer_data()
        if dataset_ids is not None:
            self.data = self.data.loc[self.data['patient_id'].isin(dataset_ids), :]
            self.data.index = [i for i in range(self.data.shape[0])]
        self.omics_type = omics_type
        self.omics_features = self.store_omics()

    def get_pancancer_data(self):
        data = pd.DataFrame()
        for cancer in self.cancer_type:
            data = pd.concat((data, pd.read_csv(os.path.join(self.TCGA_file_path, f'{cancer}/{cancer}_data_complete_modalities_preprocessed.csv'))), axis=0)
        return data

    def store_omics(self):
        omics_features = {}
        for omic in self.omics_type:
            data = pd.read_csv(os.path.join(self.TCGA_file_path, f'{self.cancer_type[0]}/{omic}_features.csv'))['features'].values.tolist()
            omics_features[omic] = data
        return omics_features

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        os_event = torch.Tensor(self.data.loc[item, ['OS']])
        os_time = torch.Tensor(self.data.loc[item, ['OS_days']])
        omics_data = {}
        for omic in self.omics_type:
            omics_data[omic] = torch.Tensor(self.data.loc[item, self.omics_features[omic]].values.tolist())
        return os_time, os_event, omics_data


class TCGA_Sur_Board(Dataset):
    def __init__(self, TCGA_file_path, cancer_type, omics_type, dataset_ids=None):
        super(TCGA_Sur_Board, self).__init__()
        self.cancer_type = cancer_type
        self.TCGA_file_path = TCGA_file_path
        self.data = pd.read_csv(os.path.join(TCGA_file_path, f'{cancer_type}/{cancer_type}_data_complete_modalities_preprocessed.csv'))
        if dataset_ids is not None:
            self.data = self.data.iloc[dataset_ids, :]
            self.data.index = [i for i in range(self.data.shape[0])]
        self.omics_type = omics_type
        self.omics_features = self.store_omics()

    def store_omics(self):
        omics_features = {}
        for omic in self.omics_type:
            data = pd.read_csv(os.path.join(self.TCGA_file_path, f'{self.cancer_type}/{omic}_features.csv'))['features'].values.tolist()
            omics_features[omic] = data
        return omics_features

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        os_event = torch.Tensor(self.data.loc[item, ['OS']])
        os_time = torch.Tensor(self.data.loc[item, ['OS_days']])
        omics_data = {}
        for omic in self.omics_type:
            omics_data[omic] = torch.Tensor(self.data.loc[item, self.omics_features[omic]].values.tolist())
        return os_time, os_event, omics_data


class CustomDataset(Dataset):
    def __init__(self, train_files, label_file, label_column):
        """
        初始化自定义数据集。

        参数：
        train_files (list of str): 包含训练数据的CSV文件列表
        label_file (str): 包含标签数据的CSV文件名
        label_column (str): 标签列名
        """
        # 从CSV文件中分别读取训练数据
        self.train_data = [pd.read_csv(file) for file in train_files]

        # 将训练数据转换为张量
        self.train_tensors = [torch.tensor(df.values, dtype=torch.float32) for df in self.train_data]

        # 从CSV文件中读取标签数据
        labels = pd.read_csv(label_file, sep='\t')[label_column]

        # 将标签数据转换为张量
        self.y = torch.LongTensor(labels.values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.train_tensors[0][idx], self.train_tensors[1][idx], self.y[idx]


class TCGA_PanCancer_Dataset(Dataset):
    def __init__(self, clinic_path, omics_path, omics_type):
        super(TCGA_PanCancer_Dataset, self).__init__()
        self.clinic_info = pd.read_csv(clinic_path)
        self.omic_data = {}
        self.omics_type = omics_type
        for omic in omics_type:
            self.omic_data[omic] = pd.read_csv(omics_path[omic])
            self.omic_data[omic].columns = ['ID'] + self.omic_data[omic].columns[1:].tolist()
            self.omic_data[omic]['ID'] = [x[:12] for x in self.omic_data[omic]['ID']]

    def __len__(self):
        return self.clinic_info.shape[0]

    def __getitem__(self, item):
        Patient_ID = self.clinic_info.iloc[item, 0]
        Cancer_type = self.clinic_info.iloc[item, 2]
        Patient_ID = Patient_ID.replace('-', '.')
        omic_data = {}
        for omic in self.omics_type:
            omic_data[omic] = torch.FloatTensor(self.omic_data[omic].loc[self.omic_data[omic]['ID'] == Patient_ID, ].values.tolist()[0][1:])
        Cancer_type = torch.LongTensor([PanCancer[Cancer_type]])
        return omic_data, Cancer_type


class TCGA_Cancer_Dataset(Dataset):
    def __init__(self, clinic_path, omics_path, omics_type, cancer_type):
        super(TCGA_Cancer_Dataset, self).__init__()
        clinic_info = pd.read_csv(clinic_path)
        clinic_info.dropna(subset=['OS.time'])
        self.clinic_info = clinic_info[clinic_info['Cancer_Type'] == cancer_type]
        self.omic_data = {}
        self.omics_type = omics_type
        for omic in omics_type:
            self.omic_data[omic] = pd.read_csv(omics_path[omic])
            self.omic_data[omic].columns = ['ID'] + self.omic_data[omic].columns[1:].tolist()
            self.omic_data[omic]['ID'] = [x[:12] for x in self.omic_data[omic]['ID']]

    def __len__(self):
        return self.clinic_info.shape[0]

    def __getitem__(self, item):
        Patient_ID = self.clinic_info.iloc[item, 0]
        clinic_ID = Patient_ID
        # Cancer_type = self.clinic_info.iloc[item, 2]
        Patient_ID = Patient_ID.replace('-', '.')
        omic_data = {}
        for omic in self.omics_type:
            omic_data[omic] = torch.FloatTensor(self.omic_data[omic].loc[self.omic_data[omic]['ID'] == Patient_ID, ].values.tolist()[0][1:])
        os_event = torch.FloatTensor(self.clinic_info.loc[self.clinic_info['ID'] == clinic_ID, 'OS'].values)
        os_time = torch.FloatTensor(self.clinic_info.loc[self.clinic_info['ID'] == clinic_ID, 'OS.time'].values)
        return omic_data, os_event, os_time


class Meta_Bric_Dataset(Dataset):
    def __init__(self, data_path):
        super(Meta_Bric_Dataset, self).__init__()
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        gene_exp = torch.Tensor(self.data.iloc[item, 31:520])
        os_event = torch.Tensor([self.data.loc[item, 'overall_survival']])
        os_time = torch.Tensor([self.data.loc[item, 'overall_survival_months']])
        return gene_exp, os_event, os_time


class argo_embedding(Dataset):
    def __init__(self, label_path, embedding_label='fusion'):
        super(argo_embedding, self).__init__()
        self.sample_ids = pd.read_csv(label_path)
        self.embedding_label = embedding_label
        if embedding_label == 'poe':
            self.embedding_data_z1 = torch.load('../data/embedding_data/latent_poe_z1.ph')
            self.embedding_data_z2 = torch.load('../data/embedding_data/latent_poe_z2.ph')

        elif embedding_label == 'moe':
            self.embedding_data = pd.read_csv('')
        else:
            self.embedding_data_z1 = torch.load('../data/embedding_data/latent_z1.ph')
            self.embedding_data_z2 = torch.load('../data/embedding_data/latent_z2.ph')
            self.embedding_data_z12 = torch.load('../data/embedding_data/latent_z2_x1_to_x2.ph')
            self.embedding_data_z21 = torch.load('../data/embedding_data/latent_z1_x2_to_x1.ph')

    def __len__(self):
        return self.sample_ids.shape[0]

    def __getitem__(self, item):

        sample_id = self.sample_ids.iloc[item, 1]
        if self.embedding_label == 'mean':
            sample_id_embedding_z1 = self.embedding_data_z1[item, :].detach().cpu()
            sample_id_embedding_z21 = self.embedding_data_z21[item, :].detach().cpu()
            sample_id_embedding_z1 = (sample_id_embedding_z1 + sample_id_embedding_z21) / 2

            sample_id_embedding_z2 = self.embedding_data_z2[item, :].detach().cpu()
            sample_id_embedding_z12 = self.embedding_data_z12[item, :].detach().cpu()
            sample_id_embedding_z2 = (sample_id_embedding_z2 + sample_id_embedding_z12) / 2

            sample_id_embedding = torch.concat((sample_id_embedding_z1, sample_id_embedding_z2), dim=0)
            # sample_id_embedding = sample_id_embedding_z2
        else:
            sample_id_embedding_z1 = self.embedding_data_z1[item, :].detach().cpu()
            sample_id_embedding_z2 = self.embedding_data_z2[item, :].detach().cpu()
            sample_id_embedding = torch.concat((sample_id_embedding_z1, sample_id_embedding_z2), dim=0)
            # sample_id_embedding = sample_id_embedding_z2

        dfs_delay = [float(self.sample_ids.loc[item, 'os.delay'])]
        dfs_event = [self.sample_ids.loc[item, 'os.event']]

        dfs_event = torch.FloatTensor(dfs_event)
        dfs_delay = torch.FloatTensor(dfs_delay)

        return sample_id_embedding, dfs_event, dfs_delay


class CancerDataset(Dataset):
    def __init__(self, omics_paths, omics_types, clinical_data_path, index_data_path, fold, cancer_types=None):
        # Check that the number of omics paths and types match
        assert len(omics_paths) == len(omics_types), "Number of omics paths and types must match"

        # Load omics data
        self.omics = {omics_type: pd.read_csv(path) for omics_type, path in zip(omics_types, omics_paths)}

        # Load clinical data
        self.clinical_data = pd.read_csv(clinical_data_path)

        # Load training data
        train_data = pd.read_csv(index_data_path)

        # If cancer_types is not specified, get all unique cancer types from the clinical data
        if cancer_types is None:
            cancer_types = self.clinical_data['Cancer_Type'].unique()
            # print(cancer_types)

        # Select data for the specified fold and cancer types
        self.train_data = train_data[(train_data['Fold'] == fold) & (train_data['Cancer_Type'].isin(cancer_types))]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # Get the index of the sample from the training data
        sample_index = self.train_data.iloc[idx]['Sample_Index']

        # Get omics data for this sample
        omics_data = {omics_type: torch.Tensor(df.iloc[sample_index, 1:].values.tolist()) for omics_type, df in self.omics.items()}

        # Get clinical data for this sample
        OS = torch.Tensor(self.clinical_data.loc[sample_index, ['OS']])
        OS_time = torch.Tensor(self.clinical_data.loc[sample_index, ['OS.time']]) / 30
        cancer = self.clinical_data.loc[sample_index, ['Cancer_Type']].values.tolist()
        cancer_type = torch.LongTensor(PanCancer[cancer[0]])
        return OS, OS_time, omics_data, cancer_type
