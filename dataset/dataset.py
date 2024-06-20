import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
import joblib 



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


class CancerDataset(Dataset):
    def __init__(self, omics_paths, omics_types, clinical_data_path, index_data_path, fold, cancer_types=None, Percent_Train=None):
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
        if Percent_Train is not None:
            self.train_data = train_data[(train_data['Fold'] == fold) & (train_data['Percent_Train'] == Percent_Train) & (train_data['Cancer_Type'].isin(cancer_types))]
        else:
            self.train_data = train_data[(train_data['Fold'] == fold) & (train_data['Cancer_Type'].isin(cancer_types))]


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # Get the index of the sample from the training data
        sample_id = self.train_data.iloc[idx]['Sample_ID']

        # Get omics data for this sample
        omics_data = {omics_type: torch.Tensor(df[df['ID'] == sample_id].values.tolist()[0][1:]) for omics_type, df in self.omics.items()}

        # Get clinical data for this sample
        sample_clinical_data = self.clinical_data[self.clinical_data['ID'] == sample_id]
        OS = torch.Tensor(sample_clinical_data['OS'].values)
        OS_time = torch.Tensor(sample_clinical_data['OS.time'].values) / 30
        cancer = sample_clinical_data['Cancer_Type'].values.tolist()
        cancer_type = torch.LongTensor([PanCancer[cancer[0]]])

        return OS, OS_time, omics_data, cancer_type


class Drug_response_Dataset(Dataset):
    def __init__(self, response_data_path, omics_paths, omics_types, drug):
        super(Dataset, self).__init__()
        response_data = pd.read_csv(response_data_path, sep='\t')
        response_data = response_data[response_data['drug'] == drug]
        self.response_data = response_data
        # Load omics data
        self.omics = {omics_type: pd.read_csv(path) for omics_type, path in zip(omics_types, omics_paths)}

    def __len__(self):
        return len(self.response_data)

    def __getitem__(self, item):

        sample_id = self.response_data.iloc[item]['sample_name']

        omics_data = {omics_type: torch.Tensor(df[df['SampleID'] == sample_id].values.tolist()[0][2:]) for omics_type, df in self.omics.items()}

        sample_clinical_data = self.response_data[self.response_data['sample_name'] == sample_id]
        response = sample_clinical_data['response'].values
        response_tensor = torch.Tensor([0] if response == 'R' else [1])

        return omics_data, response_tensor

class MetaBric_Dataset(Dataset):
    def __init__(self, omics_paths, omics_types, label_path): 
        super(MetaBric_Dataset, self).__init__()
        assert len(omics_paths) == len(omics_types), "Number of omics paths and types must match"
        
        # self.omics = {omics_type: pd.read_csv(path,index_col=0) for omics_type, path in zip(omics_types, omics_paths)}
        self.omics = {omics_type: pd.read_csv(path) for omics_type, path in zip(omics_types, omics_paths)}
        self.clinical_data = pd.read_csv(label_path,sep='\t')
    
    def __len__(self):
        return len(self.clinical_data)
    
    def __getitem__(self,idx):

        ERStatus = torch.LongTensor([self.clinical_data['ERStatus'].loc[idx]])
        HER2Status = torch.LongTensor([self.clinical_data['HER2Status'].loc[idx]])
        Pam50Subtype = torch.LongTensor([self.clinical_data['Pam50Subtype'].loc[idx]])
        Pam50SubtypeSingleLuminal = torch.LongTensor([self.clinical_data['Pam50SubtypeSingleLuminal'].loc[idx]])
        BasalNonBasal = torch.LongTensor([self.clinical_data['BasalNonBasal'].loc[idx]])
        Luminal = torch.LongTensor([self.clinical_data['Luminal'].loc[idx]])


        omics_data = {}
        for omics_type, df in self.omics.items():
            # print(omics_type, 'ID' in df.columns, df.shape)
            if 'ID' in df.columns:
                del df['ID']
            omics_data[omics_type] = torch.Tensor(df.iloc[idx].values.tolist()[1:]) 

        return omics_data, ERStatus, HER2Status, Pam50Subtype, BasalNonBasal, Pam50SubtypeSingleLuminal, Luminal


class MetastaticDataset(Dataset):
    def __init__(self, omics_paths_prim, omics_paths_meta, omics_types, split_path, fold, label_path, is_test=False):
        assert len(omics_paths_prim) == len(omics_types), "Number of omics paths and types must match"
        assert len(omics_paths_meta) == len(omics_types), "Number of omics paths and types must match"

        split = joblib.load(split_path)
        self.label = joblib.load(label_path)
        if not is_test:
            self.samples = split[f'fold{fold}_train']
        else:
            self.samples = split[f'fold{fold}_test']
        # Select data for the specified fold and cancer types
        self.omics_prim = {}
        for omics_type, path in zip(omics_types, omics_paths_prim):
            df = pd.read_csv(path,index_col=0)
            self.omics_prim[omics_type] = df

        self.omics_meta = {}
        for omics_type, path in zip(omics_types, omics_paths_meta):
            df = pd.read_csv(path,index_col=0)
            self.omics_meta[omics_type] = df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the index of the sample from the training data
        sample_id = self.samples[idx]
        omics_data = {}

        if self.label[sample_id] == 0:
            for omics_type, df in self.omics_prim.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([0])

        if self.label[sample_id] == 1:
            for omics_type, df in self.omics_meta.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([1])

        if self.label[sample_id] == 2:
            for omics_type, df in self.omics_prim.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([1])

        return omics_data, cancer_type

class CPTAC_BRCA_Dataset(Dataset):
    def __init__(self, omics_paths, omics_types, label_path, split_path, fold_i, mode='train'):
        super(CPTAC_BRCA_Dataset, self).__init__()
        self.omics = {omics_type: pd.read_csv(path,index_col=0) for omics_type, path, in zip (omics_types,omics_paths)}
        self.cases = joblib.load(split_path)[f"{mode}_{fold_i}"]
        self.clinical = pd.read_csv(label_path,index_col=1)
        self.Pam50dict = {
            'Normal-like':1,
            'Basal':1,
            'Her2':3,
            'LumA':0,
            'LumB':2 
        }
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self,idx):
        ERStatus = torch.LongTensor([self.clinical.loc[self.cases[idx]]["ER.Updated.Clinical.Status"]])
        HER2Status = torch.LongTensor([self.clinical.loc[self.cases[idx]]["Her2.Updated.Clinical.Status"]])
        Pam50Subtype = torch.LongTensor([self.Pam50dict[self.clinical.loc[self.cases[idx]]["PAM50"]]])

        omics_data = {}
        for omics_type, df in self.omics.items():
            if 'ID' in df.columns:
                del df['ID']
            omics_data[omics_type] = torch.Tensor(df.loc[self.cases[idx]].values.tolist())
        return omics_data, ERStatus, HER2Status, Pam50Subtype 
