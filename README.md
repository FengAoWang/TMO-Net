# TMO-Net:An Explainable Pretrained Multi-Omics Model for Multi-task Learning in Oncology



## Introduction

TMO-Net is a pre-trained tumor multi-omics deep learning model to learn representation across multi-omics in cancers, improving the prediction performance of oncology tasks such as survival prediction or drug response.<br> 
TMO-Net was implemented by Pytorch and details are the followning:
1. The source code of TMO-Net architecture can be found at the file of TMO_Net_model.py in 'model' module.<br>
2. The pipelines of pre-training and downstream task fine-tuning are available at the train/train_tcga_pancancer_multitask.py file. <br>
3. The implements of loss function are in the file of util/loss_function.py. <br>

## Method
Overview of TMO-Net research, including pan-cancer multi-omics collection, TMO-Net model architecture, pre-training and fine-tuning, and biological interpretation analysis.
![image](https://github.com/FengAoWang/TMO-Net/blob/master/figure1.png)

## Dataset and Data processing

Multi-omics profiling of pre-training and downstream tasks were all downloaded from the public source. The processed data can be available at https://zenodo.org/records/10944664.


## License

This source code is licensed under the MIT license.

