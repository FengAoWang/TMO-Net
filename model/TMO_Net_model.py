import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from util.loss_function import KL_loss, reconstruction_loss, KL_divergence
from functools import reduce
import numpy as np


def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def un_dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)


def product_of_experts(mu_set_, log_var_set_):
    tmp = 0
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 batchnorm: bool = False,
                 activation=None):
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None

        self.activation = None
        if activation is not None:
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'leakyrelu':
                self.activation = torch.nn.LeakyReLU()

    def forward(self, input_x):
        x = self.linear_layer(input_x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, activation):
        super(encoder, self).__init__()
        self.FeatureEncoder = nn.ModuleList([LinearLayer(input_dim, hidden_dims[0],
                                                         batchnorm=True, activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureEncoder.append(LinearLayer(hidden_dims[i], hidden_dims[i+1],
                                                   batchnorm=True, activation=activation))

        self.mu_predictor = nn.Sequential(nn.Linear(hidden_dims[-1], latent_dim),
                                          nn.ReLU()
                                          )
        self.log_var_predictor = nn.Sequential(nn.Linear(hidden_dims[-1], latent_dim),
                                               nn.ReLU()
                                               )

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def forward(self, x):
        for layer in self.FeatureEncoder:
            x = layer(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        return latent_z, mu, log_var


class decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, activation):
        super(decoder, self).__init__()
        self.FeatureDecoder = nn.ModuleList([LinearLayer(latent_dim, hidden_dims[0],
                                                         dropout=0.1, batchnorm=True,
                                                         activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureDecoder.append(LinearLayer(hidden_dims[i], hidden_dims[i+1],
                                                   dropout=0.1, batchnorm=True,
                                                   activation=activation))

        self.ReconsPredictor = LinearLayer(hidden_dims[-1], output_dim)

    def forward(self, latent_z):
        for layer in self.FeatureDecoder:
            latent_z = layer(latent_z)
        DataRecons = self.ReconsPredictor(latent_z)
        return DataRecons


class TMO_Net(nn.Module):
    def __init__(self,
                 #  number of multimodal
                 modal_num: int,
                 #  multimodal dimension number list
                 modal_dim: list[int],
                 #  dimension of latent representation
                 latent_dim: int,
                 #  dimension of encoder hidden layer
                 encoder_hidden_dims: list[int],
                 #  dimension of decoder hidden layer
                 decoder_hidden_dims: list[int],
                 #  distribution/reconstruct loss of each modality
                 omics_data_type: list[str],
                 # weight of kl loss
                 kl_loss_weight: float,
                 pretrain=False):
        super(TMO_Net, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([encoder(modal_dim[i], latent_dim, encoder_hidden_dims, 'relu') for j in range(self.k)]) for i in
            range(self.k))
        self.self_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], decoder_hidden_dims, 'relu') for i in range(self.k)])

        self.cross_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], decoder_hidden_dims, 'relu') for i in range(self.k)])

        #   modality-invariant representation
        self.share_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                           nn.BatchNorm1d(latent_dim),
                                           nn.ReLU())
        #   modal align
        self.discriminator = nn.Sequential(nn.Linear(latent_dim, 16),
                                           nn.BatchNorm1d(16),
                                           nn.ReLU(),
                                           nn.Linear(16, modal_num))

        #   infer modal and real modal align
        self.infer_discriminator = nn.ModuleList(nn.Sequential(nn.Linear(latent_dim, 16),
                                                               nn.BatchNorm1d(16),
                                                               nn.ReLU(),
                                                               nn.Linear(16, 2))
                                                 for i in range(self.k))
        #   loss function hyperparameter
        self.loss_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)

        self.omics_data_type = omics_data_type

        self.kl_loss_weight = kl_loss_weight
        #   choose to freeze the parameters of TMO-Net
        if pretrain:
            dfs_freeze(self.encoders)

    #   incomplete omics input
    def forward(self, input_x, omics):
        keys = omics.keys()
        values = list(omics.values())
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]

        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])
        share_representation = self.share_representation(output, omics)

        return output, share_representation

    def compute_generate_loss(self, input_x, batch_size, omics):
        values = list(omics.values())
        # mask_k = random.randint(0, self.k*5)
        mask_k = 10
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        self_elbo = self.self_elbo([output[i][i] for i in range(self.k)], input_x, omics, mask_k)
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics, mask_k)
        cross_infer_loss = self.cross_infer_loss(output, omics, mask_k)
        dsc_loss = self.adversarial_loss(batch_size, output, omics, mask_k)
        generate_loss = self_elbo + 0.1 * (cross_elbo + cross_infer_loss * cross_infer_loss) - (dsc_loss + cross_infer_dsc_loss) * 0.01
        return generate_loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss

    def compute_dsc_loss(self, input_x, batch_size, omics):
        mask_k = random.randint(0, self.k*5)
        values = list(omics.values())
        output = [[0 for i in range(self.k)] for j in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics, mask_k)
        dsc_loss = self.adversarial_loss(batch_size, output, omics, mask_k)
        return cross_infer_dsc_loss, dsc_loss

    def share_representation(self, output, omics):
        values = omics.values()
        share_features = [self.share_encoder(output[i][i][1]) for i in values]
        return share_features

    def self_elbo(self, input_x, input_omic, omics, mask_k):
        self_vae_elbo = 0
        keys = omics.keys()
        values = list(omics.values())
        r_squared = []
        for item, i in enumerate(values):
            if i != mask_k:
                latent_z, mu, log_var = input_x[i]
                reconstruct_omic = self.self_decoders[i](latent_z)
                self_vae_elbo += (self.kl_loss_weight * KL_loss(mu, log_var, 1.0) +
                                  reconstruction_loss(input_omic[item], reconstruct_omic, 1.0, self.omics_data_type[i]))
        return self_vae_elbo

    def cross_elbo(self, input_x, input_omic, batch_size, omics, mask_k):
        cross_elbo = 0
        cross_infer_loss = 0
        cross_modal_KL_loss = 0
        cross_modal_dsc_loss = 0

        values = list(omics.values())
        for i in range(self.k):
            if i in values:
                real_latent_z, real_mu, real_log_var = input_x[i][i]
                mu_set = []
                log_var_set = []
                for j in range(self.k):
                    if (i != j) and (j in values) and (j != mask_k):
                        latent_z, mu, log_var = input_x[j][i]
                        mu_set.append(mu)
                        log_var_set.append(log_var)

                poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
                poe_latent_z = reparameterize(poe_mu, poe_log_var)
                if i in values:
                    reconstruct_omic = self.self_decoders[i](poe_latent_z)

                    cross_elbo += (self.kl_loss_weight * KL_loss(poe_mu, poe_log_var, 1.0) +
                                   reconstruction_loss(input_omic[values.index(i)],
                                                       reconstruct_omic, 1.0, self.omics_data_type[i]))
                    cross_infer_loss += reconstruction_loss(real_mu, poe_mu, 1.0, 'gaussian')

                    cross_modal_KL_loss += KL_divergence(poe_mu, real_mu, poe_log_var, real_log_var)

                    real_modal = torch.tensor([1 for j in range(batch_size)]).cuda()
                    infer_modal = torch.tensor([0 for j in range(batch_size)]).cuda()
                    pred_real_modal = self.infer_discriminator[i](real_mu)
                    pred_infer_modal = self.infer_discriminator[i](poe_mu)

                    cross_modal_dsc_loss += F.cross_entropy(pred_real_modal, real_modal, reduction='none')
                    cross_modal_dsc_loss += F.cross_entropy(pred_infer_modal, infer_modal, reduction='none')

        cross_modal_dsc_loss = cross_modal_dsc_loss.sum(0) / (self.k * batch_size)
        return cross_elbo + cross_infer_loss + self.kl_loss_weight * cross_modal_KL_loss, cross_modal_dsc_loss

    def cross_infer_loss(self, input_x, omics, mask_k):
        values = list(omics.values())
        latent_mu = [0 for i in range(len(input_x))]
        for i in range(self.k):
            if i in values:
                latent_mu[i] = input_x[i][i][1]
        infer_loss = 0
        for i in range(len(input_x)):
            if (i in values) and (i != mask_k):
                for j in range(len(input_x)):
                    if (i != j) and (j in values):
                        latent_z_infer, latent_mu_infer, _ = input_x[j][i]
                        infer_loss += reconstruction_loss(latent_mu_infer, latent_mu[i], 1.0, 'gaussian')
        return infer_loss / len(values)

    def adversarial_loss(self, batch_size, output, omics, mask_k):
        dsc_loss = 0
        values = list(omics.values())
        for i in range(self.k):
            if (i in values) and (i != mask_k):
                latent_z, mu, log_var = output[i][i]
                shared_fe = mu

                real_modal = (torch.tensor([i for j in range(batch_size)])).cuda()
                pred_modal = self.discriminator(shared_fe)
                # print(i, pred_modal)
                dsc_loss += F.cross_entropy(pred_modal, real_modal, reduction='none')

        dsc_loss = dsc_loss.sum(0) / (self.k * batch_size)
        return dsc_loss

    def get_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.forward(input_x, omics)
        embedding_tensor = []
        keys = list(omics.keys())
        values = list(omics.values())

        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(self.k):
                if (i != j) and (j in values):
                    latent_z, mu, log_var = output[j][i]
                    mu_set.append(mu)
                    log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            if i in values:
                _, omic_mu, omic_log_var = output[i][i]
                joint_mu = (omic_mu + poe_mu) / 2
            else:
                joint_mu = poe_mu
            embedding_tensor.append(joint_mu)
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        return embedding_tensor

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1.0, distance='euclidean'):

        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = F.relu(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss


class DownStream_predictor(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, encoder_hidden_dim, decoder_hidden_dim, pretrain_model_path, task, omics_data_type, fixed, omics, kl_loss_weight):
        super(DownStream_predictor, self).__init__()
        self.k = modal_num
        #   cross encoders
        self.cross_encoders = TMO_Net(modal_num, modal_dim, latent_dim, encoder_hidden_dim, decoder_hidden_dim, omics_data_type, kl_loss_weight)
        if pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
            self.cross_encoders.load_state_dict(model_pretrain_dict)

        self.downstream_predictor = nn.Sequential(nn.Linear(latent_dim * self.k, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.Dropout(0.2),
                                                  nn.ReLU(),

                                                  nn.Linear(128, 64),
                                                  nn.BatchNorm1d(64),
                                                  nn.Dropout(0.2),
                                                  nn.ReLU(),

                                                  nn.Linear(64, task['output_dim']))
        omics_values = set(omics.values())
        for i in range(self.k):
            if i not in omics_values:
                print('fix cross-modal encoders')
                dfs_freeze(self.cross_encoders.encoders[:][i])

        if fixed:
            dfs_freeze(self.cross_encoders)

    def un_dfs_freeze_encoder(self):
        un_dfs_freeze(self.cross_encoders)

    #   return embedding
    def get_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.cross_encoders(input_x, batch_size)
        embedding_tensor = []
        keys = list(omics.keys())
        share_features = [share_representation[omics[key]] for key in keys]
        share_features = sum(share_features) / len(keys)
        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(len(omics)):
                latent_z, mu, log_var = output[omics[keys[j]]][i]
                mu_set.append(mu)
                log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)

            embedding_tensor.append(poe_mu)
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        # multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return embedding_tensor

    # omics dict, 标注输入的数据
    def forward(self, input_x, batch_size, omics):
        multi_representation = self.cross_encoders.get_embedding(input_x, batch_size, omics)
        downstream_output = self.downstream_predictor(multi_representation)

        return downstream_output


class SNN_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SNN_block, self).__init__()
        self.snn_block = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.BatchNorm1d(output_dim),
                                       nn.ReLU())

    def forward(self, input_x):
        return self.snn_block(input_x)


class SNN_encoder(nn.Module):
    def __init__(self, model_dim, hidden_dim, latent_dim):
        super(SNN_encoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(model_dim, hidden_dim[0]),
                                 nn.BatchNorm1d(hidden_dim[0]),
                                 nn.ReLU())
        self.encoders = nn.ModuleList(
            [SNN_block(hidden_dim[i], hidden_dim[i + 1]) for i in range(0, len(hidden_dim) - 1)])
        self.fc2 = nn.Linear(hidden_dim[-1], latent_dim)

    def forward(self, input_x):
        x = self.fc1(input_x)
        for snn_block in self.encoders:
            x = snn_block(x)
        x = self.fc2(x)
        return x


class PanCancer_SNN_predictor(nn.Module):
    def __init__(self, modal_num, model_dim, hidden_dim, latent_dim, task):
        super(PanCancer_SNN_predictor, self).__init__()
        self.modal_num = modal_num
        self.omics_encoders = nn.ModuleList(
            [SNN_encoder(model_dim[i], hidden_dim, latent_dim) for i in range(modal_num)])

        self.predictor = nn.Sequential(nn.Linear(modal_num * latent_dim, 64),
                                       nn.BatchNorm1d(64),
                                       nn.ReLU(),
                                       nn.Linear(64, task['output_dim']))

    def forward(self, input_x):
        output = [torch.Tensor([]) for i in range(self.modal_num)]

        for i in range(self.modal_num):
            output[i] = self.omics_encoders[i](input_x[i])
        embedding_tensor = torch.Tensor([]).cuda()
        for i in range(self.modal_num):
            embedding = output[i]
            embedding_tensor = torch.concat((embedding_tensor, embedding), dim=1)
        output = self.predictor(embedding_tensor)
        return output


class DualAutoEncoder(nn.Module):
    def __init__(self, input_dims_a, input_dims_b, hidden_dims, latent_dim):
        super(DualAutoEncoder, self).__init__()

        # Encoder for Modality A
        self.encoderA = nn.Sequential(
            nn.Linear(input_dims_a, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Decoder for Modality A
        self.decoderA = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dims_a)
        )

        # Encoder for Modality B
        self.encoderB = nn.Sequential(
            nn.Linear(input_dims_b, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Decoder for Modality B
        self.decoderB = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dims_b)
        )

    def forward(self, x_a, x_b):
        # Encode inputs from both modalities
        z_a = self.encoderA(x_a)
        z_b = self.encoderB(x_b)

        # Decode into both modalities from each latent representation
        recon_a_from_a = self.decoderA(z_a)
        recon_b_from_a = self.decoderB(z_a)
        recon_a_from_b = self.decoderA(z_b)
        recon_b_from_b = self.decoderB(z_b)

        recon_a_loss = reconstruction_loss(x_a, recon_a_from_a, 1, 'gaussian') + reconstruction_loss(x_a, recon_a_from_b, 1, 'gaussian')
        recon_b_loss = reconstruction_loss(x_b, recon_b_from_b, 1, 'gaussian') + reconstruction_loss(x_b, recon_b_from_a, 1, 'gaussian')

        return recon_a_from_a, recon_b_from_a, recon_a_from_b, recon_b_from_b, recon_a_loss, recon_b_loss


def calculate_r_squared_torch(y_true, y_pred):
    # 计算总平方和 (TSS)
    tss = torch.sum((y_true - torch.mean(y_true, axis=0))**2, axis=0)
    # 计算残差平方和 (RSS)
    rss = torch.sum((y_true - y_pred)**2, axis=0)
    # 计算R平方值
    r_squared = 1 - (rss / tss)
    return r_squared


def calc_frac(x1_mat, x2_mat):  # function to calculate FOSCTTM values
    nsamp = x1_mat.shape[0]
    total_count = nsamp * (nsamp - 1)
    rank = 0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx, :], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank += sort_euc_dist.index(true_nbr)

    frac = float(rank) / total_count
    return frac

def r_squared_pytorch(y_actual, y_predicted):
    # Ensure the input tensors are of correct shape
    if y_actual.shape != y_predicted.shape:
        raise ValueError("Shapes of y_actual and y_predicted do not match.")

    # Calculate the mean of the observed values for each dimension
    y_mean = y_actual.mean(dim=0)

    # Calculate the total sum of squares (SS_tot) for each dimension
    ss_tot = ((y_actual - y_mean) ** 2).sum(dim=0)

    # Calculate the residual sum of squares (SS_res) for each dimension
    ss_res = ((y_actual - y_predicted) ** 2).sum(dim=0)

    # Calculate the R^2 value for each dimension
    r2_values = 1 - (ss_res / ss_tot)

    return r2_values


def cosine_similarity_rows_tensor(A, B):
    # Calculate the dot product between corresponding rows of A and B
    dot_product = torch.sum(A * B, dim=1)

    # Calculate the L2 norm for A and B
    norm_A = torch.norm(A, dim=1)
    norm_B = torch.norm(B, dim=1)

    # Calculate the cosine similarity for corresponding rows
    cosine_sim = dot_product / (norm_A * norm_B)

    return cosine_sim


def corrected_pearson_correlation_rows_tensor(A, B):
    # Calculate means of A and B
    mean_A = torch.mean(A, dim=1, keepdim=True)
    mean_B = torch.mean(B, dim=1, keepdim=True)

    # Calculate the covariance between A and B
    covariance = torch.sum((A - mean_A) * (B - mean_B), dim=1) / (A.shape[1] - 1)

    # Calculate the standard deviations of A and B
    std_A = torch.std(A, dim=1, unbiased=True)
    std_B = torch.std(B, dim=1, unbiased=True)

    # Calculate the Pearson correlation coefficient for corresponding rows
    correlation = covariance / (std_A * std_B)

    return correlation