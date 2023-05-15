import torch.nn as nn
import torch
import torch.nn.functional as F
from util.loss_function import KL_loss, reconstruction_loss
from torch.autograd import Variable


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
    tmp = 1.
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]) ** 2)

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(torch.sqrt(poe_var))

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i]) ** 2) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], latent_dim),
                                     nn.BatchNorm1d(latent_dim),
                                     nn.ReLU())

        self.mu_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                          nn.ReLU())
        self.log_var_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                               nn.ReLU())

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def decode(self, latent_z):
        cross_recon_x = self.decoder(latent_z)
        return cross_recon_x

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        # recon_x = self.decoder(latent_z)
        return latent_z, mu, log_var


class decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], input_dim),
                                     nn.ReLU())

    def forward(self, latent_z):
        return self.decoder(latent_z)


class cross_encoder(nn.Module):
    def __init__(self, latent_dim):
        super(cross_encoder, self).__init__()
        self.cross_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                           nn.ReLU())

        self.cross_mu_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                                nn.ReLU())

        self.cross_log_var_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                                     nn.ReLU())

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean

    def forward(self, x):
        x = self.cross_encoder(x)
        mu = self.cross_mu_predictor(x)
        log_var = self.cross_log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        return (latent_z, mu, log_var)


class Clue_model(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, omics_data, pretrain=False):
        super(Clue_model, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([encoder(modal_dim[i], latent_dim, hidden_dim) for j in range(self.k)]) for i in
            range(len(modal_dim)))
        self.self_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])

        self.cross_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])
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

        self.omics_data = omics_data
        if pretrain:
            dfs_freeze(self.encoders)

    # 预留参数 omics，实现缺失模态训练
    # def forward(self, input_x, batch_size):
    #     output = [[0 for j in range(len(input_x))] for i in range(len(input_x))]
    #     for i in range(len(input_x)):
    #         for j in range(len(input_x)):
    #             output[i][j] = self.encoders[i][j](input_x[i])
    #
    #     self_elbo = self.self_elbo([output[i][i] for i in range(len(input_x))], input_x)
    #     cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size)
    #     cross_infer_loss = self.cross_infer_loss(output)
    #     dsc_loss = self.adversarial_loss(batch_size, output)
    #     return output, self_elbo, cross_elbo, cross_infer_loss, dsc_loss, cross_infer_dsc_loss

    def compute_generate_loss(self, input_x, batch_size):
        output = [[0 for j in range(len(input_x))] for i in range(len(input_x))]
        for i in range(len(input_x)):
            for j in range(len(input_x)):
                output[i][j] = self.encoders[i][j](input_x[i])

        self_elbo = self.self_elbo([output[i][i] for i in range(len(input_x))], input_x)
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size)
        cross_infer_loss = self.cross_infer_loss(output)
        dsc_loss = self.adversarial_loss(batch_size, output)
        generate_loss = self_elbo + cross_elbo + cross_infer_loss * cross_infer_loss - dsc_loss * 0.1 - cross_infer_dsc_loss * 0.1
        return generate_loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss

    def compute_dsc_loss(self, input_x, batch_size):
        output = [[0 for j in range(len(input_x))] for i in range(len(input_x))]
        for i in range(len(input_x)):
            for j in range(len(input_x)):
                output[i][j] = self.encoders[i][j](input_x[i])

        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size)
        dsc_loss = self.adversarial_loss(batch_size, output)
        return cross_infer_dsc_loss + dsc_loss

    def share_representation(self, output):
        share_features = []
        for i in range(self.k):
            latent_z, mu, log_var = output[i][i]
            shared_fe = self.share_encoder(mu)
            share_features.append(shared_fe)
        return share_features

    def self_elbo(self, input_x, input_omic):
        self_vae_elbo = 0
        # keys = omics.keys()
        for i in range(self.k):
            latent_z, mu, log_var = input_x[i]
            reconstruct_omic = self.self_decoders[i](latent_z)
            self_vae_elbo += KL_loss(mu, log_var, 1.0) + reconstruction_loss(input_omic[i], reconstruct_omic, 1.0,
                                                                             self.omics_data[i])
        return self_vae_elbo

    def cross_elbo(self, input_x, input_omic, batch_size):
        cross_elbo = 0
        cross_infer_loss = 0
        cross_modal_KL_loss = 0
        cross_modal_dsc_loss = 0

        for i in range(len(input_omic)):
            real_latent_z, real_mu, real_log_var = input_x[i][i]
            mu_set = []
            log_var_set = []
            for j in range(len(input_omic)):
                if i != j:
                    latent_z, mu, log_var = input_x[j][i]
                    mu_set.append(mu)
                    log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)
            reconstruct_omic = self.self_decoders[i](poe_latent_z)

            cross_elbo += KL_loss(poe_mu, poe_log_var, 1.0) + reconstruction_loss(input_omic[i], reconstruct_omic, 1.0,
                                                                                  self.omics_data[i])
            cross_infer_loss += reconstruction_loss(real_mu, poe_mu, 1.0, 'gaussian')

            # cross_modal_KL_loss += KL_loss(poe_mu, poe_log_var, 1.0)
            real_modal = torch.tensor([1 for j in range(batch_size)]).cuda()
            infer_modal = torch.tensor([0 for j in range(batch_size)]).cuda()
            pred_real_modal = self.infer_discriminator[i](real_mu)
            pred_infer_modal = self.infer_discriminator[i](poe_mu)

            cross_modal_dsc_loss += F.cross_entropy(pred_real_modal, real_modal, reduction='none')
            cross_modal_dsc_loss += F.cross_entropy(pred_infer_modal, infer_modal, reduction='none')

        cross_modal_dsc_loss = cross_modal_dsc_loss.sum(0) / (self.k * batch_size)
        return cross_elbo + cross_infer_loss, cross_modal_dsc_loss

    def cross_infer_loss(self, input_x):
        latent_mu = [input_x[i][i][0] for i in range(len(input_x))]
        infer_loss = 0
        for i in range(len(input_x)):
            for j in range(len(input_x)):
                if i != j:
                    latent_z_infer, latent_mu_infer, _ = input_x[j][i]
                    infer_loss += reconstruction_loss(latent_z_infer, latent_mu[i], 1.0, 'gaussian')
        return infer_loss / self.k

    def adversarial_loss(self, batch_size, output):
        dsc_loss = 0
        for i in range(self.k):
            latent_z, mu, log_var = output[i][i]
            shared_fe = self.share_encoder(mu)

            real_modal = (torch.tensor([i for j in range(batch_size)])).cuda()
            pred_modal = self.discriminator(shared_fe)
            # print(i, pred_modal)
            dsc_loss += F.cross_entropy(pred_modal, real_modal, reduction='none')

        dsc_loss = dsc_loss.sum(0) / (self.k * batch_size)
        return dsc_loss

    def latent_z(self, input_x, omics):
        output = [[0 for j in range(len(input_x))] for i in range(len(input_x))]
        for i in range(len(input_x)):
            for j in range(len(input_x)):
                output[i][j] = self.encoders[i][j](input_x[i])
        embedding_tensor = torch.Tensor([]).cuda()
        keys = list(omics.keys())
        for i in range(self.k):
            latent_z, mu, log_var = output[omics[keys[i]]][i]
            embedding_tensor = torch.concat((embedding_tensor, mu), dim=1)
        return embedding_tensor


class DownStream_predictor(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, pretrain_model_path, task, omics_data, fixed):
        super(DownStream_predictor, self).__init__()
        self.k = modal_num
        #   cross encoders
        self.cross_encoders = Clue_model(modal_num, modal_dim, latent_dim, hidden_dim, omics_data)
        if pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
            self.cross_encoders.load_state_dict(model_pretrain_dict)
        #   分类器
        self.downstream_predictor = nn.Sequential(nn.Linear(latent_dim * self.k, 32),
                                                  nn.BatchNorm1d(32),
                                                  nn.ReLU(),
                                                  nn.Linear(32, task['output_dim']))

        self.weights = nn.Parameter(torch.rand(self.k), requires_grad=True)

        if fixed:
            print('fix cross encoders')
            dfs_freeze(self.cross_encoders)

    def un_dfs_freeze_encoder(self):
        un_dfs_freeze(self.cross_encoders)

    # omics dict, 标注输入的数据

    def forward(self, input_x, batch_size, omics):
        output, self_elbo, cross_elbo, cross_infer_loss, dsc_loss, _ = self.cross_encoders(input_x, batch_size)
        embedding_tensor = torch.Tensor([]).cuda()
        keys = list(omics.keys())
        non_negative_weights = F.relu(self.weights)
        normalized_non_negative_weights = non_negative_weights / non_negative_weights.sum()

        # for i in range(self.k):
        #     mean_mu = []
        #     for j in keys:
        #         _, mu, _ = output[omics[j]][i]
        #         mean_mu.append(mu)
        #     mean_mu = sum(mean_mu)
        #     mu_set = []
        #     log_var_set = []
        #     for j in range(len(omics)):
        #         latent_z, mu, log_var = output[omics[keys[j]]][i]
        #         mu_set.append(mu)
        #         log_var_set.append(log_var)
        #     poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
        #     poe_latent_z = reparameterize(poe_mu, poe_log_var)
        #
        #     embedding_tensor.append(normalized_non_negative_weights[i] * mean_mu)
        # embedding_tensor = sum(embedding_tensor)
        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(len(omics)):
                latent_z, mu, log_var = output[omics[keys[j]]][i]
                mu_set.append(mu)
                log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)
            embedding_tensor = torch.concat((embedding_tensor, normalized_non_negative_weights[i] * poe_mu), dim=1)

        downstream_output = self.downstream_predictor(embedding_tensor)
        return downstream_output, self_elbo, cross_elbo, cross_infer_loss, dsc_loss


class SNN_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SNN_block, self).__init__()
        self.snn_block = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.BatchNorm1d(output_dim),
                                       nn.SELU())

    def forward(self, input_x):
        return self.snn_block(input_x)


class SNN_encoder(nn.Module):
    def __init__(self, model_dim, hidden_dim, latent_dim):
        super(SNN_encoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(model_dim, hidden_dim[0]),
                                 nn.BatchNorm1d(hidden_dim[0]),
                                 nn.SELU())
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
                                       nn.SELU(),
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

