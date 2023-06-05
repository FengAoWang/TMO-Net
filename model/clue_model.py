import torch.nn as nn
import torch
import torch.nn.functional as F
from util.loss_function import KL_loss, reconstruction_loss, KL_divergence
from functools import reduce


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
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], latent_dim),
                                     nn.BatchNorm1d(latent_dim),
                                     # nn.Dropout(0.2),
                                     nn.ReLU())

        self.mu_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                          nn.ReLU()
                                          )
        self.log_var_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                               nn.ReLU()
                                               )

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
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], input_dim),
                                     )

    def forward(self, latent_z):
        return self.decoder(latent_z)


class Clue_model(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, omics_data, pretrain=False):
        super(Clue_model, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([encoder(modal_dim[i], latent_dim, hidden_dim) for j in range(self.k)]) for i in
            range(self.k))
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
        #   loss function hyperparameter
        self.loss_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)

        self.lmf_fusion = LMF_fusion(64, 16, 4)

        self.omics_data = omics_data
        if pretrain:
            dfs_freeze(self.encoders)

    #   incomplete omics input
    def forward(self, input_x, omics):
        keys = omics.keys()
        values = list(omics.values())
        output = [[0 for i in range(self.k)] for j in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])
        share_representation = self.share_representation(output, omics)

        return output, share_representation

    def compute_generate_loss(self, input_x, batch_size, omics):
        values = list(omics.values())
        output = [[0 for i in range(self.k)] for j in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        self_elbo = self.self_elbo([output[i][i] for i in range(self.k)], input_x, omics)
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics)
        cross_infer_loss = self.cross_infer_loss(output, omics)
        dsc_loss = self.adversarial_loss(batch_size, output, omics)
        generate_loss = self_elbo + cross_elbo + cross_infer_loss * cross_infer_loss - dsc_loss * 0.1 - cross_infer_dsc_loss * 0.1
        # generate_loss = self_elbo
        return generate_loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss

    def compute_dsc_loss(self, input_x, batch_size, omics):
        values = list(omics.values())
        output = [[0 for i in range(self.k)] for j in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics)
        dsc_loss = self.adversarial_loss(batch_size, output, omics)
        return cross_infer_dsc_loss, dsc_loss

    def share_representation(self, output, omics):
        values = omics.values()
        share_features = [self.share_encoder(output[i][i][1]) for i in values]
        return share_features

    def self_elbo(self, input_x, input_omic, omics):
        self_vae_elbo = 0
        keys = omics.keys()
        values = list(omics.values())
        for item, i in enumerate(values):
            latent_z, mu, log_var = input_x[i]
            reconstruct_omic = self.self_decoders[i](latent_z)
            self_vae_elbo += 0.01 * KL_loss(mu, log_var, 1.0) + reconstruction_loss(input_omic[item], reconstruct_omic, 1.0,
                                                                             self.omics_data[i])
        return self_vae_elbo

    def cross_elbo(self, input_x, input_omic, batch_size, omics):
        cross_elbo = 0
        cross_infer_loss = 0
        cross_modal_KL_loss = 0
        cross_modal_dsc_loss = 0

        values = list(omics.values())
        keys = omics.keys()

        for i in range(self.k):
            if i in values:
                real_latent_z, real_mu, real_log_var = input_x[i][i]
                mu_set = []
                log_var_set = []
                for j in range(self.k):
                    if (i != j) and (j in values):
                        latent_z, mu, log_var = input_x[j][i]
                        mu_set.append(mu)
                        log_var_set.append(log_var)
                poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
                poe_latent_z = reparameterize(poe_mu, poe_log_var)
                if i in values:
                    reconstruct_omic = self.self_decoders[i](poe_latent_z)

                    cross_elbo += 0.01 * KL_loss(poe_mu, poe_log_var, 1.0) + reconstruction_loss(input_omic[values.index(i)], reconstruct_omic, 1.0,
                                                                                          self.omics_data[i])
                    cross_infer_loss += reconstruction_loss(real_mu, poe_mu, 1.0, 'gaussian')

                    cross_modal_KL_loss += KL_divergence(poe_mu, real_mu, poe_log_var, real_log_var)

                    real_modal = torch.tensor([1 for j in range(batch_size)]).cuda()
                    infer_modal = torch.tensor([0 for j in range(batch_size)]).cuda()
                    pred_real_modal = self.infer_discriminator[i](real_mu)
                    pred_infer_modal = self.infer_discriminator[i](poe_mu)

                    cross_modal_dsc_loss += F.cross_entropy(pred_real_modal, real_modal, reduction='none')
                    cross_modal_dsc_loss += F.cross_entropy(pred_infer_modal, infer_modal, reduction='none')

        cross_modal_dsc_loss = cross_modal_dsc_loss.sum(0) / (self.k * batch_size)
        return cross_elbo + cross_infer_loss + 0.01 * cross_modal_KL_loss, cross_modal_dsc_loss

    def cross_infer_loss(self, input_x, omics):
        values = list(omics.values())
        latent_mu = [0 for i in range(len(input_x))]
        for i in range(self.k):
            if i in values:
                latent_mu[i] = input_x[i][i][1]
        infer_loss = 0
        for i in range(len(input_x)):
            if i in values:
                for j in range(len(input_x)):
                    if (i != j) and (j in values):
                        latent_z_infer, latent_mu_infer, _ = input_x[j][i]
                        infer_loss += reconstruction_loss(latent_mu_infer, latent_mu[i], 1.0, 'gaussian')
        return infer_loss / len(values)

    def adversarial_loss(self, batch_size, output, omics):
        dsc_loss = 0
        values = list(omics.values())
        for i in range(self.k):
            if i in values:
                latent_z, mu, log_var = output[i][i]
                shared_fe = self.share_encoder(mu)

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
        share_features = sum(share_representation) / len(keys)

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
        multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return multi_representation

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1.0, distance='cosine'):

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


class LMF_fusion(nn.Module):
    def __init__(self, input_dim, ranks, modal_nums):
        super(LMF_fusion, self).__init__()
        self.input_dim = input_dim
        self.ranks = ranks
        self.k = modal_nums

        self.rank_weights = nn.ParameterList((nn.Parameter(torch.Tensor(self.ranks, self.input_dim, self.input_dim),
                                                           requires_grad=True)) for i in range(self.k))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.ranks), requires_grad=True)
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.input_dim), requires_grad=True)

        for i in range(self.k):
            nn.init.xavier_normal(self.rank_weights[i])
        nn.init.xavier_normal(self.fusion_weights)

    def forward(self, input_x):

        fusion_features = [torch.matmul(input_x[i], self.rank_weights[i]) for i in range(self.k)]
        fusion_features = reduce(torch.mul, fusion_features)
        fusion_features = fusion_features.permute(1, 0, 2).squeeze()
        output = torch.matmul(self.fusion_weights, fusion_features)
        output = output.view(-1, self.input_dim)
        return output


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims
        self.video_in = input_dims
        self.text_in = input_dims

        self.output_dim = input_dims
        self.rank = rank
        self.use_softmax = use_softmax

        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = nn.Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.video_factor = nn.Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        self.text_factor = nn.Parameter(torch.Tensor(self.rank, self.text_in + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        nn.init.xavier_normal(self.audio_factor)
        nn.init.xavier_normal(self.video_factor)
        nn.init.xavier_normal(self.text_factor)
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = audio_x
        video_h = video_x
        text_h = text_x
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), audio_h), dim=1)
        _video_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), video_h), dim=1)
        _text_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output


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
        # self.lmf = LMF(64, 16)
        self.lmf_fusion = LMF_fusion(64, 16, self.k)
        self.downstream_predictor = nn.Sequential(nn.Linear(latent_dim * (self.k + 1), 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.Dropout(0.2),
                                                  nn.ReLU(),

                                                  nn.Linear(128, 64),
                                                  nn.BatchNorm1d(64),
                                                  nn.Dropout(0.2),
                                                  nn.ReLU(),

                                                  nn.Linear(64, task['output_dim']))

        if fixed:
            print('fix cross encoders')
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
        embedding_tensor = self.lmf_fusion(embedding_tensor)
        multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return multi_representation

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
