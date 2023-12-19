import numpy as np
import torch
from torch.nn import functional as F
import pdb
import itertools
from tqdm import tqdm
import torchio as tio

from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *
from utils import center_loss, randaugment, fda

class SingleTimeMultiModalityClassificationHGIBSemiPLModel(BaseModel):
    def name(self):
        return 'SingleTimeMultiModalityClassificationHGIBSemiPLModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        class_num = 3
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        # current input size is [121, 145, 121]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']
        self.train_encoders = opt.train_encoders

        if self.using_focalloss:
            self.loss_names.append('focal')
        self.loss_names.append('kl')
        self.weight_center = opt.weight_center
        if self.weight_center > 0:
            self.loss_names.append('center')
            self.loss_center = 0
            use_gpu = len(self.gpu_ids) > 0
            self.CenterLoss = center_loss.CenterLoss(num_classes=class_num, feat_dim=1024*3, use_gpu=use_gpu)
            assert self.train_encoders, 'Center loss should be added to learnable encoders, but the encoders are fixed.'
        # self.loss_names.append('kd')

        self.model_names = ['Encoder_MRI', 'Encoder_PET', 'Encoder_NonImage', 'Decoder_HGIB']

        self.netEncoder_MRI = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_PET = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_NonImage = networks3D.init_net_update(networks3D.Encoder_NonImage(in_channels=7, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.use_strong_aug = opt.use_strong_aug
        self.netClassifier = torch.nn.Linear(1024*3, class_num)
        self.conf_thre = 0.95
        # self.netClassifier = torch.nn.Linear(1024, class_num)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netClassifier.to(self.gpu_ids[0])
            self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
        #networks3D.define_Decoder(1024*3, opt.ndf, opt.netD,
        #                                    opt.n_layers_D, class_num, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, opt.onefc)
        self.netDecoder_HGIB = networks3D.init_net_update(HGIB_v1(1024*3, 1024, 3, use_bn=False, heads=1), self.gpu_ids)

        self.criterionCE = torch.nn.CrossEntropyLoss()

        # initialize optimizers
        if self.isTrain:
            # Why not update encoders?
            if self.train_encoders:
                # TODO: use AdamW
                self.optimizer = torch.optim.Adam([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netEncoder_MRI.parameters()}, 
                                                {'params': self.netEncoder_PET.parameters()}, 
                                                {'params': self.netEncoder_NonImage.parameters()},
                                                {'params': self.netClassifier.parameters()}
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer = torch.optim.Adam([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netClassifier.parameters()}
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)
        
        self.rand_gamma = tio.RandomGamma(log_gamma=(0.70, 1.50), p=0.3)
        #self.randaug = randaugment.RandAugmentFixMatch()
        #MRI_strong_aug = self.randaug(MRI_strong_aug)

        #MRI_aug = fda.mix_amplitude(MRI_np, PET_np)
        #MRI_aug_vis = MRI_aug[0, 56:59, :, :]
        #from torchvision.utils import save_image
        #save_image(255*MRI_np[0, 56:59, :, :], 'before_aug.png')
        #save_image(255*MRI_aug_vis, 'after_aug.png')
        #from monai.transforms.utility import array
        #MRI_strong_aug = array.ToPIL()(255*MRI_np[0, 56, :, :].astype(torch.uint8))

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input, use_strong_aug=False):
        self.MRI = input[0].to(self.device)  # shape: [30, 1, 96, 96, 96]
        self.PET = input[1].to(self.device)
        self.target = input[2].to(self.device)
        self.nonimage = input[3].to(self.device)
        if use_strong_aug:
            self.MRI_strongaug = input[4].to(self.device)
            self.PET_strongaug = input[5].to(self.device)
        else:
            self.MRI_strongaug = None
            self.PET_strongaug = None

    def set_HGinput(self, input):
        self.embedding = self.embedding.to(self.device)
        self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if phase == 'test':
            with torch.no_grad():
                self.embedding_MRI = self.netEncoder_MRI(self.MRI)
                self.embedding_PET = self.netEncoder_PET(self.PET)
                self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        else:
            self.embedding_MRI = self.netEncoder_MRI(self.MRI)
            self.embedding_PET = self.netEncoder_PET(self.PET)
            self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        return self.embedding_MRI, self.embedding_PET, self.embedding_NonImage

    def HGconstruct(self, embedding_MRI, embedding_PET, embedding_NonImage):
        G = Hypergraph.from_feature_kNN(embedding_MRI, self.K_neigs, self.device)
        G.add_hyperedges_from_feature_kNN(embedding_PET, self.K_neigs)
        G.add_hyperedges_from_feature_kNN(embedding_NonImage, self.K_neigs)
        self.G = G  # construct graph for the forward pass
        self.embedding = torch.Tensor(np.hstack((embedding_MRI, embedding_PET, embedding_NonImage))).to(self.device)

    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            if self.train_encoders:
                assert train_loader is not None, 'train_loader is None, please provide train_loader for training'
                len_train_loader = len(train_loader)
                train_loader_x_iter = iter(train_loader)
                if train_loader_u is not None:
                    len_train_loader = max(len(train_loader), len(train_loader_u))
                    train_loader_u_iter = iter(train_loader_u)
                for i in range(len_train_loader): 
                    try:
                        data = next(train_loader_x_iter)
                    except StopIteration:
                        train_loader_x_iter = iter(train_loader)
                        data = next(train_loader_x_iter)
                    try:
                        data_u = next(train_loader_u_iter)
                    except StopIteration:
                        train_loader_u_iter = iter(train_loader_u)
                        data_u = next(train_loader_u_iter)
                    self.set_input(data)
                    self.ExtractFeatures(phase='train')
                    embedding = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                    # embedding = self.embedding_MRI + self.embedding_PET + self.embedding_NonImage
                    prediction = self.netClassifier(embedding)
                    self.loss_cls = self.criterionCE(prediction, self.target)
                    if self.weight_center > 0:
                        self.loss_center = self.CenterLoss(embedding, self.target)
                        if epoch is not None:
                            weight_center = self.weight_center * min(epoch / 80., 1.)
                        else:
                            weight_center = self.weight_center
                        self.loss_cls = self.loss_cls + weight_center * self.loss_center
                    self.set_input(data_u, use_strong_aug=self.use_strong_aug)
                    self.ExtractFeatures(phase='train')
                    embedding_u = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                    # embedding = self.embedding_MRI + self.embedding_PET + self.embedding_NonImage
                    prediction_u = self.netClassifier(embedding_u)
                    
                    # create hypergraph
                    self.HGconstruct(self.embedding_MRI.cpu().detach().numpy(), 
                                    self.embedding_PET.cpu().detach().numpy(), 
                                    self.embedding_NonImage.cpu().detach().numpy())
                    self.info([self.embedding_MRI.size(0), 0])
                    # consistency between the discriminative classifier and the hyper-graph net on unlabeled data
                    self.set_HGinput(self.target)
                    prediction_u_graph_net = self.netDecoder_HGIB(self.embedding, self.G)
                    prediction_u_graph = F.softmax(prediction_u_graph_net[0][-1], 1)
                    prediction_u = F.softmax(prediction_u, 1)
                    loss_u = ((prediction_u_graph - prediction_u)**2).sum(1).mean()
                    # pseudo-labeling
                    output_u = prediction_u.clone().detach()
                    max_prob, label_u = output_u.max(1)
                    mask_u = (max_prob >= self.conf_thre).float()
                    if self.MRI_strongaug is not None and self.PET_strongaug is not None:
                        embedding_MRI_aug = self.netEncoder_MRI(self.MRI_strongaug)
                        embedding_PET_aug = self.netEncoder_PET(self.PET_strongaug)
                        embedding_u_aug = torch.cat((embedding_MRI_aug, embedding_PET_aug, self.embedding_NonImage), dim=1)
                        prediction_u_aug = self.netClassifier(embedding_u_aug)
                        loss_ce_u = F.cross_entropy(prediction_u_aug, label_u, reduction='none')
                        loss_ce_u = (loss_ce_u * mask_u).mean()
                    else:
                        loss_ce_u = F.cross_entropy(prediction_u_graph_net[0][-1], label_u, reduction='none')
                        loss_ce_u = (loss_ce_u * mask_u).mean()
                    if epoch is not None:
                        weight_u = self.weight_u * min(epoch / 80., 1.)
                    else:
                        weight_u = self.weight_u
                    #self.loss_cls = self.loss_cls + weight_u * loss_u + loss_ce_u
                    if self.using_focalloss:
                        gamma = 0.5
                        alpha = 2
                        pt = torch.exp(-self.loss_cls)
                        self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                        # TODO: remove self.loss_cls below
                        self.loss = self.loss_focal + self.loss_cls
                    else:
                        self.loss = self.loss_cls
                    self.loss = self.loss + weight_u * loss_u + loss_ce_u
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
                    if i % 100 == 0:
                        print('Iteration {}, total loss for encoders {:.6f}, CE loss (unlabeled) {:.6f}, '
                              'consistency loss (unlabeled) {:.6f}, CE loss (labeled) {:.6f}'.format(
                            i, self.loss.item(), loss_ce_u.item(), (weight_u * loss_u).item(), 
                            self.loss.item() - loss_ce_u.item() - (weight_u * loss_u).item()))
                MRI, PET, Non_Img, Label, length = self.get_features([train_loader, test_loader])
                # create hypergraph
                self.HGconstruct(MRI, PET, Non_Img)
                self.info(length)
                self.set_HGinput(Label)

            num_graph_update = self.num_graph_update
            idx = torch.tensor(range(self.len_train)).to(self.device)
        elif phase == 'test':
            num_graph_update = 1
            idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
        else:
            print('Wrong in loss calculation')
            exit(-1)

        for i in range(num_graph_update):
            # prediction is  [y1, y5], (kl1 + kl5)/2.0
            prediction_encoder = self.netClassifier(self.embedding)
            self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
            if self.train_encoders:
                # the netClassifier has been train during the phase of training encoders
                self.loss_cls = 0
            else:
                self.loss_cls = self.criterionCE(prediction_encoder, self.target)
            weight = [0.5, 0.5]
            # self.prediction[0] = [y1, y5]
            for t, pred in enumerate(self.prediction[0]):
                self.loss_cls += weight[t] * self.criterionCE(pred[idx], self.target[idx])

            self.loss_kl = self.prediction[1]
            # self.loss_kd = 0

            if self.using_focalloss:
                gamma = 0.5
                alpha = 2
                pt = torch.exp(-self.loss_cls)
                self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                # TODO: remove the self.loss_cls below
                self.loss = self.loss_cls + self.loss_focal
            else:
                self.loss = self.loss_cls
            if self.weight_center > 0 and phase == 'train':
                self.loss_center = self.CenterLoss(self.embedding[idx], self.target[idx])
                if epoch is not None:
                    weight_center = self.weight_center * min(epoch / 80., 1.)
                else:
                    weight_center = self.weight_center
                self.loss = self.loss + weight_center * self.loss_center
            self.loss = self.loss + self.loss_kl * self.beta

            self.prediction_cur = self.prediction[0][-1][idx]
            self.target_cur = self.target[idx]
            self.pred_encoder = prediction_encoder[idx]
            self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            
            if phase == 'test':
                continue
            if not self.train_encoders and phase == 'train':
                if i == 0:
                    print('extract the features at the first time when updating hyper-graph net.')
                    MRI, PET, Non_Img, Label, length = self.get_features([train_loader_u, test_loader])
                # create hypergraph
                self.HGconstruct(MRI, PET, Non_Img)
                self.info(length)
                self.set_HGinput(Label)
                prediction_u_graph = self.netDecoder_HGIB(self.embedding, self.G)
                prediction_u_graph = F.softmax(prediction_u_graph[0][-1], 1)
                # embedding = self.embedding_MRI + self.embedding_PET + self.embedding_NonImage
                prediction_u = self.netClassifier(self.embedding)
                prediction_u = F.softmax(prediction_u, 1)
                loss_u = ((prediction_u_graph - prediction_u)**2).sum(1).mean()
                
                if epoch is not None:
                    weight_u = self.weight_u * min(epoch / 80., 1.)
                else:
                    weight_u = self.weight_u
                self.loss = self.loss + self.weight_u * loss_u
            if (i % 50 == 0) or (i == (num_graph_update - 1)):
                print('Update the hyper-graph net for the {} times, total loss {}'.format(i, self.loss.item()))
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def optimize_parameters(self, train_loader, test_loader, train_loader_u=None, epoch=None):
        #self.optimizer.zero_grad()
        # forward pass is here
        self.netClassifier.train()
        self.train()
        self.forward('train', train_loader, test_loader, train_loader_u, epoch)
        #self.loss.backward()
        #self.optimizer.step()

    def validation(self):
        self.netClassifier.eval()
        self.eval()
        with torch.no_grad():
            self.forward('test')

    def get_pred_encoder(self):
        return self.pred_encoder
    
    def get_acc_encoder(self):
        return self.acc_encoder
    
    def get_features(self, loaders):
        # extract featrues from pre-trained model
        # stack them 
        MRI = None
        PET = None
        Non_Img = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(loader):
                self.set_input(data)
                i_MRI, i_PET, i_Non_Img = self.ExtractFeatures()
                if MRI is None:
                    MRI = i_MRI
                    PET = i_PET
                    Non_Img = i_Non_Img
                    Label = data[2]
                else:
                    MRI = torch.cat([MRI, i_MRI], 0)
                    PET = torch.cat([PET, i_PET], 0)
                    Non_Img = torch.cat([Non_Img, i_Non_Img], 0)
                    Label = torch.cat([Label, data[2]], 0)
            length[idx] = MRI.size(0)
            if i % 10 == 0:
                print('extract features for loader {}, interation {}'.format(idx, i))
        length[1] = length[1] - length[0]
        return MRI.cpu().detach().numpy(), PET.cpu().detach().numpy(), Non_Img.cpu().detach().numpy(), Label, length
