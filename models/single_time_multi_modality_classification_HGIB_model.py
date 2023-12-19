import numpy as np
import torch
import itertools
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *
import pdb

class SingleTimeMultiModalityClassificationHGIBModel(BaseModel):
    def name(self):
        return 'SingleTimeMultiModalityClassificationHGIBModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        # current input size is [121, 145, 121]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']


        if self.using_focalloss:
            self.loss_names.append('focal')
        self.loss_names.append('kl')
        # self.loss_names.append('kd')

        self.model_names = ['Encoder_MRI', 'Encoder_PET', 'Encoder_NonImage', 'Decoder_HGIB']

        self.netEncoder_MRI = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_PET = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_NonImage = networks3D.init_net_update(networks3D.Encoder_NonImage(in_channels=7, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netDecoder_HGIB = networks3D.init_net_update(HGIB_v1(1024*3, 1024, 3, use_bn=False, heads=1), self.gpu_ids)
        self.netClassifier = torch.nn.Linear(1024*3, 3)
        self.use_modal_cls = False
        if self.use_modal_cls:
            self.netClassifier_MRI = torch.nn.Linear(1024, 3)
            self.netClassifier_PET = torch.nn.Linear(1024, 3)
            self.netClassifier_NonImage = torch.nn.Linear(1024, 3)
            self.netClassifier_Max = torch.nn.Linear(1024, 3)
            self.netClassifier_Sum = torch.nn.Linear(1024, 3)
            self.netClassifier_Avg = torch.nn.Linear(1024, 3)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netClassifier.to(self.gpu_ids[0])
            self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
            if self.use_modal_cls:
                self.netClassifier_NonImage.to(self.gpu_ids[0])
                self.netClassifier_NonImage = torch.nn.DataParallel(self.netClassifier_NonImage, self.gpu_ids)
                self.netClassifier_MRI.to(self.gpu_ids[0])
                self.netClassifier_MRI = torch.nn.DataParallel(self.netClassifier_MRI, self.gpu_ids)
                self.netClassifier_PET.to(self.gpu_ids[0])
                self.netClassifier_PET = torch.nn.DataParallel(self.netClassifier_PET, self.gpu_ids)
                self.netClassifier_Sum.to(self.gpu_ids[0])
                self.netClassifier_Sum = torch.nn.DataParallel(self.netClassifier_Sum, self.gpu_ids)
                self.netClassifier_Max.to(self.gpu_ids[0])
                self.netClassifier_Max = torch.nn.DataParallel(self.netClassifier_Max, self.gpu_ids)
                self.netClassifier_Avg.to(self.gpu_ids[0])
                self.netClassifier_Avg = torch.nn.DataParallel(self.netClassifier_Avg, self.gpu_ids)
        self.criterionCE = torch.nn.CrossEntropyLoss()

        # initialize optimizers
        if self.isTrain:
            # Why not update encoders?
            #self.optimizer = torch.optim.Adam(itertools.chain(self.netDecoder_HGIB.parameters()), #, self.netEncoder_MRI.parameters(), self.netEncoder_PET.parameters(), self.netEncoder_NonImage.parameters(), ),
            #                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.use_modal_cls:
                self.optimizer = torch.optim.Adam([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netClassifier.parameters()},
                                                {'params': self.netClassifier_PET.parameters()},
                                                {'params': self.netClassifier_MRI.parameters()},
                                                {'params': self.netClassifier_NonImage.parameters()},
                                                {'params': self.netClassifier_Sum.parameters()},
                                                {'params': self.netClassifier_Avg.parameters()},
                                                {'params': self.netClassifier_Max.parameters()},
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer = torch.optim.Adam([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netClassifier.parameters()},
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input):
        self.MRI = input[0].to(self.device)
        self.PET = input[1].to(self.device)
        self.target = input[2].to(self.device)
        self.nonimage = input[3].to(self.device)

    def set_HGinput(self, input):
        self.embedding = self.embedding.to(self.device)
        self.target = input.to(self.device)

    def ExtractFeatures(self):
        with torch.no_grad():
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

    def forward(self, phase='train'):
        prediction_encoder = self.netClassifier(self.embedding)
        embedding_MRI, embedding_PET, embedding_NonImage = torch.split(self.embedding, self.embedding.size(1)//3, dim=1)
        if self.use_modal_cls:
            prediction_MRI = self.netClassifier_MRI(embedding_MRI)
            prediction_PET = self.netClassifier_PET(embedding_PET)
            prediction_NonImage = self.netClassifier_NonImage(embedding_NonImage)
            embedding_sum = embedding_MRI + embedding_PET + embedding_NonImage
            embedding_max = torch.max(embedding_MRI, embedding_PET)
            embedding_max = torch.max(embedding_NonImage, embedding_max)
            prediction_Avg = self.netClassifier_Avg(embedding_sum / 3.0)
            prediction_Sum = self.netClassifier_Sum(embedding_sum)
            prediction_Max = self.netClassifier_Max(embedding_max)

        self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
        if phase == 'train':
            idx = torch.tensor(range(self.len_train)).to(self.device)
        elif phase == 'test':
            idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
        else:
            print('Wrong in loss calculation')
            exit(-1)

        self.loss_cls = self.criterionCE(prediction_encoder[idx], self.target[idx])
        if self.use_modal_cls:
            self.loss_cls = self.loss_cls + self.criterionCE(prediction_MRI[idx], self.target[idx]) + \
                self.criterionCE(prediction_PET[idx], self.target[idx]) + self.criterionCE(prediction_NonImage[idx], self.target[idx])
            self.loss_cls = self.loss_cls + self.criterionCE(prediction_Max[idx], self.target[idx]) + \
                self.criterionCE(prediction_Sum[idx], self.target[idx]) + self.criterionCE(prediction_Avg[idx], self.target[idx])
        weight = [0.5, 0.5]
        for t, pred in enumerate(self.prediction[0]):
            self.loss_cls += weight[t] * self.criterionCE(pred[idx], self.target[idx])

        self.loss_kl = self.prediction[1]
        # self.loss_kd = 0

        if self.using_focalloss:
            gamma = 0.5
            alpha = 2
            pt = torch.exp(-self.loss_cls)
            self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
            self.loss = self.loss_cls + self.loss_focal
        else:
            self.loss = self.loss_cls
        self.loss= self.loss_cls + self.loss_kl * self.beta

        self.prediction_cur = self.prediction[0][-1][idx]
        self.target_cur = self.target[idx]
        self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
        self.pred_encoder = prediction_encoder[idx]
        self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
        if self.use_modal_cls:
            self.pred_MRI = prediction_MRI[idx]
            self.acc_MRI = (torch.softmax(self.pred_MRI, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_PET = prediction_PET[idx]
            self.acc_PET = (torch.softmax(self.pred_PET, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_NonImage = prediction_NonImage[idx]
            self.acc_NonImage = (torch.softmax(self.pred_NonImage, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_Max = prediction_Max[idx]
            self.acc_Max = (torch.softmax(self.pred_Max, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_Avg= prediction_Avg[idx]
            self.acc_Avg = (torch.softmax(self.pred_Avg, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_Sum = prediction_Sum[idx]
            self.acc_Sum = (torch.softmax(self.pred_Sum, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
    
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        # forward pass is here
        self.forward('train')
        self.loss.backward()
        self.optimizer.step()

    def validation(self):
        with torch.no_grad():
            self.forward('test')

    def get_pred_encoder(self):
        if self.use_modal_cls:
            return self.pred_encoder, self.pred_MRI, self.pred_PET, self.pred_NonImage
        else:
            return self.pred_encoder
    
    def get_pred_reduction(self):
        return self.pred_Avg, self.pred_Sum, self.pred_Max
