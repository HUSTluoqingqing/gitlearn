#coding=utf-8
import torch.nn as nn
import torch
from functions import ReverseLayerF

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DSN(nn.Module):
    def __init__(self, in_ch, out_ch):
    # def __init__(self, code_size=100, n_class=10):
        super(DSN, self).__init__()
        # self.code_size = code_size


        # private source encoder(unet)
        '''
        self.sp_conv1 = DoubleConv(in_ch, 64)
        self.sp_pool1 = nn.MaxPool2d(2)
        self.sp_conv2 = DoubleConv(64, 128)
        self.sp_pool2 = nn.MaxPool2d(2)
        self.sp_conv3 = DoubleConv(128, 256)
        self.sp_pool3 = nn.MaxPool2d(2)
        self.sp_conv4 = DoubleConv(256, 512)
        self.sp_pool4 = nn.MaxPool2d(2)
        self.sp_conv5 = DoubleConv(512, 1024)
        '''
        self.sp = nn.Sequential(
            DoubleConv(in_ch, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )


        # private target encoder(unet)
        '''
        self.tp_conv1 = DoubleConv(in_ch, 64)
        self.tp_pool1 = nn.MaxPool2d(2)
        self.tp_conv2 = DoubleConv(64, 128)
        self.tp_pool2 = nn.MaxPool2d(2)
        self.tp_conv3 = DoubleConv(128, 256)
        self.tp_pool3 = nn.MaxPool2d(2)
        self.tp_conv4 = DoubleConv(256, 512)
        self.tp_pool4 = nn.MaxPool2d(2)
        self.tp_conv5 = DoubleConv(512, 1024)
        '''
        self.tp = nn.Sequential(
            DoubleConv(in_ch, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # shared encoder(unet)
        '''
        self.shared_conv1 = DoubleConv(in_ch, 64)
        self.shared_pool1 = nn.MaxPool2d(2)
        self.shared_conv2 = DoubleConv(64, 128)
        self.shared_pool2 = nn.MaxPool2d(2)
        self.shared_conv3 = DoubleConv(128, 256)
        self.shared_pool3 = nn.MaxPool2d(2)
        self.shared_conv4 = DoubleConv(256, 512)
        self.shared_pool4 = nn.MaxPool2d(2)
        self.shared_conv5 = DoubleConv(512, 1024)
        '''
        self.shared_encoder = nn.Sequential(
            DoubleConv(in_ch, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )


        # shared decoder(unet)
        '''
        self.shared_up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.shared_conv6 = DoubleConv(1024, 512)
        self.shared_up7 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.shared_conv7 = DoubleConv(512, 256)
        self.shared_up8 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.shared_conv8 = DoubleConv(256, 128)
        self.shared_up9 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.shared_conv9 = DoubleConv(128, 64)
        self.shared_conv10 = nn.Conv2d(64, out_ch, 1)
        '''
        self.shared_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 512, 2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            DoubleConv(128, 64),
            nn.Conv2d(64, out_ch, 1)
        )

        #segmentation(unet)
        '''
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        '''
        self.segmentation = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            DoubleConv(128, 64),
            nn.Conv2d(64, out_ch, 1)
        )


        def forward(self, input_data, mode, rec_scheme, p=0.0):
            result = []

            if mode == 'source':
                private = self.sp(input_data)
            elif mode == 'target':
                private = self.tp(input_data)
            result.append(private)

            shared = self.shared_encoder(input_data)
            result.append(shared)

            if rec_scheme == 'share':
                c1 = self.shared_encoder[0](input_data)
                p1 = self.shared_encoder[1](c1)
                c2 = self.shared_encoder[2](p1)
                p2 = self.shared_encoder[3](c2)
                c3 = self.shared_encoder[4](p2)
                p3 = self.shared_encoder[5](c3)
                c4 = self.shared_encoder[6](p3)
                p4 = self.shared_encoder[7](c4)
                c5 = self.shared_encoder[8](p4)
                up_6 = self.segmentation[0](c5)
                merge6 = torch.cat([up_6, c4], dim=1)
                c6 = self.segmentation[1](merge6)
                up_7 = self.segmentation[2](c6)
                merge7 = torch.cat([up_7, c3], dim=1)
                c7 = self.segmentation[3](merge7)
                up_8 = self.segmentation[4](c7)
                merge8 = torch.cat([up_8, c2], dim=1)
                c8 = self.segmentation[5](merge8)
                up_9 = self.segmentation[6](c8)
                merge9 = torch.cat([up_9, c1], dim=1)
                c9 = self.segmentation[7](merge9)
                c10 = self.segmentation[8](c9)
                seg_out = nn.Sigmoid()(c10)
                rec_result = seg_out

            elif rec_scheme == 'all':
                merge_all = torch.cat([private, shared], dim=1)
                rec_all = self.shared_decoder(merge_all)
                rec_result = rec_all

            elif rec_scheme == 'private':
                merge_private = torch.cat([private, torch.zeros_like(shared)], dim=1)
                rec_private = self.shared_decoder(merge_private)
                rec_result = rec_private

            result.append(rec_result)
            

            return result
        ##########################################
        # private source encoder
        ##########################################
        '''
        self.source_encoder_conv = nn.Sequential()
        self.source_encoder_conv.add_module('conv_pse1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                padding=2))
        self.source_encoder_conv.add_module('ac_pse1', nn.ReLU(True))
        self.source_encoder_conv.add_module('pool_pse1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_conv.add_module('conv_pse2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=2))
        self.source_encoder_conv.add_module('ac_pse2', nn.ReLU(True))
        self.source_encoder_conv.add_module('pool_pse2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.source_encoder_fc.add_module('ac_pse3', nn.ReLU(True))
        '''


        #########################################
        # private target encoder
        #########################################

        '''
        self.target_encoder_conv = nn.Sequential()
        self.target_encoder_conv.add_module('conv_pte1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                padding=2))
        self.target_encoder_conv.add_module('ac_pte1', nn.ReLU(True))
        self.target_encoder_conv.add_module('pool_pte1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_conv.add_module('conv_pte2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=2))
        self.target_encoder_conv.add_module('ac_pte2', nn.ReLU(True))
        self.target_encoder_conv.add_module('pool_pte2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.target_encoder_fc.add_module('ac_pte3', nn.ReLU(True))
        '''

        ################################
        # shared encoder (dann_mnist)
        ################################


        '''
        self.shared_encoder_conv = nn.Sequential()
        self.shared_encoder_conv.add_module('conv_se1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                  padding=2))
        self.shared_encoder_conv.add_module('ac_se1', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_conv.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                                                  padding=2))
        self.shared_encoder_conv.add_module('ac_se2', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=code_size))
        self.shared_encoder_fc.add_module('ac_se3', nn.ReLU(True))
        '''

        # classify 10 numbers
        '''
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))
        '''


        '''
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))
        '''

        # classify two domain
        # self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################
        '''
        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU())

        self.shared_decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd3', nn.ReLU())

        self.shared_decoder_conv.add_module('us_sd4', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd5', nn.ReLU(True))

        self.shared_decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                                                                  padding=1))
        '''

    '''
    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            # target private encoder
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 14, 14)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result
    '''