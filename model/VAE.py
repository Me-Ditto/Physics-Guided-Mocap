
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
from torch.nn import functional as F
from utils.imutils import*



def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, f, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, f, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, f,) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, f, 2) original image height and width
    :param focal_length: shape=(N, f, )
    :return:
    """
    img_h, img_w = full_img_shape[:, :, 0], full_img_shape[:, :, 1]
    cx, cy, b = center[:, :, 0], center[:, :, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, :, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, :, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, :, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


class VAE_v12(nn.Module):
    def __init__(
            self,
            smpl,
            latentD=32,
            frame_length=24,
            joint_num=26,
            n_layers=1,
            hidden_size=512,
            bidirectional=True,
            **kwargs,
    ):
        super(VAE_v12, self).__init__()

        self.smpl = smpl

        self.dropout = nn.Dropout(p=.1, inplace=False)
        self.latentD = latentD
        self.encoder_gru = nn.GRU(
            input_size=2048 + 3,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.decoder_gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        if bidirectional:
            self.hidden_dim = hidden_size * 2
        else:
            self.hidden_dim = hidden_size

        self.encoder_linear1 = nn.Sequential(nn.Linear(self.hidden_dim, 256),
                                        nn.LayerNorm(256),
                                        nn.Dropout(0.1),\
                                        )
        self.encoder_linear2 = nn.Sequential(nn.Linear(512, 256),
                                        nn.LayerNorm(256),
                                        nn.Dropout(0.1),\
                                        )
        self.encoder_residual = nn.Sequential(nn.Linear(2048 + 3, 256),
                                        nn.LayerNorm(256),
                                        nn.Dropout(0.1),\
                                        )

        self.mu_linear = nn.Sequential(nn.Linear(256, latentD),
                                        )

        self.var_linear = nn.Sequential(nn.Linear(256, latentD),
                                        )

        self.decoder_linear1 = nn.Sequential(nn.Linear(latentD, 256),\
                                            ) 
        self.decoder_linear2 = nn.Sequential(nn.Linear(self.hidden_dim, 256),
                                            nn.Dropout(0.1),\
                                            ) 
        self.decoder_linear3 = nn.Sequential(nn.Linear(512, 256),
                                            nn.Dropout(0.1),\
                                            nn.Linear(256, 24*6),
                                            )    ###23 * 6
        
        self.decoder_mlp = nn.Sequential(nn.Linear(latentD, 256),
                                            nn.Dropout(0.1),\
                                            nn.Linear(256, 24*6),
                                            )

        self.cam_head = nn.Sequential(
            nn.LayerNorm(self.latentD + 3),
            nn.Linear(self.latentD + 3 , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.latentD + 3),
            nn.Linear(self.latentD + 3, 10),
            torch.nn.Conv1d(in_channels=24, out_channels=1, kernel_size=1),
        )  #  in_channels=num_frames

    def encode(self, x, n, t):
        linear_proj = x.contiguous().view([-1, x.size(-1)])
        linear_proj = self.encoder_residual(linear_proj)
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.encoder_gru(x)
        y = y.permute(1,0,2)
        y = y.contiguous().view([-1, y.size(-1)])
        y = self.encoder_linear1(y)
        y = torch.cat([y, linear_proj], dim=1)
        y = self.encoder_linear2(y)
        mean = self.mu_linear(y).view([n, t, -1])
        std = self.var_linear(y).view([n, t, -1])

        q_z = torch.distributions.normal.Normal(mean, F.softplus(std))
        return q_z, y



    def VAE_encoder(self, data):
        with torch.no_grad():

            x = data['features']

            # input size: batch, frame length, keyp vector
            n,t,f = x.shape

            center = data["center"]
            scale = data["scale"]
            img_h = data["img_h"].reshape(n, 1)
            img_w = data["img_w"].reshape(n, 1)
            focal_length = data["focal_length"]

            cx, cy, b = center[:, :, 0], center[:, :, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            # The constants below are used for normalization, and calculated from H36M data.
            # It should be fine if you use the plain Equation (5) in the paper.
            bbox_info[:, :, :2] = bbox_info[:, :, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, :, 2] = (bbox_info[:, :, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
            
            x = torch.cat([x, bbox_info],2)
            q_z, y= self.encode(x, n, t)
            q_z_sample = q_z.sample()

            # pose
            pose6d = q_z_sample[:,:,:-3].view([n, t, -1])

            out = pose6d.reshape(-1,6)
            out = rotation_6d_to_matrix(out)
            out = matrix_to_axis_angle(out)
            out = out.view([n, t, 24, 3])

            mean_6d = q_z.mean[:,:,:-3]

            # trans
            pred_cam = q_z_sample[:,:,-3:]
            # convert the camera parameters from the crop camera to the full camera
            full_img_shape = torch.stack((img_h, img_w), dim=-1)
            pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

            pred_cam_mean = q_z.mean[:,:,-3:]
            pred_trans_mean = cam_crop2full(pred_cam_mean, center, scale, full_img_shape, focal_length)


            pred_shape = data['shape']
            pred_shape_mean = data['shape']

           
        return pose6d, pred_shape, pred_trans, mean_6d, pred_shape_mean, pred_trans_mean



def normalize(keyp, data, bs, f):
    center = data['center']
    scale = data['scale']
    # normalize
    center_tmp = center[:, :, None, :]
    center_tmp = center_tmp.expand(bs, f, keyp.shape[2], keyp.shape[3])
    scale_tmp = scale[:, :, None, None]
    scale_tmp = scale_tmp.expand(bs, f, keyp.shape[2], keyp.shape[3])
    keyp[:,:,:,:2] = (keyp[:,:,:,:2] - center_tmp) / scale_tmp / constants.IMG_RES
    return keyp