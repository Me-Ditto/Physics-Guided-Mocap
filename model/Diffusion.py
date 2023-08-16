## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
import pickle
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from utils.imutils import *
import diffusion.gaussian_diffusion as gd
from diffusion.resample import create_named_schedule_sampler
from utils import dist_util
from tqdm import tqdm
from model.VAE import VAE
from autograd import elementwise_grad as egrad

os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.append('./')
from phys_new.uhc.agents.agent_copycat_zc import AgentCopycat_zc

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Diffusion_v34(nn.Module):
    def __init__(self, smpl, frame_length=24, joint_num=24, out_chans=512, embed_dim_ratio=32, depth=4, latentD=32,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, 
                 data_rep='rot6d', njoints=144, nfeats=1, timestep=10, **kwargs):


        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            frame_length (int, tuple): input frame number
            joint_num (int, tuple): joints number
            out_chans (int): number of output channels
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            data_rep: (string): data representation
            njoints: (int): dimension of joints feature
            nfeats: (int): dimension of diffusion feature
            timestep: (int): diffusion timestep
        """
        super().__init__()
        self.smpl = smpl
        self.vae = VAE(smpl, latentD)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = 256 
        out_dim = 24 * 6    
        feature_dim = 64 
        num_features = 2048
        self.hidden_size = 256 

        self.Temporal_pos_embed = None
        self.blocks = None
        self.Temporal_norm = None
        self.motion2d_head = None
        self.Temporal_patch_to_embedding = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # pose diffusion model
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=out_chans, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(out_chans)
        self.Spatial_patch_to_embedding = nn.Linear(embed_dim, out_chans)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, frame_length + 1, out_chans))
        self.conv = None
        self.conv0_0 = None
        self.conv0_1 = None
        self.conv0_2 = None
        self.mask_token1 = None
        self.pos_drop = nn.Dropout(p=0.0)

        # trans diffusion model
        self.trans_Spatial_blocks = nn.ModuleList([
            Block(
                dim=out_chans, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.trans_Spatial_norm = norm_layer(out_chans)
        self.trans_Spatial_patch_to_embedding = nn.Linear(embed_dim, out_chans)
        self.trans_Spatial_pos_embed = nn.Parameter(torch.zeros(1, frame_length + 1, out_chans))
        self.trans_pos_drop = nn.Dropout(p=0.0)


        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats
        self.trans_rep = 'xyz'

        self.dropout = drop_rate

        self.microbatch = 32
        self.batch_size = 32

        self.nframes = frame_length

        self.num_timesteps = timestep
        steps = timestep
        scale_beta = 1.  # no scaling
        betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)

        betas = np.array(betas, dtype=np.float64)**2
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (steps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )


        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.num_timesteps)


        self.cond_process = nn.Linear(num_features, feature_dim)
        self.grad_process = nn.Linear(joint_num * 3, feature_dim)

        self.input_process = InputProcess(self.data_rep, self.input_feats, feature_dim * 2)
        self.input_trans_process = InputProcess(self.trans_rep, 3, feature_dim * 2)

        self.sequence_pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)

        self.embed_timestep = TimestepEmbedder(self.hidden_size, self.sequence_pos_encoder)


        self.head = nn.Sequential(
            nn.LayerNorm(out_chans),
            nn.Linear(out_chans , out_dim),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(out_chans),
            nn.Linear(out_chans, 10),
            torch.nn.Conv1d(in_channels=frame_length, out_channels=1, kernel_size=1),
        )
        
        self.cam_head = nn.Sequential(
            nn.LayerNorm(out_chans + 3),
            nn.Linear(out_chans + 3, 3),
        )

        # guidance
        self.proj_2d_loss_grad = egrad(self.proj_2d_loss)
        self.proj_2d_body_loss_grad = egrad(self.proj_2d_body_loss)

        # physics simulator
        self.agent = AgentCopycat_zc(num_threads=8)
        
    def proj_2d_loss(self, pred_3d_joints, trans, gt_2d_joints, data, ord=2):
        bs, f, _, _ = pred_3d_joints.shape
        
        center = data['center'].detach().cpu().numpy()
        scale = data['scale'].detach().cpu().numpy()
        img_h = data["img_h"].reshape(bs, 1)
        img_w = data["img_w"].reshape(bs, 1)
        focal_length = data["focal_length"].detach().cpu().numpy()
        cx = data["intri"][:,:,0,2]
        cy = data["intri"][:,:,1,2]
        camera_center = torch.stack([cx, cy], dim=-1)
        camera_center = camera_center.contiguous().view(bs*f,2)
        camera_center = camera_center.detach().cpu().numpy()

        pred_2d_joints = perspective_projection_np(pred_3d_joints.reshape(bs*f, -1, 3) + trans[:,:,None,:].reshape(bs*f, 1, 3),
                                                   rotation=np.tile(np.eye(3, dtype=np.float32), (bs*f, 1, 1)),
                                                   translation=np.tile(np.zeros(3, dtype=np.float32), (bs*f, 1)),
                                                   focal_length=focal_length.reshape(-1),
                                                   camera_center=camera_center)
        pred_2d_joints = pred_2d_joints.reshape(bs, f, -1, 2)
        pred_2d_joints = normalize(pred_2d_joints, center, scale)

        pred_2d_joints = pred_2d_joints[..., :2]
        conf = gt_2d_joints[..., 2]
        gt_2d_joints = gt_2d_joints[..., :2]
        
        outliers = conf[...,:] < 0.1
        curr_weighting = np.array(conf)

        if ord == 1:
            loss = np.abs(
                gt_2d_joints -
                pred_2d_joints.squeeze()).squeeze().mean()
        else: 
            diff = (pred_2d_joints - gt_2d_joints.squeeze())**2
            diff = diff.sum(axis=-1)
            curr_weighting[outliers] = 0
            loss = (diff * curr_weighting).sum() 

        return loss
    
    def proj_2d_body_loss(self, pred_3d_joints, trans, gt_2d_joints, data, ord=2):
        bs, f, _, _ = pred_3d_joints.shape
        
        center = data['center'].detach().cpu().numpy()
        scale = data['scale'].detach().cpu().numpy()
        img_h = data["img_h"].reshape(bs, 1)
        img_w = data["img_w"].reshape(bs, 1)
        focal_length = data["focal_length"].detach().cpu().numpy()
        cx = data["intri"][:,:,0,2]
        cy = data["intri"][:,:,1,2]
        camera_center = torch.stack([cx, cy], dim=-1)
        camera_center = camera_center.contiguous().view(bs*f,2)
        camera_center = camera_center.detach().cpu().numpy()

        pred_2d_joints = perspective_projection_np(pred_3d_joints.reshape(bs*f, -1, 3) + trans[:,:,None,:].reshape(bs*f, 1, 3),
                                                   rotation=np.tile(np.eye(3, dtype=np.float32), (bs*f, 1, 1)),
                                                   translation=np.tile(np.zeros(3, dtype=np.float32), (bs*f, 1)),
                                                   focal_length=focal_length.reshape(-1),
                                                   camera_center=camera_center)
        pred_2d_joints = pred_2d_joints.reshape(bs, f, -1, 2)
        pred_2d_joints = normalize(pred_2d_joints, center, scale)

        pred_2d_joints = pred_2d_joints[..., :2]
        conf = gt_2d_joints[..., 2]
        gt_2d_joints = gt_2d_joints[..., :2]
        
        outliers = conf[...,:] < 0.1

        # Has to use the current translation (to roughly put at the same position, and then zero out the translation)
        gt2d_center = gt_2d_joints[..., 19, :].copy()
        pred_2d_joints += (gt2d_center - pred_2d_joints[..., 19, :])[:,:,None,:]

        curr_weighting = np.array(conf)

        if ord == 1:
            loss = np.abs(gt_2d_joints - pred_2d_joints.squeeze()).squeeze().mean()
        else:
            diff = (gt_2d_joints - pred_2d_joints.squeeze())**2
            diff = diff.sum(axis=-1)
            curr_weighting[outliers] = 0
            loss = (diff * curr_weighting).sum()

        return loss

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            gd._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + gd._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            gd._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + gd._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = gd._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = gd._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def preprocess(self, x_start, t, data, noise=None):
        if noise is None:
            noise, _, _, _ = self.vae.VAE_encoder(data)
        x_t = self.q_sample(x_start, t, noise=noise)
        return x_t     
    

    def Spatial_forward_features(self, x):
        b, f, p = x.shape

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)

        return x
    
    def Spatial_forward_features_trans(self, x):
        b, f, p = x.shape

        x = self.trans_Spatial_patch_to_embedding(x)
        x += self.trans_Spatial_pos_embed
        x = self.trans_pos_drop(x)

        for blk in self.trans_Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)

        return x


    def forward(self, data):
        bs, f, _ = data['gt_pose_6d'].shape
        batch = data['gt_pose_6d']
        batch = batch.reshape(bs, f, -1).permute((0, 2, 1))[:,:,None,:]

        cond = data['features']
        gt_keypoints_2d = data['gt_keyp']
        gt_trans = data['trans']
        gt_shape = data['shape']

        # cam estimation
        center = data["center"]
        scale = data["scale"]
        img_h = data["img_h"].reshape(bs, 1)
        img_w = data["img_w"].reshape(bs, 1)
        focal_length = data["focal_length"]
        extri = data["extri"]

            

        if 'timestep' in data.keys():
            t = data['timestep']  # test
            trans_t = data['trans_t']
            pred_shape = data['shape']
            x_t = data['x_t']
            x_mean = data['x_mean']
            trans_mean = data['trans_mean']

            i = t.detach().cpu().numpy()[0]

            #### for the gradient of pred joints 3d
            trans_t_ori = trans_t + trans_mean
            x_t_ori = x_t + x_mean

            pose_aa = x_t_ori.permute(0,3,2,1)
            pose_aa = pose_aa.contiguous().view(-1, 6)
            pose_aa = rotation_6d_to_matrix(pose_aa)
            pose_aa = matrix_to_axis_angle(pose_aa)
            pose_aa = pose_aa.view(bs,f,24,3)
            pred_shape = gt_shape

            # if t > 0.4*num_timesteps, do not use physical imitation
            if i >= 0.4 * self.num_timesteps:
                phys_keypoints_2d, phys_verts, phys_joints = self.smpl_forward(pose_aa, pred_shape, trans_t_ori, data, proj=False)
                pose_phys = pose_aa
                trans_phys = trans_t_ori
            else:
                pose_phys, trans_phys, phys_joints = self.phys_imitation(pose_aa, trans_t_ori, pred_shape, extri, data)

            # feed physical results into diffusion model
            pose_phys = axis_angle_to_matrix(pose_phys)
            pose_phys_6d = matrix_to_rotation_6d(pose_phys)
            pose_phys_6d = pose_phys_6d.reshape(pose_phys_6d.shape[0], f, -1)
            pose_phys_6d = pose_phys_6d[:,:,None,:].permute(0,3,2,1)
            x_t_phys = pose_phys_6d - x_mean
            trans_t_phys = trans_phys - trans_mean

        # calculate gradient
        phys_joints = phys_joints.detach().cpu().numpy()
        trans_phys = trans_phys.detach().cpu().numpy()
        gt_keypoints_2d = gt_keypoints_2d.detach().cpu().numpy()

        proj2dgrad_body = self.proj_2d_body_loss_grad(phys_joints, trans_phys, gt_keypoints_2d, data, ord=2)  
        proj2dgrad = self.proj_2d_loss_grad(phys_joints, trans_phys, gt_keypoints_2d, data, ord=2)   
        proj2dgrad[..., 3:] = proj2dgrad_body[..., 3:]
        proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more

        # gradient guidance, if t > 0.4*num_timesteps, guidance=0
        grad = torch.from_numpy(proj2dgrad).to(device=x_t.device)
        grad = grad.view(bs, f, -1)

        outliers = t >= 0.4 * self.num_timesteps
        grad[outliers] = torch.zeros((1, f, grad.shape[-1]), device=x_t.device, dtype=grad.dtype)
        
        grad = self.grad_process(grad)
        grad = grad.permute(1,0,2) # [seq_len, bs, d]

        # if t > 0.4*num_timesteps, feed x_t into diffusion model
        x = x_t_phys
        x[outliers] = x_t[outliers]
        y = cond
        timesteps = t

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        y = self.cond_process(y)
        y = y.permute(1,0,2) # [seq_len, bs, d]

        ## pose diffusion
        x = self.input_process(x)
        input_x = torch.cat((y, grad, x), axis=2)

        xseq = torch.cat((emb, input_x), axis=0)  # [seq_len + 1, bs, d]
        xseq = self.encoder(xseq)

        pose_feat = xseq[:, 1:, :] 
        pose = self.head(pose_feat)

        decoded = pose.view(-1, 6)
        Xout = rotation_6d_to_matrix(decoded)
        Xout = matrix_to_axis_angle(Xout)
        Xout = Xout.view(bs,f,24,3)

        
        ## trans diffusion
        cx, cy, b = center[:, :, 0], center[:, :, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        bbox_info[:, :, :2] = bbox_info[:, :, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, :, 2] = (bbox_info[:, :, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        # if t > 0.4*num_timesteps, feed trans_t into diffusion model
        trans = trans_t_phys
        trans[outliers] = trans_t[outliers]
        trans = self.input_trans_process(trans)
        input_trans = torch.cat((y, grad, trans), axis=2)

        trans_seq = torch.cat((emb, input_trans), axis=0)  # [seq_len + 1, bs, d]
        trans_seq = self.trans_encoder(trans_seq)

        trans_feat = trans_seq[:, 1:, :]
        trans_feat = torch.cat([trans_feat, bbox_info],2)  
        pred_cam = self.cam_head(trans_feat)

        # convert the camera parameters from the crop camera to the full camera
        intri_cx = data["intri"][:,:,0,2]
        intri_cy = data["intri"][:,:,1,2]
        full_img_shape = torch.stack((intri_cy*2, intri_cx*2), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)


        # convert denoised distribution x ~ N(0, σ) to original distribution x ~ N(μ, σ)
        pose6d_ori = pose + x_mean.permute(0,3,2,1).squeeze(2)
        pred_pose_ori = pose6d_ori.view(-1, 6)
        pred_pose_ori = rotation_6d_to_matrix(pred_pose_ori)
        pred_pose_ori = matrix_to_axis_angle(pred_pose_ori)
        pred_pose_ori = pred_pose_ori.view(bs,f,24,3)

        pred_trans_ori = pred_trans + trans_mean

        # smpl
        tmp_pose = pred_pose_ori.contiguous().view(-1,72)
        tmp_shape = pred_shape.contiguous().view(-1,10)
        tmp_shape = tmp_shape[:,None,:]
        tmp_shape = tmp_shape.expand(bs, f, 10).contiguous().view(-1,10)
        temp_trans = torch.zeros((tmp_pose.shape[0], 3), dtype=tmp_pose.dtype, device=tmp_pose.device)
        
        pred_verts, pred_joints = self.smpl(tmp_shape, tmp_pose, temp_trans, halpe=True)
        pred_verts = pred_verts.view(bs, f, -1, 3)

        # keyp estimation
        camera_center = torch.stack([intri_cx, intri_cy], dim=-1)
        camera_center = camera_center.contiguous().view(bs*f,2)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans_ori[:,:,None,:].view(bs*f, 1, 3),
                                                   rotation=torch.eye(3, device=pred_joints.device, dtype=pred_joints.dtype).unsqueeze(0).expand(bs*f, -1, -1),
                                                   translation=torch.zeros(3, device=pred_joints.device, dtype=pred_joints.dtype).unsqueeze(0).expand(bs*f, -1),
                                                   focal_length=focal_length.view(-1),
                                                   camera_center=camera_center)
        pred_keypoints_2d = pred_keypoints_2d.view(bs, f, -1, 2)
        pred_keypoints_2d = normalize(pred_keypoints_2d, center, scale)

        results = {'pred_pose':pred_pose_ori, 'shape':pred_shape, 'pose6d':pose6d_ori, 'pred_verts':pred_verts, 'pred_keyp':pred_keypoints_2d, 'pred_trans':pred_trans_ori, 'pred_pose_denoised':Xout, 'pose6d_denoised':pose, 'pred_trans_denoised':pred_trans}


        return results

    def inference(self, data):
        cond = data['features']

        bs, f, _ = cond.shape
        shape = (bs, self.njoints, self.nfeats, f)

        # cam estimation
        center = data["center"]
        scale = data["scale"]
        img_h = data["img_h"].reshape(bs, 1)
        img_w = data["img_w"].reshape(bs, 1)
        focal_length = data["focal_length"]
        extri = data['extri']
        intri = data['intri']
        gt_trans = data["trans"]
        gt_shape = data["shape"]
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        assert isinstance(shape, (tuple, list))

        # init noise input    
        x_prior, vae_shape, trans_prior, x_mean, shape_mean, trans_mean = self.vae.VAE_encoder(data)
        
        # x0' = x0 - μ
        x_mean = x_mean.permute(0,2,1).unsqueeze(2)
        x_prior = x_prior.permute(0,2,1).unsqueeze(2)
        x = x_prior - x_mean
        transl = trans_prior - trans_mean
        
        smpl_shape = gt_shape
        smpl_trans = transl

        indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                B, C = x.shape[:2]
                assert t.shape == (B,)

                input_data = {}
                input_data['x_t'] = x
                input_data['x_mean'] = x_mean
                input_data['trans_mean'] = trans_mean
                input_data['features'] = cond
                input_data['timestep'] = t
                input_data['gt_pose_6d'] = data['gt_pose_6d']
                input_data['gt_keyp'] = data['gt_keyp']

                # cam estimation
                input_data['shape'] = smpl_shape
                input_data['trans'] = gt_trans
                input_data['trans_t'] = smpl_trans
                input_data['center'] = center
                input_data['scale'] = scale
                input_data['img_h'] = img_h
                input_data['img_w'] = img_w
                input_data['focal_length'] = focal_length
                input_data['extri'] = extri
                input_data['intri'] = intri
                
                model_output = self.forward(input_data)
                pred_xstart = model_output['pose6d_denoised'].permute(0,2,1)[:,:,None,:]
                pred_trans_start = model_output['pred_trans_denoised']
                
                model_variance, model_log_variance = (self.posterior_variance, self.posterior_log_variance_clipped)

                model_variance_trans = gd._extract_into_tensor(model_variance, t, smpl_trans.shape)
                model_log_variance_trans = gd._extract_into_tensor(model_log_variance, t, smpl_trans.shape)

                model_variance = gd._extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = gd._extract_into_tensor(model_log_variance, t, x.shape)
                
                # the mean of x_t-1
                model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

                # the mean of trans_t-1
                model_mean_trans, _, _ = self.q_posterior_mean_variance(x_start=pred_trans_start, x_t=smpl_trans, t=t)

                assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
                assert (model_mean_trans.shape == model_log_variance_trans.shape == pred_trans_start.shape == smpl_trans.shape)

                x_prior, pred_shape, trans_prior, x_mean, shape_mean, trans_mean = self.vae.VAE_encoder(data)
                
                x_mean = x_mean.permute(0,2,1).unsqueeze(2)
                x_prior = x_prior.permute(0,2,1).unsqueeze(2)
                noise = x_prior - x_mean

                noise_transl = trans_prior - trans_mean

                
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                nonzero_mask_trans = (
                    (t != 0).float().view(-1, *([1] * (len(smpl_trans.shape) - 1)))
                )
                
                # add noise to x_t-1 and trans_t-1
                sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
                x = sample
                sample_trans = model_mean_trans + nonzero_mask_trans * torch.exp(0.5 * model_log_variance_trans) * noise_transl
                smpl_trans = sample_trans

                
        # 6d to aa
        output = sample.reshape(bs, self.njoints, f)
        output = output.permute(0,2,1)
        pose6d = output.contiguous()
        output = output.reshape(-1, 6)

        output = rotation_6d_to_matrix(output)
        output = matrix_to_axis_angle(output)
        output = output.view(bs,f,24,3)

        # convert denoised distribution x ~ N(0, σ) to original distribution x ~ N(μ, σ)
        pose6d_ori = sample + x_mean
        pose6d_ori = pose6d_ori.reshape(bs, self.njoints, f)
        pose6d_ori = pose6d_ori.permute(0,2,1)
        pred_pose_ori = pose6d_ori.view(-1, 6)
        pred_pose_ori = rotation_6d_to_matrix(pred_pose_ori)
        pred_pose_ori = matrix_to_axis_angle(pred_pose_ori)
        pred_pose_ori = pred_pose_ori.view(bs,f,24,3)

        pred_trans_ori = sample_trans + trans_mean

        
        return dict(pred_pose=pred_pose_ori, pose6d=pose6d_ori, pred_trans=pred_trans_ori, shape=model_output['shape'],
                    pred_verts=model_output['pred_verts'], pred_keyp=model_output['pred_keyp'], pred_pose_denoised = output, pose6d_denoised=pose6d, pred_trans_denosied=pred_trans_start)
    
    def encoder(self, x):
        f, b, _ = x.shape

        x = x.permute(1,0,2)

        x = self.Spatial_forward_features(x)

        return x

    def trans_encoder(self, x):
        f, b, _ = x.shape
        x = x.permute(1,0,2)

        x = self.Spatial_forward_features_trans(x)

        return x

    def smpl_forward(self, pose, shape, trans, data, proj=True):
        bs, f, _, _ = pose.shape

        center = data['center']
        scale = data['scale']
        img_w = data['img_w'].reshape(bs, 1)
        img_h = data['img_h'].reshape(bs, 1)
        focal_length = data["focal_length"]
        cx = data["intri"][:,:,0,2]
        cy = data["intri"][:,:,1,2]
        camera_center = torch.stack([cx, cy], dim=-1)
        camera_center = camera_center.contiguous().view(bs*f,2)

        tmp_pose = pose.contiguous().view(-1,72)
        tmp_shape = shape.contiguous().view(-1,10)
        tmp_shape = tmp_shape[:,None,:]
        tmp_shape = tmp_shape.expand(bs, f, 10).contiguous().view(-1,10)
        temp_trans = torch.zeros((tmp_pose.shape[0], 3), dtype=tmp_pose.dtype, device=tmp_pose.device)
        
        pred_verts, pred_joints = self.smpl(tmp_shape, tmp_pose, temp_trans, halpe=True)

        # keyp estimation
        if proj:
            pred_keypoints_2d = perspective_projection(pred_joints + trans[:,:,None,:].view(bs*f, 1, 3),
                                                    rotation=torch.eye(3, device=trans.device).unsqueeze(0).expand(bs*f, -1, -1),
                                                    translation=torch.zeros(3, device=trans.device).unsqueeze(0).expand(bs*f, -1),
                                                    focal_length=focal_length.view(-1),
                                                    camera_center=camera_center)
            pred_keypoints_2d = pred_keypoints_2d.view(bs, f, -1, 2)
            pred_keypoints_2d = normalize(pred_keypoints_2d, center, scale)
        else:
            pred_keypoints_2d = None

        pred_joints = pred_joints.view(bs, f, -1, 3)
        pred_verts = pred_verts.view(bs, f, -1, 3)

        return pred_keypoints_2d, pred_verts, pred_joints

    def phys_imitation(self, poses, transes, shapes, extris, data):
        experts = []
        for seq_id, (pose, shape, trans, extri) in enumerate(zip(poses, shapes, transes, extris)):

            '''Convert to World Coordinate System'''
            matrix = torch.linalg.inv(extri.clone())
            f = pose.shape[0]
            shape = shape[None,:]
            shape = shape.expand(f, 10).contiguous().view(-1,10)

            zeros_trans = torch.zeros((f, 3), device=pose.device, dtype=pose.dtype)
            verts_zero_cam, joints_zero_cam = self.smpl(shape, pose, zeros_trans, halpe=False)
            root = joints_zero_cam[:,0,:]

            
            # convert pose params
            oritation = axis_angle_to_matrix(pose[:,0,:].clone())
            oritation = torch.matmul(matrix[:,:3,:3], oritation)
            oritation = matrix_to_axis_angle(oritation)
            pose[:,0,:] = oritation


            # rot root joint
            root_cam = root + trans
            root_world = torch.einsum('bij,bkj->bki', matrix[:,:3,:3], root_cam[:,None,:])
            root_world = root_world.squeeze(-2) + matrix[:,:3,3]
            trans = root_world - root

            ### interpolate for 30fps
            interpolated_pose, interpolated_trans = interpolate(pose, trans)
            
            expert = self.agent.smpl2expert(interpolated_pose, interpolated_trans, shape[0].detach().cpu().numpy(), 'seq_%s' %seq_id)    
            experts.append(expert)
            

        self.agent.tasks.append(experts)
        res_dicts, results = self.agent.eval_policy(epoch=19000, dump=True)

        phys_pose_batch, phys_trans_batch, joints_batch = [],[],[]
        for i, (key, res) in enumerate(results.items()):
            # Transform data to SMPL parameters
            res_pose, res_trans = self.agent.res2smpl(results, key)

            # extrapolate for original 10 fps
            phys_pose, phy_trans = extrapolate(res_pose, res_trans)

            # vis
            visualize = False
            if visualize :
                from phys_new.uhc.utils.copycat_visualizer import CopycatVisualizer
                vis = CopycatVisualizer(R'./assets/test_new.xml', self.agent, {key:res})
                vis.record_video(batch_id=i)

            # convert pose params
            oritation = axis_angle_to_matrix(phys_pose[:,0,:].clone())
            oritation = torch.matmul(extris[i][:,:3,:3], oritation)
            oritation = matrix_to_axis_angle(oritation)
            phys_pose[:,0,:] = oritation

            # rot root joint
            root_world = root + phy_trans
            root_cam = torch.einsum('bij,bkj->bki', extris[i][:,:3,:3], root_world[:,None,:])
            root_cam = root_cam.squeeze(-2) + extris[i][:,:3,3]

            zeros_trans = torch.zeros((f, 3), device=pose.device, dtype=pose.dtype)
            verts_zero, joints_zero = self.smpl(shape, phys_pose, zeros_trans, halpe=True)
            phy_trans = root_cam - root

            phys_pose_batch.append(phys_pose)
            phys_trans_batch.append(phy_trans)
            joints_batch.append(joints_zero)
        
        phys_pose_batch = torch.stack(phys_pose_batch)
        phys_trans_batch = torch.stack(phys_trans_batch)
        joints_batch = torch.stack(joints_batch)

        ### clear phys reference
        self.agent.tasks.clear()
   
        return phys_pose_batch, phys_trans_batch, joints_batch
    
    def load_pkl(self, path):
        """"
        load pkl file
        """
        with open(path, 'rb') as f:
            param = pickle.load(f, encoding='iso-8859-1')
        # param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
        return param

    

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



def normalize(keyp, center, scale):
    # normalize
    center_tmp = center[:, :, None, :]
    scale_tmp = scale[:, :, None, None]
    keyp = (keyp - center_tmp) / scale_tmp / constants.IMG_RES
    return keyp


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    

class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        if len(x.shape) == 4:
            bs, njoints, nfeats, nframes = x.shape
            x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        elif len(x.shape) == 3:
            bs, nframes, nfeats = x.shape
            x = x.permute((1, 0, 2)).reshape(nframes, bs, nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)