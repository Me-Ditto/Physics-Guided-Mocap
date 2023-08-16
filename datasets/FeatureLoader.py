import os
from os.path import join
import sys
import numpy as np
import pickle
import torch
import cv2
import time
from tqdm import tqdm
import torch.utils.data as data
from utils.imutils import *
from torchvision.transforms import Normalize
from utils.rotation_conversions import *
# from utils.dataset_handle import create_video_heatmap

class FeatureData(data.Dataset):
    def __init__(self, train=True, data_folder='', smpl=None, dataset='', frame_length=16, num_joint=24, mocap=False, phys_folder='', offline_phys=False):
        self.is_train = train
        self.is_mocap = mocap
        self.data_type = torch.float32
        self.np_type = np.float32
        
        # Augmentation parameters
        self.threshold = 0.7
        self.noise_factor = 0.4
        self.rot_factor = 30
        self.scale_factor = 0.25
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.img_res = 224

        self.frame_length = frame_length
        # # ECCV
        if num_joint == 24:
            self.PoseInd = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        # J15
        elif num_joint == 15:
            self.PoseInd = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        elif num_joint == 23:
            self.PoseInd = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23]
        elif num_joint == 21:
            self.PoseInd = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21]
            self.flip_index = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,18,20,19]
        elif num_joint == 26:
            self.PoseInd = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

        
        self.dataset_dir = os.path.join(data_folder, dataset)

        self.dataset_name = dataset

        self.offline_phys = offline_phys
        if self.offline_phys:
            self.phys_dir = phys_folder

        
        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir,'annot/train.pkl')

            self.eval = False
            pose2d_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'pose2ds.pkl')
            pose2ds_pred_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'pose2ds_pred.pkl')
            feature_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'img_features.pkl')
            motions_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'motions.pkl')
            imnames_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'imnames.pkl')
            img_size_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'img_size.pkl')
            shapes_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'shapes.pkl')
            intris_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'intris.pkl')
            extris_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'extris.pkl')
            trans_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'trans.pkl')
            center_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'centers.pkl')
            scale_dir = os.path.join(data_folder, 'TMM_annotations', dataset, 'patch_scales.pkl')

            self.pose2ds = self.load_pkl(pose2d_dir) if os.path.exists(pose2d_dir) else self.pose2ds_pred
            self.features = self.load_pkl(feature_dir)
            self.motions = self.load_pkl(motions_dir)
            self.imnames = self.load_pkl(imnames_dir)
            self.img_size = self.load_pkl(img_size_dir)
            self.shapes = self.load_pkl(shapes_dir)
            self.intris = self.load_pkl(intris_dir)
            self.extris = self.load_pkl(extris_dir)
            self.trans = self.load_pkl(trans_dir)
            self.centers = self.load_pkl(center_dir)
            self.scales = self.load_pkl(scale_dir)

        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir,'annot/test.pkl')

            params = self.load_pkl(dataset_annot)
            self.pose2ds_pred, self.pose2ds, self.motions, self.shapes, self.imnames, self.masks, self.img_size, self.intris, self.extris, self.trans, self.features, self.centers, self.scales = [], [], [], [], [], [], [], [], [], [], [], [], []
            for seq in params:
                if len(seq) < 1:
                    continue
                pose2d_pred, pose2d, motion, imname, mask, intri, extri, transl, feature, center, scale = [], [], [], [], [], [], [], [], [], [], []
                for i, frame in enumerate(seq):
                    if frame['0']['pose'] is None:
                        break
                    if i == 0:
                        self.shapes.append(np.array(frame['0']['betas'], dtype=self.np_type))
                        self.img_size.append(frame['h_w'])
                    motion.append(np.array(frame['0']['pose'], dtype=self.np_type))
                    pose2d.append(np.array(frame['0']['halpe_joints_2d'], dtype=self.np_type))
                    pose2d_pred.append(np.array(frame['0']['halpe_joints_2d_pred'], dtype=self.np_type).reshape(-1,3))
                    feature.append(np.array(frame['0']['gt_box_cliff_features'], dtype=self.np_type))
                    imname.append(frame['img_path'])
                    intri.append(frame['0']['intri'])
                    extri.append(frame['0']['extri'])
                    transl.append(frame['0']['trans'])
                    center.append(np.array(frame['0']['center'], dtype=self.np_type))
                    scale.append(np.array(frame['0']['patch_scale'], dtype=self.np_type))
                    
                self.pose2ds.append(pose2d)
                self.pose2ds_pred.append(pose2d_pred)
                self.features.append(feature)
                self.motions.append(motion)
                self.imnames.append(imname)
                self.intris.append(intri)
                self.extris.append(extri)
                self.trans.append(transl)
                self.centers.append(center)
                self.scales.append(scale)
            del frame
            del params



        self.iter_list = []
        
        for i in range(len(self.motions)):
            if self.is_train:
                for n in range(0, (len(self.motions[i]) - self.frame_length)):
                    self.iter_list.append([i, n])
            else:
                for n in range(0, (len(self.motions[i]) - self.frame_length), self.frame_length):
                    self.iter_list.append([i, n])
        
        self.device = torch.device('cpu')
        self.smpl = smpl
        self.ratio = 256 / 24.0
        self.len = len(self.iter_list)
        
    def save_pkl(self, path, result):
        """"
        save pkl file
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, 'wb') as result_file:
            pickle.dump(result, result_file, protocol=2)

    def load_pkl(self, path):
        """"
        load pkl file
        """
        with open(path, 'rb') as f:
            param = pickle.load(f, encoding='iso-8859-1')
        # param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
        return param
 


    def create_data_newaug(self, index=0):
        load_data = {}

        # Load data index
        seq_ind, start    = self.iter_list[index]
        gt_keyps          = np.array(self.pose2ds[seq_ind], dtype=self.np_type).reshape(-1, 26, 3)
        poses             = np.array(self.motions[seq_ind], dtype=self.np_type)
        gt_intri          = np.array(self.intris[seq_ind], dtype=self.np_type)
        gt_extri          = np.array(self.extris[seq_ind], dtype=self.np_type) 
        features          = np.array(self.features[seq_ind], dtype=self.np_type)
        centers           = np.array(self.centers[seq_ind], dtype=self.np_type)
        scales            = np.array(self.scales[seq_ind], dtype=self.np_type)
        transes             = np.array(self.trans[seq_ind], dtype=self.np_type)


        # Get augmentation parameters
        gap = 1
        ind = [start+k*gap for k in range(self.frame_length)]
        flip, pn, rot, sc, ind, gt_input= 0, np.ones(3), 0, 1, ind, 1

        # Load image information
        img_h, img_w = self.img_size[seq_ind]
        
        # Load 2D keyps and poses and trans
        pose = poses[ind].copy()
        trans = transes[ind].copy()
        gt_keyp = gt_keyps[ind].copy()
        gt_keyp = gt_keyp[:,self.PoseInd]
        gt_keyp[:,:,0] = np.clip(gt_keyp[:,:,0], 0, img_w-1)
        gt_keyp[:,:,1] = np.clip(gt_keyp[:,:,1], 0, img_h-1)

        # Load image features
        features = features[ind].copy()
        center = centers[ind].copy()
        scale = scales[ind].copy()
        
        gt_extrisinc = gt_extri[ind].copy()
        gt_extrisinc = torch.from_numpy(gt_extrisinc).float()
        gt_intrinsic = gt_intri[ind].copy()
        focal_length = gt_intrinsic[:,0,0]
        gt_intrinsic = torch.from_numpy(gt_intrinsic).float()
        focal_length = torch.from_numpy(focal_length).float()


        # Get 2D keypoints and apply augmentation transforms
        ori_keyp_2d = gt_keyp.copy()
        ori_keyp_2d = torch.from_numpy(ori_keyp_2d).float()

        
        # normalize
        center_tmp = center[:, None, :]
        scale_tmp = scale[:, None, None]
        gt_keyp[:,:,:2] = (gt_keyp[:,:,:2] - center_tmp) / scale_tmp / constants.IMG_RES
        gt_keyp = torch.from_numpy(gt_keyp).float()


        pose = pose.reshape(self.frame_length, -1, 3)
        pose = torch.from_numpy(pose).float()
        trans = torch.from_numpy(trans).float()
        gt_shape = torch.from_numpy(self.shapes[seq_ind]).float()

        center = torch.from_numpy(np.array(center)).float()

        ##estimate new gt trans according to new intri=[f, img_w/2, img_h/2]
        temp_pose = pose.clone().reshape(-1, 72)
        temp_shape = gt_shape.clone()[None, :]
        temp_shape = temp_shape.expand(temp_pose.shape[0], 10).reshape(-1, 10)
        temp_trans = torch.zeros((temp_pose.shape[0], 3), dtype=temp_pose.dtype, device=temp_pose.device)
        verts, joints = self.smpl(temp_shape, temp_pose, temp_trans, halpe=True)
        verts = verts.squeeze(0).view(self.frame_length, -1, 3)
        joints = joints.squeeze(0)

        ### convert to 6d
        gt_pose_6d = pose.clone()
        gt_pose_6d = axis_angle_to_matrix(gt_pose_6d)
        gt_pose_6d = matrix_to_rotation_6d(gt_pose_6d).reshape(pose.shape[0], -1)

        # load phys results according to the seq_ind and frame_ind
        if self.offline_phys:
            param_dir = os.path.join(self.phys_dir, '%03d' %int(seq_ind), '%04d' %int(ind[0]))


        img_paths = self.imnames[seq_ind]
        img_pathes = [os.path.join(self.dataset_dir, img_paths[path]) for path in ind]
        load_data['img_path'] = img_pathes
        load_data['gt_keyp'] = gt_keyp   ## halpe 26
        
        load_data['pose'] = pose
        load_data['gt_pose_6d'] = gt_pose_6d
        load_data['features'] = features
        load_data['shape'] = gt_shape
        load_data['trans'] = trans
        load_data['verts'] = verts
        
        load_data["center"] = center
        load_data["scale"] = sc*scale
        load_data["img_h"] = img_h
        load_data["img_w"] = img_w
        load_data["focal_length"] = focal_length
        load_data["extri"] = gt_extrisinc
        load_data["intri"] = gt_intrinsic

        load_data['seq_ind'] = np.array(seq_ind)
        load_data['frame_ind'] = np.array(ind)

        # phys results
        if self.offline_phys:
            load_data['phys_dir'] = param_dir
   

        return load_data


    def __getitem__(self, index):

        data = self.create_data_newaug(index)
        return data

    def __len__(self):
        return self.len