import time
import os

from tqdm import tqdm
from utils.logger import Logger
import yaml
from utils.smpl_torch_batch import SMPLModel
import torch
from loss_func import *
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from utils.rotation_conversions import *
from datasets.FeatureLoader import FeatureData
import random
from utils.renderer_pyrd import Renderer
import sys
import json
import pickle
import utils.optimizers.cyclic_scheduler  as cyclic_scheduler
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

keypoint_names = ['RA', 'RK', 'RH', 'LH', 'LK', 'LA', 'RW', 'RE', 'RS', 'LS', 'LE','LW', 'Neck', 'Head']
font = cv2.FONT_HERSHEY_SIMPLEX

# The number of samples in the dataset
sample_num = {'Human36M_MOSH': 297788, 'VCLOcclusion': 275184, 'PennAction': 107914, 'PoseTrack': 17941, 'InstaVariety': 1844866, 'NeuralAnnot3DHP': 1039515, 'AIST':846322, 'OcMotion':288000}


def seed_worker(seed):
    worker_seed = 7
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def init(note='occlusion', dtype=torch.float32, device=torch.device('cpu'), **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="vposer")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )

    return out_dir, logger, model_smpl


class DatasetLoader():
    def __init__(self, trainset=None, testset=None, smpl_model=None, generator=None, data_folder='./data', dtype=torch.float32, frame_length=16, num_joint=24, task=None, offline_phys=False, **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.model = smpl_model
        self.generator = generator
        self.frame_length = frame_length
        self.num_joint = num_joint
        self.task = task
        if task in ['mocap']:
            self.mocap = True
        else:
            self.mocap = False
        self.offline_phys = offline_phys
        if self.offline_phys:
            self.phys_folder = kwargs['phys_dir']
        else:
            self.phys_folder=''

    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task in ['diffusion', 'vae', 'demo']:
                test_dataset.append(FeatureData(False, self.data_folder, self.model, self.testset[i], self.frame_length, self.num_joint))
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset




class ModelLoader():
    def __init__(self, model=None, lr=0.001, dtype=torch.float32, device=torch.device('cpu'), pretrain=False, pretrain_dir='', output='', smpl=None, frame_length=16, num_joint=24, use_vae_prior=False, vae_dir=None, batchsize=32, trainset='', **kwargs):
        self.smpl = smpl
        self.output = output
        self.threshold = 0.7
        self.data_shape = [1,kwargs.pop('data_shape'),3]
        self.num_neurons = kwargs.pop('num_neurons')
        self.latentD = kwargs.get('latentD')
        self.frame_length = frame_length
        self.joint_num = num_joint

        # load smpl model 
        self.model_smpl_gpu = SMPLModel(
                            device=torch.device('cuda'),
                            model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )
        
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(smpl=self.model_smpl_gpu, frame_length=self.frame_length, joint_num=self.joint_num, **kwargs)
        self.lsp_regressor = np.load('data/J_regressor_lsp.npy')
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.epoch_size = 0

        trainset = trainset.split(' ')
        for set in trainset:
            if set in sample_num.keys():
                self.epoch_size += sample_num[set]

        for parameter in self.model.vae.parameters():
            parameter.requires_grad = False

        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count:', model_params)

        self.device = device
        
        print('load model: %s' %self.model_type)

        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=batchsize, epoch_size=self.epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {'prior.' + k: v for k ,v in premodel_dict.items() if 'prior.' + k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print('Load pre-trained prior from %s with %d layers' %(pretrain_dir, len(premodel_dict)))

        # load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(params['optimizer'])
            print("load pretrain parameters from %s" %pretrain_dir)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # load fixed model
        if use_vae_prior:
            model_dict = self.model.state_dict()
            params = torch.load(vae_dir)
            premodel_dict = params['model']
            premodel_dict = {'vae.%s' %k: v for k ,v in premodel_dict.items() if 'vae.%s' %k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print('Load pre-trained vae prior from %s with %d layers' %(vae_dir, len(premodel_dict)))

    
    def create_gaussian_diffusion(self, **kwargs):
        # default params
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = 1000
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
        loss_type = gd.LossType.MSE

        sigma_small = True
        lambda_vel = 0.0
        lambda_rcxyz = 0.0
        lambda_fc = 0.0

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=lambda_vel,
            lambda_rcxyz=lambda_rcxyz,
            lambda_fc=lambda_fc,
        )

    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    

    def save_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images', results['mode'])
        if not os.path.exists(output):
            os.makedirs(output)
        img_pathes = [[f[n] for f in results['img_path']] for n in range(len(results['img_path'][0]))]


        if 'pred_poses' in results.keys():
            pred_poses = results['pred_poses']
            gt_poses = results['gt_poses']
            pred_shape = results['pred_shape']
            gt_shape = results['gt_shape']
            gt_trans = results['gt_trans']
            pred_trans = results['pred_trans']
            focals = results['focal_length']

            pred_keyps = results['pred_keyp']
            gt_keyps = results['gt_keyp']
            scales = results['scales']
            centers = results['centers']
            extris = results['extris']
            intris = results['intris']


            for index, (pred_pose, gt_pose, img_path, pshape, gshape, focal, gt_transl, pred_transl) in enumerate(zip(pred_poses, gt_poses, img_pathes, pred_shape, gt_shape, focals, gt_trans, pred_trans)):
                if index > 0:
                    continue
                for frame, (pred_p, gt_p, img, f, gt_t, pred_t) in enumerate(zip(pred_pose, gt_pose, img_path, focal, gt_transl, pred_transl)):
                    if results['instance_loss'] is not None:
                        error = results['instance_loss'][index][frame]
                        if error < 100:
                            continue
                    else:
                        if frame > 5:
                            continue
                    # convert to world coordinate
                    pred_p_w, pred_t_w = self.convert_world_coord(pred_p, pred_t, pshape, extris[index][frame])
                    gt_p_w, gt_t_w = self.convert_world_coord(gt_p, gt_t, pshape, extris[index][frame])

                    pred_poses = torch.tensor(pred_p, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)
                    gt_poses = torch.tensor(gt_p, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)
                    pred_poses_w = torch.tensor(pred_p_w, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)
                    gt_poses_w = torch.tensor(gt_p_w, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)

                    pred_shapes = torch.tensor(pshape, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 10)
                    gt_shapes = torch.tensor(gshape, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 10)

                    pred_translation = torch.tensor(pred_t, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)
                    gt_translation = torch.tensor(gt_t, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)
                    pred_translation_w = torch.tensor(pred_t_w, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)
                    gt_translation_w = torch.tensor(gt_t_w, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)


                    pred_meshes, _ = self.smpl(pred_shapes, pred_poses, pred_translation)
                    gt_meshes, _ = self.smpl(gt_shapes, gt_poses, gt_translation)
                    pred_meshes_w, _ = self.smpl(pred_shapes, pred_poses_w, pred_translation_w)
                    gt_meshes_w, _ = self.smpl(gt_shapes, gt_poses_w, gt_translation_w)
                    
                    pred_meshes = pred_meshes.detach().cpu().numpy()[0]
                    gt_meshes = gt_meshes.detach().cpu().numpy()[0]
                    pred_meshes_w = pred_meshes_w.detach().cpu().numpy()[0]
                    gt_meshes_w = gt_meshes_w.detach().cpu().numpy()[0]
                    
                    pred_joints = np.dot(self.lsp_regressor, pred_meshes)
                    gt_joints = np.dot(self.lsp_regressor, gt_meshes)


                    name = img.split('images')[-1].replace('\\', '_').replace('/', '_')
                    img = cv2.imread(img)
                    img_h, img_w = img.shape[:2]
                    renderer = Renderer(focal_length=f, center=(intris[index][frame][0,2], intris[index][frame][1,2]), img_w=img.shape[1], img_h=img.shape[0],
                                        faces=self.smpl.faces,
                                        same_mesh_color=True)

                    pred_smpl = renderer.render_front_view(pred_meshes[np.newaxis,:,:],
                                                        bg_img_rgb=img.copy())

                    gt_smpl = renderer.render_front_view(gt_meshes[np.newaxis,:,:],
                                                        bg_img_rgb=img.copy())
                    
                    pred_meshes_w_render = np.insert(pred_meshes_w, 3, values=1., axis=1)
                    pred_meshes_w_render = np.dot(extris[index][frame], pred_meshes_w_render.T).T[:,:3]
                    pred_smpl_w = renderer.render_front_view(pred_meshes_w_render[np.newaxis,:,:],
                                                        bg_img_rgb=img.copy())
                    
                    ## draw keyp
                    import constants
                    gt_keyp = gt_keyps[index][frame]
                    pred_keyp = pred_keyps[index][frame]
                    scale = scales[index][frame]
                    center = centers[index][frame]

                    ## denormalize
                    gt_keyp[:,:2] = gt_keyp[:,:2] * scale * constants.IMG_RES + center 
                    pred_keyp[:,:2] = pred_keyp[:,:2] * scale * constants.IMG_RES + center 
                    gt_keyp_img = self.draw_keyps([gt_keyp], img)
                    pred_keyp_img = self.draw_keyps([pred_keyp], img)

                    render_name = "%05d_%02d_pred_keyp.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, render_name), pred_keyp_img)

                    render_name = "%05d_%02d_gt_keyp.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, render_name), gt_keyp_img)

                    render_name = "%05d_%02d_pred_render.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, render_name), pred_smpl)

                    render_name = "%05d_%02d_gt_smpl.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, render_name), gt_smpl)

                    render_name = "%05d_%02d_pred_w_smpl.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, render_name), pred_smpl_w)

                    img_name = "%05d_%02d_image.jpg" % (iter * batchsize + index, frame)
                    cv2.imwrite(os.path.join(output, img_name), img)

                    self.smpl.write_obj(
                        pred_meshes, os.path.join(output, 'meshes/%05d_%05d_pred_mesh.obj' %(iter * batchsize + index, frame) )
                    )

                    self.smpl.write_obj(
                        gt_meshes, os.path.join(output, 'meshes/%05d_%05d_gt_mesh.obj' %(iter * batchsize + index, frame) )
                    )

                    self.smpl.write_obj(
                        pred_meshes_w, os.path.join(output, 'meshes/%05d_%05d_pred_w_mesh.obj' %(iter * batchsize + index, frame) )
                    )

                    self.smpl.write_obj(
                        gt_meshes_w, os.path.join(output, 'meshes/%05d_%05d_gt_w_mesh.obj' %(iter * batchsize + index, frame) )
                    )

                    renderer.delete()
    
    def convert_world_coord(self, pose, trans, shape, extri):
        '''Convert to World Coordinate System'''
        pose = torch.tensor(pose, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)
        trans = torch.tensor(trans, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)
        shape = torch.tensor(shape, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 10)
        extri = torch.from_numpy(extri)

        matrix = torch.linalg.inv(extri.clone())
        f = pose.shape[0]

        zeros_trans = torch.zeros((f, 3), device=pose.device, dtype=pose.dtype)
        verts_zero_cam, joints_zero_cam = self.smpl(shape, pose, zeros_trans)
        root = joints_zero_cam[:,0,:]

        
        # convert pose params
        oritation = axis_angle_to_matrix(pose[:,:3].clone())
        oritation = torch.matmul(matrix[:3,:3], oritation)
        oritation = matrix_to_axis_angle(oritation)
        pose[:,:3] = oritation


        # rot root joint
        root_cam = root + trans
        root_world = torch.einsum('ij,kj->ki', matrix[:3,:3], root_cam)
        root_world = root_world + matrix[:3,3]
        trans = root_world - root

        pose = pose.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()

        return pose, trans

    def save_parameter(self, results, batchsize):
        output = os.path.join(self.output, 'params')
        if not os.path.exists(output):
            os.makedirs(output)
        img_pathes = [[f[n] for f in results['img_path']] for n in range(len(results['img_path'][0]))]

        pred_poses = results['pred_poses']
        gt_poses = results['gt_poses']
        # raw_poses = results['raw_pose']
        pred_shape = results['pred_shape']
        gt_shape = results['gt_shape']
        gt_trans = results['gt_trans']
        pred_trans = results['pred_trans']
        focals = results['focal_length']

        pred_keyps = results['pred_keyp']
        gt_keyps = results['gt_keyp']
        scales = results['scales']
        centers = results['centers']

        seq_id = results['seq_ind']
        frame_id = results['frame_ind']
        extris = results['extris']

        for index, (pred_pose, gt_pose, img_path, pshape, gshape, focal, gt_transl, pred_transl) in enumerate(zip(pred_poses, gt_poses, img_pathes, pred_shape, gt_shape, focals, gt_trans, pred_trans)):
                param = {}
                param['pred_pose'] = pred_pose
                param['ge_pose'] = gt_pose 
                param['pred_shape'] = pshape 
                param['gt_shape'] = gshape
                param['pred_trans'] = pred_transl
                param['gt_trans'] = gt_transl
                param['extris'] = extris[index]

                seq_name = seq_id[index]
                frame_name = frame_id[index]

                out_path = os.path.join(output, 'param_seq_%s_frame_%03d_%03d.pkl' %(int(seq_name), frame_name[0], frame_name[-1]))
                self.save_pkl(out_path, param)

    def save_pkl(self, path, result):
        """"
        save pkl file
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, 'wb') as result_file:
            pickle.dump(result, result_file, protocol=2)

    def draw_keyps(self, keypoints, img):
        # if len(keypoints) > 1:
        img = img.copy()
        for keyps in keypoints:
            for point in keyps:
                point = (int(point[0]), int(point[1]))
                img = cv2.circle(img, point, 3, (0, 255, 255), -1)
                # vis_img('keyp', img)
        return img
    


class LossLoader():
    def __init__(self, train_loss='L1', test_loss='L1', device=torch.device('cpu'),  batchsize=1, dtype=torch.float32, generator=None, task=None, **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device
        self.generator = generator
        self.task = task
        self.smpl = SMPLModel(
                                device=device,
                                model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        self.smpl_gpu = SMPLModel(
                                device=torch.device('cuda'),
                                model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        self.kl_coef = kwargs.get('kl_coef')
        self.latentD = kwargs.get('latentD')
        self.frame_length = kwargs.get('frame_length')

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(self.device))
            if loss == 'PA_MPJPE':
                self.test_loss.update(PA_MPJPE=MPJPE(self.device))
                

    def calcul_diffusion_testloss(self, pred, data):

        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_verts'], data['verts']))
            elif ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred['pred_pose'], data['pose']))
            elif ltype == 'PA_MPJPE':
                loss_dict.update(PA_MPJPE=self.test_loss['PA_MPJPE'].pa_mpjpe(pred['pred_verts'], data['verts']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        loss = 0
        for k in loss_dict:
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 6)
        return loss, loss_dict


