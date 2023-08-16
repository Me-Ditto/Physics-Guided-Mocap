from cv2 import imread
import torch
from torch.autograd import Variable
import numpy as np
import os
import cv2
import torch.nn as nn
from tqdm import tqdm
import time
import pickle

# from utils.recorder import Recorder
from utils import dist_util


def viz_input(batch_x, batch_y, rgb_img=None, pred=None):
    oc_image = batch_x.detach().data.cpu().numpy().astype(np.float32)
    image = batch_y.detach().data.cpu().numpy().astype(np.float32)
    rgb_image = rgb_img.detach().data.cpu().numpy().astype(np.float32)
    img_decoded = pred.detach().data.cpu().numpy().astype(np.float32)
    for img, oc_img, rgb, img_d in zip(image, oc_image, rgb_image, img_decoded):
        img = img.transpose(1,2,0)
        oc_img = oc_img.transpose(1,2,0)
        rgb = rgb.transpose(1,2,0)
        img_d = img_d.transpose(1,2,0)
        cv2.imshow("img",(img+0.5))
        cv2.imshow("oc_img",(oc_img+0.5))
        cv2.imshow("rgb_img",rgb)
        cv2.imshow("d_img",(img_d+0.5))
        cv2.waitKey()

def viz_masks(m0, m1, m2, m3, mask, gt):
    m_0 = m0.detach().data.cpu().numpy().astype(np.float32)
    m_1 = m1.detach().data.cpu().numpy().astype(np.float32)
    m_2 = m2.detach().data.cpu().numpy().astype(np.float32)
    m_3 = m3.detach().data.cpu().numpy().astype(np.float32)
    mask_viz = mask.detach().data.cpu().numpy().astype(np.float32)
    mask_gt = gt.detach().data.cpu().numpy().astype(np.float32)
    for m0, m1, m2, m3, mask, gt in zip(m_0, m_1, m_2, m_3, mask_viz, mask_gt):

        m0 = m0.transpose(1,2,0)
        m1 = m1.transpose(1,2,0)
        m2 = m2.transpose(1,2,0)
        m3 = m3.transpose(1,2,0)
        mask = mask.transpose(1,2,0)
        gt = gt.transpose(1,2,0)

        cv2.imshow("m0",m0)
        cv2.imshow("m1",m1)
        cv2.imshow("m2",m2)
        cv2.imshow("m3",m3)
        cv2.imshow("mask",mask)
        cv2.imshow("gt",gt)
        cv2.waitKey()

def to_device(data, device):
    if 'phys_dir' in data.keys():
        img_path = {'img_path':data['img_path']} 
        phys_dir = {'phys_dir':data['phys_dir']}
        data = {k:v.to(device).float() for k, v in data.items() if k not in ['img_path', 'phys_dir']}
        data = {**img_path, **phys_dir, **data}
    else:
        img_path = {'img_path':data['img_path']} 
        data = {k:v.to(device).float() for k, v in data.items() if k not in ['img_path', 'phys_dir']}
        data = {**img_path, **data}

    return data

        

def diffusion_test(model, loss_func, loader, epoch, viz=False, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            
            data = to_device(data, device)

            batchsize = data['pose'].size(0)

            # forward
            pred = model.model.inference(data)
            # pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_diffusion_testloss(pred, data)
            instance_loss = None

            if i < 1:
                results = {}
                results.update(gt_keyp=data['gt_keyp'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_poses=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['shape'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_trans=pred['pred_trans'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_poses=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(img_path=data['img_path'])
                results.update(instance_loss=instance_loss)
                results.update(gt_trans=data['trans'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(mode='eval')
                results.update(pred_keyp=pred['pred_keyp'].detach().cpu().numpy().astype(np.float32))
                results.update(scales=data['scale'].detach().cpu().numpy().astype(np.float32))
                results.update(centers=data['center'].detach().cpu().numpy().astype(np.float32))

                results.update(seq_ind=data['seq_ind'].detach().cpu().numpy().astype(np.float32))
                results.update(frame_ind=data['frame_ind'].detach().cpu().numpy().astype(np.float32))
                results.update(extris=data['extri'].detach().cpu().numpy().astype(np.float32))
                results.update(intris=data['intri'].detach().cpu().numpy().astype(np.float32))

                model.save_results(results, epoch, batchsize)

                
            loss_batch = loss.detach() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

