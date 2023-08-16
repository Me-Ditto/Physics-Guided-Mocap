import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues, rot6d_to_rotmat

class L1(nn.Module):
    def __init__(self, device):
        super(L1, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(size_average=False)

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L1Loss(x, y)
        diff = diff / b
        return diff



class MPJPE(nn.Module):
    def __init__(self, device):
        super(MPJPE, self).__init__()
        self.device = device
        self.regressor = torch.from_numpy(np.load('data/J_regressor_lsp.npy')).to(torch.float32).to(device)
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_verts, gt_verts):
        loss_dict = {}

        pred_joints = torch.matmul(self.regressor, pred_verts)
        gt_joints = torch.matmul(self.regressor, gt_verts)



        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]))
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_verts, gt_verts):
        loss_dict = {}


        bs, f , _, _ = pred_verts.shape

        pred_joints = torch.matmul(self.regressor, pred_verts).reshape(bs*f, -1, 3)
        gt_joints = torch.matmul(self.regressor, gt_verts).reshape(bs*f, -1, 3)


        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')


        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]))
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff


    def pa_mpjpe(self, pred_verts, gt_verts):
        loss_dict = {}

        bs, f , _, _ = pred_verts.shape

        pred_joints = torch.matmul(self.regressor, pred_verts).reshape(bs*f, -1, 3)
        gt_joints = torch.matmul(self.regressor, gt_verts).reshape(bs*f, -1, 3)

 
        pred_joints = pred_joints.detach().cpu()
        gt_joints = gt_joints.detach().cpu()

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        # diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]))
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device, dtype=S1.dtype).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)

