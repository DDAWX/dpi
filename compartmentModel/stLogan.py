################################################################################
#
# dwx 2021-01-01
#
# all tensor should be float32
# the unit of t is minutes
# 
#################################################################################
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dpi.function import cumtrapz
from dpi.compartmentModel.classical import Logan
from dpi.compartmentModel.partialconv3d import PartialConv3d


###############################################################################################################
#
# embed the self2self net to STLogan
# 
##############################################################################################################
class STLogan():
    def __init__(self,imageSize,in_c,out_c=2,mid_c=64,bP=0.5,dropP=0.5,lr_kcm=1e-3,lr_net=1e-4,log='log',device='cuda',method='ols'):
        """ method:'ols','ma1' """
        self.device = device
        self.im_S = imageSize.astype(np.int)
        self.model_kcm = Logan(im_L=self.im_S[0]*self.im_S[1]*self.im_S[2])
        self.optim_kcm = optim.Adam([{'params':self.model_kcm.k}], lr=lr_kcm)
        self.bP = bP
        self.model_net = Self2self(in_c=in_c,out_c=out_c,mid_c=mid_c,dropP=dropP).to(device)
        self.optim_net = optim.Adam(self.model_net.parameters(), lr=lr_net)
        self.loss_fn = Self2selfLoss()
        self.start_its = [0, 0, 0, 0] # 分别对应vw,tv，pre_train,train
        self.writer = SummaryWriter(log)
        self.method = method

    def setdata(self,t,dy_im,ref,tf_idx,t1_idx=0):
        """设置数据：从全部动态图像中得到想要的部分"""
        self.A,self.B,self.C,self.t_ = self.data_prepare(t,dy_im,ref,tf_idx,t1_idx)
        if self.method == 'ma1':
            self.B,self.C = self.C,self.B # 改为MA1: swap B with C
        self.C = self.C.reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
        self.dy_im_ = dy_im[t1_idx:]         # 实际采集的动态图像

    def data_prepare(self,t,dy_im,ref,tf_idx,t1_idx=0):
        """
        功能：将完整的动态数据转换为想要拟合的部分
        t 时间,1Dtensor;
        dy_im 动态图像,4Dtensor;
        ref 参考区tac,1Dtensor
        tf_idx (用于拟合的)初始平衡时间的idx
        t1_idx t1初始时间的idx,0表示从头扫到尾;
        """
        A = cumtrapz(ref,t,device=self.device).unsqueeze(1) # 参考区积分
        B = dy_im.reshape(dy_im.shape[0],-1) # 组织活度曲线
        C = cumtrapz(B,t,device=self.device) # 组织活度曲线积分
        if t1_idx != 0:                      # 不等于0说明不是从头开始扫描
            A = A - A[t1_idx,:]
            B = B - B[t1_idx,:]
            C = C - C[t1_idx,:]
            if t1_idx < tf_idx:
                return A[tf_idx:,:],B[tf_idx:,:],C[tf_idx:,:],t[tf_idx:]
            else:
                return A[t1_idx+1:,:],B[t1_idx+1:,:],C[t1_idx+1:,:],t[t1_idx+1:]
        else:
            return A[tf_idx:,:],B[tf_idx:,:],C[tf_idx:,:],t[tf_idx:]

    def fit(self,niter=10000,roi=None):
        for it in tqdm(range(self.start_its[0],niter)):
            C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
            loss = self.loss_fn.mseloss(C_pred,self.C,roi)
            loss.backward()
            self.optim_kcm.step()
            self.model_kcm.k[0,:].data.clamp_(min=0.) # DVR > 0
            self.optim_kcm.zero_grad()
            self.writer.add_scalar('vw',loss.data.item(),it)
        self.start_its[0] = it + 1
    
    def pre_train(self,k_label,niter=10000,roi=None):
        """" k_label = self.model_kcm.k.detach().reshape(2,self.im_S[0],self.im_S[1],self.im_S[2])*roi, (nb_k, d, h, w)"""
        for it in tqdm(range(self.start_its[2],niter)):
            im,mask = bernsample(self.dy_im_,bP=self.bP)
            k_pred = self.model_net(im,mask)
            loss = self.loss_fn.partloss(k_pred,k_label,mask[0],roi)
            loss.backward()
            self.optim_net.step()
            self.optim_net.zero_grad()
            self.writer.add_scalar('s2snn_pre',loss.data.item(),it)
        self.start_its[2] = it + 1
    
    def train(self,niter=10000,roi=None):
        for it in tqdm(range(self.start_its[3],niter)):
            im,mask = bernsample(self.dy_im_,bP=self.bP)
            self.model_kcm.k = self.model_net(im,mask).reshape(2,-1)
            C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
            loss = self.loss_fn.partloss(C_pred,self.C,mask[0,-self.C.shape[0]:],roi)
            loss.backward()
            self.optim_net.step()
            self.optim_net.zero_grad()
            self.writer.add_scalar('s2snn',loss.data.item(),it)
        self.start_its[3] = it + 1
    
    def predict(self,niter=50,roi=None):
        k_pred = torch.zeros(1,2,self.im_S[0],self.im_S[1],self.im_S[2],device=self.device)
        with torch.no_grad():
            for it in range(niter):
                im, mask = bernsample(self.dy_im_,bP=self.bP)
                k_pred += self.model_net(im,mask)
            k_pred /= niter
            if roi is not None:
                k_pred = k_pred * roi
        return k_pred[0]

    def save(self,path):
        """ e.g.:'./checkpoint.pth' """
        checkpoint = {
            'model_kcm':self.model_kcm.k.detach(),
            'optim_kcm':self.optim_kcm.state_dict(),
            'model_net':self.model_net.state_dict(),
            'optim_net':self.optim_net.state_dict(),
            'start_its':self.start_its
            }
        torch.save(checkpoint, path)

    def load(self,path):
        checkpoint = torch.load(path)
        self.model_kcm.k = checkpoint['model_kcm'].requires_grad_(True)
        self.optim_kcm = optim.Adam([{'params':self.model_kcm.k}]) # 断点fit需重新实例化optimizer(目的是设置待优化的参数；不需要lr，因为下一步会load)
        self.optim_kcm.load_state_dict(checkpoint['optim_kcm'])
        self.model_net.load_state_dict(checkpoint['model_net'])
        self.optim_net.load_state_dict(checkpoint['optim_net'])
        self.start_its = checkpoint['start_its']

    def fit_tv(self,niter=10000,w=1e-2,roi=None):
        for it in tqdm(range(self.start_its[1],niter)):
            C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
            # penalty term
            k = self.model_kcm.k.reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2]) * torch.tensor([[[[1.]]],[[[0.1]]]],device='cuda')
            penalty = torch.sqrt( ( (k[:,:-1,:-1,:-1] - k[:,1:,:-1,:-1]).pow(2) +
                                    (k[:,:-1,:-1,:-1] - k[:,:-1,1:,:-1]).pow(2) +
                                    (k[:,:-1,:-1,:-1] - k[:,:-1,:-1,1:]).pow(2) ) )
            # loss
            if roi is not None:
                loss = ((C_pred - self.C).pow(2) * roi).mean() + ( penalty * roi[:-1,:-1,:-1] ).mean() * w
            else:
                loss = ((C_pred - self.C).pow(2)).mean() + penalty.mean()
            loss.backward()
            self.optim_kcm.step()
            # self.model_kcm.k[0,:].data.clamp_(min=0.) # DVR > 0
            self.optim_kcm.zero_grad()
            self.writer.add_scalar('tv',loss.data.item(),it)
        self.start_its[1] = it + 1

    def fit_reg(self,niter=10000,w=1e-2,roi=None):
        for it in tqdm(range(self.start_its[1],niter)):
            C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
            # penalty term
            k = self.model_kcm.k.reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2]) * torch.tensor([[[[1.]]],[[[0.1]]]],device='cuda')
            c = k[:,1:-1,1:-1,1:-1]
            penalty1 = ( torch.abs( c - k[:,:-2,1:-1,1:-1] ) + torch.abs( c - k[:,2:,1:-1,1:-1] ) +
                        torch.abs( c - k[:,1:-1,:-2,1:-1] ) + torch.abs( c - k[:,1:-1,2:,1:-1] ) +
                        torch.abs( c - k[:,1:-1,1:-1,:-2] ) + torch.abs( c - k[:,1:-1,1:-1,2:] ) ) / 6
            penalty2 = ( torch.abs( c - k[:,:-2,:-2,1:-1] ) + torch.abs( c - k[:,2:,2:,1:-1]  ) +
                        torch.abs( c - k[:,2:,:-2,1:-1]  ) + torch.abs( c - k[:,:-2,2:,1:-1] ) +              
                        torch.abs( c - k[:,1:-1,:-2,:-2] ) + torch.abs( c - k[:,1:-1,2:,2:]  ) +
                        torch.abs( c - k[:,1:-1,2:,:-2]  ) + torch.abs( c - k[:,1:-1,:-2,2:] ) +               
                        torch.abs( c - k[:,:-2,1:-1,:-2] ) + torch.abs( c - k[:,2:,1:-1,2:]  ) +
                        torch.abs( c - k[:,2:,1:-1,:-2]  ) + torch.abs( c - k[:,:-2,1:-1,2:] ) ) / (12 * 1.414)
            penalty3 = ( torch.abs( c - k[:,:-2,:-2,:-2] ) + torch.abs( c - k[:,2:,2:,2:]  ) +
                        torch.abs( c - k[:,2:,:-2,:-2]  ) + torch.abs( c - k[:,:-2,2:,2:] ) +
                        torch.abs( c - k[:,:-2,2:,:-2]  ) + torch.abs( c - k[:,2:,:-2,2:] ) +
                        torch.abs( c - k[:,:-2,:-2,2:]  ) + torch.abs( c - k[:,2:,2:,:-2] ) ) / (8 * 1.732)
            penalty = penalty1 + penalty2 + penalty3
            # loss
            if roi is not None:
                loss = ((C_pred - self.C).pow(2) * roi).mean() + ( penalty * roi[1:-1,1:-1,1:-1] ).mean() * w
            else:
                loss = ((C_pred - self.C).pow(2)).mean() + penalty.mean()
            loss.backward()
            self.optim_kcm.step()
            self.model_kcm.k[0,:].data.clamp_(min=0.) # DVR > 0
            self.optim_kcm.zero_grad()
            self.writer.add_scalar('reg',loss.data.item(),it)
        self.start_its[1] = it + 1     



###############################################################################################################
#
# self2self net first, then do the STLogan
# 
##############################################################################################################
class NN_STLogan(STLogan):
    def __init__(self,imageSize,in_c,out_c=2,mid_c=64,bP=0.5,dropP=0.5,lr_kcm=1e-3,lr_net=1e-4,log='log',device='cuda',method='ols'):
        super(NN_STLogan,self).__init__(imageSize,in_c,out_c=out_c,mid_c=mid_c,bP=bP,dropP=dropP,lr_kcm=lr_kcm,lr_net=lr_net,log=log,device=device,method=method)
        
    def train(self,dy_im_,niter=10000,roi=None):
        for it in tqdm(range(self.start_its[3],niter)):
            im,mask = bernsample(dy_im_,bP=self.bP)     
            dy_im_pred = self.model_net(im,mask)
            loss = self.loss_fn.partloss(dy_im_pred,dy_im_,mask[0,-dy_im_.shape[0]:],roi)
            loss.backward()
            self.optim_net.step()   
            self.optim_net.zero_grad()
            self.writer.add_scalar('dy_im_pred',loss.data.item(),it)
        self.start_its[3] = it + 1
        
    def predict(self,dy_im_,niter=50,roi=None):
        dy_im_pred = torch.unsqueeze(torch.zeros(dy_im_.shape,device=self.device), 0)
        with torch.no_grad():
            for it in range(niter):
                im, mask = bernsample(dy_im_,bP=self.bP)
                dy_im_pred += self.model_net(im,mask)
            dy_im_pred /= niter
            if roi is not None:
                dy_im_pred = dy_im_pred * roi
        return dy_im_pred[0]


###############################################################################################################
#
# Self2self NN
# 
##############################################################################################################
def bernsample(im,bP=0.5,device='cuda'):
    """ im: ( c, d, h, w ), device = 'cuda' or 'cpu' """
    c, d, h, w = im.shape
    masks = torch.zeros(1, c, d, h, w, device=device)
    # masks[0,:,:,:,:] = torch.bernoulli( torch.ones(d, h, w, device=device) * bP )
    masks[0,:,:,:,:] = torch.bernoulli( torch.empty(d, h, w, device=device).uniform_(0, 1), bP)
    images = im.unsqueeze(0) * masks
    return images,masks


class PConvLR3d(nn.Module): # Pconv+LeakyReLU
    def __init__(self,in_c,out_c,stride=1):
        super(PConvLR3d,self).__init__()
        self.pc = PartialConv3d(in_c,out_c,kernel_size=3,stride=stride,padding=1,return_mask=True,multi_channel=True)
        self.act = nn.LeakyReLU(inplace=True)
        
    def forward(self,image,mask):
        output,new_mask = self.pc(image,mask)
        output = self.act(output)
        return output, new_mask


class SConvLRdrop3d(nn.Module): # drop+conv+LeakyReLU
    def __init__(self,in_c,out_c,dropP,stride=1):
        super(SConvLRdrop3d,self).__init__()
        self.drop = nn.Dropout(p=dropP)
        self.sc = nn.Conv3d(in_c,out_c,kernel_size=3,stride=stride,padding=1)
        self.act = nn.LeakyReLU(inplace=True)
        
    def forward(self,x):
        x = self.drop(x)
        x = self.sc(x)
        x = self.act(x)
        return x


class SConvdrop3d(nn.Module): # drop+conv
    def __init__(self,in_c,out_c,dropP,stride=1):
        super(SConvdrop3d,self).__init__()
        self.drop = nn.Dropout(p=dropP)
        self.sc = nn.Conv3d(in_c,out_c,kernel_size=3,stride=stride,padding=1)
        
    def forward(self,x):
        x = self.drop(x)
        x = self.sc(x)
        return x


class Self2self(nn.Module):
    def __init__(self,in_c,out_c,mid_c=64,dropP=0.5):
        super(Self2self,self).__init__() 
        self.pc1 = PConvLR3d(in_c,mid_c)
        self.pc2 = PConvLR3d(mid_c,mid_c,stride=2)
        self.pc3 = PConvLR3d(mid_c,mid_c)
        self.pc4 = PConvLR3d(mid_c,mid_c,stride=2)
        self.pc5 = PConvLR3d(mid_c,mid_c)
        self.pc6 = PConvLR3d(mid_c,mid_c,stride=2)
        self.pc7 = PConvLR3d(mid_c,mid_c)
        self.pc8 = PConvLR3d(mid_c,mid_c)

        self.ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.uc2 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc4 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc6 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc7 = SConvdrop3d(mid_c,out_c,dropP=dropP) # 没有激活函数

    def forward(self,im,mask):
        record = []
        # enconder
        im,mask = self.pc1(im,mask)
        record.append(im)
        im,mask = self.pc2(im,mask)
        im,mask = self.pc3(im,mask)
        record.append(im)
        im,mask = self.pc4(im,mask)
        im,mask = self.pc5(im,mask)
        record.append(im)
        im,mask = self.pc6(im,mask)
        im,mask = self.pc7(im,mask)
        im,mask = self.pc8(im,mask)
        # deconder
        im = self.ups(im)
        im = torch.cat([im, record.pop()], 1)
        im = self.uc2(im)
        im = self.ups(im)
        im = torch.cat([im, record.pop()], 1)
        im = self.uc4(im)
        im = self.ups(im)
        im = torch.cat([im, record.pop()], 1)
        im = self.uc6(im)
        im = self.uc7(im)
        return im


class Self2selfLoss():
    def mseloss(self,y1,y2,roi=None):
        if roi is not None:
            return ((y1 - y2).pow(2) * roi).mean()
        else:
            return ((y1 - y2).pow(2)).mean()

    def partloss(self,y1,y2,mask,roi=None):
        if roi is not None:
            return ((1 - mask) * roi * (y1 - y2).pow(2)).mean()
        else:
            return ((1 - mask) * (y1 - y2).pow(2)).mean()



###############################################################################################################
#
# u-net
# 
##############################################################################################################
# class STLogan_unet(STLogan):
#     def __init__(self,imageSize,nb_frames,mid_c=64,bP=0.5,dropP=0.5,lr_kcm=1e-3,lr_net=1e-4,log='log',device='cuda',method='ols'):
#         super(STLogan_unet,self).__init__(imageSize,nb_frames,mid_c=mid_c,bP=bP,dropP=dropP,lr_kcm=lr_kcm,lr_net=lr_net,log=log,device=device)
#         self.model_net = UNet(in_c=nb_frames,out_c=2,mid_c=mid_c,dropP=dropP).to(device)
#         self.optim_net = optim.Adam(self.model_net.parameters(), lr=lr_net)

#     def pre_train(self,k_label,niter=10000,roi=None):
#         """" k_label = self.model_kcm.k.detach().reshape(2,self.im_S[0],self.im_S[1],self.im_S[2])*roi, (nb_k, d, h, w)"""
#         im = self.dy_im_.unsqueeze(0)
#         for it in tqdm(range(self.start_its[2],niter)):
#             k_pred = self.model_net(im)
#             loss = self.loss_fn.mseloss(k_pred,k_label,roi)
#             loss.backward()
#             self.optim_net.step()
#             self.optim_net.zero_grad()
#             self.writer.add_scalar('s2snn_pre',loss.data.item(),it)
#         self.start_its[2] = it + 1
    
#     def train(self,niter=10000,roi=None):
#         im = self.dy_im_.unsqueeze(0)
#         for it in tqdm(range(self.start_its[3],niter)):
#             self.model_kcm.k = self.model_net(im).reshape(2,-1)
#             C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
#             loss = self.loss_fn.mseloss(C_pred,self.C,roi)
#             loss.backward()
#             self.optim_net.step()   
#             self.optim_net.zero_grad()
#             self.writer.add_scalar('s2snn',loss.data.item(),it)
#         self.start_its[3] = it + 1
    
#     def predict(self,roi=None):
#         im = self.dy_im_.unsqueeze(0)
#         with torch.no_grad():
#             k_pred = self.model_net(im)
#             if roi is not None:
#                 k_pred = k_pred * roi
#         return k_pred[0]


# class SConvLR3d(nn.Module): # conv+LeakyReLU
#     def __init__(self,in_c,out_c,stride=1):
#         super(SConvLR3d,self).__init__()
#         self.sc = nn.Conv3d(in_c,out_c,kernel_size=3,stride=stride,padding=1)
#         self.act = nn.LeakyReLU(inplace=True)
        
#     def forward(self,image):
#         output = self.sc(image)
#         output = self.act(output)
#         return output


# class UNet(nn.Module):
#     def __init__(self,in_c,out_c,mid_c=64,dropP=0.5):
#         super(UNet,self).__init__() 
#         self.pc1 = SConvLR3d(in_c,mid_c)
#         self.pc2 = SConvLR3d(mid_c,mid_c,stride=2)
#         self.pc3 = SConvLR3d(mid_c,mid_c)
#         self.pc4 = SConvLR3d(mid_c,mid_c,stride=2)
#         self.pc5 = SConvLR3d(mid_c,mid_c)
#         self.pc6 = SConvLR3d(mid_c,mid_c,stride=2)
#         self.pc7 = SConvLR3d(mid_c,mid_c)
#         self.pc8 = SConvLR3d(mid_c,mid_c)

#         self.ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         self.uc2 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
#         self.uc4 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
#         self.uc6 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
#         self.uc7 = SConvdrop3d(mid_c,out_c,dropP=dropP) # 没有激活函数

#     def forward(self,im):
#         record = []
#         # enconder
#         im = self.pc1(im)
#         record.append(im)
#         im = self.pc2(im)
#         im = self.pc3(im)
#         record.append(im)
#         im = self.pc4(im)
#         im = self.pc5(im)
#         record.append(im)
#         im = self.pc6(im)
#         im = self.pc7(im)
#         im = self.pc8(im)
#         # deconder
#         im = self.ups(im)
#         im = torch.cat([im, record.pop()], 1)
#         im = self.uc2(im)
#         im = self.ups(im)
#         im = torch.cat([im, record.pop()], 1)
#         im = self.uc4(im)
#         im = self.ups(im)
#         im = torch.cat([im, record.pop()], 1)
#         im = self.uc6(im)
#         im = self.uc7(im)
#         return im