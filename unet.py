from dpi.compartmentModel.stLogan import STLogan, Self2selfLoss
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class STLogan_unet(STLogan):
    def __init__(self,imageSize,nb_frames,mid_c=64,bP=0.5,dropP=0.5,lr_kcm=1e-3,lr_net=1e-4,log='log',device='cuda'):
        super(STLogan_unet,self).__init__(imageSize,nb_frames,mid_c=mid_c,bP=bP,dropP=dropP,lr_kcm=lr_kcm,lr_net=lr_net,log=log,device=device)
        self.model_net = UNet(in_c=nb_frames,out_c=2,mid_c=mid_c,dropP=dropP).to(device)
        self.optim_net = optim.Adam(self.model_net.parameters(), lr=lr_net)

    def pre_train(self,k_label,niter=10000,roi=None):
        """" k_label = self.model_kcm.k.detach().reshape(2,self.im_S[0],self.im_S[1],self.im_S[2])*roi, (nb_k, d, h, w)"""
        im = self.dy_im_.unsqueeze(0)
        for it in tqdm(range(self.start_its[2],niter)):
            k_pred = self.model_net(im)
            loss = self.loss_fn.mseloss(k_pred,k_label,roi)
            loss.backward()
            self.optim_net.step()
            self.optim_net.zero_grad()
            self.writer.add_scalar('s2snn_pre',loss.data.item(),it)
        self.start_its[2] = it + 1
    
    def train(self,niter=10000,roi=None):
        im = self.dy_im_.unsqueeze(0)
        for it in tqdm(range(self.start_its[3],niter)):
            self.model_kcm.k = self.model_net(im).reshape(2,-1)
            C_pred = self.model_kcm(self.t_,self.A,self.B).reshape(-1,self.im_S[0],self.im_S[1],self.im_S[2])
            loss = self.loss_fn.mseloss(C_pred,self.C,roi)
            loss.backward()
            self.optim_net.step()   
            self.optim_net.zero_grad()
            self.writer.add_scalar('s2snn',loss.data.item(),it)
        self.start_its[3] = it + 1
    
    def predict(self,roi=None):
        im = self.dy_im_.unsqueeze(0)
        with torch.no_grad():
            k_pred = self.model_net(im)
            if roi is not None:
                k_pred = k_pred * roi
        return k_pred[0]


#####################################################################################################################
class SConvLR3d(nn.Module): # conv+LeakyReLU
    def __init__(self,in_c,out_c,stride=1):
        super(SConvLR3d,self).__init__()
        self.sc = nn.Conv3d(in_c,out_c,kernel_size=3,stride=stride,padding=1)
        self.act = nn.LeakyReLU(inplace=True)
        
    def forward(self,image):
        output = self.sc(image)
        output = self.act(output)
        return output


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


class UNet(nn.Module):
    def __init__(self,in_c,out_c,mid_c=64,dropP=0.5):
        super(UNet,self).__init__() 
        self.pc1 = SConvLR3d(in_c,mid_c)
        self.pc2 = SConvLR3d(mid_c,mid_c,stride=2)
        self.pc3 = SConvLR3d(mid_c,mid_c)
        self.pc4 = SConvLR3d(mid_c,mid_c,stride=2)
        self.pc5 = SConvLR3d(mid_c,mid_c)
        self.pc6 = SConvLR3d(mid_c,mid_c,stride=2)
        self.pc7 = SConvLR3d(mid_c,mid_c)
        self.pc8 = SConvLR3d(mid_c,mid_c)

        self.ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.uc2 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc4 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc6 = SConvLRdrop3d(mid_c*2,mid_c,dropP=dropP)
        self.uc7 = SConvdrop3d(mid_c,out_c,dropP=dropP) # 没有激活函数

    def forward(self,im):
        record = []
        # enconder
        im = self.pc1(im)
        record.append(im)
        im = self.pc2(im)
        im = self.pc3(im)
        record.append(im)
        im = self.pc4(im)
        im = self.pc5(im)
        record.append(im)
        im = self.pc6(im)
        im = self.pc7(im)
        im = self.pc8(im)
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