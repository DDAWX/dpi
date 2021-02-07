################################################################################
#
# dwx 2020-10-06
#
# lors_scanner,emap,voxelSize: dtype = np.float32
#
#################################################################################
import numpy as np
import math
from tqdm import tqdm
from numba import cuda
from dpi.reconstruction.p_hao3D import projection3D_cuda
from dpi.reconstruction.bp_hao3D import bprojection3D_cuda
import time as tt


class MLEM():
    def __init__(self,imageSize,voxelSize,emap):
        """ imageSize == emap.shape """
        self.imageSize = imageSize
        self.voxelSize = voxelSize
        self.emap = emap
        self.imageInit()

    def imageInit(self):
        """image initialization"""
        self.image = np.ones(self.imageSize,dtype=np.float32)

    def osem(self,lors_scanner,nb_iter=5,nb_sub=5):
        lors_all = np.array_split(lors_scanner,nb_sub)
        for it in tqdm(range(nb_iter)):
            for lors in lors_all: 
                # 当前图像投影
                P_now = np.zeros(lors.shape[0], dtype=np.float32)
                projection3D_cuda[(lors.shape[0]+255)//256, 256](lors, self.image, self.voxelSize, P_now)
                # 比值
                p_ratio = np.divide(np.ones(lors.shape[0],dtype=np.float32), P_now, out=np.zeros_like(P_now), where=P_now!=0)
                # 比值反投影
                I = np.zeros(self.imageSize, dtype=np.float32)
                bprojection3D_cuda[(lors.shape[0]+255)//256, 256](lors, p_ratio, self.voxelSize, I)
                # 与emap比值 
                scale = np.divide(I, self.emap, out=np.zeros_like(I), where=self.emap!=0)
                # 更新图像
                self.image *= scale

    def mapem_t(self,lors_scanner,pred_u,p,nb_iter=5,nb_sub=5):
        """
        Iterative reconstruction with compartment model regularization
        pred_u: f(k,t)-u, where u is Lagrange multiplier, pred_u.shape = imageSize 
        p: penalty coefficient, scalar
        """
        lors_all = np.array_split(lors_scanner,nb_sub)
        for it in tqdm(range(nb_iter)):
            for lors in lors_all:
                # 当前图像投影
                P_now = np.zeros(lors.shape[0], dtype=np.float32)
                projection3D_cuda[(lors.shape[0]+255)//256, 256](lors, self.image, self.voxelSize, P_now)
                # 比值
                p_ratio = np.divide(np.ones(lors.shape[0],dtype=np.float32), P_now, out=np.zeros_like(P_now), where=P_now!=0)
                # 比值反投影
                I = np.zeros(self.imageSize, dtype=np.float32)
                bprojection3D_cuda[(lors.shape[0]+255)//256, 256](lors, p_ratio, self.voxelSize, I)
                # 与emap比值 
                scale = np.divide(I, self.emap+p*(self.image-pred_u), out=np.zeros_like(I), where=self.emap!=0)
                # 更新图像
                self.image *= scale


class DyMLEM(MLEM):
    def __init__(self,imageSize,voxelSize,emap,nb_frames):
        """ imageSize == emap.shape """
        super(DyMLEM,self).__init__(imageSize,voxelSize,emap)
        self.nb_frames = nb_frames
        self.dyInit()

    def dyInit(self):
        """dynamic image initialization"""
        self.dy_im = np.ones((self.nb_frames,self.imageSize[0],self.imageSize[1],self.imageSize[2]),dtype=np.float32)

    def dyosem(self,path,nb_iter=5,nb_sub=5):    
        for i in range(self.nb_frames):
            lors_scanner = np.fromfile(path+'frame'+str(i)+'.bin',dtype=np.float32).reshape(-1,6)
            self.image = self.dy_im[i]   # frame initialization
            self.osem(lors_scanner,nb_iter=nb_iter,nb_sub=nb_sub)
            self.dy_im[i] = self.image

    def dymapem(self,path,pred_u,p,nb_iter=1,nb_sub=5):
        """
        pred_u: f(k,t)-u, where u is Lagrange multiplier, pred_u.shape == dy_im.shape
        p: penalty coefficient, scalar
        """
        for i in range(self.nb_frames):
            lors_scanner = np.fromfile(path+'frame'+str(i)+'.bin',dtype=np.float32).reshape(-1,6)
            self.image = self.dy_im[i]   # frame initialization
            self.mapem(lors_scanner,pred_u[i],p[i],nb_iter=nb_iter,nb_sub=nb_sub)
            self.dy_im[i] = self.image


def recon2activity(dyim,tS,tE,scale=1):
    """"""
    for i in range(len(tS)):
        dyim[i,:,:,:] /= (tE[i] - tS[i])
    return dyim / scale


def activity2recon(dyim,tS,tE,scale=1):
    for i in range(len(tS)):
        dyim[i,:,:,:] *= (tE[i] - tS[i])
    return dyim * scale