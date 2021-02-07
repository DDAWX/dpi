################################################################################
#
# dwx 2020-10-06
#
# for gate simulation
#
#################################################################################
import numpy as np
import math
from numba import cuda
from tqdm import tqdm
from dpi.reconstruction.bp_hao3D import bprojection3D,bprojection3D_cuda


class System_20panel():
    """ 针对20panel系统 """
    def __init__(self):
        self.all_crystal,self.all_submodule,self.all_panel = self.geometry()

    def gate_trans(self,lors_global):
        """
        输入：仿真全局坐标[[x1,y1,z1,x2,y2,z2]], 输出：晶体中心坐标[[x1,y1,z1,x2,y2,z2]]
        方法依次查找 -> 1.panel -> 2.submodule -> 3.crystal
        """
        lors_scanner = np.zeros(lors_global.shape,dtype=np.float32)
        for i in range(len(lors_global)):
            # 确定panel,共20个panel
            idx1 = np.sum(np.square(self.all_panel[:,:2] - lors_global[i,0:2]),axis=1).argmin()
            idx2 = np.sum(np.square(self.all_panel[:,:2] - lors_global[i,3:5]),axis=1).argmin()
            # 确定submodule，每个panel有48个submodule
            idx3_ = np.sum(np.square(self.all_submodule[idx1*48:(idx1+1)*48] - lors_global[i,0:3]),axis=1).argmin()
            idx3 = idx3_ + idx1*48
            idx4_ = np.sum(np.square(self.all_submodule[idx2*48:(idx2+1)*48] - lors_global[i,3:6]),axis=1).argmin()
            idx4 = idx4_ + idx2*48
            # 确定crystal，每个submodule有36个crystal
            idx5_ = np.sum(np.square(self.all_crystal[idx3*36:(idx3+1)*36] - lors_global[i,0:3]),axis=1).argmin()
            idx5 = idx5_ + idx3*36
            idx6_ = np.sum(np.square(self.all_crystal[idx4*36:(idx4+1)*36] - lors_global[i,3:6]),axis=1).argmin()
            idx6 = idx6_ + idx4*36
            # result
            lors_scanner[i,0:3] = self.all_crystal[idx5]
            lors_scanner[i,3:6] = self.all_crystal[idx6]
        return lors_scanner

    def cal_emap(self,imageSize,voxelSize,device='cuda'):
        """ 
        imageSize = np.array([x,y,z],dtype=np.int16 or int32)
        voxelSize = np.array([a,b,c],dtype=np.float32)
        """
        nb_crystals = len(self.all_crystal)
        octiles_ind = np.where(np.logical_and.reduce((self.all_crystal[:,0]>0, self.all_crystal[:,1]>0, self.all_crystal[:,2]>0)))[0]
        ind1, ind2 = np.meshgrid(octiles_ind, np.arange(nb_crystals,dtype=np.uint16), sparse=False, indexing='ij')
        ind1 = np.squeeze(ind1.reshape(1,-1))
        ind2 = np.squeeze(ind2.reshape(1,-1))
        lors = np.zeros((ind1.size, 6), dtype = np.float32) # 全部lor
        lors[:, :3] = self.all_crystal[ind1, :]
        lors[:, 3:] = self.all_crystal[ind2, :]
        proj_1 = np.ones(len(lors),dtype=np.float32)        # 全1投影
        emap = np.zeros(imageSize,dtype = np.float32)       # emap
        if device == 'cpu':
            bprojection3D(lors, proj_1, voxelSize, emap)
        elif device == 'cuda':
            bprojection3D_cuda[(lors.shape[0]+255)//256, 256](lors, proj_1, voxelSize, emap)
        emap += np.flip(emap,0)
        emap += np.flip(emap,1)
        emap += np.flip(emap,2)
        return emap / emap.max()

    def geometry(self):
        """ 20panel系统的结构，输出所有晶体、submodule和ring的中心坐标 """
        r = 399.72
        # crystal相对坐标
        y = 3.2 * np.arange(6,dtype = np.float32) - 3.2*(6-1)/2
        z = 3.2 * np.arange(6,dtype = np.float32) - 3.2*(6-1)/2
        yv, zv = np.meshgrid(y, z, sparse=False, indexing='ij')
        depth1 = np.zeros([6*6,3],dtype=np.float32)
        depth1[:,0] = r
        depth1[:,1] = np.squeeze(np.reshape(yv,(1,-1)))
        depth1[:,2] = np.squeeze(np.reshape(zv,(1,-1)))
        # submodule相对坐标
        y = 20 * np.arange(3,dtype = np.float32) - 20*(3-1)/2
        z = 20 * np.arange(4,dtype = np.float32) - 20*(4-1)/2
        yv, zv = np.meshgrid(y, z, sparse=False, indexing='ij')
        depth2 = np.zeros([3*4,3],dtype=np.float32)
        depth2[:,0] = r
        depth2[:,1] = np.squeeze(np.reshape(yv,(1,-1)))
        depth2[:,2] = np.squeeze(np.reshape(zv,(1,-1)))
        # module相对坐标
        depth3 = np.array([[r,-30,-40],[r,-30,40],[r,30,-40],[r,30,40]],dtype = np.float32)
        # module的绝对坐标
        module = depth3
        # submodule的绝对坐标
        a = module[0,1:3] + depth2[:,1:3]
        for i in range(1,len(module)):
            b = module[i,1:3] + depth2[:,1:3]
            a = np.concatenate((a,b),axis=0)
        submodule = np.zeros((4*12,3),dtype=np.float32)
        submodule[:,0] = r
        submodule[:,1:3] = a
        # crystal的绝对坐标
        a = submodule[0,1:3] + depth1[:,1:3]
        for i in range(1,len(submodule)):
            b = submodule[i,1:3] + depth1[:,1:3]
            a = np.concatenate((a,b),axis=0)
        crystal = np.zeros((4*12*36,3),dtype=np.float32)
        crystal[:,0] = r
        crystal[:,1:3] = a
        # 旋转
        theta = np.linspace(0,2*math.pi,20+1)[:-1]
        all_panel = np.zeros((20,3),dtype=np.float32)
        all_panel[:,0:2] = r * np.concatenate((np.cos(theta),np.sin(theta))).reshape(2,-1).T
        all_panel[:,2] = 0
        all_panel = all_panel.astype(dtype = np.float32)
        all_submodule = np.zeros((20*48,3),dtype=np.float32)
        all_crystal = np.zeros((20*48*36,3),dtype=np.float32)
        for i in range(20):
            all_submodule[i*48:(i+1)*48,0] = submodule[:,0] * np.cos(theta[i]) - submodule[:,1] * np.sin(theta[i])
            all_submodule[i*48:(i+1)*48,1] = submodule[:,0] * np.sin(theta[i]) + submodule[:,1] * np.cos(theta[i])
            all_submodule[i*48:(i+1)*48,2] = submodule[:,2]
            all_crystal[i*48*36:(i+1)*48*36,0] = crystal[:,0] * np.cos(theta[i]) - crystal[:,1] * np.sin(theta[i])
            all_crystal[i*48*36:(i+1)*48*36,1] = crystal[:,0] * np.sin(theta[i]) + crystal[:,1] * np.cos(theta[i])
            all_crystal[i*48*36:(i+1)*48*36,2] = crystal[:,2]
        return all_crystal,all_submodule,all_panel


class listmodeProcess():
    def __init__(self):
        """ for dynamic gate simulation data; the unit of t is second!!! """
        self.offset = 10000
            
    def trans(self,in_path,out_path,t1,t2,pet_sys='20panel'):
        if pet_sys=='20panel':
            pet = System_20panel()
            for i in tqdm(range(self.offset+t1,self.offset+t2)):
                data = np.fromfile(in_path+str(i)+'.bin',dtype=np.float32).reshape(-1,7)[:,:-1]
                lor = pet.gate_trans(data)
                lor.tofile(out_path+str(i)+'.bin')

    def frame_split(self,in_path,out_path,t_start,t_end,r=1.):
        for frame in tqdm(range(len(t_start))):
            lors_scanner = self.read_listmode(in_path,t_start[frame],t_end[frame],r=r)
            lors_scanner.tofile(out_path+'frame'+str(frame)+'.bin')
    
    def read_listmode(self,in_path,t1,t2,r=1.): # r表示使用多少比重的投影数据
        lors_scanner = np.zeros((1,6),dtype=np.float32)
        for i in range(self.offset+t1,self.offset+t2):
            data = np.fromfile(in_path+str(i)+'.bin',dtype=np.float32).reshape(-1,6)
            idx = int(np.round(data.shape[0] * r))
            data = data[:idx]
            lors_scanner = np.concatenate((lors_scanner,data),axis=0)
        return lors_scanner[1:,:]


class System_rings():
    """圆柱系统"""
    def __init__(self,r,nb_crys_per_ring,nb_ring=1,ring_interval=1): 
        self.crystals,self.a_ring,self.z = self.geometry(r,nb_crys_per_ring,nb_ring=1,ring_interval=1)

    def geometry(self,r,nb_crys_per_ring,nb_ring=1,ring_interval=1):
        nb_crystals = nb_crys_per_ring*nb_ring
        crystals = np.zeros([nb_crystals,3],dtype = np.float32)
        theta = np.linspace(0,2*math.pi,nb_crys_per_ring+1)[:-1]
        a_ring = r * np.concatenate((np.cos(theta),np.sin(theta))).reshape(2,-1).T
        a_ring = a_ring.astype(dtype = np.float32)
        z = np.linspace(0,(nb_ring-1)*ring_interval,nb_ring,dtype = np.float32) - (nb_ring-1)*ring_interval/2
        for i in range(nb_ring):
            crystals[i*nb_crys_per_ring:(i+1)*nb_crys_per_ring,0:2] = a_ring
            crystals[i*nb_crys_per_ring:(i+1)*nb_crys_per_ring,2] = z[i]
        return crystals,a_ring,z

    def gate_trans(self,lors_global):
        lors_scanner = np.zeros(lors_global.shape,dtype = np.float32)
        for i in range(len(lors_global)):
            idx1 = np.sum(np.square(self.a_ring - lors_global[i,0:2]),axis=1).argmin()
            idx1_z = (np.abs(self.z - lors_global[i,2])).argmin()
            idx2 = np.sum(np.square(self.a_ring - lors_global[i,3:5]),axis=1).argmin()
            idx2_z = (np.abs(self.z - lors_global[i,5])).argmin()
            lors_scanner[i,0:2] = self.a_ring[idx1]
            lors_scanner[i,2] = self.z[idx1_z]
            lors_scanner[i,3:5] = self.a_ring[idx2]
            lors_scanner[i,5] = self.z[idx2_z]
        return lors_scanner