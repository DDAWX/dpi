##########################################################
#
# dwx 2021-01-01
#
# all tensor should be float32
# the unit of t is minutes
#
###########################################################
import numpy as np
import torch
import torch.nn.functional as F
from scipy import integrate
from scipy.interpolate import interp1d,interpn
from skimage.restoration import richardson_lucy
from skimage.filters import gaussian
from matplotlib import pyplot as plt
import nibabel as nib
from tqdm import tqdm



def myConv1d(a,b,interval=None,t=None,method='efficient',device=None):
    """
    convolution by each column
    a: torch.tensor([[]]) (n,m)
    b: torch.tensor([[]]) (n,j), (j=1 or m if use 'efficient' else j=1)
    interval:scalar
    t: torch.tensor([]) (resampling), 1D
    method='efficient'(require the interval and t)
    method='full'     (require the interval)
    """
    b = b.flip(0)   #翻转
    if method == 'efficient':
        x = torch.round(t/interval).long()
        result = torch.zeros(t.shape[0],a.shape[1],device=device)
        for i in range(t.shape[0]):
            result[i,:] = (a[:x[i]+1,:] * b[-x[i]-1:,:]).sum(0)
        return result * interval
    elif method == 'full':
        h = torch.cat((torch.zeros(a.shape[0]-1,a.shape[1],device=device),a),dim=0)
        return F.conv1d(h.T.unsqueeze(1), b.T.unsqueeze(1)).squeeze().T * interval
    else:
        print('please use method efficient or full')


def cumtrapz(y,t,same=True,device=None):
    if y.dim() == 1:
        out = torch.cumsum((y[:-1] + y[1:]) * (t[1:] - t[:-1]),dim=0) / 2
        if same:
            out = torch.cat((torch.tensor([0.],device=device),out),dim=0)
    elif y.dim() == 2:
        out = torch.cumsum((y[:-1] + y[1:]) * (t[1:] - t[:-1]).unsqueeze(1),dim=0) / 2
        if same:
            out = torch.cat((torch.zeros(1,out.shape[1],device=device),out),dim=0)
    else:
        print('wrong dimemsion!')
    return out


def subtrapz(y,tao,tS,tE):
    """
    subsection integral by the first dimension
    y: torch.tensor([[]]), shape is (tao,im_L)
    tao: tensor,sample time,should be uniformly-spaced
    tS: start time, tS[0]>=tao[0]
    tE: end time, tE[-1]<=tao[-1]
    """
    interval = tao[1] - tao[0]
    ind_S = torch.round((tS-tao[0])/interval).long() # the index of tS in tao
    ind_E = torch.round((tE-tao[0])/interval).long() # the index of tE in tao
    y_integral = torch.zeros(tE.shape[0],y.shape[1])
    for i in range(tE.shape[0]):
        y_integral[i,:] = torch.trapz(y[ind_S[i]:ind_E[i]+1,:],tao[ind_S[i]:ind_E[i]+1],dim=0)
    return y_integral


def cumintegrate(y,t,method='trapz'):
    """
    accumulate integral by the first dimension
    y: torch.tensor([[]]), shape is (t,im_L)
    t: torch.tensor([]), 1D
    """
    if method=='trapz':
        return torch.from_numpy(integrate.cumtrapz(y,t,axis=0))
    elif method=='simps':
        result = np.zeros((y.shape[0]-1,y.shape[1]),dtype=np.float32)
        for i in range(t.shape[0]-1):
            result[i,:] = integrate.simps(y[:i+2,:], t[:i+2],axis=0)
        return torch.from_numpy(result)
    else:
        print('method = trapz or simps')


def myInterp1d(x,y,xnew,kind='cubic',fill_value='extrapolate',axis=0):
    return torch.from_numpy(interp1d(x, y, kind=kind, fill_value=fill_value, axis=axis)(xnew)).to(torch.float32)


#--------------------------------------------------------------------------------------------------------------------------
def imshow(image,figsize=[9,7],vmin=None,vmax=None):
    plt.figure(figsize=figsize)
    plt.imshow(image,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.show()


def imshow3d(image,figsize=[20,6],vmin=None,vmax=None):
    a = int(image.shape[0] / 2)
    b = int(image.shape[1] / 2)
    c = int(image.shape[2] / 2)
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(image[a,:,:],vmin=vmin,vmax=vmax)
    plt.subplot(132)
    plt.imshow(image[:,b,:],vmin=vmin,vmax=vmax)
    plt.subplot(133)
    plt.imshow(image[:,:,c],vmin=vmin,vmax=vmax)
    plt.show()


def read_nii(file_path):
    im = nib.load(file_path).get_fdata().astype(dtype=np.float32)
    im[np.where(im!=im)] = 0
    if np.ndim(im)==4:
        im = np.swapaxes(im,2,3)
        im = np.swapaxes(im,1,2) 
        im = np.swapaxes(im,0,1) # (C, D, H, W)
    return im


def resize_crop(im,out_size):
    """ im: 3D tensor, out_size: int [x,y,z] """
    a = out_size[-3] - im.shape[-3]
    b = out_size[-2] - im.shape[-2]
    c = out_size[-1] - im.shape[-1]
    a_ = int(a / 2.)
    b_ = int(b / 2.)
    c_ = int(c / 2.)
    return F.pad(im, (c_,c-c_,b_,b-b_,a_,a-a_), "constant", 0)


def two_mode_registration(in_image,out_shape,A1,A2):
    """
    将in_image配到out_image,A1:affine matrix of in_image,A2:affine matrix of out_image
    e.g.: A_pet = np.array([[2.08626,     0,         0,         -358.217],
                            [0,           2.08626,   0,         -355.642],
                            [0,           0,         2.03125,   -151.865],
                            [0,           0,         0,          1]],dtype=np.float32)
          A_mri = np.array([[-0.999952,   0.00340223,0.00334638,  86.4786],
                            [ 0.00770569, 0.485066,  0.0558198, -109.194],
                            [ 0.00601166,-0.0558429, 0.485069,  -143.611],
                            [ 0,          0,         0,          1]],dtype=np.float32)
          out_image = two_mode_registration(T2,mean.shape,A_mri,A_pet)
    """
    A = np.linalg.inv(A1).dot(A2)
    out_image = np.zeros(out_shape)
    points = (np.arange(0,in_image.shape[0],1),np.arange(0,in_image.shape[1]),np.arange(0,in_image.shape[2]))
    for i in tqdm(range(out_shape[0])):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                point = A.dot([[i],[j],[k],[1]]).squeeze()[:-1]
                if point[0] > 0 and point[0] < in_image.shape[0]-1 and point[1] > 0 and point[1] < in_image.shape[1]-1 and point[2] > 0 and point[2] < in_image.shape[2]-1:
                    out_image[i,j,k] = interpn(points, in_image, point)
    return out_image


def gussian_kernel(kernelSize=[5,5,5],sigma=0.5):
    kernel = np.zeros(kernelSize,np.float32)
    kernel[int(kernelSize[0]/2),int(kernelSize[1]/2),int(kernelSize[2]/2)] = 1
    kernel = gaussian(kernel, sigma=sigma)
    kernel = data_norm(kernel) / data_norm(kernel).sum()
    return kernel


def deconv_RL(in_im,psf,niter=30):
    """in_im, psf: numpy array"""
    out_im = data_norm(in_im,0,1)
    if in_im.ndim == psf.ndim:
        out_im = richardson_lucy(out_im, psf, iterations=niter)
    elif in_im.ndim == psf.ndim + 1:
        for i in range(in_im.shape[0]):
            out_im[i] = richardson_lucy(out_im[i], psf, iterations=niter)
    else:
        print('the dim is wrong')
    out_im = out_im * (in_im.max() - in_im.min()) + in_im.min()
    return out_im


def get_tac(dy_im,mask):
    tac = torch.zeros(dy_im.shape[0])
    for i in range(dy_im.shape[0]):
        im = dy_im[i]
        tac[i] = im[torch.where(mask==1)].mean()
    return tac


def decay_factor(t,t_half = 109.771):
    return np.exp( - ( np.log(2) / t_half ) * t )


#--------------------------------------------------------------------------------------------------------------------------
def plotShere(image,center,r,value):
    """plot a shere on the image"""
    out = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if (i-center[0]) ** 2 + (j-center[1]) ** 2 + (k-center[2]) ** 2 < r * r:
                    out[i,j,k] = value
    return out


def plotCylinder(image,center,r,halfz,value):
    """plot a cylinder on the image"""
    out = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i-center[0]) ** 2 + (j-center[1]) ** 2 < r * r:
                out[i,j,center[2]-halfz:center[2]+halfz] = value
    return out