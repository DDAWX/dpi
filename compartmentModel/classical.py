################################################################################
#
# dwx 2021-01-01
#
# all tensor should be float32
# the unit of t is minutes
# 
#################################################################################
import numpy as np
import torch
from dpi.function import myConv1d, subtrapz, myInterp1d, cumintegrate, cumtrapz


###############################################################################################################
#
# input function
# 
###############################################################################################################
def fengInpFun(p,t):
    """ param:torch.tensor([]), time:torch.tensor(any dimension) """
    y = ( ( p[0] * t - p[1] - p[2] ) * torch.exp( p[3] * t ) + p[1] * torch.exp( p[4] * t ) + p[2] * torch.exp( p[5] * t ) )
    return y

def fengInpFun_int(p,t,from_zero=False):
    """ One integral of feng input, p:torch.tensor([]), t:torch.tensor([]) """
    y = ( 1 / p[3] * ( p[0] * t * torch.exp( p[3] * t ) + ( p[0] / p[3] + p[1] + p[2] ) * ( 1 - torch.exp( p[3] * t ) ) )
        + p[1] / p[4] * ( torch.exp( p[4] * t ) - 1 ) + p[2] / p[5] * ( torch.exp( p[5] * t ) - 1 ) )
    return y if from_zero else y[1:] - y[:-1]

def fengInpFun_int2(p,t,from_zero=True):
    """ Two integral of feng input, p:torch.tensor([]), t:torch.tensor([]) """
    y = ( 1 / p[3].pow(2) * ( p[0] * t * torch.exp( p[3] * t ) - ( 2 * p[0] / p[3] + p[1] + p[2] ) * ( torch.exp( p[3] * t ) - 1 ) )
        + p[1] / p[4].pow(2) * ( torch.exp( p[4] * t ) - 1 ) + p[2] / p[5].pow(2) * ( torch.exp( p[5] * t ) - 1 ) 
        + ( ( p[0] / p[3] + p[1] + p[2] ) / p[3] - p[1] / p[4] - p[2] / p[5] ) * t )
    return y if from_zero else y[1:] - y[:-1]


def oneExp(p,t):
    return p[0] * torch.exp(p[1]*t)

def twoExp(p,t):
    return p[0] * torch.exp(p[1]*t) + p[2] * torch.exp(p[3]*t)

def threeExp(p,t):
    return p[0] * torch.exp(p[1]*t) + p[2] * torch.exp(p[3]*t) + p[4] * torch.exp(p[5]*t)

def oneExp_int(p,tl,tu):
    """tl(lower bound) and tu(upper bound) of the integral"""
    return ( p[0]/p[1] ) * ( torch.exp(p[1]*tu) - torch.exp(p[1]*tl) )

def twoExp_int(p,tl,tu):
    """tl(lower bound) and tu(upper bound) of the integral"""
    return ( p[0]/p[1] ) * ( torch.exp(p[1]*tu) - torch.exp(p[1]*tl) ) + ( p[2]/p[3] ) * ( torch.exp(p[3]*tu) - torch.exp(p[3]*tl) )

def threeExp_int(p,tl,tu):
    """tl(lower bound) and tu(upper bound) of the integral"""
    return ( p[0]/p[1] ) * ( torch.exp(p[1]*tu) - torch.exp(p[1]*tl) ) + ( p[2]/p[3] ) * ( torch.exp(p[3]*tu) - torch.exp(p[3]*tl) ) + ( p[4]/p[5] ) * ( torch.exp(p[5]*tu) - torch.exp(p[5]*tl) )


###############################################################################################################
#
# compartment model
# 
###############################################################################################################
class D_T_KCM(torch.nn.Module):
    def __init__(self,im_L=1,requires_fv=False,device='cuda'):
        super(D_T_KCM,self).__init__()
        self.device = device
        self.im_L = im_L
        self.requires_fv = requires_fv
    
    def prevent_zero(self,c,eps=1e-7):
        c[torch.where( torch.logical_and(-eps<c,c<0) )] -= eps
        c[torch.where( torch.logical_and(0<=c,c<eps) )] += eps
        return c


class D1T2KCM(D_T_KCM):
    """  1 Tissue 2K compartment model  """
    def __init__(self,im_L=1,requires_fv=False,device='cuda'):
        super(D1T2KCM,self).__init__(im_L=im_L,requires_fv=requires_fv,device=device)
        self.k = self.paramInit()

    def paramInit(self):
        nb_parameters = 3 if self.requires_fv else 2 # [k1,k2] or [k1,k2,fv]
        p_ = torch.ones(nb_parameters,self.im_L,device=self.device) * 0.5
        return p_.requires_grad_(True)

    def analytical(self,p,t):
        t = t.unsqueeze(1)
        c1 = self.prevent_zero( p[3] + self.k[1,:] )
        c2 = p[1] / self.prevent_zero( p[4] + self.k[1,:] )
        c3 = p[2] / self.prevent_zero( p[5] + self.k[1,:] )
        B = p[0] / c1.pow(2) + ( p[1] + p[2] ) / c1
        y = self.k[0,:] * ( ( p[0] / c1 * t - B ) * torch.exp( p[3] * t ) 
                            + c2 * torch.exp( p[4] * t ) 
                            + c3 * torch.exp( p[5] * t )
                            + ( B - c2 - c3 ) * torch.exp( - self.k[1,:] * t ) )
        if self.requires_fv:
                y = ( 1 - self.k[2,:] ) * y + self.k[2,:] * fengInpFun(p,t).unsqueeze(1)
        return y


class D2T3KCM(D_T_KCM):
    """  2 Tissue 3K compartment model  """
    def __init__(self,im_L=1,requires_fv=False,device='cuda'):
        super(D2T3KCM,self).__init__(im_L=im_L,requires_fv=requires_fv,device=device)
        self.k = self.paramInit()

    def paramInit(self):
        nb_parameters = 4 if self.requires_fv else 3 # [k1,k2,k3] or [k1,k2,k3,fv]
        p_ = torch.ones(nb_parameters,self.im_L,device=self.device) * 0.05
        return p_.requires_grad_(True)
    
    def macroParam(self):
        A2 = self.prevent_zero(self.k[1,:] + self.k[2,:])
        B1 = self.k[0,:] * self.k[2,:] / A2 
        B2 = self.k[0,:] * self.k[1,:] / A2
        return A2,B1,B2
    
    def analytical(self,p,t,integral=False):
        """
        1) if integral=False, t is torch.tensor([]), return the activity at time 't'
        2) if integral=True , t is cat(tS, tE[-1]),  return the integral within two adjacent time points
        """
        A2,B1,B2 = self.macroParam()
        t = t.unsqueeze(1)
        c2 = self.prevent_zero( A2 + p[3] )
        c4 = self.prevent_zero( A2 + p[4] )
        c6 = self.prevent_zero( A2 + p[5] )
        D1 = ( p[0] + ( p[1] + p[2] ) * p[3] ) / p[3].pow(2)
        D2 = ( p[0] + ( p[1] + p[2] ) * c2 ) / c2.pow(2)
        if integral == False:
            y = ( B1 * ( D1 - p[1] / p[4] - p[2] / p[5] )
                + B2 * ( D2 - p[1] / c4 - p[2] / c6 ) * torch.exp( - A2 * t )
                + ( p[0] * ( B1 / p[3] + B2 / c2 ) * t - B1 * D1 - B2 * D2 ) * torch.exp( p[3] * t )
                + p[1] * ( B1 / p[4] + B2 / c4 ) * torch.exp( p[4] * t )
                + p[2] * ( B1 / p[5] + B2 / c6 ) * torch.exp( p[5] * t ) )
            if self.requires_fv:
                y = ( 1 - self.k[3,:] ) * y + self.k[3,:] * fengInpFun(p,t).unsqueeze(1)
        else:
            y = ( B1 * ( D1 - p[1] / p[4] - p[2] / p[5] ) * t
                + B2 / A2 * ( p[1] / c4 + p[2] / c6 - D2 ) * ( torch.exp( - A2 * t ) - 1 )
                + p[0] / p[3].pow(2) * ( B1 / p[3] + B2 / c2 ) * ( ( p[3] * t - 1 ) * torch.exp( p[3] * t ) + 1 )
                - ( B1 * D1 + B2 * D2 ) / p[3] * ( torch.exp( p[3] * t ) - 1 )
                + p[1] / p[4] * ( B1 / p[4] + B2 / c4 ) * ( torch.exp( p[4] * t ) - 1 )
                + p[2] / p[5] * ( B1 / p[5] + B2 / c6 ) * ( torch.exp( p[5] * t ) - 1 ) )
            y = y[1:,:] - y[:-1,:]
            if self.requires_fv:
                y = ( 1 - self.k[3,:] ) * y + self.k[3,:] * fengInpFun_int(p,t).unsqueeze(1)
        return y

    def numerical(self,inp,t,interval=1/60,inpFun=fengInpFun):
        """
        1) if inpFun is fengInpFun, inp is the six parameters of fengInputFun
        2) if inpFun is 
        """
        A2,B1,B2 = self.macroParam()
        tao = torch.arange(0,t[-1]+interval,interval,device=self.device).unsqueeze(1)
        g = B1 + B2 * torch.exp( - A2 * tao )
        f = inpFun(inp,tao)
        y = myConv1d(g,f,interval,t,method='efficient',device=self.device)
        if self.requires_fv:
            y = ( 1 - self.k[3,:] ) * y + self.k[3,:] * inpFun(inp,t).unsqueeze(1)
        return y


class D2T4KCM(D_T_KCM):
    """  2 Tissue 4K compartment model  """
    def __init__(self,im_L=1,requires_fv=False,device='cuda'):
        super(D2T4KCM,self).__init__(im_L=im_L,requires_fv=requires_fv,device=device)
        self.k = self.paramInit()
    
    def paramInit(self):
        nb_parameters = 5 if self.requires_fv else 4 # [k1,k2,k3,k4] or [k1,k2,k3,k4,fv]
        p_ = torch.ones(nb_parameters,self.im_L,device=self.device) * 0.05
        p_[3,:] *= 0.1
        return p_.requires_grad_(True)

    def macroParam(self):
        c_ = torch.sqrt( ( torch.square( self.k[1,:] + self.k[2,:] + self.k[3,:] ) - 4 * self.k[1,:] * self.k[3,:] ).clamp_(1e-7) )
        A1 = ( self.k[1,:] + self.k[2,:] + self.k[3,:] - c_ ) / 2
        A2 = ( self.k[1,:] + self.k[2,:] + self.k[3,:] + c_ ) / 2
        B1 = self.k[0,:] * ( self.k[2,:] + self.k[3,:] - A1 ) / c_
        B2 = self.k[0,:] * ( A2 - self.k[2,:] - self.k[3,:] ) / c_
        return A1,A2,B1,B2
        
    def analytical(self,p,t,integral=False):
        """
        1) if integral=False, t is torch.tensor([]), return the activity at time 't'
        2) if integral=True , t is cat(tS, tE[-1]),  return the integral within two adjacent time points
        """
        A1,A2,B1,B2 = self.macroParam()
        t = t.unsqueeze(1)
        c1 = self.prevent_zero( A1 + p[3] )
        c2 = self.prevent_zero( A2 + p[3] )
        c3 = self.prevent_zero( A1 + p[4] )
        c4 = self.prevent_zero( A2 + p[4] )
        c5 = self.prevent_zero( A1 + p[5] )
        c6 = self.prevent_zero( A2 + p[5] )
        D1 = ( p[0] + ( p[1] + p[2] ) * c1 ) / c1.pow(2)
        D2 = ( p[0] + ( p[1] + p[2] ) * c2 ) / c2.pow(2)
        if integral == False:
            y = ( B1 * ( D1 - p[1] / c3 - p[2] / c5 ) * torch.exp( - A1 * t )
                + B2 * ( D2 - p[1] / c4 - p[2] / c6 ) * torch.exp( - A2 * t )
                + ( p[0] * ( B1 / c1 + B2 / c2 ) * t - B1 * D1 - B2 * D2 ) * torch.exp( p[3] * t )
                + p[1] * ( B1 / c3 + B2 / c4 ) * torch.exp( p[4] * t )
                + p[2] * ( B1 / c5 + B2 / c6 ) * torch.exp( p[5] * t ) )
            if self.requires_fv:
                y = ( 1-self.k[4,:] ) * y + self.k[4,:] * fengInpFun(p,t).unsqueeze(1)
        else:
            y = ( B1 / A1 * ( p[1] / c3 + p[2] / c5 - D1 ) * ( torch.exp( - A1 * t ) - 1 )
                + B2 / A2 * ( p[1] / c4 + p[2] / c6 - D2 ) * ( torch.exp( - A2 * t ) - 1 )
                + p[0] / p[3].pow(2) * ( B1 / c1 + B2 / c2 ) * ( ( p[3] * t - 1 ) * torch.exp( p[3] * t ) + 1 )
                - ( B1 * D1 + B2 * D2 ) / p[3] * ( torch.exp( p[3] * t ) - 1 )
                + p[1] / p[4] * ( B1 / c3 + B2 / c4 ) * ( torch.exp( p[4] * t ) - 1 )
                + p[2] / p[5] * ( B1 / c5 + B2 / c6 ) * ( torch.exp( p[5] * t ) - 1 ) )
            y = y[1:,:] - y[:-1,:] # Returns the integral within two adjacent time points
            if self.requires_fv:
                y = ( 1-self.k[4,:] ) * y + self.k[4,:] * fengInpFun_int(p,t).unsqueeze(1)
        return y

    def numerical(self,inp,t,interval=1/60,inpFun=fengInpFun):
        """
        1) if inpFun is fengInpFun, inp is the six parameters of fengInputFun
        2) if inpFun is 
        """
        A1,A2,B1,B2 = self.macroParam()
        tao = torch.arange(0,t[-1]+interval,interval,device=self.device).unsqueeze(1)
        g = B1 * torch.exp( - A1 * tao ) + B2 * torch.exp( - A2 * tao )
        f = inpFun(inp,tao)
        y = myConv1d(g,f,interval,t,method='efficient',device=self.device)
        if self.requires_fv:
            y = ( 1-self.k[4,:] ) * y + self.k[4,:] * inpFun(inp,t).unsqueeze(1)
        return y


###############################################################################################################
#
# linear model
# 
###############################################################################################################
class Patlak(torch.nn.Module):
    """  inpParam: torch.tensor([]) or torch.tensor([],requires_grad=True) """
    def __init__(self,im_L=1,device='cuda'): 
        super(Patlak,self).__init__()
        self.device = device
        self.im_L = im_L
        self.k = self.paramInit()
    
    def paramInit(self):
        p_ = torch.ones(2,self.im_L,device=self.device) * 0.2
        p_[0,:] *= 0.1
        return p_.requires_grad_(True)

    def forward(self,Cpt_int,Cpt):
        """
        Cpt: linear part of input function
        Cpt_int: linear part of integral of the input function from 0min 
        """
        return Cpt_int.unsqueeze(1) * self.k[0,:] + Cpt.unsqueeze(1) * self.k[1,:]


class Logan(torch.nn.Module):
    def __init__(self,im_L=1,device='cuda'):
        super(Logan,self).__init__()
        self.device = device
        self.im_L = im_L
        self.k = self.paramInit()
    
    def paramInit(self):
        p = torch.ones(2,self.im_L,device=self.device) * torch.tensor([[1.],[-1.]],device=self.device)
        return p.requires_grad_(True)
    
    def forward(self,A,B):
        """
        ordinary least square (OLS)
            A: linear part of ref_inte, 2D ( _ ,1)
            B: linear part of CT(t) or [CT(t)-CT(t1)], 2D ( _ ,im_L)
            C: linear part of y_inte, 2D ( _ ,im_L)
            y_inte = DVR * ref_inte + d * y, k = [DVR, d]
        multilinear analysis 1 (MA1)
            swap B with C, then the k = [-DVR/d, 1/d]
        """
        C = A * self.k[0,:] + B * self.k[1,:]
        return C


###############################################################################################################
#
# other model
# 
###############################################################################################################
class SRTM(torch.nn.Module):
    """  Simplified Reference Tissue Model: [R1,k2,BPND], cuda is not supported for now  """
    def __init__(self,im_L=1,device='cuda'):
        super(SRTM,self).__init__()
        self.device = device
        self.im_L = im_L
        self.k = self.paramInit()

    def paramInit(self): 
        p_ = torch.ones(3,self.im_L,device=self.device) * 0.1  # [R1,k2,BPND]
        return p_.requires_grad_(True)
    
    def forward(self,t,ref_tac,method='numerical',interval=1/60):
        """
        ref_tac: torch.tensor([]) is the activity at time 't'
        1) For conventional methods, t: torch.tensor([]), return the activity at time 't'
        2) For integral methods, t is cat(tS, tE[-1]), return the integral within two adjacent time points
        """
        B = self.k[1,:] * ( 1 - self.k[0,:] / ( 1 + self.k[2,:] ) )
        A = - self.k[1,:] / ( 1 + self.k[2,:] )
        if method == 'numerical':
            return self.y_numerical(t,A,B,ref_tac,interval=interval)
        elif method == 'numerical_int':
            return self.y_nume_int(t,A,B,ref_tac,interval=interval)
        elif method == 'numerical_fast':
            return self.y_nume_fast(t,A,B,ref_tac)
        else:
            print(" method = 'numerical', 'numerical_int' or 'numerical_fast' ")
    
    def y_numerical(self,t,A,B,ref_tac,interval=1/60):
        tao = torch.arange(0,t[-1]+interval,interval,device=self.device).unsqueeze(1)      
        g = torch.exp( A * tao )
        f = myInterp1d(t,ref_tac,tao).clamp_(0)
        y = ref_tac.unsqueeze(1) * self.k[0,:] + B * myConv1d(g,f,interval,t,method='efficient',device=self.device)
        # x = torch.round(t/interval).long()
        # y = ref_tac.unsqueeze(1) * self.k[0,:] + B * myConv1d(g,f,interval,t,method='full',device=self.device)[x]
        return y

    def y_nume_int(self,t,A,B,ref_tac,interval=1/60):
        tao = torch.arange(0,t[-1]+interval,interval,device=self.device).unsqueeze(1)      
        g = torch.exp( A * tao )
        f = myInterp1d((t[1:]+t[:-1])/2,ref_tac,tao).clamp_(0)
        y = f * self.k[0,:] + B * myConv1d(g,f,interval,method='full',device=self.device)
        y = subtrapz(y,tao,t[:-1],t[1:])
        return y

    def y_nume_fast(self,t,A,B,ref_tac):
        """t should be uniform"""
        g = torch.exp( t.unsqueeze(1) * A )
        y = ref_tac.unsqueeze(1) * self.k[0,:] + B * myConv1d(g,ref_tac.unsqueeze(1),interval=1.0,method='full',device=self.device)
        return y


class DSCM(torch.nn.Module):
    """ Simplified Compartment Model: [k2,BPND] and r """
    def __init__(self,inpParam,im_L=1,device='cuda'):
        super(DSCM,self).__init__()
        self.device = device
        self.im_L = im_L
        self.inpParam = inpParam
        self.k = ( torch.ones(2,im_L,device=self.device) * 0.1 ).requires_grad_(True) # [k2,BPND]
        self.r = torch.tensor([1.],device=self.device,requires_grad=True)             # r = k1/k2, all ROI share the same r

    def forward(self,t):
        """ t: torch.tensor([]), return the activity at time 't' """
        t = t.unsqueeze(1)
        p = self.inpParam
        k2a = self.k[0,:] / ( 1 + self.k[1,:] )
        c1 = self.prevent_zero( p[3] + k2a )
        c2 = p[1] / self.prevent_zero( p[4] + k2a )
        c3 = p[2] / self.prevent_zero( p[5] + k2a )
        B = p[0] / c1.pow(2) + ( p[1] + p[2] ) / c1
        y = self.r * self.k[0,:] * ( ( p[0] / c1 * t - B ) * torch.exp( p[3] * t ) 
                                    + c2 * torch.exp( p[4] * t ) 
                                    + c3 * torch.exp( p[5] * t ) 
                                    + ( B - c2 - c3 ) * torch.exp( - k2a * t ) )
        return y

    def prevent_zero(self,c,eps=1e-7):
        c[torch.where( torch.logical_and(-eps<c,c<0) )] -= eps
        c[torch.where( torch.logical_and(0<=c,c<eps) )] += eps
        return c


class GLLS():
    """ kmodel: 2t4k or 2t3k, ‘cuda’ is not supported for now """
    def __init__(self,inpParam,kmodel='2t4k'):
        if kmodel!='2t4k' and kmodel!='2t3k':
            print('kmodel should be 2t4k or 2t3k')
        self.inpParam = inpParam
        self.kmodel = kmodel
        self.nb_parameters = 4 if kmodel=='2t4k' else 3

    def forward(self,t,y,iter=3,interval=1/60):
        """
        t: cat(tS, tE[-1]); 
        y: torch.tensor([[]]), 2D shape is (t,im_L), it is the integral within two adjacent time points"""
        P = self.lls(t,y)
        for it in range(1,iter):
            P = self.glls(t,y,P,interval=interval)
        return self.convertP2k(P)

    def lls(self,t,y):
        cit = myInterp1d((t[1:]+t[:-1])/2, y/(t[1:]-t[:-1]).unsqueeze(1), t[1:], axis=0)
        cit_int1 = torch.cumsum(y, dim=0)
        A = torch.zeros(y.shape[1],y.shape[0],self.nb_parameters)
        A[:,:,0] = fengInpFun_int(self.inpParam,t[1:],from_zero=True)  # cpt_int1
        A[:,:,1] = fengInpFun_int2(self.inpParam,t[1:],from_zero=True) # cpt_int2
        A[:,:,2] = cit_int1.T
        if self.kmodel=='2t4k':
            A[:,:,3] = cumintegrate(torch.cat((torch.zeros(1,cit_int1.shape[1]),cit_int1),dim=0),t,method='trapz').T # cit_int2.T
        P = self.leastsq(A,cit.T.unsqueeze(2))
        return P

    def glls(self,t,y,P,interval=1/60):
        tao = torch.arange(0,t[-1]+interval,interval)
        phi1, phi2 = self.phi(P,tao)
        cpt_tao = fengInpFun(self.inpParam,tao).unsqueeze(1)
        y_ = y/(t[1:]-t[:-1]).unsqueeze(1)
        cit_tao = myInterp1d((t[1:]+t[:-1])/2, y_, tao, axis=0).clamp_(0)
        B = torch.zeros(y.shape[1],y.shape[0],self.nb_parameters)
        B[:,:,0] = myConv1d(phi1.T,cpt_tao,interval,t=t[1:],method='efficient').T
        B[:,:,1] = myConv1d(phi2.T,cpt_tao,interval,t=t[1:],method='efficient').T
        B[:,:,2] = myConv1d(phi1.T,cit_tao,interval,t=t[1:],method='efficient').T
        if self.kmodel=='2t4k':
            B[:,:,3] = myConv1d(phi2.T,cit_tao,interval,t=t[1:],method='efficient').T
            Z = ( y_ + myConv1d((P[:,2]*phi1+P[:,3]*phi2).T,cit_tao,interval,t=t[1:],method='efficient') ).T.unsqueeze(2)
        elif self.kmodel=='2t3k':
            Z = ( y_ + myConv1d((P[:,2]*phi1).T,cit_tao,interval,t=t[1:],method='efficient') ).T.unsqueeze(2)
        P = self.leastsq(B,Z)
        return P

    def leastsq(self,A,y):
        """
        solve Ax=y, 'batch' means 'im_L'
        input  --> A: [batch, nb_frames, nb_paramters]; y: [batch, nb_frames, 1]
        output --> x: [batch, nb_paramters, 1]
        """
        return torch.bmm(torch.bmm(torch.bmm(A.transpose(1,2),A).inverse(),A.transpose(1,2)), y)

    def phi(self,P,t): 
        """P: [batch, nb_paramters, 1]; t: torch.tensor([]),1D; phi1 or phi2: [batch, t.shape[0]]"""
        if self.kmodel=='2t4k':
            c = torch.sqrt( P[:,2].pow(2) + 4 * P[:,3] )
            lambda1 = ( P[:,2] + c ) / 2
            lambda2 = ( P[:,2] - c ) / 2
            phi1 = ( lambda1 * torch.exp( lambda1 * t ) - lambda2 * torch.exp( lambda2 * t ) ) / c
            phi2 = ( torch.exp( lambda1 * t ) - torch.exp( lambda2 * t ) ) / c
        elif self.kmodel=='2t3k':
            phi1 = torch.exp( P[:,2] * t )
            phi2 = ( torch.exp( P[:,2] * t ) - 1 ) / P[:,2]
        return phi1, phi2

    def convertP2k(self,P):
        if self.kmodel=='2t4k':
            k1 = P[:,0]
            k2 = - P[:,1] / P[:,0] - P[:,2]
            k4 = - P[:,3] / k2
            k3 = - P[:,2] - k2 - k4
            k = torch.cat((k1,k2,k3,k4),dim=1).T
        elif self.kmodel=='2t3k':
            k1 = P[:,0]
            k3 = P[:,1] / P[:,0]
            k2 = -P[:,2] - k3
            k = torch.cat((k1,k2,k3),dim=1).T
        return k


class DualTracer(torch.nn.Module):
    """
    dual tracer compartment model, default:D2T3KCM + D2T4KCM without fv
    t:torch.tensor([]), delay:scalar
    """
    def __init__(self,inpParamA,inpParamB,t,delay):
        super(DualTracer,self).__init__()
        self.modelA = D2T3KCM(inpParamA)
        self.modelB = D2T4KCM(inpParamB)
        self.delay = delay
        self.T = torch.where(t>=self.delay)[0][0]
        self.tA = t
        self.tB = t[self.T:]
    
    def dual_Cit(self):
        Cit = self.modelA(self.tA)
        Cit[self.T:,:] += self.modelB(self.tB-self.delay)
        return Cit
        
    def dual_Cpt(self): 
        Cpt = fengInpFun(self.modelA.inpParam, self.tA)
        Cpt[self.T:] += fengInpFun(self.modelB.inpParam, self.tB-self.delay)
        return Cpt
    
    def params_k(self):
        return [{'params':self.modelA.k},{'params':self.modelB.k}]
    
    def params_p(self):
        return [{'params':self.modelA.inpParam},{'params':self.modelB.inpParam}]











