import numpy as np
from numba import jit, njit, prange, cuda
import math

@njit(parallel=True)
def projection(lors, image, voxelSize, proj):
    dx,dy = voxelSize
    nx,ny = image.shape[0],image.shape[1]
    nx2,ny2 = nx/2,ny/2
    for i in prange(lors.shape[0]): 
        x1,y1,x2,y2 = lors[i]/dx #转换成以dx为单位的坐标
        if (x1-x2)**2+(y1-y2)**2 >= 10:  #排除一些不可能的lor
            if abs(x2 - x1) > abs(y2 - y1):
                ky = (y2 - y1)/(x2 - x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky >= 0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = math.floor(yy1)
                    cy2 = math.floor(yy2)

                    if cy1 == cy2:
                        if 0 <= cy1 < ny:
                            iy = cy1
                            weight = (1 + ky*ky) ** 0.5
                            proj[i] += image[ix, iy] * weight
                    else:
                        if -1 <= cy1 < ny:
                            if cy1 >= 0:
                                iy = cy1
                                weight = ((1 + ky * ky) ** 0.5) * ((cy2 - yy1)/(yy2 - yy1))
                                proj[i] += image[ix, iy] * weight
                            if cy2 < ny:
                                iy = cy2
                                weight = ((1 + ky * ky) ** 0.5) * ((yy2 - cy2)/(yy2 - yy1))
                                proj[i] += image[ix, iy] * weight

            else:
                kx = (x2 - x1)/(y2 - y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx >= 0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = math.floor(xx1)
                    cx2 = math.floor(xx2)

                    if cx1 == cx2:
                        if 0 <= cx1 < nx:
                            ix = cx1
                            weight = (1 + kx * kx) ** 0.5
                            proj[i] += image[ix, iy] * weight
                    else:
                        if -1 <= cx1 < nx:
                            if cx1 >= 0:
                                ix = cx1
                                weight = ((1 + kx * kx) ** 0.5) * ((cx2 - xx1)/(xx2 - xx1))
                                proj[i] += image[ix, iy] * weight
                            if cx2 < nx:
                                ix = cx2
                                weight = ((1 + kx * kx) ** 0.5) * ((xx2 - cx2)/(xx2 - xx1))
                                proj[i] += image[ix, iy] * weight


@cuda.jit
def projection_cuda(lors, image, voxelSize, proj):
    i = cuda.grid(1)
    dx,dy = voxelSize
    nx,ny = image.shape[0],image.shape[1]
    nx2,ny2 = nx/2,ny/2
    if i < lors.shape[0]:
        x1 = lors[i,0]/dx #转换成以dx为单位的坐标
        y1 = lors[i,1]/dx
        x2 = lors[i,2]/dx
        y2 = lors[i,3]/dx
        if (x1-x2)**2+(y1-y2)**2 >= 10:    
            if abs(x2 - x1) > abs(y2 - y1):
                ky = (y2 - y1)/(x2 - x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky >= 0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = int(math.floor(yy1))
                    cy2 = int(math.floor(yy2))

                    if cy1 == cy2:
                        if 0 <= cy1 < ny:
                            iy = cy1
                            weight = (1 + ky*ky) ** 0.5
                            proj[i] += image[ix, iy] * weight
                    else:
                        if -1 <= cy1 < ny:
                            if cy1 >= 0:
                                iy = cy1
                                weight = ((1 + ky * ky) ** 0.5) * ((cy2 - yy1)/(yy2 - yy1))
                                proj[i] += image[ix, iy] * weight
                            if cy2 < ny:
                                iy = cy2
                                weight = ((1 + ky * ky) ** 0.5) * ((yy2 - cy2)/(yy2 - yy1))
                                proj[i] += image[ix, iy] * weight

            else:
                kx = (x2 - x1)/(y2 - y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx >= 0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = int(math.floor(xx1))
                    cx2 = int(math.floor(xx2))

                    if cx1 == cx2:
                        if 0 <= cx1 < nx:
                            ix = cx1
                            weight = (1 + kx * kx) ** 0.5
                            proj[i] += image[ix, iy] * weight
                    else:
                        if -1 <= cx1 < nx:
                            if cx1 >= 0:
                                ix = cx1
                                weight = ((1 + kx * kx) ** 0.5) * ((cx2 - xx1)/(xx2 - xx1))
                                proj[i] += image[ix, iy] * weight
                            if cx2 < nx:
                                ix = cx2
                                weight = ((1 + kx * kx) ** 0.5) * ((xx2 - cx2)/(xx2 - xx1))
                                proj[i] += image[ix, iy] * weight
 