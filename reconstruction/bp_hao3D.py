import numpy as np
from numba import jit, njit, prange, cuda
import math

@njit
def bprojection3D(lors, proj, voxelSize, image):
    dx,dy,dz = voxelSize
    dz = dz/dx
    nx,ny,nz = image.shape[0],image.shape[1],image.shape[2]
    nx2,ny2,nz2 = nx/2,ny/2,nz/2
    for i in range(lors.shape[0]):
        x1,y1,z1,x2,y2,z2 = lors[i]/dx #转换成以dx为单位的坐标
        if (x1-x2)**2+(y1-y2)**2>=10:  #排除一些不可能的lor
            if abs(x2-x1)>abs(y2-y1) and abs(x2-x1)>abs(z2-z1):
                ky = (y2-y1)/(x2-x1)
                kz = (z2-z1)/(x2-x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky>=0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = math.floor(yy1)
                    cy2 = math.floor(yy2)

                    if kz>=0:
                        zz1 = (z1 + kz * (xx1 - x1)) / dz + nz2
                        zz2 = (z1 + kz * (xx2 - x1)) / dz + nz2
                    else:
                        zz1 = (z1 + kz * (xx2 - x1)) / dz + nz2
                        zz2 = (z1 + kz * (xx1 - x1)) / dz + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)

                    if cy2 == cy1:
                        if 0 <= cy1 < ny:
                            if cz2 == cz1:  #figure 4(b) 
                                if 0 <= cz1 < nz: 
                                    iy = cy1
                                    iz = cz1
                                    weight = (1 + ky * ky + kz * kz) ** 0.5
                                    image[ix, iy, iz] += proj[i] * weight
                            else:           #figure 4(c)
                                if -1 <= cz1 < nz:
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if cz1 >= 0:
                                        iy = cy1
                                        iz = cz1
                                        weight = rz * (1 + ky * ky + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                                    if cz2 < nz:
                                        iy = cy1
                                        iz = cz2
                                        weight = (1 - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                    else:
                        if -1 <= cy1 < ny:
                            if cz1 == cz2:  #figure 4(d)
                                if 0 <= cz1 < nz:
                                    ry = (cy2 - yy1)/(yy2 - yy1)
                                    if cy1 >= 0:
                                        iy = cy1
                                        iz = cz1
                                        weight = ry * (1 + ky * ky + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                                    if cy2 < ny:
                                        iy = cy2
                                        iz = cz1
                                        weight = (1 - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                            else:            #figure 4(e)或4(f)
                                if -1 <= cz1 < nz:
                                    ry = (cy2 - yy1)/(yy2 - yy1)
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if ry > rz:  #figure 4(e)
                                        if cy1 >= 0 and cz1 >= 0:
                                            iy = cy1
                                            iz = cz1
                                            weight = rz * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cy1 >= 0 and cz2 < nz:
                                            iy = cy1
                                            iz = cz2
                                            weight = (ry - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cy2 < ny and cz2 < nz:
                                            iy = cy2
                                            iz = cz2
                                            weight = (1 - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                    else:   #figure 4(f)
                                        if cy1 >= 0 and cz1 >= 0:
                                            iy = cy1
                                            iz = cz1
                                            weight = ry * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cy2 < ny and cz1 >= 0:
                                            iy = cy2
                                            iz = cz1
                                            weight = (rz - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cy2 < ny and cz2 < nz:
                                            iy = cy2
                                            iz = cz2
                                            weight = (1 - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight

            elif abs(y2-y1)>=abs(x2-x1) and abs(y2-y1)>abs(z2-z1):
                kx = (x2-x1)/(y2-y1)
                kz = (z2-z1)/(y2-y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx>=0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = math.floor(xx1)
                    cx2 = math.floor(xx2)

                    if kz>=0:
                        zz1 = (z1 + kz * (yy1 - y1)) / dz + nz2
                        zz2 = (z1 + kz * (yy2 - y1)) / dz + nz2
                    else:
                        zz1 = (z1 + kz * (yy2 - y1)) / dz + nz2
                        zz2 = (z1 + kz * (yy1 - y1)) / dz + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)

                    if cx2 == cx1:
                        if 0 <= cx1 < nx:
                            if cz2 == cz1:  #figure 4(b) 
                                if 0 <= cz1 < nz: 
                                    ix = cx1
                                    iz = cz1
                                    weight = (1 + kx * kx + kz * kz) ** 0.5
                                    image[ix, iy, iz] += proj[i] * weight
                            else:           #figure 4(c)
                                if -1 <= cz1 < nz:
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if cz1 >= 0:
                                        ix = cx1
                                        iz = cz1
                                        weight = rz * (1 + kx * kx + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                                    if cz2 < nz:
                                        ix = cx1
                                        iz = cz2
                                        weight = (1 - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                    else:
                        if -1 <= cx1 < nx:
                            if cz1 == cz2:  #figure 4(d)
                                if 0 <= cz1 < nz:
                                    rx = (cx2 - xx1)/(xx2 - xx1)
                                    if cx1 >= 0:
                                        ix = cx1
                                        iz = cz1
                                        weight = rx * (1 + kx * kx + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                                    if cx2 < nx:
                                        ix = cx2
                                        iz = cz1
                                        weight = (1 - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                        image[ix, iy, iz] += proj[i] * weight
                            else:            #figure 4(e)或4(f)
                                if -1 <= cz1 < nz:
                                    rx = (cx2 - xx1)/(xx2 - xx1)
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if rx > rz:  #figure 4(e)
                                        if cx1 >= 0 and cz1 >= 0:
                                            ix = cx1
                                            iz = cz1
                                            weight = rz * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cx1 >= 0 and cz2 < nz:
                                            ix = cx1
                                            iz = cz2
                                            weight = (rx - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cx2 < nx and cz2 < nz:
                                            ix = cx2
                                            iz = cz2
                                            weight = (1 - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                    else:   #figure 4(f)
                                        if cx1 >= 0 and cz1 >= 0:
                                            ix = cx1
                                            iz = cz1
                                            weight = rx * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cx2 < nx and cz1 >= 0:
                                            ix = cx2
                                            iz = cz1
                                            weight = (rz - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight
                                        if cx2 < nx and cz2 < nz:
                                            ix = cx2
                                            iz = cz2
                                            weight = (1 - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                            image[ix, iy, iz] += proj[i] * weight

@cuda.jit
def bprojection3D_cuda(lors, proj, voxelSize, image):
    i = cuda.grid(1)
    dx,dy,dz = voxelSize
    dz = dz/dx
    nx,ny,nz = image.shape[0],image.shape[1],image.shape[2]
    nx2,ny2,nz2 = nx/2,ny/2,nz/2
    if i < lors.shape[0]:
        x1 = lors[i,0]/dx #转换成以dx为单位的坐标
        y1 = lors[i,1]/dx
        z1 = lors[i,2]/dx
        x2 = lors[i,3]/dx
        y2 = lors[i,4]/dx
        z2 = lors[i,5]/dx
        if (x1-x2)**2+(y1-y2)**2>=10:  #排除一些不可能的lor
            if abs(x2-x1)>abs(y2-y1) and abs(x2-x1)>abs(z2-z1):
                ky = (y2-y1)/(x2-x1)
                kz = (z2-z1)/(x2-x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky>=0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = int(math.floor(yy1))
                    cy2 = int(math.floor(yy2))

                    if kz>=0:
                        zz1 = (z1 + kz * (xx1 - x1)) / dz + nz2
                        zz2 = (z1 + kz * (xx2 - x1)) / dz + nz2
                    else:
                        zz1 = (z1 + kz * (xx2 - x1)) / dz + nz2
                        zz2 = (z1 + kz * (xx1 - x1)) / dz + nz2
                    cz1 = int(math.floor(zz1))
                    cz2 = int(math.floor(zz2))

                    if cy2 == cy1:
                        if 0 <= cy1 < ny:
                            if cz2 == cz1:  #figure 4(b) 
                                if 0 <= cz1 < nz: 
                                    iy = cy1
                                    iz = cz1
                                    weight = (1 + ky * ky + kz * kz) ** 0.5
                                    cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                            else:           #figure 4(c)
                                if -1 <= cz1 < nz:
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if cz1 >= 0:
                                        iy = cy1
                                        iz = cz1
                                        weight = rz * (1 + ky * ky + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    if cz2 < nz:
                                        iy = cy1
                                        iz = cz2
                                        weight = (1 - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                    else:
                        if -1 <= cy1 < ny:
                            if cz1 == cz2:  #figure 4(d)
                                if 0 <= cz1 < nz:
                                    ry = (cy2 - yy1)/(yy2 - yy1)
                                    if cy1 >= 0:
                                        iy = cy1
                                        iz = cz1
                                        weight = ry * (1 + ky * ky + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    if cy2 < ny:
                                        iy = cy2
                                        iz = cz1
                                        weight = (1 - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                            else:            #figure 4(e)或4(f)
                                if -1 <= cz1 < nz:
                                    ry = (cy2 - yy1)/(yy2 - yy1)
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if ry > rz:  #figure 4(e)
                                        if cy1 >= 0 and cz1 >= 0:
                                            iy = cy1
                                            iz = cz1
                                            weight = rz * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cy1 >= 0 and cz2 < nz:
                                            iy = cy1
                                            iz = cz2
                                            weight = (ry - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cy2 < ny and cz2 < nz:
                                            iy = cy2
                                            iz = cz2
                                            weight = (1 - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    else:   #figure 4(f)
                                        if cy1 >= 0 and cz1 >= 0:
                                            iy = cy1
                                            iz = cz1
                                            weight = ry * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cy2 < ny and cz1 >= 0:
                                            iy = cy2
                                            iz = cz1
                                            weight = (rz - ry) * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cy2 < ny and cz2 < nz:
                                            iy = cy2
                                            iz = cz2
                                            weight = (1 - rz) * (1 + ky * ky + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)

            elif abs(y2-y1)>=abs(x2-x1) and abs(y2-y1)>abs(z2-z1):
                kx = (x2-x1)/(y2-y1)
                kz = (z2-z1)/(y2-y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx>=0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = int(math.floor(xx1))
                    cx2 = int(math.floor(xx2))

                    if kz>=0:
                        zz1 = (z1 + kz * (yy1 - y1)) / dz + nz2
                        zz2 = (z1 + kz * (yy2 - y1)) / dz + nz2
                    else:
                        zz1 = (z1 + kz * (yy2 - y1)) / dz + nz2
                        zz2 = (z1 + kz * (yy1 - y1)) / dz + nz2
                    cz1 = int(math.floor(zz1))
                    cz2 = int(math.floor(zz2))

                    if cx2 == cx1:
                        if 0 <= cx1 < nx:
                            if cz2 == cz1:  #figure 4(b) 
                                if 0 <= cz1 < nz: 
                                    ix = cx1
                                    iz = cz1
                                    weight = (1 + kx * kx + kz * kz) ** 0.5
                                    cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                            else:           #figure 4(c)
                                if -1 <= cz1 < nz:
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if cz1 >= 0:
                                        ix = cx1
                                        iz = cz1
                                        weight = rz * (1 + kx * kx + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    if cz2 < nz:
                                        ix = cx1
                                        iz = cz2
                                        weight = (1 - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                    else:
                        if -1 <= cx1 < nx:
                            if cz1 == cz2:  #figure 4(d)
                                if 0 <= cz1 < nz:
                                    rx = (cx2 - xx1)/(xx2 - xx1)
                                    if cx1 >= 0:
                                        ix = cx1
                                        iz = cz1
                                        weight = rx * (1 + kx * kx + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    if cx2 < nx:
                                        ix = cx2
                                        iz = cz1
                                        weight = (1 - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                        cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                            else:            #figure 4(e)或4(f)
                                if -1 <= cz1 < nz:
                                    rx = (cx2 - xx1)/(xx2 - xx1)
                                    rz = (cz2 - zz1)/(zz2 - zz1)
                                    if rx > rz:  #figure 4(e)
                                        if cx1 >= 0 and cz1 >= 0:
                                            ix = cx1
                                            iz = cz1
                                            weight = rz * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cx1 >= 0 and cz2 < nz:
                                            ix = cx1
                                            iz = cz2
                                            weight = (rx - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cx2 < nx and cz2 < nz:
                                            ix = cx2
                                            iz = cz2
                                            weight = (1 - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                    else:   #figure 4(f)
                                        if cx1 >= 0 and cz1 >= 0:
                                            ix = cx1
                                            iz = cz1
                                            weight = rx * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cx2 < nx and cz1 >= 0:
                                            ix = cx2
                                            iz = cz1
                                            weight = (rz - rx) * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)
                                        if cx2 < nx and cz2 < nz:
                                            ix = cx2
                                            iz = cz2
                                            weight = (1 - rz) * (1 + kx * kx + kz * kz) ** 0.5
                                            cuda.atomic.add(image,(ix,iy,iz),proj[i] * weight)