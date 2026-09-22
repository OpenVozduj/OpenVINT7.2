import numpy as np
from cv2 import split

def invScalerX(Xs, xmin, xmax):
    X = Xs*(xmax-xmin)+xmin
    return X

def searchCoeffswithAlphaRe(alpha, Re, graphs):    
    alpha1 = -4.0 
    pxa1 = 0
    alpha2 = 10.0
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    if px_alpha >= 256:
        px_alpha = 255
    
    Re1 = 500000 
    pxr1 = 0
    Re2 = 10000
    pxr2 = 255 
    px_Re = (Re-Re1)*(pxr2-pxr1)/(Re2-Re1)+pxr1
    px_Re = round(px_Re)
    if px_Re >= 256:
        px_Re = 255
    if px_Re < 0:
        px_Re = 0
    
    CY, CX = split(graphs)
    cyn = CY[px_Re, px_alpha]
    cxn = CX[px_Re, px_alpha]
    
    cy = invScalerX(cyn, xmin=-0.7, xmax=1.5)
    cx = invScalerX(cxn, xmin=0.001, xmax=0.22)
    
    return cy, cx
