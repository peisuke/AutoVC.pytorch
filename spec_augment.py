import random
import numpy as np
from scipy import interpolate
from scipy.spatial.distance import pdist, cdist, squareform

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K+3, K+3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K+1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1 # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T

def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K+3))
    pLift[:,0] = 1
    pLift[:,1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:,3:] = R
    return pLift

def spec_augment(spec, pitch, W=32, T=24, F=16):
    # Nframe : number of spectrum frame
    Nframe = spec.shape[1]

    # Nbin : number of spectrum freq bin
    Nbin = spec.shape[0]

    # check input length
    if Nframe < W*2+1:
        W = int(Nframe/4)

    if Nframe < T*2+1:
        T = int(Nframe/mt)

    if Nbin < F*2+1:
        F = int(Nbin/mf)

    w = random.randint(-W,W)
    center = random.randint(W,Nframe-W)

    src = np.asarray([[ center,  1], [ center,  0], [ center,  2], [0, 0], [0, 1], [0, 2], [Nframe-1, 0], [Nframe-1, 1], [Nframe-1, 2]])
    dst = np.asarray([[ center+w,  1], [ center+w,  0], [ center+w,  2], [0, 0], [0, 1], [0, 2], [Nframe-1, 0], [Nframe-1, 1], [Nframe-1, 2]])
    src = src.astype(np.float)
    dst = dst.astype(np.float)

    # source control points
    xs, ys = src[:,0], src[:,1]
    cps = np.vstack([xs, ys]).T

    # target control points
    xt, yt = dst[:,0],dst[:,1]

    # construct TT
    TT = makeT(cps)

    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = np.linalg.solve(TT, xtAug) # [K+3]
    cy = np.linalg.solve(TT, ytAug)

    # dense grid
    x = np.linspace(0, Nframe-1,Nframe)
    y = np.linspace(1,1,1)
    x, y = np.meshgrid(x, y)

    xgs, ygs = x.flatten(), y.flatten()

    gps = np.vstack([xgs, ygs]).T

    # transform
    pgLift = liftPts(gps, cps) # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)     
    spec_warped = np.zeros_like(spec)
    pitch_warped = np.zeros_like(pitch)
    
    for f_ind in range(Nbin):
        spec_tmp = spec[f_ind,:]
        func = interpolate.interp1d(xgt, spec_tmp, fill_value="extrapolate")
        xnew = np.linspace(0, Nframe-1,Nframe)
        spec_warped[f_ind,:] = func(xnew)
        
    func = interpolate.interp1d(xgt, pitch, fill_value="extrapolate")
    xnew = np.linspace(0, Nframe-1,Nframe)
    pitch_warped = func(xnew)
    
    return spec_warped, pitch_warped