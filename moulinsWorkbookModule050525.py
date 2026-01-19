# -*- coding: utf-8 -*-
##############################################################################
#### Import statements #######################################################
##############################################################################

import os
from sys import exit
import time
import numpy as np
import shapely.geometry as shpg
from pyproj import Transformer
from math import ceil, hypot
from PIL import Image
from shutil import copy
import rasterio

import warnings
warnings.filterwarnings('ignore')

##############################################################################
#### Functions ###############################################################
##############################################################################

def path_maker(file_path):
# A function to check for the specified file and create it if it doesn't exist.
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def extract_num(s, inst=False):
# Returns numbers from a string.
    if inst:
        num = ''.join([a for a in s if a.isnumeric()])
        code = s.split(num)[0]
        return num, code
    return ''.join([a for a in s if a.isnumeric()])

def time_stamp(string):
# Prints the input string with a timestamp.
    print(string + ' at ' + time.strftime('%H:%M:%S', time.localtime()))

def mmddyy2dec(s):
#Converts MMDDYY format dates to decimal format. Input=string, output=float.
    daysMonth = [0,31,59,90,120,151,181,212,243,273,304,334]  #cum. days/month
    y, d, m = int(s[-2:]), int(s[-4:-2]), int(s[:-4])
    y = 2000 + y if y < 50 else 1900 + y           #Code will fail after 2050!
    if y % 4 == 0:                                     #Account for leap years
        daysYear = 366
        daysMonth[2:] = [x+1 for x in daysMonth[2:]]
    else:
        daysYear = 365
    return y +round((d + daysMonth[m - 1]) / daysYear, 4)

def s2ff(inVal, n=3, w=0, s='f'):
# Accepts a floating point number or a numeric string and returns it with
# specified formatting. n is the number of decimal places to include, w is the
# minimum width for the string to fill (empty space will be added to the left
# if needed to reach this number), and s is the output style (use 'e' for
# scientific notation).
    num = float(inVal)
    form = '{:' + str(w) + '.' + str(n) + s + '}'
    return str(form.format(num))

def utc2hhmmss(s, mode):
#Convert time in UTC seconds of day to HHMMSS time. Input and output strings.
#Different modes for ATM, ICESat, and ICESat2.
    if mode == 'ATM':
        t = float(s)
        h = str(int(t/3600))
        m = '{:0>2}'.format(str(int(t%3600/60)))
        s = '{:0>2}'.format(str(int(t%3600%60)))
    elif mode == 'ICE':
        t = float(s)
        tm = list(time.gmtime(946684800+t))
        h = str(tm[3])
        m ='{:0>2}'.format(str(tm[4]))
        s ='{:0>2}'.format(str(tm[5]))
    elif mode == 'IC2':
        stamp = time.gmtime(float(s))
        h = str(stamp[3])
        m = '{:0>2}'.format(str(stamp[4]))
        s = '{:0>2}'.format(str(stamp[5]))
    return h + m + s

def tif_reader(path, border=False, raw=False, gt2=[], shape2=(0,)):
# Reads a GeoTIFF and converts into a numpy array. Output is the array of DEM
# values a shapely polygon of the convex hull of non-nodata pixels, the nodata
# value, the geotransform (an object with the corner coords and pixel size),
# and the map projection. If "border" is False, the convex hull is not
# calculated. If raw is False the array is a list of X,Y,Z point, and if it is
# True, it is a grid of DEM values. gt2 and shape2 are the geotransform and
# array dimension for a second array; if given values, they will crop the
# input DEM to match.
    with rasterio.open(path) as f:
        gt, proj, ndv = f.transform, f.crs, f.nodata #transform, projection, nodata
        data, outVector = np.array(f.read(1)), []          #read raster values
    
    X = np.arange(gt[2], gt[2] + gt[0] * (len(data[0])), gt[0]) #get list of all X coords
    Y = np.arange(gt[5] + gt[4], gt[5] + gt[4] * (len(data)+1), gt[4]) #get list of all Y coords
    
    for j in reversed(range(len(Y))):              #iterate through the raster
        for i in range(len(X)):   #turn data raster into array of X,Y,Z points
            outVector.append([X[i], Y[j], data[j,i]])
    outVector = np.array(outVector, dtype=np.float32) #convert the matrix into a numpy array
    
    if border:                             #Get non-nodata values in the array
        cloud = shpg.MultiPoint(outVector[:,:2][np.where(outVector[:,2]>ndv)])
        polygon = cloud.convex_hull               #Get the shapely convex hull
    else:
        polygon = [] #Or generate a blank value to save a fraction of a second
    
    if gt2 != []:
        from rasterio.warp import reproject, Resampling
        temp = np.ones(shape2, dtype=np.float32)   #Dummy array for new values
        reproject(data, temp, src_transform=gt, src_crs=proj, src_nodata=ndv,
                  dst_transform=gt2, dst_crs=proj, dst_nodata=ndv,
                  resampling=Resampling.bilinear)
        data = temp                                    #Resample the input DEM
    
    if raw:
        outVector = data
    
    outVector[outVector == ndv] = np.nan
    return outVector, polygon, ndv, gt, proj

def exp_combine(shapeList, buffer=0, line=False):
# Turns .exp file coordinates into a Shapely geometric object. Giving a value 
# for 'buffer' will enlarge the polygons in all directions by the specified
# amount. 'line' should be true for reading linear shapes.
    import shapely
    for x in shapeList:
        coords = []             #Generate a polygon from each list of vertices
        with open(x, 'r') as f:
            content = f.readlines()
        for y in content[1:]:
            coords.append((float(y.split()[0]), float(y.split()[1])))
        if line:
            temp = shpg.LineString(coords)
        else:
            temp = shpg.Polygon(coords).buffer(buffer)     #Add a small buffer
        try:                #Merge all polygons into one polygon or collection
            polygon = shapely.union(polygon, temp)
        except:             #If this is the first polygon, the union will fail
            polygon = temp
    return polygon

def getBins(cent, binS, patchS, xmin, ymin):
#For a given point, finds all bins that a surface patch centered on that point
#would contain. "cent" = point coordinates x,y; "binS" = bin size; "patchS" =
#surface patch size; "xmin/ymin" = minimum coords of DEM bounding box.
    cx, cy = (int((cent[0]-xmin)/binS), int((cent[1]-ymin)/binS)) #bin indices
    bxs = int(cent[0]-(cent[0]-xmin)%binS)             #min coords of that bin
    bys = int(cent[1]-(cent[1]-ymin)%binS)
    bxf, byf = bxs + binS, bys + binS                  #max coords of that bin
    xs, ys = cent[0] - patchS/2, cent[1] - patchS/2              #patch bounds
    xf, yf = xs + patchS, ys + patchS
         #Get the number of bins required forward and back for both dimensions
    xi, xn = max(ceil((bxs-xs)/binS),0), max(ceil((xf-bxf)/binS),0)
    yj, yn = max(ceil((bys-ys)/binS),0), max(ceil((yf-byf)/binS),0)
    ind = []
    for i in range(-xi, xn+1):  #Iterate through everything in those intervals
        for j in range(-yj, yn+1):
            ind.append((cx+i, cy+j))
    return ind

def square_domain(xmin, xmax, ymin, ymax):
# For the given minumum and maximum x and y values, returns new values such 
#that the range of both dimensions will be equal to whichever of the two is 
#larger. (So surfaces can be plotted without distortion.)
    xrng, yrng = xmax - xmin, ymax - ymin
    if xrng > yrng:
        ypad = (xrng - yrng) / 2                   #Amount of "padding" to add
        ymin, ymax = ymin - ypad, ymax + ypad
    elif yrng > xrng:
        xpad = (yrng - xrng) / 2
        xmin, xmax = xmin - xpad, xmax + xpad
    return xmin, xmax, ymin, ymax

def mat_2_pil(fig):
#Turns a MATLAB figure into an exportable image.
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    temp = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    temp.shape=(w, h, 3)
    temp = np.roll(temp, 3, axis=2)
    return Image.frombytes("RGB", (w,h), temp.tostring())

def poly_surface(data, order=3, time=False):
# This function fits a polynomial surface of the specified order (default=3) to
#   the input dataset. (i.e., z = h + a1x + a2y + a3x^2 +a4y^2 + a5xy + ...)
# data: 3 x m array of [x,y,z] with 1 row/observation. x,y,z are relative to
#   the centroid, in meters; if time=True, there must be a 4th column of
#   times relative to a given reference date in decimal years, and h has
#   multiple time-dependent terms.
# Since x,y,z,t are known, this is a system of linear equations where the
#   polynomial shape parameters and origin heights are unknown.
# Solve the equation Ax=B by minimizing the sum of squares of B-Ax.
# A: design/input matrix (i.e. rows as [1 x y x^2 y^2 xy ...], n x m)
# x: vector of unknowns (length n = #shape parameters + #times)
# B: observation vector (elevations, length m)
# A*x=B -> A.T*A*x=A.T*B -> N*x = b -> x = N(inv)*b

    if time:               #Create multiple A columns corresponding to heights
        ordict = {3:9, 2:5, 1:3, 0:0}        #Number of shape params per order
        num = ordict[order]
        ind = np.unique(data[:,3], return_index=True)[1]
        unqT = [data[:,3][i] for i in sorted(ind)]       #Get all unique dates
        unqT = np.array(unqT)
        for j in unqT:
            temp = np.where(data[:,3] == j, 1, 0)
            try:
                A0 = np.c_[A0, temp]      #A0 is time dependent, with multiple 
            except:           #columns corresponding to each date that equal 1
                A0 = temp   #for data from that date, and zero everywhere else
    else:                                 #Create just one A column for height
        A0 = np.ones((data.shape[0], 1))   #A0 is constant, a column of all 1s

    if order > 0:
        A = np.c_[A0, data[:,:2]]                           #A1 and A2 columns
    if order > 1:
        A = np.c_[A, data[:,:2]**2, np.prod(data[:,:2], axis=1)]   #A3, A4, A5
    if order > 2:
        temp1 = np.prod(np.c_[data[:,0]**2, data[:,1]], axis=1)
        temp2 = np.prod(np.c_[data[:,0], data[:,1]**2], axis=1)
        A = np.c_[A, data[:,:2]**3, temp1, temp2]  #A6, A7, A8, and A9 columns
    x, res, sig, errs, kap = least_sq(A, data[:,2])

    if time:
        H1 = x[:-num]
        try:
            iB, iA = np.where(unqT < 0)[0][-1], np.where(unqT > 0)[0][0]
            tB, tA, hB, hA = unqT[iB], unqT[iA], H1[iB], H1[iA]
            z0 = (tA * hB - tB * hA) / (tA - tB)         #Interpolate h at t=0
        except:
            z0 = H1[0]
        H =  H1 - z0             #height with respect to reference date height
        return x, H, res, unqT, sig, errs[:-num], kap
    else:
        return x, res, sig, errs, kap

def least_sq(A, B):
#A least squares fitting algorithm. A is a 2D matrix of linear equation
#x and y values, and B is a list of the corresponding desired solutions.
#Outputs: 'x' is the matrix such that A*x=B. 'res' is the residue for each row.
#'sig' is the standard deviation (degrees of freedom = length of x). 
#'kap' is kappa: the ratio of max to min eigenvalue.)
    N = np.dot(A.T, A)                      #A.T*A = N, an n x n square matrix
    Ninv = np.linalg.inv(N)
    b = np.dot(A.T, B)                        #A.T*B = b, a vector of length n
    x = np.dot(Ninv, b)                    #N(inv)*b = x, a vector of length n
    norms, errs = [], []           #Errors can be estimated from the diagonals
    for i in range(len(x)):
        norms.append(N[i,i]**0.5)
        errs.append(Ninv[i,i]**0.5)
    for i in range(len(x)):
        for j in range(len(x)):
            N[i,j] = N[i,j] / norms[i] / norms[j]
    res = B - np.dot(A,x)                                     #residual errors
    sig = np.std(res,ddof=len(x))
    errs = sig * np.array(errs)
    eVec = np.abs(np.linalg.eigvals(N))                  #Get condition number
    kap = max(eVec) / min(eVec)                   #ratio of max/min eigenvalue
    return x, res, sig, errs, kap

def prog_rep(num1, num2, string):
# Prints a percenage completion alert.
    time_stamp('{0:.0f}% of {1} '.format(ceil(100*num1/num2), string))

def time_series(path):
# Reads data from a the time series file created by this software.
    ts, cp = {}, {}
    with open(path) as f:
        content = f.readlines()
    for x in content[3:]:
        y = x.split()
        if len(y) == 8:
            ind = int(y[0])
            ts[ind], cp[ind] = [], [float(y[2]), float(y[3])]
        elif len(y) == 10:
            ts[ind].append([float(y[1]), float(y[4]), int(y[2]), int(y[9])])
    for x in ts:
        ts[x] = np.array(ts[x])
    return ts, cp

def rms(inArray):
# Returns the root mean square of an array.
    return np.sqrt(np.mean(inArray**2))

def poly_value(x, y, c):
# Calculates z values of a polynomial surface for a series and x and y points, where x and y are arrays of x and y values (with matching
# indices, and c is an array of polynomial coefficients, as outlines in the header for poly_surface, above. The return value is an array
# of z values of the same length as x and y.
    ordict = {10:3, 6:2, 3:1, 1:0}
    order = ordict[len(c)]   #infer order from number of coefficients
    z = np.full(x.shape,c[0])
    if order > 0:
        z = z + c[1] * x + c[2] * y
    if order > 1:
        z = z + c[3] * x**2 + c[4] * y**2 + c[5] * x * y
    if order > 2:
        z = z + c[6] * x**3 + c[7] * y**3 + c[8] * x**2 * y + c[9] * x * y**2
    return z

def photo_shoot(fig, ax, viewList=np.array([[90,270],[25,-45],[25,0],[25,45],
                                   [25,90],[25,135],[25,225],[0,270],[0,0]])):
# Returns a list of images of a 3d matlab plot with a variety of view angles.
    imageList = []
    for x in viewList:
        ax.view_init(elev = x[0], azim = x[1])
        imageList.append(mat_2_pil(fig))
    return imageList

def linear_sort(array, x_col, y_col):
    from scipy.spatial import distance
    xsum, ysum, dist = sum(array[:,x_col]), sum(array[:,y_col]), len(array)
    xc, yc =  xsum / dist, ysum / dist
    ifar = distance.cdist([(xc, yc)], np.c_[array[:,x_col], array[:,y_col]]).argmax()
    xi, yi, out_array = array[ifar,x_col], array[ifar,y_col], [array[ifar]]
    array = np.delete(array, ifar, axis=0)
    for i in range(dist-1):
        inear = distance.cdist([(xi, yi)], np.c_[array[:,x_col], array[:,y_col]]).argmin()
        xi, yi = array[inear,x_col], array[inear,y_col]
        out_array.append(array[inear])
        array = np.delete(array, inear, axis=0)
    return np.array(out_array)

def order_by_date(in_list):
# A function to put DEM dates in order when they are in SERAC format. Input
# and output are lists of strings.
    inter_list = []
    out_list = []
    for x in in_list:
        inter_list.append(mmddyy2dec(x))
    inter_list.sort()
    for x in inter_list:
        if x[4] == '0':
            out_list.append(x[5] + x[6:8] + x[2:4] + x[8:])
        else:
            out_list.append(x[4:6] + x[6:8] + x[2:4] + x[8:0])

    return out_list

##############################################################################
#### ALPS Packages - by P. Shekhar ###########################################
##############################################################################

def quantile_mine(Data, q, k):
# Author: P Shekhar
    n1 = Data.shape[0]
    fac = (q/k) * n1
    if fac%1 != 0:
        val = Data[round(fac)-1,0]
    else:
        val = Data[round(fac)-1,0] if fac == n1 else (Data[round(fac)-1,0]+\
                                                      Data[round(fac),0]) / 2
    return val

def Kno_pspline_opt(Data, p, n):
    U, count = np.zeros([n+2*p+1,]), p+1
    U[p], U[n+p] = Data[0,0], Data[-1,0]
    dist = (float(Data[-1,0])-float(Data[0,0])) / n
    for d in range(n+p):
        U[count] = U[count-1] + dist
        count = count+1
    count = p+1
    for d in range(n):
        U[count] = quantile_mine(Data, d+1, n)
        count = count+1
    count = p-1
    for d in range(p):
        U[count] = U[count+1] - dist
        count = count-1
    return U

def Bspline_Basis(p, i, u, U):
    m = len(U)-1
    n = m - p-1
    if (u == U[0] and i == 0) or (u == U[m] and i == n): return 1
    if (u == U[0] and i != 0) or (u == U[m] and i != n): return 0
    if (u < U[i] or u >= U[i+p+1]): return 0
    for d in range(i, i+p+1):
        if u >= U[d] and u < U[d+1]: interval = d
    N1 = np.zeros([p+1, len(U)-1])
    N1[0,interval], reduce_len = 1, i + p
    for deg in range(1, p+1):
        for i1 in range(i, reduce_len):
            if N1[deg-1, i1] != 0 and N1[deg-1, i1+1] != 0:
                N1[deg,i1] = ((u-U[i1]) / (U[i1+deg]-U[i1])) * N1[deg-1,i1] +\
                    ((U[i1+deg+1]-u) / (U[i1+deg+1]-U[i1+1])) * N1[deg-1, i1+1]
            if N1[deg-1, i1] != 0 and N1[deg-1, i1+1] == 0:
                N1[deg,i1] = ((u-U[i1]) / (U[i1+deg]-U[i1])) * N1[deg-1, i1]
            if N1[deg-1,i1] == 0 and N1[deg-1, i1+1] != 0:
                N1[deg, i1] = ((U[i1+deg+1]-u) / (U[i1+deg+1]-U[i1+1])) *\
                    N1[deg-1, i1+1]
        reduce_len = reduce_len-1
    return N1[p, i]

def Basis_Pspline(n, p, U, loc):
    num = len(loc)
    B, c1 = np.zeros([num, n+p]), 0
    for i in range(n+p):
        c2 = 0
        for u in loc:
            B[c2,c1] = Bspline_Basis(p, i, u, U)
            c2 = c2+1
        c1 = c1+1
    return B

def Penalty_p(q,c):
    if q == 1:
        D = np.zeros([c-1,c])
        for i in range(c-1):
            D[i,i], D[i,i+1] = -1, 1
    if q == 2:
        D = np.zeros([c-2,c])
        for i in range(c-2):
            D[i,i], D[i,i+1], D[i,i+2] = 1, -2, 1
    if q == 3:
        D = np.zeros([c-3,c])
        for i in range(c-3):
            D[i,i], D[i,i+1] = -1, 3
            D[i,i+2], D[i,i+3] = -3, 1
    if q == 4:
        D = np.zeros([c-4,c])
        for i in range(c-4):
            D[i,i], D[i,i+1], D[i,i+2] = 1, -4, 4
            D[i,i+3], D[i,i+3] = -4, 1
    P = D.T.dot(D)
    return P

def Smoothing_cost(lamb, Data, B, q, c, choice):
    P = lamb * Penalty_p(q,c)
    H = B.dot(np.linalg.inv(B.T.dot(B)+P)).dot(B.T)
    y_cap, n, t = H.dot(Data[:,1].reshape(-1,1)), Data.shape[0], 0
    if choice == 1:
        for i in range(n):
            t = t + ((Data[i,1] - y_cap[i])/(1-H[i,i]))**2
        obj = t
    elif choice == 2:
        d = sum(np.diag(H))/n
        for i in range(n):
            t = t + ((Data[i,1] - y_cap[i])/(1-d))**2
        obj = t
    elif choice == 3:
        del1, ed = 2, np.trace(H)
        for i in range(n):
            t = t+ (Data[i,1] - y_cap[i])**2
        obj = t + del1*ed
    elif choice == 4:
        del1, ed = np.log(n), np.trace(H)
        for i in range(n):
            t = t+ (Data[i,1] - y_cap[i])**2
        obj = t + del1*ed
    return obj

def Smoothing_par(Data, B, q, c, lamb, choice):
    from scipy.optimize import minimize
    args = (Data, B, q, c, choice)
    bnds, lamb = [(1.0e-2, None)], [lamb]
    lam = minimize(Smoothing_cost,lamb,args,bounds=bnds,method='SLSQP')
    return lam

def full_search_nk(Data, p, q):
    n, comp, choice = 1, 1.0e+9, 2
    while n < Data.shape[0]:
        c, lamb, U = n+p, 0.1, Kno_pspline_opt(Data, p, n)
        B = Basis_Pspline(n, p, U, Data[:,0])
        lam = Smoothing_par(Data, B, q, c, lamb, choice)
        if lam.fun < comp:
            comp, opt_n, opt_lam = lam.fun, n, lam.x[0]
        n = n+1
    c = opt_n+p
    P, U = opt_lam * Penalty_p(q,c), Kno_pspline_opt(Data, p, opt_n)
    B_dat = Basis_Pspline(opt_n, p, U, Data[:,0])
    theta = np.linalg.solve(B_dat.T.dot(B_dat) +\
                            P, B_dat.T.dot(Data[:,1].reshape(-1,1)))
    nr = (Data[:,1].reshape(-1,1) - B_dat.dot(theta)).reshape(-1,1)
    term = np.linalg.inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T.dot(B_dat))
    n = Data.shape[0]
    df_res = n - 2*np.trace(term) + np.trace(term.dot(term.T))
    sigmasq = (nr.T.dot(nr)) / (df_res)
    sigmasq = sigmasq[0][0]
    return [opt_n, opt_lam, sigmasq]

def Var_bounds(Data,B,B_dat,theta,P,lamb,conf = 0.95):
    P, n = lamb*P, Data.shape[0]
    nr = (Data[:,1].reshape(-1,1) - B_dat.dot(theta)).reshape(-1,1)
    std = np.std(nr)
    #term = inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T.dot(B_dat))
    #df_res = n - 2*np.trace(term) + np.trace(term.dot(term.T))
    #sigmasq = (nr.T.dot(nr))/(df_res)
    #sigmasq = sigmasq[0][0]
    #std = np.sqrt(np.diag(sigmasq*B.dot(inv(B_dat.T.dot(B_dat)+P)).dot(B.T)))
    #import scipy.stats
    #stdev_t = scipy.stats.t._ppf((1+conf)/2.,df_res)*std
    return std #stdev_t

##############################################################################
#### MOULINS Pre-Step 1 - File architecture ##################################
#### Generates the file architecture expected by MOULINS. ####################
#### Altimetry and DEMs in the home directory will be automatically moved. ###
##############################################################################

def pre_step1(cwd):

    folders = ['Altimetry','DEMs','Shapes','Step1','Step2','Step3',
               'Step4','Step5','Step6','Step7','Step8','Step9','Step10',
               'Step11','TideModel','ClimateModel','IceSheetBoundary',
               'BedMachine']
    for x in folders:
        path_maker(cwd + x)

    print('Please check your home directory to verify architecture creation.')

##############################################################################
#### MOULINS Pre-Step 2 - Shape file generation ##############################
#### Store DEM convex hull coordinates for rapid retrieval later. ############
#### Creates one file per DEM. ###############################################
##############################################################################

def pre_step2(cwd, prefix):
    tstart = time.time()
    path = cwd + 'Shapes\\' + prefix + 'Shapes\\'
    path_maker(path)                    #Generate shape subfolder for this run
    os.chdir(cwd + 'DEMs\\')
    temp, border = os.listdir(), []           #Get list of all DEMs to be used
    fileList = [x for x in temp if '.tif' in x]
    if len(fileList)==0:            #Exit if the DEMs folder has no .tif files
        exit('Error: Please add DEMs to the DEMs folder.')

    for x in fileList:                      #Iterate through the provided DEMs
        date = extract_num(x)
        if len(date) < 5 or len(date) > 6:
            exit('Error: Please check DEM file name specifications')
        time_stamp('Reading DEM on ' + date)
        _, temp, _, _,_ = tif_reader(x, border=True)   #Get DEM border coords
        border.append(temp)       #This may be a collection of multiple shapes
        coords = np.array(list(temp.exterior.coords))           #Get XY coords
        with open(path + prefix + '_' + date + 'shape.exp', 'w') as f:
            f.write(str(len(coords)) + ' 1\n')
            for y in coords:             #Make a .exp file with the coord list
                f.write(str(round(y[0])) + ' ' + str(round(y[1])) + '\n')
    
    polygon = exp_combine([cwd + 'IceSheetBoundary\\GRE_IceSheet_IMBIE2_v1.exp'])

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot2grid((1,1),(0,0))
    Xp, Yp = polygon.exterior.xy
    ax.plot(Xp, Yp, c='black')
    for x in border:
        Xb, Yb = x.exterior.xy
        ax.plot(Xb, Yb, c='red')
    
    time_stamp('Completed shape file generation')
    tend = time.time()
    print('Pre-Step 2 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Pre-Step 3 - Altimetry subsetting ###############################
#### Extract Icesat and ATM altimetry in the DEM convex hulls. ###############
#### Altimetry files should have the sensor name in the filename. ############
##############################################################################

def pre_step3(cwd, prefix, demProj, patchS=1000):
    tstart = time.time()

    os.chdir(cwd + 'Shapes\\' + prefix + 'Shapes\\')  #Retrieve the DEM shapes
    sList = [x for x in os.listdir() if not 'shelf' in x.lower()]
    polygon = exp_combine(sList, buffer=patchS/2)       #Combine them into one
    xmin, ymin, xmax, ymax = polygon.bounds
    transformer = Transformer.from_crs(32624, demProj)

    #Read the ice sheet boundary shapefile so points outside can be excluded.
    #from shapely import intersection
    #polygon2 = exp_combine([cwd + 'IceSheetBoundary\\GRE_IceSheet_IMBIE2_v1.exp'])
    #bounds = intersection(polygon, polygon2)
    bounds = polygon
    
    if bounds.contains(shpg.Point(43543.620,-1073495.424)):
        print('something works')

    os.chdir(cwd + 'Altimetry\\')
    data, fileList = [], os.listdir()            #Get altimetry file locations
    for x in fileList:                               #Read each altimetry file
        time_stamp('Reading ' + x)
        with open(x, 'r') as f:
            content = f.readlines()
        testStr = content[0].split(',')[0] #Check the first word of the header

        if testStr == 'DATE_MMDDYY':                                      #ATM
            n, mis = 1, 'ATM'           #n=number of header lines; mis=mission
            indList = [1,2,0,3,5,0]          #indeces of variables of interest
            useBaro = False         #Whether barometric correction is possible
        elif testStr == 'DS_UTCTime_40':                            #ICESat (1)
            n, mis = 2, 'ICE'
            indList = [-6, -5, 9, 5, 0, -7]    #[x, y, date, z, time, baroCorr]
            useBaro = True
        elif testStr == 'delta_time':                                #ICESat-2
            n, mis = 1, 'IC2'
            indList = [-6, -5, -12, 2, 1, -8]
            useBaro = True
        #elif testStr == 'thinned LVIS laser points':                     #LVIS
        #    n, mis = 2, 'LVIS'
        #    indList = [0, 1, 3, 2]
        #    useBaro = False
        else: continue                      #Skip files that aren't recognized

        for y in content[n:]:            #Split each text line into components
            if ',' in y:               #Some format have commas, others do not
                s = y.replace('\n','').split(',')
            else:
                s = y.replace('\n','').split()
            xo, yo = float(s[indList[0]]), float(s[indList[1]])      #get x, y
            xi,yi = transformer.transform(xo,yo)         #transform, if needed
            
            if xi > xmin and xi < xmax and yi > ymin and yi < ymax and \
                bounds.contains(shpg.Point(xi,yi)):     #Check if point in DEM
                try:                 #could fail if the file line had an error
                    date, elev = s[indList[2]], float(s[indList[3]])
                    if len(date) == 8:              #Icesat1 date is different
                        date = date[4:] + date[2:4]   #for some fucking reason
                    dDate = mmddyy2dec(date)
                    hhmmss = int(utc2hhmmss(s[indList[4]],mis))
                    if useBaro:
                        baroCorr = float(s[indList[5]])
                    else:
                        baroCorr = 0
                    data.append([mis,date,dDate,hhmmss,xi,yi,elev,baroCorr])
                except:           #skip the line if there was an error with it
                    continue
    data = np.array(data)
    data = data[data[:,2].argsort(kind='stable')]           #Sort data by time

    time_stamp('Writing altimetry subset')
    with open(prefix + '_alt.txt', 'w') as f:
        f.write('MIS   MMDDYY     DecDate   HHMMSS         UTM_X         ' +\
                'UTM_Y       Elev   BaroCorr\n')
        for x in data:             #Output relevant information to a text file
            outStr = '{:3}'.format(x[0]) + s2ff(x[1],0,9) + s2ff(x[2],4,12) +\
                     s2ff(x[3],0,9) + s2ff(x[4],3,14) + s2ff(x[5],3,14) +\
                     s2ff(x[6],3,11) + s2ff(x[7],3,11) + '\n'
            f.write(outStr)

    time_stamp('Completed altimetry subsetting')
    tend = time.time()
    print('Pre-step 3 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 1 - Control point generation ###############################
#### Finds optimum control points locations based on spatial and temporal ####
#### density of altimetry data. ##############################################
##############################################################################

def step1(cwd, prefix, dRad, tBuf, eBuf, gMax, patchS=1000, binS=500):
    print('Step 1: Generating control points...')
    cpPath, tstart = cwd + 'Step1\\' + prefix + 'CPs\\', time.time()
    path_maker(cpPath)

    os.chdir(cwd + 'Shapes\\' + prefix + 'Shapes\\')
    time_stamp('Reading shapefiles')
    sList = [x for x in os.listdir() if not 'shelf' in x.lower()]
    polygon = exp_combine(sList, buffer=patchS)
    xmin, ymin, xmax, ymax = polygon.bounds      #Get bounding box of all DEMs

    demDates = []                            #Get the date range of input DEMs
    for x in os.listdir(cwd + 'DEMs'):
        demDates.append(mmddyy2dec(extract_num(x)))
    dMin, dMax = min(demDates), max(demDates)

    altDic, plotDic = {}, {}            #Read altimetry and sort into the bins
    os.chdir(cwd + 'Altimetry')
    time_stamp('Reading altimetry')
    with open(prefix + '_alt.txt', 'r') as f:
        content = f.readlines()
    for x in content[1:]:
        s = x.replace('\n','').split()            #Get date and xy coordinates
        date, utmx, utmy = float(s[2]), float(s[4]), float(s[5])
        if abs(date-dMin) < dRad and abs(date-dMax) < dRad:
            ind = (int((utmx-xmin)/binS), int((utmy-ymin)/binS))    #Bin index
            try:                        #Save coordinates by date for plotting
                plotDic[int(date)].append([utmx, utmy]) #Just the integer year
            except:
                plotDic[int(date)] = [[utmx, utmy]]
            try:        #Save everything by bin for control point optimization
                altDic[ind].append([utmx, utmy, date])
            except:
                altDic[ind] = [[utmx, utmy, date]]
    for x in altDic:               #Turn each subset into a proper NumPy array
        altDic[x] = np.array(altDic[x])
    for x in plotDic:
        plotDic[x] = np.array(plotDic[x])

    data, binNum = {}, 0
    for y in altDic:                                 #Iterate through the bins
        binNum += 1                     #Get the center coordinates of the bin
        centr = (int(y[0]*binS+xmin+binS/2), int(y[1]*binS+ymin+binS/2))
        temp = altDic[y]  #Ignore data outside the temporal window of interest
        temp = temp[(temp[:,2] > dMin - dRad) & (temp[:,2] < dMax + dRad)]
        if len(temp) < 1:
            continue                    #Skip the bin if this removes all data
                    #Starting with bin centers, refine control point locations
        impr, i, oldCentr = binS, 0, (1e12,1e12)       #Iterate up to 50 times
        while impr > 1 and i < 50:  #or until improvement is less than 1 meter
                     #Get the centroid coords of all the altimetry in that bin
            centr = (int(np.mean(temp[:,0])),int(np.mean(temp[:,1])))
            binList = getBins(centr, binS, patchS, xmin, ymin)  #neighbor bins
            temp, i = [], i+1
            for z in binList:           #Pull altimetry from the neighbor bins
                if z in altDic:
                    if len(temp) == 0:
                        temp = altDic[z]
                    else:
                        temp = np.r_[temp, altDic[z]]
                      #Get data in a new patch centered on the centroid coords
            temp = temp[(temp[:,0]>(centr[0]-patchS/2)) & (temp[:,0]<(
                   centr[0]+patchS/2)) & (temp[:,1]>(centr[1]-patchS/2)) &
                   (temp[:,1]<(centr[1]+patchS/2))]
                    #Improvement is the distance between old and new centroids
            impr = hypot(centr[0]-oldCentr[0], centr[1]-oldCentr[1])
        unqD = np.sort(np.unique(temp[:,2]))       #Get unique altimetry dates
                   #Get number of unique dates before and after DEM date range
        before, after = unqD[unqD < dMin], unqD[unqD > dMax]
        during = unqD[(unqD > dMin) & (unqD < dMax)]     #And during DEM range
        preTime = 0 if len(before) == 0 else max(dMin - before)   #Time before
        postTime = 0 if len(after) == 0 else max(after - dMax)     #Time after
           #Get time gap between DEM dates and temporally nearest altim. dates
        gapBefore = 0 if preTime == 0 else dMin - max(before)
        gapAfter = 0 if postTime == 0 else min(after) - dMax
        if len(during) == 0:        #If only 1 DEM, combine gaps for total gap
            gap = gapAfter + gapBefore
        else:                           #Otherwise, combine with internal gaps
            inGap1, inGap2 = max(during - dMin), max(dMax - during)
            gap = max(gapBefore + inGap1, gapAfter + inGap2)
          #Save point if it meets minimum criteria and not too close to others
        useFlag = True               #Use points that meet criteria by default
        if preTime >= tBuf and postTime >= tBuf and len(before) >= eBuf \
                and len(after) >= eBuf and gap <= gMax and preTime + postTime > 3:
            for z in data:            #Disable if it overlaps a previous point
                if hypot(z[0]-centr[0], z[1]-centr[1]) < patchS:
                    useFlag = False
                    break
            if useFlag:                      #Save data for each control point
                data[(centr[0],centr[1],binNum)] = temp

    os.chdir(cpPath)
    import matplotlib.pyplot as plt
    images = []

    #Plot DEM outline, control points, and altimetry.
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot2grid((1,1),(0,0))
    x0, xf, y0, yf = square_domain(xmin, xmax, ymin, ymax)
    ax.set_xlim(x0, xf)
    ax.set_ylim(y0, yf)
    years = [y for y in plotDic if max(dMin-y, y-dMax) < dRad]
      #Below is an algorithm for generating evenly-spaced colors for each year
    cStep, i, images = int(255 / (len(years)/6)), 0, []
    hexes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for j in range(int(len(years)/6)+1):     #Generate list of color hex codes
        s = '{0:0<2}'.format(hex(255-(i*cStep))[2:])
        s2 = '{0:0<2}'.format(hex(i*cStep)[2:])
        hexes[0].append('#' + s +'00ff')
        hexes[1].append('#' + '00' + s2 +'ff')
        hexes[2].append('#' + '00ff' + s)
        hexes[3].append('#' + s2 + 'ff00')
        hexes[4].append('#' + 'ff' + s + '00')
        hexes[5].append('#' + 'ff00' + s2)
        i += 1
    i = 0
    clist = hexes[0]+hexes[1]+hexes[2]+hexes[3]+hexes[4]+hexes[5]
    for y in years:                 #Plot altimetry by year for the master map
        temp = plotDic[y]
        #temp = temp[(temp[:,0] >= xmin) & (temp[:,0] <= xmax) &
           #         (temp[:,1] >= ymin) & (temp[:,1] <= ymax)]
        ax.scatter(temp[:,0], temp[:,1], s=0.5, c=clist[i], label=str(y))
        i += 1
    plt.legend(markerscale=5, loc='upper right')
    plotData = list(data.keys())            #Now add control points to the map
    X, Y = [z[0] for z in plotData], [z[1] for z in plotData]  #Get x,y coords
    Z = [str(100000000+z[2]) for z in plotData]      #Generate CP reference ID
    ax.scatter(X, Y, s=20, c='black')                               #Plot them
    try:                        #Add the combined input DEM outline to the map
        Xb, Yb = polygon.exterior.xy
        ax.plot(Xb, Yb, c='gray')
    except:           #Exception for if DEMs required multiple separate shapes
        for y in polygon.geoms:
            Xb, Yb = y.exterior.xy
            ax.plot(Xb, Yb, c='gray')
    plt.show()
    for k, txt in enumerate(Z):             #Annotate each point with the CPID
        ax.annotate(txt, (X[k],Y[k]), size=6)
    images.append(mat_2_pil(fig))            #Append to list for export to PDF
    plt.close()

    for y in data:     #Plot surface patch w/ altimetry for each control point
        i = 0
        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot2grid((1,1),(0,0))
        for z in years:
            temp = data[y][data[y][:,2].astype(int) == z]        #Plot by year
            if len(temp) > 0:
                ax.scatter(temp[:,0], temp[:,1], s=10, c=clist[i], label=str(z))
            i += 1
        plt.legend(loc='lower right')
        ax.set_xlim(y[0]-patchS/2, y[0]+patchS/2)
        ax.set_ylim(y[1]-patchS/2, y[1]+patchS/2)
        ax.scatter(y[0], y[1], s= 100, marker='+', c='black')  #Mark CP locale
        ax.text(0.02, 0.04, 'bin '+str(100000000+y[2])+'\ncenter: '+
                str(y[0])+', '+str(y[1])+'\nNum points: '+str(len(data[y])),
                fontsize=10, transform=ax.transAxes, backgroundcolor='w')
        images.append(mat_2_pil(fig))
        plt.close()

    images[0].save(prefix+'_cp.pdf', "PDF" , resolution=100.0,     #Export PDF
                       save_all=True, append_images=images[1:])

    with open(prefix + '_cp.csv', 'w') as f:  #Write control point information
        for y in data:
            cpid = str(100000000 + y[2])
            f.write(cpid + ',' + str(y[0]) + ',' + str(y[1]) + '\n')

    time_stamp('Completed control point generation')
    tend = time.time()
    print('Step 1 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 2 - Altimetry extraction by surface patch ##################
#### Extracts altimetry data within each surface patch. One file per CP. #####
##############################################################################

def step2(cwd, prefix, patchS=1000):
    print('Beginning Step 2: extracting laser altimetry by surface patch')
    tstart = time.time()
    time_stamp('Reading altimetry')
    with open('Altimetry\\' + prefix + '_alt.txt', 'r') as f:
        content = f.readlines()         #Read in all lines from altimetry file
    altdata, hp = [], patchS/2

    for i in range(len(content)-1):          #Note x, y, line number, and date
        x = content[i+1].split()
        altdata.append([float(x[4]), float(x[5]), i+1, int(x[1])])
    altdata = np.array(altdata)

    with open('Step1\\' + prefix + 'CPs\\' + prefix + '_cp.csv', 'r') as f:
        cps = f.readlines()            #Read all lines from control point file
    path_maker(cwd + 'Step2\\' + prefix + 'Patches')
    os.chdir(cwd + 'Step2\\' + prefix + 'Patches')

    time_stamp('Writing altimetry by surface patch')
    for x in cps:
        y = x.split(',')
        cpid, xi, yi = y[0], int(y[1]), int(y[2])
                                #Search for altimetry inside the surface patch
        temp = altdata[np.where((altdata[:,0] > xi-hp) & (altdata[:,0] < xi+hp)
                            & (altdata[:,1] > yi-hp) & (altdata[:,1] < yi+hp))]
        unqD = np.unique(temp[:,3])
        for z in unqD:                      #Remove epochs with too few points
            if len(np.where(temp[:,3] == int(z))[0]) < 2:
                temp = np.delete(temp, np.where(temp[:,3] == int(z)), 0)
              #Prepare the header for this surface patch (cpid, #points, x, y)
        line = cpid + s2ff(len(temp),0,7) + s2ff(xi,3,15) + s2ff(yi,3,15) +'\n'
        #try:                                    #Keep track of altimetry dates
        #    epochs = np.unique(np.r_[epochs, unqD])
        #except:
        #    epochs = unqD

        with open(prefix + 'Patch' + cpid + '.txt', 'w') as f2:
            f2.write(line)                                  #Write header line
            for y in temp:              #Copy lines from altimetry subset file
                f2.write(content[int(y[2])])
    #Write the list of dates used, for later reference.
    #with open(prefix + 'Epochs.txt', 'w') as f:
     #   for x in epochs:
      #      f.write(str(int(x)) + '\n')

    time_stamp('Completed extracting altimetry data')
    tend = time.time()
    print('Step 2 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 3/6: Surface patch time series generation ##################
#### Uses least squares to fit a surface of constant shape but changing ######
#### height to the data. Runs twice, without (3) and with (6) DEM data. ######
##############################################################################

def read_patch_file(path, data, sdic, isDEM=0, omit=[]):
#This function reads the files produced by steps 2 and 5.
#"path is the file path of the patch file, "data" and "sdic" are a numerical
#array and string dictionary, respectively, that are appended to. "isDEM' flags
#whether this is a DEM patch. "omit" is a dictionary of CPIDs and any DEM dates
#to be removed from each.
    with open(path, 'r') as f:                              #Read file content
        content = f.readlines()
    cpid = content[0].split()[0]         #Get control point ID from first line
    flag = 1 if isDEM and cpid in omit else 0      #If any DEM dates are to be
    for x in content[1:]:                 #removed, toggle flag to check dates
        y = x.split()
        date = y[1]                        #Get date for each line in the file
        if isDEM and flag and date in omit[cpid][1]:
            continue        #Ignore that line if it's from a DEM to be removed
        mis, ddate = y[0], float(y[2])    #Get mission string and decimal date
        if isDEM:     #Fudge DEM dec. dates to never duplicate altimetry dates
            ddate += 0.00001          #This represents an offset of ~5 minutes
            xi, yi, zi = float(y[3]), float(y[4]), float(y[5])
        else:
            xi, yi, zi = float(y[4]), float(y[5]), float(y[6])           #x,y,elev
        if ddate not in sdic:    #Save mission and MMDDYY for each unique date
            sdic[ddate] = [mis, date]
        data.append([xi, yi, zi, ddate])   #Add x,y,elev,decimal date to array
    return data, sdic      #Return the numeric array and dictionary of strings

def step3(cwd, prefix, tZero='83006', withDEM=0, maxIter=2, flot=False):
    print('Beginning Step 3: time series generation...')
    tstart = time.time()
    if flot:
        cpList = []
        for x in os.listdir(cwd + 'Step10\\' + prefix + 'Floating\\Patches'):
            cpList.append(x.replace(prefix,'').replace('Patch','').replace('.txt',''))

    deduct = 0     #Number of CPs with all DEMs removed (to deduct from total)
    altMis = ['ATM', 'ICE', 'IC2', 'LVIS']    #Altimetry mission tag reference
    if withDEM: #If using DEMs, check for bad epochs to remove (per Step5)
        os.chdir(cwd + 'Step5\\' + prefix + 'Analysis\\')
        files, remo = os.listdir(), {}            #Remo = DEM epochs to remove
        for x in files:    #Get control point IDs from Step 5 output filenames
            cpid = x.replace(prefix,'').replace('Patch','').replace('.txt','')
            if flot and not cpid in cpList:
                continue
            bads= []                                   #List of DEMs to remove
            with open(x, 'r') as f:                        #Read file contents
                content = f.readlines()
            for y in content[1:]:                 #Use [1:] to skip the header
                z = y.split()                #Split line into separate strings
                if int(z[-1]):  #Check the quality flag for each DEM in the CP
                    bads.append(z[1])
            if len(bads) > 0:                    #Move on if none were flagged
                      #Check to see if any DEMs using that CP were NOT flagged
                flag = 1 if len(bads) == len(content)-1 else 0
                 #Store DEMs to remove per CP, and flag CP if no DEMs are left
                remo[cpid], deduct = [flag, bads], deduct + 1

    inFolder = 'Step4\\' if withDEM else 'Step2\\'   #Set folder based on step
    files = os.listdir(cwd + inFolder + prefix + 'Patches')
    if flot:
        #files = os.listdir(cwd + 'Step10\\' + prefix + 'Floating\\Patches')
        path = cwd + 'Step10\\' + prefix + 'Floating\\DHDT\\'
    else:
        outFolder = 'Step6\\' if withDEM else 'Step3\\'
        #files = os.listdir(cwd + inFolder + prefix + 'Patches')
        path = cwd + outFolder + prefix + 'DHDT\\'

    path_maker(path)
    if os.path.exists(path + prefix + 'Stats.txt'):
        os.remove(path + prefix + 'Stats.txt')   #Remove redundant output file
    os.chdir(path)                                #Create the time series file
    count, num = 0, len(files)-deduct #Track "num" of CPs in case some are cut
    with open(prefix + 'DHDT.txt', 'w') as f:            #and write the header
        f.write('cpid,numItems,x,y,z,kappa,sigma0,referenceTime'+
                '\nmission,decDate,MMDDYY,months,relativeZ,absoluteZ,sigma,'+
                'numPoints,blunders,demFlag,\nnumXovers=' + str(num) + '\n')

    for x in files:                                     #Iterate through files
        data, sdic = [], {}   #Create separate objects for strings and numbers
        cpid = x.replace(prefix,'').replace('Patch','').replace('.txt','')
        #if flot and not cpid in cpList:
        #    continue
        if withDEM:
            if cpid in remo and remo[cpid][0] == 1:
                continue                   #Skip CPs that had all DEMs removed
            else: #Read unskipped data from Program 4 output
                os.chdir(cwd + 'Step4\\' + prefix + 'Patches')
                data,sdic = read_patch_file(x, data, sdic, isDEM=1, omit=remo)

        if flot and cpid in cpList:
            os.chdir(cwd + 'Step10\\' + prefix + 'Floating\\Patches')
        else:
            os.chdir(cwd + 'Step2\\' + prefix + 'Patches')
        data, sdic = read_patch_file(x, data, sdic)       #Read altimetry data
        data = np.array(data)
        if withDEM:           #Sort by date, since DEM data won't start sorted
            data = data[data[:,3].argsort()]
        os.chdir(path)

        #Adjust to centroid coordinates and time difference from ref. date
        xc,yc,zc = np.mean(data[:,0]), np.mean(data[:,1]), np.mean(data[:,2])
        xn, yn, zn = data[:,0] - xc, data[:,1] - yc, data[:,2] - zc
        tn = data[:,3] - mmddyy2dec(tZero)
        data = np.c_[xn, yn, zn, tn, data[:,4:]]          #Recollate the array

        blun, bind = 0, [[]]             #No. of blunders; indices of blunders
        idic, temp = {}, data          #Container for data from each iteration
        for i in range(maxIter+1):     #Begin iterations of lst. sq. solutions
            C, H, res, unqT, sig, errs, kap = poly_surface(
                                                       temp,order=3,time=True)
            idic[i] = [H, C, sig]                      #heights, coeffs, sigma
            if i < maxIter:              #Find outliers with 3*sigma tolerance
                bind = np.where(np.abs(res) > 3*sig)
                if len(bind[0]) == 0:
                    break       #If no blunders were detected, stop iterations
                else:
                    badEpochs = []                        #Unsalvageable dates
                     #Remove dates from time series if over 1/4 of pts are bad
                    for i in range(len(unqT)):
                        tempA = temp[np.where(temp[:,3] == unqT[i])]
                        tempA2 = data[np.where(data[:,3] == unqT[i])]
                        tempB = temp[bind]
                        tempB = tempB[np.where(tempB[:,3] == unqT[i])]
                        if len(tempA) > 0 and len(tempB) > 0 and\
                          len(tempA2)/(len(tempB)+len(tempA2)-len(tempA)) < 4:
                            badEpochs.append(unqT[i])
                            blun += len(tempA) - len(tempB)
                    blun += len(bind[0])               #Increase blunder count
                    temp = np.delete(temp, bind, 0)           #Remove blunders
                    for y in badEpochs:
                        temp = np.delete(temp, np.where(temp[:,3]==y), 0)
        #Writing to the fitting statistical summary file
        iters = len(idic)                 #Total iterations actually performed
        wstr = 'CPID' + s2ff(cpid,0,23) + '\n\nheight at centroid' +\
               s2ff(zc,3,12) + '\n#points' + s2ff(len(data),0,23) +\
               '\n#blunders' + s2ff(blun,0,21) + '\nkappa' + s2ff(kap,3,25) +\
               '\n\niteration '
        for i in range(iters):
            wstr += s2ff(i+1,0,18)
        wstr += '\n\nsigma_0        '             #Std dev from each iteration
        for i in range(iters):
            wstr += s2ff(idic[i][2],3,18)
        wstr += '\n\n'
        for i in range(9):                                   #Shape parameters
            wstr += 'a0' + str(i+1) + '            '
            for j in range(iters):
                wstr += s2ff(idic[j][1][i-9],10,18)
            wstr += '\n'
        wstr += '\n'
        for i in range(len(H)):            #Centroid height at each time epoch
            wstr += 'h' + ('' if i > 9 else '0') + str(i) + '            '
            for j in range(iters):
                wstr += s2ff(idic[j][1][i],3,18)
            wstr += '\n'
        wstr += '\n--------------------------------------------------------------\n'
        with open(prefix + 'Stats.txt', 'a+') as f:
            f.write(wstr)                    #Append all that data for each cp

        #Write to the actual time series file
        wstr = cpid + s2ff(len(unqT),0,8) + s2ff(xc,3,15) + s2ff(yc,3,15) +\
               s2ff(zc,3,12) + s2ff(kap,2,12) + s2ff(sig,3,10) +\
               s2ff(tZero,0,9) + '\n'                      #Header for each CP
        for i in range(len(unqT)):  #Iterate through epochs in the time series
            date = unqT[i] + mmddyy2dec(tZero)
            thisEpoch = np.where(temp[:,3] == unqT[i])
            pts = temp[thisEpoch]
            if len(pts) < 2: continue       #Ignore epochs with <2 data points
            blnds = len(np.where(data[:,3] == unqT[i])[0]) - len(pts)
            demFlag = 1 if sdic[date][0] not in altMis else 0
            wstr += '{:3s}'.format(sdic[date][0]) + s2ff(date,4,12) +\
                    s2ff(sdic[date][1],0,9) + s2ff(unqT[i]*12,3,11) +\
                    s2ff(H[i],3,10) + s2ff(C[i]+zc,3,11) + s2ff(errs[i],3,8) +\
                    s2ff(len(pts),0,7) + s2ff(blnds,0,6) +\
                    s2ff(demFlag,0,4) + '\n'
        with open(prefix + 'DHDT.txt', 'a+') as f:
            f.write(wstr)                     #Append to dhdt file for each cp
        count += 1
        try:
            if count % int(num/10) == 0:
                prog_rep(count, num, 'surface patches processed')
        except:
            continue

    time_stamp('Completed generating time series')
    tend = time.time()
    print('Step 3 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 4: DEM data extraction by surface patch ####################
#### Extracts DEM pixels within each surface patch. One file per CP. #########
##############################################################################

def step4(cwd, prefix, patchS=1000):
    print('Beginning Step 4: extracting surface patches from DEMs')
    tstart = time.time()

    time_stamp('Parsing control points')
    cps, data, hp = {}, {}, patchS / 2       #Containers for cp data, dem data
    #if not iproj == oproj:
     #   transformer = Transformer.from_crs(iproj, oproj)
    with open('Step1\\' + prefix + 'CPs\\' + prefix + '_cp.csv', 'r') as f:
        content = f.readlines()                           #Read CP information
    for x in content:
        y = x.replace('\n','').split(',')
        cpx, cpy = float(y[1]), float(y[2])                #Store x, y by CPID
        cps[y[0]] = np.array([cpx, cpy])

    os.chdir(cwd + 'DEMs')
    demList, demDic = os.listdir(), {}
    for x in demList:
        date = extract_num(x)                        #Get dates from filenames
                                   #Extract the mission code from the filename
        mission = ''.join((y for y in x.split('_')[0] if not y.isdigit()))
        demDic[mmddyy2dec(date)] = [date, mission, x]   #Save date, code, file
    for x in sorted (demDic.keys()):          #Go through DEMs chronologically
        time_stamp('Reading DEM on date ' + str(demDic[x][0]))
        surf, polygon, ndv, _, _ = tif_reader(demDic[x][2], border=True)
        #if not iproj == oproj:
         #   X, Y, Z = surf[:,0], surf[:,1], surf[:,2]
          #  X1, Y1 = transformer.transform(X, Y)
           # surf = np.c_[X1, Y1, Z]
         #   coords = np.array(list(polygon.exterior.coords))
          #  X2, Y2 = transformer.transform(coords[:,0], coords[:,1])
           # polygon = Polygon(np.c_[X2, Y2])

        for y in cps:                  #For each DEM, check each control point
            X, Y = cps[y]
            if polygon.contains(shpg.Point((X,Y))):           #Is CP in DEM bounds?
                temp = surf[(surf[:,0] >= X-hp) & (surf[:,0] <= X+hp) &\
                            (surf[:,1] >= Y-hp) & (surf[:,1] <= Y+hp) &\
                            (surf[:,2] != ndv)]   #Get pixels in surface patch
                temp = np.c_[temp, np.full(len(temp), x)]  #Save with DEM date
                try:                            #Collate data by control point
                    data[y] = np.r_[data[y], temp]
                except:
                    data[y] = temp

    print('Writing output files...')
    path_maker(cwd + 'Step4\\' + prefix + 'Patches')
    os.chdir(cwd + 'Step4\\' + prefix + 'Patches')
    for x in data:                             #Iterate through control points
        with open(prefix + 'Patch' + x + '.txt', 'w') as f:
            f.write(x + s2ff(len(data[x]),0,8) + '\n')    #Write no. of points
            for y in data[x]:
                                    #Write mission, mmddyy, dec. date, x, y, z
                wstr = '{:3s}'.format(demDic[y[3]][1]) +\
                       s2ff(demDic[y[3]][0],0,8) + s2ff(y[3],4,12) +\
                       s2ff(y[0],3,14) + s2ff(y[1],3,14) +s2ff(y[2],3,10)+'\n'
                f.write(wstr)

    time_stamp('Completed extracting DEM data')
    tend = time.time()
    print('Step 4 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 5: DEM surface patch examination ###########################
#### Checks to see if DEM data provides tenable surface patch shapes. ########
##############################################################################

def step5(cwd, prefix, thresh=10):
    print('Beginning Step 5: Surface patch shape analysis...')
    files = os.listdir(cwd + 'Step4\\' + prefix +'Patches')
    path_maker(cwd + 'Step5\\' + prefix + 'Analysis')
    num, count, tstart = len(files), 0, time.time()

    time_stamp('Analyzing surface patches')
    for x in files:                           #Iterate through DEM patch files
        os.chdir(cwd + 'Step4\\' + prefix + 'Patches')
        data, zstats, i, count = [], [], 0, count + 1
        with open(x, 'r') as f:
            content = f.readlines()
        for y in content[1:]:
            z = y.split()
            data.append([z[1], z[3], z[4], z[-1]])         #Save date, x, y, z
        if len(data) == 0:
            continue             #Failsafe for the case that the file is empty
        data = np.array(data, dtype=float)
        #ind = np.unique(data[:,0], return_index=True)[1]
        #unqD = [data[:,0][i] for i in sorted(ind)]
        unqD = np.unique(data[:,0])                      #Get unique DEM dates
                                            #Find the centroid of the DEM data
        xc,yc,zc = np.mean(data[:,1]), np.mean(data[:,2]), np.mean(data[:,3])

        for i in range(len(unqD)):                      #Iterate through dates
            temp = data[np.where(data[:,0] == unqD[i])]
                                       #Compute x,y,z relative to the centroid
            xn, yn, zn = temp[:,1] - xc, temp[:,2] - yc, temp[:,3] - zc
            #Do a least squares fitting for the surface patch and save results
            C, res, sig, _, kap = poly_surface(np.c_[xn, yn, zn])
            zstats.append([i+1, unqD[i], len(temp), zc, sig, C[1], C[2],
                         C[0] + zc])

        zstats = np.array(zstats)
        os.chdir(cwd + 'Step5\\' + prefix + 'Analysis')
        with open(x, 'w') as f:
            f.write('timeEpoch  date  #points  averageZ  sigma   shapePar1  ' +
                    ' shapePar2  absoluteZ   flag\n')
            for y in zstats:
                                   #Flag DEM for removal from CP if sigma > 10
                flg = '1' if y[4] > thresh else '0'
                wstr = s2ff(y[0],0,5) + s2ff(y[1],0,10) + s2ff(y[2],0,8) +\
                       s2ff(y[3],2,10) + s2ff(y[4],2,8) + s2ff(y[5],8,12) +\
                       s2ff(y[6],8,12) + s2ff(y[7],2,11) + s2ff(flg,0,7) +'\n'
                f.write(wstr)

        if num > 10:
            if count % int(num / 10) == 0:
                prog_rep(count, num, 'surface patches processed')

    time_stamp('Completed reconstruction of surface patches')
    tend = time.time()
    print('Step 5 elapsed time: ' + str(tend-tstart) + ' seconds.')
    os.chdir(cwd)

##############################################################################
#### MOULINS Step 7: Time series curve fitting ###############################
#### Fits a function to the time series and calculates DEM corrections. ######
##############################################################################

def step7(cwd, prefix, refd, climateMode=False, mosaicMode=False, refDate=0, dh=25, dp=1, flot=False):
    import matplotlib.pyplot as plt
    tstart = time.time()
    if flot:
        mainPath= cwd + 'Step10\\' + prefix + 'Floating\\Corrections'
    else:
        mainPath = cwd + 'Step7\\' + prefix + 'Corrections'
    path_maker(mainPath)
    print('Beginning Step 7: calculating correction vectors...')
    if flot:
        subPath = cwd + 'Step10\\' + prefix + 'Floating\\DHDT\\'
    else:
        subPath = cwd +'Step6\\' +prefix +'DHDT\\'
    ts,cp = time_series(subPath + prefix +'DHDT.txt')
    print('Reading time series...')
    ind = [x for x in ts]     #Get all control points; use as indices for data
    ind.sort()

    dems, images, ddic, count, num = np.empty(0), [], {}, 0, len(ind)
    #This array manually defines the elements of a plot legend
    lgndElem = [plt.Line2D([0],[0],marker='o',color='w',label='Altim.',
                           markeredgecolor='r',markerfacecolor='none',
                           markersize=10,linewidth=2),
                plt.Line2D([0],[0],marker='o',color='w',label='DEM',
                           markeredgecolor='b',markerfacecolor='none',
                           markersize=10, linewidth=2),
                plt.Line2D([0],[0],color='k',lw=3,label='ALPS'),
                plt.Line2D([0],[0],color='g',lw=3,label='Polynomial'),
                plt.Line2D([0],[0],alpha=0.2,color='k',lw=10,label='ALPSErr'),
                plt.Line2D([0],[0],alpha=0.2,color='g',lw=10,label='PolyErr')]

    time_stamp('Beginning spline fitting')
    os.chdir(mainPath)

    if climateMode:
        images2 = []
        import netCDF4 as nc                       #Open the climate data file
        ncin = nc.Dataset(cwd + 'ClimateModel\\gsfc_fdm_v1_2_1_gris_Dec22.nc')
        climX = np.array(ncin['x'])                             #Read x coords
        climY = np.array(ncin['y'])                             #Read y coords
        climT = np.array(ncin['time'])                             #Read dates
        h_a = np.array(ncin['h_a'])                  #Read firn height anomaly

    for x in ind:                         #Iterate through all control points
        count += 1
        try:
            if count % int(num/10) == 0:
                prog_rep(count, num, 'control points processed')
        except:
            pass

        data, dtext = ts[x], 'DEMS in this series:\n'
        #Separate altimetry; don't use DEM data to fit the curve
        adata, ddata = data[:,:2][data[:,3]==0], data[:,:2][data[:,3]==1]
        if len(ddata) == 0: continue
        dtemp = [str(int(y)) for y in data[:,2][data[:,3]==1]]  #All DEM dates
        dems = np.unique(np.array(dtemp))                #All unique DEM dates
        for y in dems :       #Make a label to show which DEMs are in the plot
            dtext += y + '\n'

        #Remove altimetry outside of temporal bounds, skip this control point
        #if too little altimetry is left
        dtmin, dtmax = np.min(ddata[:,0]), np.max(ddata[:,0])
        adata = adata[(adata[:,0] > dtmin-dh) & (adata[:,0] < dtmax+dh) &
                      (adata[:,1] < np.mean(adata[:,1])+100) &
                      (adata[:,1] > np.mean(adata[:,1])-100)]
        if len(adata[adata[:,0] < dtmin]) < dp or\
            len(adata[adata[:,0] > dtmax]) < dp: continue

        #Removing seasonal effects.
        if climateMode:
            tc, zc = np.mean(adata[:,0]), np.mean(adata[:,1])
            tt, zt = adata[:,0] - tc, adata[:,1] - zc
            temp = np.c_[np.ones(len(tt)),tt,tt**2,tt**3]
            C, _, sigP, _, kap = least_sq(temp, zt)
            sigP = 0 if sigP > 1e10 else sigP
            xpred = np.linspace(np.min(tt), np.max(tt), 200)
            ypredP = C[0] + C[1]*xpred + C[2]*xpred**2 + C[3]*xpred**3
            if mosaicMode:
                dtt = np.full(ddata[:,0].shape, mmddyy2dec(str(refDate))) - tc
            else:
                dtt = ddata[:,0] - tc
            dempredP = C[0] + C[1]*dtt + C[2]*dtt**2 + C[3]*dtt**3
            xpred, ypredP, dempredP = xpred + tc, ypredP + zc, dempredP + zc
            corrVecP = dempredP - ddata[:,1]

            p, q, k = 4, 2, 200     #do the spline computation
            try:
                [n, lamb, sigmasq] = full_search_nk(adata,p,q)
            except:
                continue
            c = n + p
            U = Kno_pspline_opt(adata, p, n)
            B = Basis_Pspline(n,p,U,adata[:,0])
            P = Penalty_p(q,c)
            theta = np.linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(
                                                        adata[:,1].reshape(-1,1)))
            Bpred = Basis_Pspline(n, p, U, xpred)
            ypredA = Bpred.dot(theta)
            sigA = Var_bounds(adata, Bpred, B, theta, P, lamb)
            temppred = Basis_Pspline(n,p,U,dtt+tc)
            dempredA = temppred.dot(theta)
            corrVecA = dempredA.flatten() - ddata[:,1]

            fig = plt.figure(figsize=(15,10))
            ax = plt.subplot2grid((1,1),(0,0))

            ax.scatter(adata[:,0], adata[:,1], color='r', facecolors='none',
                       s=100, linewidth=2)                         #plot altimetry
            ax.scatter(ddata[:,0], ddata[:,1], color='b', facecolors='none',
                       s=100, linewidth=2)           #plot dems in different color
            ax.plot(xpred,ypredA,linewidth=3,color = 'k')           #plot the curve
            ax.plot(xpred,ypredP,linewidth=3,color = 'g')
            ax.set_title('Time Series: ' + str(x), size = 25)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize = 20)
            ax.set_xlabel('Time',size=20)
            ax.set_ylabel('Relative elevation (m)',size = 22)
            ax.text(0.02, 0.04, dtext, fontsize=10, transform=ax.transAxes,
                    backgroundcolor='w')                   #list the included dems
            ax.fill_between(xpred,ypredA.flatten()-sigA,ypredA.flatten()+sigA,
                            alpha = 0.2,color ='k')
            ax.fill_between(xpred,ypredP-sigP,ypredP+sigP,alpha = 0.2,color ='g')
            ax.grid(True)
            ax.legend(handles = lgndElem,bbox_to_anchor=(0.99,1),loc='upper left')
            images2.append(mat_2_pil(fig)) #compile list of figures to export later
            plt.close()
            
            xind = np.absolute(cp[x][0] - climX).argmin()   #Get index of CP x
            yind = np.absolute(cp[x][1] - climY).argmin()   #Get index of CP y
            refInd = np.absolute(mmddyy2dec(refd) - climT).argmin() #Ref. date ind
            tempH_a = h_a[:,xind,yind]       #Get h_a time series at CP coords
            hOffset = tempH_a[refInd]      #Find value of h_a at the ref. date
            tempH_a = tempH_a - hOffset        #Normalize h_a to the ref. date
            corrArray = []      #An array for holding the seasonal corrections
            for y in adata:                            #Iterate altimetry data
                tempInd = np.absolute(y[0] - climT).argmin() #Nearest h_a time
                corrArray.append([0,tempH_a[tempInd]])   #Get h_a at that date
            adata = np.array(adata - corrArray)          #Apply the correction
            if mosaicMode:
                corrArray = []                            #Repeat for DEM data
                for y in ddata:
                    tempInd = np.absolute(y[0] - climT).argmin()
                    corrArray.append([0,tempH_a[tempInd]])
                ddata = np.array(ddata + corrArray)

        tc, zc = np.mean(adata[:,0]), np.mean(adata[:,1])
        tt, zt = adata[:,0] - tc, adata[:,1] - zc
        temp = np.c_[np.ones(len(tt)),tt,tt**2,tt**3]
        C, _, sigP, _, kap = least_sq(temp, zt)
        sigP = 0 if sigP > 1e10 else sigP
        xpred = np.linspace(np.min(tt), np.max(tt), 200)
        ypredP = C[0] + C[1]*xpred + C[2]*xpred**2 + C[3]*xpred**3
        if mosaicMode:
            dtt = np.full(ddata[:,0].shape, mmddyy2dec(str(refDate))) - tc
        else:
            dtt = ddata[:,0] - tc
        dempredP = C[0] + C[1]*dtt + C[2]*dtt**2 + C[3]*dtt**3
        xpred, ypredP, dempredP = xpred + tc, ypredP + zc, dempredP + zc
        corrVecP = dempredP - ddata[:,1]

        p, q, k = 4, 2, 200     #do the spline computation
        try:
            [n, lamb, sigmasq] = full_search_nk(adata,p,q)
        except:
            continue
        c = n + p
        U = Kno_pspline_opt(adata, p, n)
        B = Basis_Pspline(n,p,U,adata[:,0])
        P = Penalty_p(q,c)
        theta = np.linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(
                                                    adata[:,1].reshape(-1,1)))
        Bpred = Basis_Pspline(n, p, U, xpred)
        ypredA = Bpred.dot(theta)
        sigA = Var_bounds(adata, Bpred, B, theta, P, lamb)#[0]
        #temppred = Basis_Pspline(n,p,U,ddata[:,0])
        temppred = Basis_Pspline(n,p,U,dtt+tc)
        dempredA = temppred.dot(theta)
        corrVecA = dempredA.flatten() - ddata[:,1]

        fig = plt.figure(figsize=(15,10))
        ax = plt.subplot2grid((1,1),(0,0))

        ax.scatter(adata[:,0], adata[:,1], color='r', facecolors='none',
                   s=100, linewidth=2)                         #plot altimetry
        ax.scatter(ddata[:,0], ddata[:,1], color='b', facecolors='none',
                   s=100, linewidth=2)           #plot dems in different color
        ax.plot(xpred,ypredA,linewidth=3,color = 'k')           #plot the curve
        ax.plot(xpred,ypredP,linewidth=3,color = 'g')
        ax.set_title('Time Series: ' + str(x), size = 25)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize = 20)
        ax.set_xlabel('Time',size=20)
        ax.set_ylabel('Relative elevation (m)',size = 22)
        ax.text(0.02, 0.04, dtext, fontsize=10, transform=ax.transAxes,
                backgroundcolor='w')                   #list the included dems
        ax.fill_between(xpred,ypredA.flatten()-sigA,ypredA.flatten()+sigA,
                        alpha = 0.2,color ='k')
        ax.fill_between(xpred,ypredP-sigP,ypredP+sigP,alpha = 0.2,color ='g')
        ax.grid(True)
        ax.legend(handles = lgndElem,bbox_to_anchor=(0.99,1),loc='upper left')
        images.append(mat_2_pil(fig)) #compile list of figures to export later
        plt.close()

        for i in range(len(dempredA)):
            try:
                ddic[dems[i]][x] = [corrVecA[i], sigA, corrVecP[i], sigP]
            except:
                ddic[dems[i]] = {}
                ddic[dems[i]][x] = [corrVecA[i], sigA, corrVecP[i], sigP]

    pdfname = prefix + 'TimeSeriesPlots.pdf'
    time_stamp('Exporting figures to pdf')
    if len(images) == 1:          #Have to adjust call if there's only 1 image
        images[0].save(pdfname, "PDF", resolution=100.0, save_all=True)
    elif len(images) > 0:                      #write the images to a pdf file
        images[0].save(pdfname, "PDF", resolution=100.0, save_all=True,
                       append_images=images[1:])
    
    if climateMode:
        pdfname = prefix + 'TimeSeriesPlotsNoFDC.pdf'
        time_stamp('Exporting figures to pdf')
        if len(images2) == 1:          #Have to adjust call if there's only 1 image
            images2[0].save(pdfname, "PDF", resolution=100.0, save_all=True)
        elif len(images2) > 0:                      #write the images to a pdf file
            images2[0].save(pdfname, "PDF", resolution=100.0, save_all=True,
                           append_images=images2[1:])

    print('Writing correction vectors and control point locations...')
    demList = os.listdir(cwd + 'Shapes\\' + prefix + 'Shapes')
    images = []
    for x in ddic.keys():
        vecs = []
        for y in ddic[x]:
            vecs.append(ddic[x][y][0])
        vecs = np.array(vecs)
        std, avg = np.std(vecs), np.mean(vecs)
        wstr = 'cpid        x             y             corrALPS    resALPS' +\
               '     corrPoly    resPoly    flag\n'
        plotArray = []
        for y in ddic[x]:
            vecA, confA  = ddic[x][y][0], ddic[x][y][1]
            vecP, confP  = ddic[x][y][2], ddic[x][y][3]
            flag = 1 if abs(avg-vecA) > 3*std else 0
            wstr += s2ff(y,0,9) + s2ff(cp[y][0],3,14) + s2ff(cp[y][1],3,14) +\
                    s2ff(vecA,3,10) + s2ff(confA,3,10) + s2ff(vecP,3,10) +\
                    s2ff(confP,3,10) +s2ff(flag,0,5)+'\n'
            if flag == 0:
                plotArray.append([cp[y][0], cp[y][1], vecA])
        with open(x + 'Corrections.txt', 'w') as f:
            f.write(wstr)
        plotArray = np.array(plotArray)
        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot2grid((1,1),(0,0))

        demFile = [s for s in demList if x in s]
        polygon = exp_combine([cwd+'Shapes\\'+prefix+'Shapes\\'+demFile[0]])
        xmin, ymin, xmax, ymax = polygon.bounds
        xmin, xmax, ymin, ymax = square_domain(xmin, xmax, ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        Xp, Yp = polygon.exterior.xy
        ax.plot(Xp, Yp, c='black')

        sc=ax.scatter(plotArray[:,0], plotArray[:,1], c=plotArray[:,2], cmap='hsv')
        plt.colorbar(sc)
        ax.set_title('DEM on date ' + str(x), size = 25)
        images.append(mat_2_pil(fig))

    pdfname = prefix + 'DEMPlots.pdf'
    time_stamp('Exporting figures to pdf')
    if len(images) == 1:          #Have to adjust call if there's only 1 image
        images[0].save(pdfname, "PDF", resolution=100.0, save_all=True)
    elif len(images) > 0:                      #write the images to a pdf file
        images[0].save(pdfname, "PDF", resolution=100.0, save_all=True,
                       append_images=images[1:])

    os.chdir(cwd)
    tend = time.time()
    print('Step 7 elapsed time: ' + str(tend-tstart) + ' seconds.')
    time_stamp('Completed fitting curves to time series')

##############################################################################
#### MOULINS Step 8: Correction surface fitting ##############################
#### Fits a polynomial surface to the correction vectors. ####################
##############################################################################

def step8(cwd, prefix, cfit='alps', mImp=0.5, flagged=[], flot=False):
    import matplotlib.pyplot as plt
    tstart = time.time()
    print('Beginning step 11: computing correction surface...')
    os.chdir('DEMs')
    dems, cpDict = os.listdir(), {}
    if flot:
        path_maker(cwd + 'Step10\\' + prefix + 'Floating\\Surface')
        with open(cwd + 'Step10\\' + prefix + 'Floating\\' + prefix + 'DEMList.txt', 'r') as f:
            content = f.readlines()
        for i, x in enumerate(content):
            content[i] = x.replace('\n','').replace('Out','')
        temp = []
        for x in dems:
            if x in content:
                temp.append(x)
        dems = temp
    else:
        path_maker(cwd + 'Step8\\' + prefix + 'Surface')
    legend = [plt.Line2D([0],[0],marker='o',color='r',label='Control points',
                         markerfacecolor='r',markersize=10),
              plt.Line2D([0],[0],marker='.',color='k',label='DEM boundary',
                         markerfacecolor='k',markersize=10),
              plt.Line2D([0],[0],color='b',lw=3,label='Polynomial surface')]

    for x in dems:            #iterate through each item in the DEM folder
        if flot:
            os.chdir(cwd + 'Step10\\' + prefix + 'Floating\\Corrections')
        else:
            os.chdir(cwd + 'Step7\\'+ prefix + 'Corrections')
        date, coord = extract_num(x), []
        if flagged != [] and date not in flagged:
            continue
        if len(date) < 4 or not date.isnumeric():
            exit('\aDEM does not have readable date. Start file with MMDDYY.')
        time_stamp('Loading control points for DEM on ' + date)
        with open(str(int(date)) + 'Corrections.txt','r') as f:
            content = f.readlines()
        for y in content[1:]:
            s = y.split()          #read the correction file data for this DEM
            if s[7] == '0':
                if cfit == 'alps':
                    coord.append([int(s[0]), float(s[1]), float(s[2]),
                                  float(s[3])])
                elif cfit == 'poly':
                    coord.append([int(s[0]), float(s[1]), float(s[2]),
                                  float(s[5])])
        coord = np.array(coord)

        print('Loading DEM on ' + date)
        os.chdir(cwd + 'DEMs')
        temp = []
        surf, _, ndv, _, _ = tif_reader(x)
        polygon = exp_combine([cwd + 'Shapes\\' + prefix + 'Shapes\\' +\
                               prefix + '_' + date + 'shape.exp'], buffer=100)
        dx = round(np.diff(np.unique(surf[:,0])).min())
        dy = round(np.diff(np.unique(surf[:,1])).min())
        for y in surf:
            if polygon.contains(shpg.Point((y[0],y[1]))):
                temp.append(y)
        surf = np.array(temp)
        xc, yc = np.mean(surf[:,0]), np.mean(surf[:,1])
        surf = np.c_[surf[:,0] - xc, surf[:,1] - yc, surf[:,2]]      #recenter
        coord = np.c_[coord[:,0], coord[:,1]-xc, coord[:,2]-yc, coord[:,3]]    
        xmin, ymin, xmax, ymax = polygon.bounds

        xpred = np.linspace(xmin-xc, xmax-xc, int((xmax-xmin)/dx))
        ypred = np.linspace(ymin-yc, ymax-yc, int((ymax-ymin)/dy))
        X1, Y1 = np.meshgrid(xpred, ypred, indexing='xy')
        xmin,xmax,ymin,ymax = square_domain(xmin-xc,xmax-xc,ymin-yc,ymax-yc)
        border = np.array([polygon.exterior.interpolate(z).xy for z in
             np.linspace(0,int(polygon.length),int(polygon.length/dx),False)])
        bx, by = border[:,0] - xc, border[:,1] - yc

        print('Computing approximation for DEM on ' + date)
        if flot:
            os.chdir(cwd + 'Step10\\' + prefix + 'Floating\\Surface')
        else:
            os.chdir(cwd + 'Step8\\' + prefix + 'Surface')
        if os.path.exists(prefix + date + 'Stats.txt'):
            os.remove(prefix + date + 'Stats.txt')

        with open(prefix + date + 'Stats.txt', 'a+') as f:
            f.write('Centroid:' + s2ff(xc,3,22) + s2ff(yc,3,21) + '\n\n')
        maxWeight, useDegree, cpsUsed = 0, 0, {}
        cpTemp = {1: [], 2: [], 3: []}
        for i in range(1,4):         #check each polynomial degree from 1 to 3
            cps, resDict, iterDict = coord, {}, {}
            for y in cps[:,0]:
                resDict[y] = []  #keep track of residues at each control point
            for j in range(2):    #reiterate after removing blunders
                rmsCpVect = round(rms(cps[:,3]),3)
                meanCpVect = round(np.mean(cps[:,3]),3)
                try:
                    C,res,sigma,_,_ = poly_surface(cps[:,1:],order=i)
                    cz = poly_value(cps[:,1], cps[:,2], C)
                    bz, rmsRes = poly_value(bx, by, C), rms(res)
                    zpred, iterDict[j+1] = poly_value(X1, Y1, C), C
                except:
                    continue  #skip this degree if the linear regression fails
                cps = np.c_[cps, cz]   #get standard deviation of the residues
                rmsCpCorr = round(rms(cz),3)  #get rms surface value at cps
                for k in range(len(cps)):
                    resDict[cps[k,0]].append(res[k])

                temp, sample = surf[np.where(surf[:,2] > ndv)], []
                rando = np.random.choice(range(len(temp)),int(len(temp) / 20),
                                         replace=False)   #randomly sample 1/5
                for y in rando:
                    sample.append([temp[y,0], temp[y,1]])
                sample = np.array(sample)  #get surface values at these pixels
                sz = poly_value(sample[:,0], sample[:,1], C)
                rmsGenCorr = round(rms(sz), 3) #the the rms of these values

                improve = rmsCpVect - rmsRes                      #improvement
                fitRatio = min(rmsGenCorr / rmsCpCorr, rmsCpCorr / rmsGenCorr)
                weight = (improve / rmsCpVect) * fitRatio**3
                if improve >= mImp and fitRatio > 0.8 and weight > maxWeight\
                                                            and len(cps) > 15:
                    maxWeight, useDegree = weight, i
                if j == 0:
                    ind = np.where(abs(res) < 3 * sigma)
                    if len(ind[0]) == len(cps):                #if no blunders
                        break
                    cps = cps[ind]
                #remove this later
                useDegree = 1
                #remove this later

            cpsUsed[i] = len(cps)     #final number of control points used
            wstr = 'Deg ' +str(i) + '\n\nIteration:' + s2ff(1,0,21)
            if len(resDict[cps[k,0]]) > 1:
                wstr += s2ff(2,0,21)
            wstr += '\n'
            for j in range(len(iterDict[1])):
                wstr += '\n' +'A' +str(j) +':' +s2ff(iterDict[1][j],5,28,'e')
                if 2 in iterDict:
                    wstr += s2ff(iterDict[2][j],5,21,'e')
            wstr += '\n'
            for y in resDict:
                wstr += '\n' + s2ff(y,0,9) + ':' + s2ff(resDict[y][0],1,21)
                if len(resDict[y]) > 1:
                    wstr += s2ff(resDict[y][1],1,21)
            wstr += '\n\nMean Control vector is' + s2ff(meanCpVect,3,9) +\
                   '\nRMS Control vector is' + s2ff(rmsCpVect,3,10) +\
                    '\nRMS Residual is' + s2ff(rmsRes,3,16) +\
                    '\nRMS Correction value is' + s2ff(rmsGenCorr,3,8) +\
                    '\nImprovement is' + s2ff(improve,3,17) +\
                    '\nFit ratio is' + s2ff(fitRatio,3,19) +\
                    '\nDecision weight:' + s2ff(weight,3,15) +\
                    '\n\n*  *  *  *  *  *  *  *  *  *  *  *  *\n\n'
            with open(prefix + date + 'Stats.txt', 'a+') as f:
                f.write(wstr)

            fig = plt.figure(figsize=(15,15))              #3d plot of surface
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cps[:,1], cps[:,2], cps[:,3], c='r',marker='o')    #cps
            ax.plot(bx.flatten(), by.flatten(), bz.flatten(),
                    c='k', linewidth=3)           #draw dem border
            ax.plot_wireframe(X1, Y1, zpred, rcount=50, ccount=50)  #wireframe
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.legend(handles = legend)
            imageList = photo_shoot(fig, ax) #take snapshots from diff. angles
            pdfFilename = prefix + date + 'Deg' + str(i) + 'Plots.pdf'
            imageList[0].save(pdfFilename, "PDF", resolution=100.0,
                              save_all=True, append_images=imageList[1:])
            plt.close()                               #make pdf with all plots

            for y in cps:
                xdem = int(((y[1] - xmin + xc) / dx) + 0.5) * dx + xmin - xc
                ydem = int(((y[2] - ymin + yc) / dy) + 0.5) * dy + ymin - yc
                zdem = surf[:,2][np.where((surf[:,0] == xdem) &
                                          (surf[:,1] == ydem))]
                if zdem > 0:
                    cpTemp[i].append([y[0], zdem + y[4]])

        if useDegree != 0:
            for y in cpTemp[useDegree]:       #record the final recommendation
                try:
                    cpDict[str(int(y[0]))].append([date, y[1][0]])
                except:
                    cpDict[str(int(y[0]))]  = [[date, y[1][0]]]

        with open(prefix + date + 'Stats.txt', 'a+') as f:
            if useDegree == 0:
                f.write('\nControl points used:' + s2ff(cpsUsed[3],0,8))
                f.write('\nUse degree                NA')
            else:
                f.write('\nControl points used:' + s2ff(cpsUsed[useDegree],0,8))
                f.write('\nUse degree ' + s2ff(useDegree,0,17))

    os.chdir(cwd)
    tend = time.time()
    print('Step 8 elapsed time: ' + str(tend-tstart) + ' seconds.')
    time_stamp('Completed estimating correction surfaces')

##############################################################################
#### MOULINS Step 9: DEM Correction ##########################################
#### Apply the correction surface to the DEMs ################################
##############################################################################

def step9(cwd, prefix, flagged=[], flot=False):
    print('Beginning step 9: DEM correction...')
    tstart = time.time()
    os.chdir(cwd + 'DEMs')
    demList, cdic = os.listdir(), {1: 3, 2: 6, 3: 10}       #DEMs; # of coeffs
    if flot:
        path_maker(cwd + 'Step10\\' + prefix + 'Floating\\Corrected')
        with open(cwd + 'Step10\\' + prefix + 'Floating\\' + prefix + 'DEMList.txt', 'r') as f:
            content = f.readlines()
        for i, x in enumerate(content):
            content[i] = x.replace('\n','').replace('Out','')
        temp = []
        for x in demList:
            if x in content:
                temp.append(x)
        demList = temp
    else:
        path_maker(cwd + 'Step9\\' + prefix + 'Corrected')    #make output dir.

    for x in demList:                                #iterate through each DEM
        date, inst = extract_num(x, inst=True)        #get date and instrument
        if flagged != [] and date not in flagged:      #check for flagged DEMs
            continue
        time_stamp('Checking stats for DEM on ' + date)
        if flot:
            os.chdir(cwd + 'Step10\\' + prefix + 'Floating\\Surface')
        else:
            os.chdir(cwd + 'Step8\\' + prefix + 'Surface')
        with open(prefix + date + 'Stats.txt', 'r') as f:  #read the stat file
            content = f.readlines()

        C, deg = [], content[-1].split()[2]        #retrieve the degree to use
        xc,yc = float(content[0].split()[1]), float(content[0].split()[2])
        if deg == 'NA':
               #get RMS control vector as error surrogate if no recommendation
            ind = [i for i, s in enumerate(content) if 'S Con' in s]
               #if it's less than 1, copy DEM and use it, otherwise do nothing
            if float(content[ind[0]].split()[-1]) < 0.6:
                if flot:
                    tempPath = cwd + 'Step10\\' + prefix + 'Floating\\Corrected\\' + x
                else:
                    tempPath = cwd + 'Step9\\' + prefix + 'Corrected\\' + x
                copy(cwd + 'DEMs\\' + x, tempPath)
                continue
            else:
                ind = [i for i, s in enumerate(content) if 'n Con' in s]
                cVec = float(content[ind[0]].split()[-1])
                os.chdir(cwd + 'DEMs')
                surf, _, ndv, gt, proj = tif_reader(x, raw=True)
                outVector = surf + cVec
                yLen, xLen = outVector.shape
                
        else:
            i = content.index('Deg ' + str(deg) + '\n') + 4
            for y in content[i: i + cdic[int(deg)]]:             #get coefficients
                C.append(float(y.split()[1]))
    
            print('Extracting DEM values for ' + date + '...')
            os.chdir(cwd + 'DEMs')
            surf, _, ndv, gt, proj = tif_reader(x)
            surf = np.c_[surf[:,0] - xc, surf[:,1] - yc, surf[:,2]]     #translate
            surf = np.c_[surf, surf[:,2] + poly_value(surf[:,0], surf[:,1], C) *
                         (surf[:,2] != np.full(surf.shape[0], ndv))]   #correction
            surf = np.c_[surf[:,0] + xc, surf[:,1] + yc, surf[:,2:]]  #untranslate
    
            time_stamp('Writing corrected DEM for ' + date)
            xi, yi = list(set(surf[:,0])), list(set(surf[:,1]))    #write new file
            xLen, yLen = len(xi), len(yi)
            xi.sort()
            yi.sort(reverse=True)
            outVector = np.zeros([len(yi),len(xi)])
            for j in range(len(yi)):
                temp = surf[np.where(surf[:,1] == yi[j])]
                for i in range(len(xi)):
                    temp2 = temp[np.where(temp[:,0] == xi[i])]
                    outVector[j,i] = temp2[0,3]
        if flot:
            outPath = cwd + 'Step10\\' + prefix + 'Floating\\Corrected\\' + x.replace('.','Out.')
        else:
            outPath = cwd + 'Step9\\' + prefix + 'Corrected\\' + x.replace('.','Out.')
        outVector[np.isnan(outVector)] = ndv
        with rasterio.open(outPath, 'w', driver='GTiff', height=yLen, \
                           width=xLen, count=1, dtype=outVector.dtype, crs=proj, \
                           transform=gt, nodata=ndv) as dst:
            dst.write(outVector,1)

    os.chdir(cwd)
    tend = time.time()
    print('Step 9 elapsed time: ' + str(tend-tstart) + ' seconds.')
    time_stamp('Completed writing corrected DEMs')

##############################################################################
#### MOULINS Step 10: Floating Ice Corrections ###############################
#### Identify floating CPs and account for tides/pressure ####################
##############################################################################

def step10(cwd, prefix, rhoW, rhoI, demEPSG):
    print('Beginning step 10a: Floating Ice Corrections...')
    import pyMTDForMoulinsModule050525 as pyMTD
    import datetime
    epoch = (1970,1,1,0,0,0)
    tstart, path = time.time(), cwd + 'TideModel\\'
    epochT = datetime.datetime(epoch[0], epoch[1], epoch[2])
    utcEp = epochT.replace(tzinfo=datetime.timezone.utc).timestamp()
    path_maker(cwd + 'Step10\\' + prefix + 'Floating')
    thickRatio = np.float32((rhoW - rhoI) / rhoW)
    underRatio = 1 - (1/thickRatio)           #Ratio of submerged/floating ice
    os.chdir(cwd + 'BedMachine')                 #Read the bed and geoid grids
    bedData,_,ndvbm,gtbm,prjbm = tif_reader('BedMachineGreenland-v5_bed.tif', raw=True)
    geoidData,_,_,_,_ = tif_reader('BedMachineGreenland-v5_geoid.tif', raw=True)

    cpDict, floatDict = {}, {}              #Store all CPs; store floating CPs
    floatDems = []
    os.chdir(cwd + 'Step1\\' + prefix + 'CPs')
    with open(prefix + '_cp.csv', 'r') as f:
        content = f.readlines()
    for x in content:                               #Get CPs from Step1 output
        cpid, X, Y = x.replace('\n','').split(',')
        cpDict[cpid] = [float(X), float(Y)]

    os.chdir(cwd + 'Step9\\' + prefix + 'Corrected')
    for x in os.listdir():                     #Iterate through corrected DEMs
        time_stamp('Finding floating ice for ' + x)              #Read the DEM
        data,poly,ndv,gt,proj = tif_reader(x, border=True, raw=True)
        temp = np.ones(data.shape, dtype=np.float32)
        from rasterio.warp import reproject, Resampling
        temp = np.ones(data.shape, dtype=np.float32)
        reproject(bedData, temp, src_transform=gtbm, src_crs=prjbm, src_nodata=ndvbm,
                  dst_transform=gt, dst_crs=proj, dst_nodata=ndv,
                  resampling=Resampling.bilinear)  #Reproject the bed to match
        bedTemp = temp                             #the DEM extents
        temp = np.ones(data.shape, dtype=np.float32)
        reproject(geoidData, temp, src_transform=gtbm, src_crs=prjbm, src_nodata=ndvbm,
                  dst_transform=gt, dst_crs=proj, dst_nodata=ndv,
                  resampling=Resampling.bilinear)  #Do the same with the geoid
        geoidTemp = temp
        data = data - geoidTemp       #Correct the DEM vertically to the geoid
        floatMask = np.where(data < bedTemp/underRatio, 1, 0)
        floatMask = floatMask.astype(np.float32)         #Find floating pixels
        outPath = cwd + 'Step10\\' + prefix + 'Floating\\' + x.replace('Out','FloatMask')
        data[np.isnan(data)] = ndv
        with rasterio.open(outPath, 'w', driver='GTiff', height=floatMask.shape[0], \
                           width=floatMask.shape[1], count=1, dtype=floatMask.dtype, crs=proj, \
                           transform=gt, nodata=ndv) as dst:
            dst.write(floatMask,1)              #Write a floating mask geotiff
        demFlag = False
        for y in cpDict.keys():             #Find CPs within that DEM's bounds
            cpLoc = shpg.Point(cpDict[y])
            if poly.contains(cpLoc):
                xc = round((cpDict[y][0]-gt[2])/gt[0])      #Get grid location
                yc = round((cpDict[y][1]-gt[5])/gt[4]) - 1
                if floatMask[yc,xc] == 1:              #Check if the CP is floating
                    print('Control point ' + y + ' is floating.')
                    demFlag = True
                    dist = 1e15               #Find the closest grounded pixel
                    for i in range (floatMask.shape[0]):
                        for j in range (floatMask.shape[1]):
                            if not i == yc and not j == xc:
                                if floatMask[i,j] == 0:
                                    minDist = ((yc-i)**2 + (xc-j)**2)**0.5
                                    if minDist < dist: dist = minDist
                    floatDict[y] = dist                         #And record it
        if demFlag:
            floatDems.append(x + '\n')
    time_stamp('Floating Mask calculated ' + x)

    os.chdir(cwd + 'Step2\\' + prefix + 'Patches')
    path_maker(cwd + 'Step10\\' + prefix + 'Floating\\Patches')
    for x in floatDict.keys():
        time_stamp('Correcting CP ' + x)
        timeList = []
        with open(prefix + 'Patch' + x + '.txt', 'r') as f:
            content = f.readlines()
        s = content[0].replace('\n','').split()
        X, Y = float(s[2]), float(s[3])
        for y in content[1:]:
            s = y.replace('\n','').split()
            yr, day, mon = int(s[1][-2:]), int(s[1][-4:-2]), int(s[1][:-4])
            yr = 1900 + yr if yr > 50 else 2000 + yr
            hr = 0 if len(s[3]) < 5 else int(s[3][:-4])
            mins = 0 if len(s[3]) < 3 else int(s[3][-4:-2])
            sec = round(int(s[3][-2:]))
            dt = datetime.datetime(yr, mon, day, hr, mins, sec)
            utcDt = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
            ###delete this later
            X, Y = 443729.000, -1063997.000
            ###delete the thing above later!
            timeList.append([X, Y, utcDt - utcEp])
        timeList = np.array(timeList)
        tide = pyMTD.compute_tide_corrections(timeList[:,0],timeList[:,1],timeList[:,2],path,
                                              EPSG=demEPSG,EPOCH=epoch)
        #if not tide[int(len(tide)/2)] < 100:
        #    tide = pyMTD.compute_tide_corrections(timeList[:,0],timeList[:,1],timeList[:,2],path,
        #                                          EPSG=demEPSG,EPOCH=epoch,NEAREST=True)
        for i, y in enumerate(content[1:]):
            s = y.replace('\n','').split()
            if not tide.mask[i] and abs(tide[i]) < 50:
                tideCor = tide[i]
            else:
                tideCor = 0
            corr = float(s[6]) + float(s[7]) - tideCor
            #if not -1e6 < corr < 1e6:
            #    corr = float(s[6]) #+ float(s[7])
            newLine = '{:3}'.format(s[0]) + s2ff(s[1],0,9) + s2ff(s[2],4,12) +\
                     s2ff(s[3],0,9) + s2ff(s[4],3,14) + s2ff(s[5],3,14) +\
                     s2ff(corr,3,11) + s2ff(s[7],3,11) + '\n'
            content[i+1] = newLine
        outPath = cwd + 'Step10\\' + prefix + 'Floating\\'
        with open(outPath +'Patches\\' + prefix + 'Patch' + x + '.txt', 'w') as f:
            f.writelines(content)
    if not floatDems == []:
        with open(outPath + prefix + 'DEMList' + '.txt', 'w') as f:
            f.writelines(floatDems)

def step10a(cwd, prefix, referenceDate, climateMode, mosaicMode, refDate):
    step3(cwd, prefix, referenceDate, withDEM=1, flot=True)
    step7(cwd, prefix, referenceDate, climateMode=climateMode,
          mosaicMode=mosaicMode, refDate=refDate, flot=True)
    #step8(cwd, prefix, flot=True)
    #step9(cwd, prefix, flot=True)

##############################################################################
#### MOULINS Step 11: DEM Mosaicking #########################################
#### Find the optimal order for a mosaic. ####################################
##############################################################################

def step11(cwd, prefix, raw=False):
    from rasterio.warp import reproject, Resampling
    tstart = time.time()
    print('Mosaicking data...')
    path_maker(cwd + 'Step11\\' + prefix + 'Mosaic')
    try:               #if a mosaic order document exists, get order from that
        os.chdir(cwd + 'Step11\\' + prefix + 'Mosaic')
        dems = []
        with open(prefix + 'MosaicOrder.txt', 'r') as f:
            for x in f:
                dems.append(x.split()[1])
    except:                              #otherwise base order on RMS residual
        os.chdir(cwd + 'Step9\\' + prefix + 'Corrected')
        dems = os.listdir()
        dems, wgts = np.array(dems), []
        
        os.chdir(cwd + 'Step8\\' + prefix + 'Surface')
        time_stamp('Fetching stats')
        for x in dems:
            date = extract_num(x)
            with open(prefix + date + 'Stats.txt', 'r') as f:
                content = f.readlines()
            deg = content[-1].split()[-1]
            if deg == 'NA':
                for y in content:
                    if y != '\n' and y[:5] == 'RMS C':
                        wgts.append(10 + float(y.split()[-1]))
                        break
            else:
                i = content.index('Deg ' + deg + '\n')
                for y in content[i:]:
                    if y != '\n' and y[:5] == 'RMS R':
                        wgts.append(float(y.split()[-1]))
                        break
        wgts = np.array(wgts)
        dems = np.c_[dems, wgts]
        dems = dems[:,0][dems[:,1].argsort()]

        os.chdir(cwd + 'Step11\\' + prefix + 'Mosaic')
        with open(prefix + 'MosaicOrder.txt', 'w') as f:
            for i in range(len(dems)):
                f.write(str(i+1) + ' ' + dems[i] + '\n')

    os.chdir(cwd + 'Step9\\' + prefix + 'Corrected')
    surf0, _, ndv0, gt0, proj = tif_reader(dems[0], raw=True)
    xmax, ymin = gt0[2] + gt0[0] * surf0.shape[1], gt0[5] + gt0[4] * surf0.shape[0]
    lm0 = [gt0[2], xmax, ymin, gt0[5], gt0[0], -gt0[4]]
    lm1 = [gt0[2], xmax, ymin, gt0[5], gt0[0], -gt0[4]]
    ddic, ldic = {0: surf0}, {0: lm0}

    time_stamp('Reading DEMs')
    for i in range(1, len(dems)):
        surf, _, ndv, gt, proj = tif_reader(dems[i], raw=True)
        xmax, ymin = gt[2] + gt[0] * surf.shape[1], gt[5] + gt[4] * surf.shape[0]
        lm = [gt[2], xmax, ymin, gt[5], gt[0], -gt[4]]
        xadj, yadj = (gt0[2] - gt[2]) % gt0[0], (gt0[5] - gt[5]) % -gt0[4]
        lm[0], lm[1] = lm[0] + xadj, lm[1] + xadj
        lm[2], lm[3] = lm[2] + yadj, lm[3] + yadj
        lm1[0] = lm[0] if lm1[0] > lm[0] else lm1[0]
        lm1[1] = lm[1] if lm1[1] < lm[1] else lm1[1]
        lm1[2] = lm[2] if lm1[2] > lm[2] else lm1[2]
        lm1[3] = lm[3] if lm1[3] < lm[3] else lm1[3]
        if xadj == 0 and yadj == 0:
            surft = surf
        else:
            gtt = gt[0], gt[1], gt[2] + xadj, gt[3], gt[4], gt[5] + yadj
            surft = np.ones(surf.shape)
            reproject(surf, surft, src_transform=gt, src_crs=proj, src_nodata=ndv,
                      dst_transform=gtt, dst_crs=proj, dst_nodata=ndv0,
                      resampling=Resampling.bilinear)
        ddic[i], ldic[i] = surft, lm

    os.chdir(cwd + 'Step11\\' + prefix + 'Mosaic')
    lm1[1] += (lm1[1] - lm1[0]) % gt0[0]
    lm1[3] += (lm1[3] - lm1[2]) % -gt0[4]
    mosShp = (int((lm1[3] - lm1[2]) / lm1[5]), int((lm1[1] - lm1[0]) / lm1[4]))
    mosaic = np.full(mosShp, ndv0)
    dateKey = np.full(mosShp, ndv0)
    ind = [x for x in ddic]
    for x in reversed(ind):
        lm, surf, date = ldic[x], ddic[x], int(extract_num(dems[x]))
        time_stamp('Adding ' + dems[x] + ' to mosaic')
        xi, yi = int((lm[0] - lm1[0]) / lm[4]), int((lm1[3] - lm[3]) / lm[5])
        xf, yf = int((lm[1] - lm1[0]) / lm[4]), int((lm1[3] - lm[2]) / lm[5])
        for j in range(xf-xi):
            for i in range(yf-yi):
                if surf[i,j] > 0:
                    mosaic[i+yi, j+xi] = surf[i,j]
                    dateKey[i+yi, j+xi] = date

    gtf = gt0[0], 0.0, lm1[0], 0.0, gt0[4], lm1[3]
    with rasterio.open(prefix+'Mosaic.tif', 'w', driver='GTiff', height=mosShp[0], \
                       width=mosShp[1], count=1, dtype=mosaic.dtype, crs=proj, \
                       transform=gtf, nodata=ndv0) as dst:
        dst.write(mosaic,1)
    with rasterio.open(prefix+'DateKey.tif', 'w', driver='GTiff', height=mosShp[0], \
                       width=mosShp[1], count=1, dtype=mosaic.dtype, crs=proj, \
                       transform=gtf, nodata=ndv0) as dst:
        dst.write(dateKey,1)

    with open(prefix + 'MosaicOrder.txt', 'w') as f:
        for i in range(len(dems)):
            f.write(str(i+1) + ' ' + dems[i] + '\n')

    if raw:
        time_stamp('Making uncorrected mosaic')
        os.chdir(cwd + 'DEMs\\')
        demList = os.listdir()
        for x in demList:
            if extract_num(dems[0]) in x:
                demName = x
                break
        surf0, _, ndv0, gt0, proj = tif_reader(demName, raw=True)
        xmax, ymin = gt0[2] + gt0[0] * surf0.shape[1], gt0[5] + gt0[4] * surf0.shape[0]
        lm0 = [gt0[2], xmax, ymin, gt0[5], gt0[0], -gt0[4]]
        lm1 = [gt0[2], xmax, ymin, gt0[5], gt0[0], -gt0[4]]
        ddic, ldic = {0: surf0}, {0: lm0}

        for i in range(1, len(dems)):
            for x in demList:
                if extract_num(dems[i]) in x:
                    demName = x
                    break
            surf, _, ndv, gt, proj = tif_reader(demName, raw=True)
            xmax, ymin = gt[2] + gt[0] * surf.shape[1], gt[5] + gt[4] * surf.shape[0]
            lm = [gt[2], xmax, ymin, gt[5], gt[0], -gt[4]]
            xadj, yadj = (gt0[2] - gt[2]) % gt0[0], (gt0[5] - gt[5]) % -gt0[4]
            lm[0], lm[1] = lm[0] + xadj, lm[1] + xadj
            lm[2], lm[3] = lm[2] + yadj, lm[3] + yadj
            lm1[0] = lm[0] if lm1[0] > lm[0] else lm1[0]
            lm1[1] = lm[1] if lm1[1] < lm[1] else lm1[1]
            lm1[2] = lm[2] if lm1[2] > lm[2] else lm1[2]
            lm1[3] = lm[3] if lm1[3] < lm[3] else lm1[3]
            if xadj == 0 and yadj == 0:
                surft = surf
            else:
                gtt = gt[0], gt[1], gt[2] + xadj, gt[3], gt[4], gt[5] + yadj
                surft = np.ones(surf.shape)
                reproject(surf, surft, src_transform=gt, src_crs=proj, src_nodata=ndv,
                          dst_transform=gtt, dst_crs=proj, dst_nodata=ndv0,
                          resampling=Resampling.bilinear)
            ddic[i], ldic[i] = surft, lm

        lm1[1] += (lm1[1] - lm1[0]) % gt0[0]
        lm1[3] += (lm1[3] - lm1[2]) % -gt0[4]
        mosShp = (int((lm1[3] - lm1[2]) / lm1[5]), int((lm1[1] - lm1[0]) / lm1[4]))
        mosaic = np.full(mosShp, ndv0)
        ind = [x for x in ddic]
        for x in reversed(ind):
            lm, surf = ldic[x], ddic[x]
            time_stamp('Adding ' + demList[x] + ' to mosaic')
            xi, yi = int((lm[0] - lm1[0]) / lm[4]), int((lm1[3] - lm[3]) / lm[5])
            xf, yf = int((lm[1] - lm1[0]) / lm[4]), int((lm1[3] - lm[2]) / lm[5])
            for j in range(xf-xi):
                for i in range(yf-yi):
                    if surf[i,j] > 0:
                        mosaic[i+yi, j+xi] = surf[i,j]

        os.chdir(cwd + 'Step11\\' + prefix + 'Mosaic')
        gtf = gt0[0], 0.0, lm1[0], 0.0, gt0[4], lm1[3]
        with rasterio.open(prefix+'Mosaic_uncorr.tif', 'w', driver='GTiff', height=mosShp[0], \
                           width=mosShp[1], count=1, dtype=mosaic.dtype, crs=proj, \
                           transform=gtf, nodata=ndv0) as dst:
            dst.write(mosaic,1)

    os.chdir(cwd)
    tend = time.time()
    print('Step 11 elapsed time: ' + str(tend-tstart) + ' seconds.')
    time_stamp('Completed mosaicking DEMs')

##############################################################################
#### MOULINS Step 12: Altimetry Comparison ###################################
#### Compare corrected DEMs with cotemporal altimetry ########################
##############################################################################

def post_compare(prefix, directory, wgs_sl=28):
#Looks for altimetry data within three months of each DEM to compare them
#before and after correction.
    print('Beginning Step 12: DEM validation...')
    import matplotlib.pyplot as plt
    path = directory + 'Step12\\' + prefix + 'Comparison'
    path_maker(path)
    legend = [plt.Line2D([0], [0], color='r', lw=3, label='Altimetry'),
              plt.Line2D([0], [0], color='g', lw=3, label='Original DEM'),
              plt.Line2D([0], [0], color='b', lw=3, label='Corrected DEM')]
    
    time_stamp('Loading altimetry')
    with open('Altimetry\\' + prefix + '_alt.txt', 'r') as f:
        content = f.readlines()         #Read in all lines from altimetry file
    alts = []
    for x in content[1:]:
        y = x.split()
        alts.append([float(y[4]), float(y[5]), float(y[6]), float(y[2])])
    alts = np.array(alts)

    os.chdir(directory + 'Step9\\' + prefix + 'Corrected')
    dems = os.listdir()
    os.chdir(directory + 'DEMs')
    dems2 = os.listdir()

    for x in dems:
        date, subs = extract_num(x), []
        for y in dems2:
            if date in y:
                surf1,_,_,_,_ = tif_reader(y)
                print('Match found')
        os.chdir(directory + 'Step9\\' + prefix + 'Corrected')
        surf2, border,_, gt,_ = tif_reader(x, border=True)
        surf2 = np.c_[surf2, surf1[:,2]]
        bx,by = border.exterior.xy

        time_stamp('Comparing DEM on ' + date + ' with altimetry')
        dec_date = mmddyy2dec(date)
        temp = alts[np.where((alts[:,3] > dec_date - 0.25) & (alts[:,3] < dec_date + 0.25))]
        print(str(len(temp)) + ' altimetry points found...')
        xo, yo = 0, 0
        for y in temp:
            if hypot(y[0] - xo, y[1] - yo) > 500:
                if border.contains(shpg.Point(y[0], y[1])):
                    subs.append(y)
                    xo, yo = y[0], y[1]
        subs = np.array(subs)

        temp, image_list = [], []
        lm = [border.bounds[0], border.bounds[2], border.bounds[1], border.bounds[3], gt[0], -gt[4]]
        xmin, xmax, ymin, ymax = square_domain(lm[0], lm[1], lm[2], lm[3])
        if len(subs) > 0:
            xo, yo, num, points = 0, 0, 0, []
            for y in subs:
                xa, ya = y[0], y[1]
                xdem = int(((xa - lm[0]) / lm[4]) + 0.5) * lm[4] + lm[0]
                ydem = int(((ya - lm[2]) / lm[5]) + 0.5) * lm[5] + lm[2]
                data = surf2[np.where((surf2[:,0] == xdem) & (surf2[:,1]==ydem)
                                      & (surf2[:,2]>wgs_sl) & (surf2[:,3]>wgs_sl))]
                if len(data) == 0:
                    continue
                if hypot(xa - xo, ya - yo) > 5000:
                    num = num+1
                oz, cz, az = data[0][3], data[0][2], float(y[2])
                odiff, cdiff = az - oz, az - cz
                points.append([num, xa, ya, az, oz, cz, odiff, cdiff])
                xo, yo = xa, ya
            points = np.array(points)
            segs = np.unique(points[:,0])
            nseg = len(segs)
            points2 = points

            for y in segs:
                temp = points[np.where(points[:,0] == y)]
                if len(temp) < 20 or np.max(abs(temp[:,6])) > 15:
                    points2 = np.delete(points2, np.where(points2[:,0] == y), axis=0)
                    nseg = nseg - 1
                    continue
            
                temp = linear_sort(temp,1,2)
                dist, xo, yo, t_dist = [0], temp[0,1], temp[0,2], 0
                for z in temp[1:]:
                    t_dist += hypot(z[1] - xo, z[2] - yo)
                    dist.append(t_dist)
                    xo, yo = z[1], z[2]
                fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2,figsize=(20,20))
                ax1.set_xlim(xmin, xmax)
                ax1.set_ylim(ymin, ymax)
                ax1.plot(bx, by, c='k', linewidth=3)
                xlen = (temp[-1,1] - temp[0,1]) / 10
                ylen = (temp[-1,2] - temp[0,2]) / 10
                ax1.arrow(temp[0,1], temp[0,2], xlen, ylen, shape='full', width=100,
                          head_width=400, color='r')
                ax1.plot(temp[:,1], temp[:,2], c='c', linewidth=1)
                ax1.set_title('Profile location')
                if 'Out' in x:
                    ax2.plot(dist, temp[:,3], linewidth=1, c='r')
                    ax2.plot(dist,temp[:,5],linewidth=1,c='b',linestyle='--')
                    ax2.set_title('Corrected profile elevation vs altimetry')
                else:
                    ax2.plot(dist, temp[:,3], linewidth=1, c='r')
                    ax2.plot(dist, temp[:,4],linewidth=1,c='g',linestyle='--')
                    ax2.set_title('Uncorrected profile elevation vs altimetry')
                ax2.grid(True)
                ax2.legend(handles = legend)
                ax3.plot(dist, temp[:,6],linewidth=1,c='g',linestyle='--')
                if 'Out' in x:
                    ax3.plot(dist, temp[:,7], linewidth=1, c='b')
                ax3.set_title('Difference between altimetry and DEM')
                ax3.grid(True)
                aboz, abcz = abs(temp[:,6]), abs(temp[:,7])
                ax4.plot(dist, aboz,linewidth=1,c='g',linestyle='--')
                if 'Out' in x:
                    ax4.plot(dist, abcz, linewidth=1, c='b')
                ax4.grid(True)
                ax4.set_title('Abs Difference between altimetry and DEM')
                image_list.append(mat_2_pil(fig))
                plt.close()
            

            orms, crms = rms(points2[:,6]), rms(points2[:,7])
            oavg, cavg = np.mean(abs(points2[:,6])), np.mean(abs(points2[:,7]))

        with open(directory + 'Step8\\' + prefix + 'Surface\\' +\
                  prefix + date + 'Stats.txt', 'r') as f:
            temp = f.readlines()
        degree, cpnum = temp[-1].split()[-1], temp[-2].split()[-1]
        os.chdir(path)
        
        pdf = prefix + date + 'Plots.pdf'
        if len(image_list) == 1:
            image_list[0].save(pdf, "PDF" , resolution=100.0, save_all=True)
        elif len(image_list) > 0:
            image_list[0].save(pdf, "PDF" , resolution=100.0, save_all=True, 
                               append_images=image_list[1:])
        
        wstr = '*  *  *  *  *  *  *  *  *  *  *  *  *\n\nDEM ' + date
        if degree == 'NA':
            wstr += '\n\nPolynomial Degree used:          NA'
        else:
            wstr += '\n\nPolynomial Degree used:' + s2ff(degree,0,17)
        wstr += '\nNumber of control points:' + s2ff(cpnum,0,15)
        if len(subs) == 0:
            wstr += '\nNo Altimetry available for comparison.\n\n'
        else:
            wstr += '\nAltimetry segments found:' + s2ff(nseg,0,15) +\
                    '\nData points compared:' + s2ff(len(points),0,19) +\
                    '\nUncorrected mean abs. dif.:' + s2ff(oavg,3,13)
            if degree == 'NA':
                wstr += '\nCorrected sigma:                 NA\n\n'
            else:
                wstr += '\nCorrected mean abs. dif.:' + s2ff(cavg,3,15) +\
                        '\nUncorrected std. dev.:' + s2ff(orms,3,18) +\
                        '\nCorrected std. dev.:' + s2ff(crms,3,20) + '\n\n'

        with open(directory + 'Step12\\' + prefix + 'Comparison\\' + prefix + 'Report.txt', 'a+') as f:
            f.write(wstr)
        os.chdir(directory + 'DEMs')

    os.chdir(directory)
    time_stamp('Completed comparison with altimetry')