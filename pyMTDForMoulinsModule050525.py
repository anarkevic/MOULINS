# -*- coding: utf-8 -*-

import os
import pyproj
import numpy as np
import datetime
import time
import inspect
import scipy.interpolate
import gzip
import uuid
import netCDF4
import re

def convert_delta_time(delta_time, epoch1=None, epoch2=None, scale=1.0):
    epoch1 = datetime.datetime(*epoch1)
    epoch2 = datetime.datetime(*epoch2)
    delta_time_epochs = (epoch2 - epoch1).total_seconds()
    return scale*(delta_time - delta_time_epochs)

def get_data_path(relpath):
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    if isinstance(relpath,list):
        return os.path.join(filepath,*relpath)
    elif isinstance(relpath,str):
        return os.path.join(filepath,relpath)

def polynomial_sum(coefficients, t):
    t = np.atleast_1d(t)
    return np.sum([c * (t ** i) for i,c in enumerate(coefficients)],axis=0)

#### Get Leap Seconds ########################################################

leapSeconds = np.array([[2.2720608e+09, 10.0], [2.2877856e+09, 11.0],
                        [2.3036832e+09, 12.0], [2.3352192e+09, 13.0],
                        [2.3667552e+09, 14.0], [2.3982912e+09, 15.0],
                        [2.4299136e+09, 16.0], [2.4614496e+09, 17.0],
                        [2.4929856e+09, 18.0], [2.5245216e+09, 19.0],
                        [2.5717824e+09, 20.0], [2.6033184e+09, 21.0],
                        [2.6348544e+09, 22.0], [2.6980128e+09, 23.0],
                        [2.7769824e+09, 24.0], [2.8401408e+09, 25.0],
                        [2.8716768e+09, 26.0], [2.9189376e+09, 27.0],
                        [2.9504736e+09, 28.0], [2.9820096e+09, 29.0],
                        [3.0294432e+09, 30.0], [3.0767040e+09, 31.0],
                        [3.1241376e+09, 32.0], [3.3450624e+09, 33.0],
                        [3.4397568e+09, 34.0], [3.5500896e+09, 35.0],
                        [3.6446976e+09, 36.0], [3.6922176e+09, 37.0]])
expiry = 3.8496384e+09
today = time.time() + 2208988800
if (expiry < today):
    print('Warning: leap second list out of date.')
leap_UTC,TAI_UTC = leapSeconds.T
TAI_GPS = 19.0
leap_GPS = convert_delta_time(leap_UTC+TAI_UTC-TAI_GPS-1,
           epoch1=(1900,1,1,0,0,0), epoch2=(1980,1,6,0,0,0))
leaps = leap_GPS[leap_GPS >= 0].astype(np.float64)

##############################################################################

def count_leap_seconds(GPS_Time):
    n_leaps = np.zeros_like(GPS_Time,dtype=np.float64)
    for i,leap in enumerate(leaps):
        count = np.count_nonzero(GPS_Time >= leap)
        if (count > 0):
            indices = np.nonzero(GPS_Time >= leap)
            n_leaps[indices] += 1.0
    return n_leaps

def convert_calendar_dates(year, month, day, hour=0.0, minute=0.0, second=0.0,
    epoch=(1992,1,1,0,0,0), scale=1.0):
    MJD = 367.0*year - np.floor(7.0*(year + np.floor((month+9.0)/12.0))/4.0) - \
        np.floor(3.0*(np.floor((year + (month - 9.0)/7.0)/100.0) + 1.0)/4.0) + \
        np.floor(275.0*month/9.0) + day + hour/24.0 + minute/1440.0 + \
        second/86400.0 + 1721028.5 - 2400000.5
    epoch1 = datetime.datetime(1858,11,17,0,0,0)
    epoch2 = datetime.datetime(*epoch)
    delta_time_epochs = (epoch2 - epoch1).total_seconds()
    return scale*np.array(MJD - delta_time_epochs/86400.0,dtype=np.float64)

def calc_delta_time(delta_file,idays):
    dinput = np.loadtxt(os.path.expanduser(delta_file))
    days = convert_calendar_dates(dinput[:,0],dinput[:,1],dinput[:,2],
           epoch=(1992,1,1,0,0,0))
    spl = scipy.interpolate.UnivariateSpline(days,dinput[:,3],k=1,s=0,ext=0)
    return spl(idays)/86400.0

def calc_astrol_longitudes(MJD, MEEUS=False, ASTRO5=False):
    circle = 360.0
    if MEEUS:
        T = MJD - 51544.5
        lunar_longitude = np.array([218.3164591, 13.17639647754579,
            -9.9454632e-13, 3.8086292e-20, -8.6184958e-27])
        s = polynomial_sum(lunar_longitude,T)
        solar_longitude = np.array([280.46645, 0.985647360164271,
            2.2727347e-13])
        h = polynomial_sum(solar_longitude,T)
        lunar_perigee = np.array([83.3532430, 0.11140352391786447,
            -7.7385418e-12, -2.5636086e-19, 2.95738836e-26])
        p = polynomial_sum(lunar_perigee,T)
        lunar_node = np.array([125.0445550, -0.052953762762491446,
            1.55628359e-12, 4.390675353e-20, -9.26940435e-27])
        N, PP = polynomial_sum(lunar_node,T), 282.94 + 1.7192 * T
    elif ASTRO5:
        T = (MJD - 51544.5)/36525.0
        lunar_longitude = np.array([218.3164477, 481267.88123421, -1.5786e-3,
             1.855835e-6, -1.53388e-8])
        s = polynomial_sum(lunar_longitude,T)
        lunar_elongation = np.array([297.8501921, 445267.1114034, -1.8819e-3,
             1.83195e-6, -8.8445e-9])
        h = polynomial_sum(lunar_longitude-lunar_elongation,T)
        lunar_perigee = np.array([83.3532465, 4069.0137287, -1.032e-2,
            -1.249172e-5])
        p = polynomial_sum(lunar_perigee,T)
        lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
        N, PP = polynomial_sum(lunar_node,T), 282.94 + 1.7192 * T
    else:
        T, PP = MJD - 51544.4993, 282.8
        s, h = 218.3164 + 13.17639648 * T, 280.4661 + 0.98564736 * T
        p, N =  83.3535 + 0.11140353 * T, 125.0445 - 0.05295377 * T

    s, h = np.mod(s, circle), np.mod(h, circle)
    p, N = np.mod(p, circle), np.mod(N, circle)
    return (s,h,p,N,PP)

def infer_minor_corrections(t,zmajor,constituents,DELTAT=0.0,CORRECTIONS=''):
    dtr = np.pi/180.0
    npts,nc = np.shape(zmajor)
    nt = len(np.atleast_1d(t))
    n = nt if ((npts == 1) & (nt > 1)) else npts
    dh = np.ma.zeros((n))
    MJD = 48622.0 + t
    cindex = ['q1','o1','p1','k1','n2','m2','s2','k2','2n2']
    z = np.ma.zeros((n,9),dtype=np.complex64)
    nz = 0
    for i,c in enumerate(cindex):
        j = [j for j,val in enumerate(constituents) if (val == c)]
        if j:
            j1, = j
            z[:,i] = zmajor[:,j1]
            nz += 1
    if (nz < 6):
        raise Exception('Not enough constituents for inference')
    minor = ['2q1','sigma1','rho1','m12','m11','chi1','pi1','phi1','theta1',
        'j1','oo1','2n2','mu2','nu2','lambda2','l2','l2','t2','eps2','eta2']
    minor_indices = [i for i,m in enumerate(minor) if m not in constituents]

    zmin = np.zeros((n,20),dtype=np.complex64)
    zmin[:,0] = 0.263*z[:,0] - 0.0252*z[:,1]#-- 2Q1
    zmin[:,1] = 0.297*z[:,0] - 0.0264*z[:,1]#-- sigma1
    zmin[:,2] = 0.164*z[:,0] + 0.0048*z[:,1]#-- rho1
    zmin[:,3] = 0.0140*z[:,1] + 0.0101*z[:,3]#-- M12
    zmin[:,4] = 0.0389*z[:,1] + 0.0282*z[:,3]#-- M11
    zmin[:,5] = 0.0064*z[:,1] + 0.0060*z[:,3]#-- chi1
    zmin[:,6] = 0.0030*z[:,1] + 0.0171*z[:,3]#-- pi1
    zmin[:,7] = -0.0015*z[:,1] + 0.0152*z[:,3]#-- phi1
    zmin[:,8] = -0.0065*z[:,1] + 0.0155*z[:,3]#-- theta1
    zmin[:,9] = -0.0389*z[:,1] + 0.0836*z[:,3]#-- J1
    zmin[:,10] = -0.0431*z[:,1] + 0.0613*z[:,3]#-- OO1
    zmin[:,11] = 0.264*z[:,4] - 0.0253*z[:,5]#-- 2N2
    zmin[:,12] = 0.298*z[:,4] - 0.0264*z[:,5]#-- mu2
    zmin[:,13] = 0.165*z[:,4] + 0.00487*z[:,5]#-- nu2
    zmin[:,14] = 0.0040*z[:,5] + 0.0074*z[:,6]#-- lambda2
    zmin[:,15] = 0.0131*z[:,5] + 0.0326*z[:,6]#-- L2
    zmin[:,16] = 0.0033*z[:,5] + 0.0082*z[:,6]#-- L2
    zmin[:,17] = 0.0585*z[:,6]#-- t2
    if CORRECTIONS in ('FES',):
        mu2 = [0.069439968323, 0.351535557706, -0.046278307672]
        nu2 = [-0.006104695053, 0.156878802427, 0.006755704028]
        l2 = [0.077137765667, -0.051653455134, 0.027869916824]
        t2 = [0.180480173707, -0.020101177502, 0.008331518844]
        lda2 = [0.016503557465, -0.013307812292, 0.007753383202]
        zmin[:,12] = mu2[0]*z[:,7] + mu2[1]*z[:,4] + mu2[2]*z[:,5]#-- mu2
        zmin[:,13] = nu2[0]*z[:,7] + nu2[1]*z[:,4] + nu2[2]*z[:,5]#-- nu2
        zmin[:,14] = lda2[0]*z[:,7] + lda2[1]*z[:,4] + lda2[2]*z[:,5]#-- lambda2
        zmin[:,16] = l2[0]*z[:,7] + l2[1]*z[:,4] + l2[2]*z[:,5]#-- L2
        zmin[:,17] = t2[0]*z[:,7] + t2[1]*z[:,4] + t2[2]*z[:,5]#-- t2
        zmin[:,18] = 0.53285*z[:,8] - 0.03304*z[:,4]#-- eps2
        zmin[:,19] = -0.0034925*z[:,5] + 0.0831707*z[:,7]#-- eta2

    hour = (t % 1)*24.0
    t1 = 15.0*hour
    t2 = 30.0*hour
    ASTRO5 = True if CORRECTIONS in ('GOT','FES') else False
    S,H,P,omega,pp = calc_astrol_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)

    arg = np.zeros((n,20))
    arg[:,0] = t1 - 4.0*S + H + 2.0*P - 90.0#-- 2Q1
    arg[:,1] = t1 - 4.0*S + 3.0*H - 90.0#-- sigma1
    arg[:,2] = t1 - 3.0*S + 3.0*H - P - 90.0#-- rho1
    arg[:,3] = t1 - S + H - P + 90.0#-- M12
    arg[:,4] = t1 - S + H + P + 90.0#-- M11
    arg[:,5] = t1 - S + 3.0*H - P + 90.0#-- chi1
    arg[:,6] = t1 - 2.0*H + pp - 90.0#-- pi1
    arg[:,7] = t1 + 3.0*H + 90.0#-- phi1
    arg[:,8] = t1 + S - H + P + 90.0#-- theta1
    arg[:,9] = t1 + S + H - P + 90.0#-- J1
    arg[:,10] = t1 + 2.0*S + H + 90.0#-- OO1
    arg[:,11] = t2 - 4.0*S + 2.0*H + 2.0*P#-- 2N2
    arg[:,12] = t2 - 4.0*S + 4.0*H#-- mu2
    arg[:,13] = t2 - 3.0*S + 4.0*H - P#-- nu2
    arg[:,14] = t2 - S + P + 180.0#-- lambda2
    arg[:,15] = t2 - S + 2.0*H - P + 180.0#-- L2
    arg[:,16] = t2 - S + 2.0*H + P#-- L2
    arg[:,17] = t2 - H + pp#-- t2
    arg[:,18] = t2 - 5.0*S + 4.0*H + P #-- eps2
    arg[:,19] = t2 + S + 2.0*H - pp #-- eta2

    sinn = np.sin(omega*dtr)
    cosn = np.cos(omega*dtr)
    sin2n = np.sin(2.0*omega*dtr)
    cos2n = np.cos(2.0*omega*dtr)

    f = np.ones((n,20))
    f[:,0] = np.sqrt((1.0 + 0.189*cosn - 0.0058*cos2n)**2 +
        (0.189*sinn - 0.0058*sin2n)**2)#-- 2Q1
    f[:,1] = f[:,0]#-- sigma1
    f[:,2] = f[:,0]#-- rho1
    f[:,3] = np.sqrt((1.0 + 0.185*cosn)**2 + (0.185*sinn)**2)#-- M12
    f[:,4] = np.sqrt((1.0 + 0.201*cosn)**2 + (0.201*sinn)**2)#-- M11
    f[:,5] = np.sqrt((1.0 + 0.221*cosn)**2 + (0.221*sinn)**2)#-- chi1
    f[:,9] = np.sqrt((1.0 + 0.198*cosn)**2 + (0.198*sinn)**2)#-- J1
    f[:,10] = np.sqrt((1.0 + 0.640*cosn + 0.134*cos2n)**2 +
        (0.640*sinn + 0.134*sin2n)**2)#-- OO1
    f[:,11] = np.sqrt((1.0 - 0.0373*cosn)**2 + (0.0373*sinn)**2)#-- 2N2
    f[:,12] = f[:,11]#-- mu2
    f[:,13] = f[:,11]#-- nu2
    f[:,15] = f[:,11]#-- L2
    f[:,16] = np.sqrt((1.0 + 0.441*cosn)**2 + (0.441*sinn)**2)#-- L2

    u = np.zeros((n,20))
    u[:,0] = np.arctan2(0.189*sinn - 0.0058*sin2n,
        1.0 + 0.189*cosn - 0.0058*sin2n)/dtr#-- 2Q1
    u[:,1] = u[:,0]#-- sigma1
    u[:,2] = u[:,0]#-- rho1
    u[:,3] = np.arctan2( 0.185*sinn, 1.0 + 0.185*cosn)/dtr#-- M12
    u[:,4] = np.arctan2(-0.201*sinn, 1.0 + 0.201*cosn)/dtr#-- M11
    u[:,5] = np.arctan2(-0.221*sinn, 1.0 + 0.221*cosn)/dtr#-- chi1
    u[:,9] = np.arctan2(-0.198*sinn, 1.0 + 0.198*cosn)/dtr#-- J1
    u[:,10] = np.arctan2(-0.640*sinn - 0.134*sin2n,
        1.0 + 0.640*cosn + 0.134*cos2n)/dtr#-- OO1
    u[:,11] = np.arctan2(-0.0373*sinn, 1.0 - 0.0373*cosn)/dtr#-- 2N2
    u[:,12] = u[:,11]#-- mu2
    u[:,13] = u[:,11]#-- nu2
    u[:,15] = u[:,11]#-- L2
    u[:,16] = np.arctan2(-0.441*sinn, 1.0 + 0.441*cosn)/dtr#-- L2

    if CORRECTIONS in ('FES',):
        II = np.arccos(0.913694997 - 0.035692561*np.cos(omega*dtr))
        at1 = np.arctan(1.01883*np.tan(omega*dtr/2.0))
        at2 = np.arctan(0.64412*np.tan(omega*dtr/2.0))
        xi = -at1 - at2 + omega*dtr
        xi[xi > np.pi] -= 2.0*np.pi
        nu = at1 - at2
        I2 = np.tan(II/2.0)
        Ra1 = np.sqrt(1.0 - 12.0*(I2**2)*np.cos(2.0*(P - xi)) + 36.0*(I2**4))
        P2 = np.sin(2.0*(P - xi))
        Q2 = 1.0/(6.0*(I2**2)) - np.cos(2.0*(P - xi))
        R = np.arctan(P2/Q2)

        f[:,0] = np.sin(II)*(np.cos(II/2.0)**2)/0.38 #-- 2Q1
        f[:,1] = f[:,0] #-- sigma1
        f[:,2] = f[:,0] #-- rho1
        f[:,3] = f[:,0] #-- M12
        f[:,4] = np.sin(2.0*II)/0.7214 #-- M11
        f[:,5] = f[:,4] #-- chi1
        f[:,9] = f[:,5] #-- J1
        f[:,10] = np.sin(II)*np.power(np.sin(II/2.0),2.0)/0.01640 #-- OO1
        f[:,11] = np.power(np.cos(II/2.0),4.0)/0.9154 #-- 2N2
        f[:,12] = f[:,11] #-- mu2
        f[:,13] = f[:,11] #-- nu2
        f[:,14] = f[:,11] #-- lambda2
        f[:,15] = f[:,11]*Ra1 #-- L2
        f[:,18] = f[:,11] #-- eps2
        f[:,19] = np.power(np.sin(II),2.0)/0.1565 #-- eta2

        u[:,0] = (2.0*xi - nu)/dtr #-- 2Q1
        u[:,1] = u[:,0] #-- sigma1
        u[:,2] = u[:,0] #-- rho1
        u[:,3] = u[:,0] #-- M12
        u[:,4] = -nu/dtr #-- M11
        u[:,5] = u[:,4] #-- chi1
        u[:,9] = u[:,4] #-- J1
        u[:,10] = (-2.0*xi - nu)/dtr #-- OO1
        u[:,11] = (2.0*xi - 2.0*nu)/dtr #-- 2N2
        u[:,12] = u[:,11] #-- mu2
        u[:,13] = u[:,11] #-- nu2
        u[:,14] = (2.0*xi - 2.0*nu)/dtr #-- lambda2
        u[:,15] = (2.0*xi - 2.0*nu - R)/dtr#-- L2
        u[:,18] = u[:,12] #-- eps2
        u[:,19] = -2.0*nu/dtr #-- eta2

    for k in minor_indices:
        th = (arg[:,k] + u[:,k])*dtr
        dh += zmin.real[:,k]*f[:,k]*np.cos(th)-zmin.imag[:,k]*f[:,k]*np.sin(th)
    return dh

def load_constituent(c):
    cindex = ['m2','s2','k1','o1','n2','p1','k2','q1','2n2','mu2','nu2','l2',
        't2','j1','m1','oo1','rho1','mf','mm','ssa','m4','ms4','mn4','m6','m8',
        'mk3','s6','2sm2','2mk3']
    species_all = np.array([2,2,1,1,2,1,2,1,2,2,2,2,2,1,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0])
    alpha_all = np.array([0.693,0.693,0.736,0.695,0.693,0.706,0.693,0.695,0.693,
        0.693,0.693,0.693,0.693,0.695,0.695,0.695,0.695,0.693,0.693,0.693,
        0.693,0.693,0.693,0.693,0.693,0.693,0.693,0.693,0.693])
    omega_all = np.array([1.405189e-04,1.454441e-04,7.292117e-05,6.759774e-05,
        1.378797e-04,7.252295e-05,1.458423e-04,6.495854e-05,1.352405e-04,
        1.355937e-04,1.382329e-04,1.431581e-04,1.452450e-04,7.556036e-05,
        7.028195e-05,7.824458e-05,6.531174e-05,0.053234e-04,0.026392e-04,
        0.003982e-04,2.810377e-04,2.859630e-04,2.783984e-04,4.215566e-04,
        5.620755e-04,2.134402e-04,4.363323e-04,1.503693e-04,2.081166e-04])
    phase_all = np.array([1.731557546,0.000000000,0.173003674,1.558553872,
        6.050721243,6.110181633,3.487600001,5.877717569,4.086699633,
        3.463115091,5.427136701,0.553986502,0.052841931,2.137025284,
        2.436575100,1.929046130,5.254133027,1.756042456,1.964021610,
        3.487600001,3.463115091,1.731557546,1.499093481,5.194672637,
        6.926230184,1.904561220,0.000000000,4.551627762,3.809122439])
    amplitude_all = np.array([0.2441,0.112743,0.141565,0.100661,0.046397,
        0.046848,0.030684,0.019273,0.006141,0.007408,0.008811,0.006931,0.006608,
        0.007915,0.007915,0.004338,0.003661,0.042041,0.022191,0.019567,0.,0.,0.,
        0.,0.,0.,0.,0.,0.])
    j = [j for j,val in enumerate(cindex) if (val == c.lower())]
    if j:
        amplitude, = amplitude_all[j]
        phase, = phase_all[j]
        omega, = omega_all[j]
        alpha, = alpha_all[j]
        species, = species_all[j]
    else:
        amplitude = 0.0; phase = 0.0; omega = 0.0; alpha = 0.0; species = 0
    return (amplitude,phase,omega,alpha,species)

def load_nodal_corrections(MJD,constituents,DELTAT=0.0,CORRECTIONS='OTIS'):
    cindex = ['sa','ssa','mm','msf','mf','mt','alpha1','2q1','sigma1','q1',
        'rho1','o1','tau1','m1','chi1','pi1','p1','s1','k1','psi1','phi1',
        'theta1','j1','oo1','2n2','mu2','n2','nu2','m2a','m2','m2b','lambda2',
        'l2','t2','s2','r2','k2','eta2','mns2','2sm2','m3','mk3','s3','mn4',
        'm4','ms4','mk4','s4','s5','m6','s6','s7','s8','m8','mks2','msqm',
        'mtm','n4','eps2','z0']
    ASTRO5 = True if CORRECTIONS in ('GOT','FES') else False
    dtr, hour = np.pi/180.0, (MJD % 1)*24.0
    s,h,p,omega,pp = calc_astrol_longitudes(MJD+DELTAT, ASTRO5=ASTRO5)
    t1, t2, nt = 15.0*hour, 30.0*hour, len(np.atleast_1d(MJD))

    arg = np.zeros((nt,60))
    arg[:,0], arg[:,1] = h - pp, 2.0*h #-- Sa, Ssa
    arg[:,2], arg[:,3] = s - p, 2.0*s - 2.0*h #-- Mm, MSf
    arg[:,4], arg[:,5] = 2.0*s, 3.0*s - p #-- Mf, Mt
    arg[:,6] = t1 - 5.0*s + 3.0*h + p - 90.0 #-- alpha1
    arg[:,7] = t1 - 4.0*s + h + 2.0*p - 90.0 #-- 2Q1
    arg[:,8] = t1 - 4.0*s + 3.0*h - 90.0 #-- sigma1
    arg[:,9] = t1 - 3.0*s + h + p - 90.0 #-- q1
    arg[:,10] = t1 - 3.0*s + 3.0*h - p - 90.0 #-- rho1
    arg[:,11] = t1 - 2.0*s + h - 90.0 #-- o1
    arg[:,12] = t1 - 2.0*s + 3.0*h + 90.0 #-- tau1
    arg[:,13] = t1 - s + h + 90.0 #-- M1
    arg[:,14] = t1 - s + 3.0*h - p + 90.0 #-- chi1
    arg[:,15] = t1 - 2.0*h + pp - 90.0 #-- pi1
    arg[:,16] = t1 - h - 90.0 #-- p1
    if CORRECTIONS in ('OTIS','ATLAS','netcdf'):
        arg[:,17] = t1 + 90.0 #-- s1
    elif CORRECTIONS in ('GOT','FES'):
        arg[:,17] = t1 + 180.0 #-- s1 (Doodson's phase)
    arg[:,18] = t1 + h + 90.0 #-- k1
    arg[:,19] = t1 + 2.0*h - pp + 90.0 #-- psi1
    arg[:,20] = t1 + 3.0*h + 90.0 #-- phi1
    arg[:,21] = t1 + s - h + p + 90.0 #-- theta1
    arg[:,22] = t1 + s + h - p + 90.0 #-- J1
    arg[:,23] = t1 + 2.0*s + h + 90.0 #-- OO1
    arg[:,24] = t2 - 4.0*s + 2.0*h + 2.0*p #-- 2N2
    arg[:,25] = t2 - 4.0*s + 4.0*h #-- mu2
    arg[:,26] = t2 - 3.0*s + 2.0*h + p #-- n2
    arg[:,27] = t2 - 3.0*s + 4.0*h - p #-- nu2
    arg[:,28] = t2 - 2.0*s + h + pp #-- M2a
    arg[:,29] = t2 - 2.0*s + 2.0*h #-- M2
    arg[:,30] = t2 - 2.0*s + 3.0*h - pp #-- M2b
    arg[:,31] = t2 - s + p + 180.0 #-- lambda2
    arg[:,32] = t2 - s + 2.0*h - p + 180.0 #-- L2
    arg[:,33], arg[:,34] = t2 - h + pp, t2 #-- T2, S2
    arg[:,35] = t2 + h - pp + 180.0 #-- R2
    arg[:,36] = t2 + 2.0*h #-- K2
    arg[:,37] = t2 + s + 2.0*h - pp #-- eta2
    arg[:,38] = t2 - 5.0*s + 4.0*h + p #-- MNS2
    arg[:,39] = t2 + 2.0*s - 2.0*h #-- 2SM2
    arg[:,40] = 1.5*arg[:,29] #-- M3
    arg[:,41] = arg[:,18] + arg[:,29] #-- MK3
    arg[:,42] = 3.0*t1 #-- S3
    arg[:,43] = arg[:,26] + arg[:,29] #-- MN4
    arg[:,44] = 2.0*arg[:,29] #-- M4
    arg[:,45] = arg[:,29] + arg[:,34] #-- MS4
    arg[:,46] = arg[:,29] + arg[:,36] #-- MK4
    arg[:,47], arg[:,48] = 4.0*t1, 5.0*t1 #-- S4, S5
    arg[:,49], arg[:,50] = 3.0*arg[:,29], 3.0*t2 #-- M6, S6
    arg[:,51], arg[:,52] = 7.0*t1, 4.0*t2 #-- S7, S8
    arg[:,53] = 4.0*arg[:,29] #-- m8
    arg[:,54] = arg[:,29] + arg[:,36] - arg[:,34] #-- mks2
    arg[:,55], arg[:,56] = 4.0*s - 2.0*h, 3.0*s - p #-- msqm, mtm
    arg[:,57] = 2.0*arg[:,26] #-- n4
    arg[:,58], arg[:,59] = t2 - 5.0*s + 4.0*h + p, 0.0 #-- eps2, Z0

    sinn, cosn = np.sin(omega*dtr), np.cos(omega*dtr)
    sin2n, cos2n = np.sin(2.0*omega*dtr), np.cos(2.0*omega*dtr)
    sin3n = np.sin(3.0*omega*dtr)
    f = np.zeros((nt,60))
    u = np.zeros((nt,60))
    if CORRECTIONS in ('OTIS','ATLAS','netcdf'):
        f[:,0], f[:,1] = 1.0, 1.0 #-- Sa, Ssa
        f[:,2], f[:,3] = 1.0 - 0.130*cosn, 1.0 #-- Mm, MSf
        f[:,4] = 1.043 + 0.414*cosn #-- Mf
        temp1 = (1.0 + 0.203*cosn + 0.040*cos2n)**2
        temp2 = (0.203*sinn + 0.040*sin2n)**2
        f[:,5], f[:,6] = np.sqrt(temp1 + temp2), 1.0 #-- Mt, alpha1
        f[:,7] = np.sqrt((1.0 + 0.188*cosn)**2 + (0.188*sinn)**2) #-- 2Q1
        f[:,8], f[:,9], f[:,10] = f[:,7], f[:,7], f[:,7] #-- sigma1, q1, rho1
        temp1 = (1.0 + 0.189*cosn - 0.0058*cos2n)**2
        temp2 = (0.189*sinn - 0.0058*sin2n)**2
        f[:,11], f[:,12] = np.sqrt(temp1 + temp2), 1.0 #-- O1, tau1
        Mtmp1 = 1.36*np.cos(p*dtr) + 0.267*np.cos((p-omega)*dtr)
        Mtmp2 = 0.64*np.sin(p*dtr) + 0.135*np.sin((p-omega)*dtr)
        f[:,13] = np.sqrt(Mtmp1**2 + Mtmp2**2) #-- M1
        f[:,14] = np.sqrt((1.0+0.221*cosn)**2+(0.221*sinn)**2) #-- chi1
        f[:,15], f[:,16], f[:,17] = 1.0, 1.0, 1.0 #-- pi1, P1, S1
        temp1 = (1.0 + 0.1158*cosn - 0.0029*cos2n)**2
        temp2 = (0.1554*sinn - 0.0029*sin2n)**2
        f[:,18] = np.sqrt(temp1 + temp2) #-- K1
        f[:,19], f[:,20], f[:,21] = 1.0, 1.0, 1.0 #-- psi1, ph1, theta1
        f[:,22] = np.sqrt((1.0+0.169*cosn)**2 + (0.227*sinn)**2) #-- J1
        temp1 = (1.0 + 0.640*cosn + 0.134*cos2n)**2
        temp2 = (0.640*sinn + 0.134*sin2n)**2
        f[:,23] = np.sqrt(temp1 + temp2) #-- OO1
        temp1 = (1.0 - 0.03731*cosn + 0.00052*cos2n)**2
        temp2 = (0.03731*sinn - 0.00052*sin2n)**2
        f[:,24] = np.sqrt(temp1 + temp2) #-- 2N2
        f[:,25], f[:,26], f[:,27] = f[:,24], f[:,24], f[:,24] #-- mu2, N2, nu2
        f[:,28], f[:,29] = 1.0, f[:,24] #-- M2a, M2
        f[:,30], f[:,31] = 1.0, 1.0 #-- M2b, lambda2
        Ltmp1 = 1.0 - 0.25*np.cos(2*p*dtr) - 0.11*np.cos((2.0*p-omega)*dtr) - 0.04*cosn
        Ltmp2 = 0.25*np.sin(2*p*dtr) + 0.11*np.sin((2.0*p-omega)*dtr) + 0.04*sinn
        f[:,32] = np.sqrt(Ltmp1**2 + Ltmp2**2) #-- L2
        f[:,33], f[:,34], f[:,35] = 1.0, 1.0, 1.0 #-- T2, S2, R2
        temp1 = (1.0 + 0.2852*cosn + 0.0324*cos2n)**2
        temp2 = (0.3108*sinn + 0.0324*sin2n)**2
        f[:,36] = np.sqrt(temp1 + temp2) #-- K2
        f[:,37] = np.sqrt((1.0 + 0.436*cosn)**2 + (0.436*sinn)**2) #-- eta2
        f[:,38], f[:,39] = f[:,29]**2, f[:,29] #-- MNS2, 2SM2
        f[:,40], f[:,41] = 1.0, f[:,18]*f[:,29] #-- M3 (wrong), MK3
        f[:,42], f[:,43] = 1.0, f[:,29]**2 #-- S3, MN4
        f[:,44], f[:,45] = f[:,43], f[:,43] #-- M4, MS4
        f[:,46] = f[:,29]*f[:,36] #-- MK4
        f[:,47], f[:,48] = 1.0, 1.0 #-- S4, S5
        f[:,49] = f[:,29]**3 #-- M6
        f[:,50], f[:,51], f[:,52] = 1.0, 1.0, 1.0 #-- S6, S7, S8
        f[:,53] = f[:,29]**4 #-- m8
        f[:,54] = f[:,29]*f[:,36] #-- mks2
        f[:,55], f[:,56] = f[:,4], f[:,4] #-- msqm, mtm
        f[:,57] = f[:,29]**2 #-- n4
        f[:,58], f[:,59] = f[:,29], 1.0 #-- eps2, Z0

        u[:,0], u[:,1] = 0.0, 0.0 #-- Sa, Ssa
        u[:,2], u[:,3] = 0.0, 0.0 #-- Mm, MSf
        u[:,4] = -23.7*sinn + 2.7*sin2n - 0.4*sin3n #-- Mf
        temp1 = -(0.203*sinn + 0.040*sin2n)
        temp2 = (1.0 + 0.203*cosn + 0.040*cos2n)
        u[:,5] = np.arctan(temp1/temp2)/dtr #-- Mt
        u[:,6] = 0.0 #-- alpha1
        u[:,7] = np.arctan(0.189*sinn/(1.0 + 0.189*cosn))/dtr #-- 2Q1
        u[:,8], u[:,9], u[:,10] = u[:,7], u[:,7], u[:,7] #-- sigma1, q1, rho1
        u[:,11] = 10.8*sinn - 1.3*sin2n + 0.2*sin3n #-- O1
        u[:,12] = 0.0 #-- tau1
        u[:,13] = np.arctan2(Mtmp2,Mtmp1)/dtr #-- M1
        u[:,14] = np.arctan(-0.221*sinn/(1.0+0.221*cosn))/dtr #-- chi1
        u[:,15], u[:,16], u[:,17] = 0.0, 0.0, 0.0 #-- pi1, P1 S1
        temp1 = (-0.1554*sinn + 0.0029*sin2n)
        temp2 = (1.0 + 0.1158*cosn - 0.0029*cos2n)
        u[:,18] = np.arctan(temp1/temp2)/dtr #-- K1
        u[:,19], u[:,20], u[:,21] = 0.0, 0.0, 0.0 #-- psi1, phi1, theta1
        u[:,22] = np.arctan(-0.227*sinn/(1.0+0.169*cosn))/dtr #-- J1
        temp1 = -(0.640*sinn + 0.134*sin2n)
        temp2 = (1.0 + 0.640*cosn + 0.134*cos2n)
        u[:,23] = np.arctan(temp1/temp2)/dtr #-- OO1
        temp1 = (-0.03731*sinn + 0.00052*sin2n)
        temp2 = (1.0 - 0.03731*cosn + 0.00052*cos2n)
        u[:,24] = np.arctan(temp1/temp2)/dtr #-- 2N2
        u[:,25], u[:,26], u[:,27] = u[:,24], u[:,24], u[:,24] #-- mu2, N2, nu2
        u[:,28], u[:,29] = 0.0, u[:,24] #-- M2a, M2
        u[:,30], u[:,31] = 0.0, 0.0 #-- M2b, lambda2
        u[:,32] = np.arctan(-Ltmp2/Ltmp1)/dtr #-- L2
        u[:,33], u[:,34], u[:,35] = 0.0, 0.0, 0.0 #-- T2, S2, R2
        temp1 = -(0.3108*sinn+0.0324*sin2n)
        temp2 = (1.0 + 0.2852*cosn + 0.0324*cos2n)
        u[:,36] = np.arctan(temp1/temp2)/dtr #-- K2
        u[:,37] = np.arctan(-0.436*sinn/(1.0 + 0.436*cosn))/dtr #-- eta2
        u[:,38], u[:,39] = u[:,29]*2.0, u[:,29] #-- MNS2, 2SM2
        u[:,40] = 1.50*u[:,29] #-- M3
        u[:,41] = u[:,29] + u[:,18] #-- MK3
        u[:,42], u[:,43] = 0.0, 2.0*u[:,29] #-- S3, MN4
        u[:,44], u[:,45] = u[:,43], u[:,29] #-- M4, MS4
        u[:,46] = u[:,29] + u[:,36] #-- MK4
        u[:,47], u[:,48] = 0.0, 0.0 #-- S4, S5
        u[:,49] = 3.0*u[:,29] #-- M6
        u[:,50], u[:,51] = 0.0, 0.0 #-- S6, S7
        u[:,52], u[:,59] = 0.0, 0.0 #-- S8, Z0

    elif CORRECTIONS in ('FES',):
        II = np.arccos(0.913694997 - 0.035692561*np.cos(omega*dtr))
        at1 = np.arctan(1.01883*np.tan(omega*dtr/2.0))
        at2 = np.arctan(0.64412*np.tan(omega*dtr/2.0))
        xi = -at1 - at2 + omega*dtr
        xi[xi > np.pi] -= 2.0*np.pi
        nu, I2 = at1 - at2, np.tan(II/2.0)
        Ra1 = np.sqrt(1.0 - 12.0*(I2**2)*np.cos(2.0*(p - xi)) + 36.0*(I2**4))
        P2 = np.sin(2.0*(p - xi))
        Q2 = 1.0/(6.0*(I2**2)) - np.cos(2.0*(p - xi))
        R = np.arctan(P2/Q2)
        P_prime = np.sin(2.0*II)*np.sin(nu)
        Q_prime = np.sin(2.0*II)*np.cos(nu) + 0.3347
        nu_prime = np.arctan(P_prime/Q_prime)
        P_sec = (np.sin(II)**2)*np.sin(2.0*nu)
        Q_sec = (np.sin(II)**2)*np.cos(2.0*nu) + 0.0727
        nu_sec = 0.5*np.arctan(P_sec/Q_sec)

        f[:,0], f[:,1] = 1.0, 1.0 #-- Sa, Ssa
        f[:,2] = (2.0/3.0 - np.power(np.sin(II),2.0))/0.5021 #-- Mm
        f[:,3] = 1.0 #-- MSf
        f[:,4] = np.power(np.sin(II),2.0)/0.1578  #-- Mf
        f[:,7] = np.sin(II)*(np.cos(II/2.0)**2)/0.38 #-- 2Q1
        f[:,8], f[:,9] = f[:,7], f[:,7] #-- sigma1, q1
        f[:,10], f[:,11] = f[:,7], f[:,7] #-- rho1, O1
        Mtmp1 = 1.36*np.cos(p*dtr) + 0.267*np.cos((p-omega)*dtr)
        Mtmp2 = 0.64*np.sin(p*dtr) + 0.135*np.sin((p-omega)*dtr)
        f[:,13] = np.sqrt(Mtmp1**2 + Mtmp2**2) #-- M1
        f[:,14] = np.sin(2.0*II) / 0.7214 #-- chi1
        f[:,15], f[:,16], f[:,17] = 1.0, 1.0, 1.0 #-- pi1, P1, S1
        temp1 = 0.8965*np.power(np.sin(2.0*II),2.0)
        temp2 = 0.6001*np.sin(2.0*II)*np.cos(nu)
        f[:,18] = np.sqrt(temp1 + temp2 + 0.1006) #-- K1
        f[:,19], f[:,20] = 1.0, 1.0 #-- psi1, phi1
        f[:,21], f[:,22] = f[:,14], f[:,14] #-- theta1
        f[:,23] = np.sin(II)*np.power(np.sin(II/2.0),2.0)/0.01640 #-- OO1
        f[:,24] = np.power(np.cos(II/2.0),4.0)/0.9154 #-- 2N2
        f[:,25], f[:,26] = f[:,24], f[:,24] #-- mu2, N2
        f[:,27], f[:,28] = f[:,24], 1.0 #-- nu2, M2a
        f[:,29], f[:,30] = f[:,24], 1.0 #-- M2, M2b
        f[:,31], f[:,32] = f[:,29], f[:,29]*Ra1 #-- lambda2, L2
        f[:,33], f[:,34], f[:,35] = 1.0, 1.0, 1.0 #-- T2, S2, R2
        temp1 = 19.0444 * np.power(np.sin(II),4.0)
        temp2 = 2.7702 * np.power(np.sin(II),2.0) * np.cos(2.0*nu)
        f[:,36] = np.sqrt(temp1 + temp2 + 0.0981) #-- K2
        f[:,37] = np.power(np.sin(II),2.0)/0.1565 #-- eta2
        f[:,38], f[:,39] = f[:,29]**2, f[:,29] #-- MNS2, 2SM2
        f[:,40] = np.power(np.cos(II/2.0), 6.0) / 0.8758 #-- M3
        f[:,41], f[:,42] = f[:,18]*f[:,29], 1.0 #-- MK3, S3
        f[:,43] = f[:,29]**2 #-- MN4
        f[:,44], f[:,45] = f[:,43], f[:,29] #-- M4, MS4
        f[:,46] = f[:,29]*f[:,36] #-- MK4
        f[:,47], f[:,48] = 1.0, 1.0 #-- S4, S5
        f[:,49] = f[:,29]**3 #-- M6
        f[:,50], f[:,51], f[:,52] = 1.0, 1.0, 1.0 #-- S6, S7, S8
        f[:,53] = f[:,29]**4 #-- m8
        f[:,54] = f[:,29]*f[:,36] #-- mks2
        f[:,55], f[:,56] = f[:,4], f[:,4] #-- msqm, mtm
        f[:,57] = f[:,29]**2 #-- n4
        f[:,58], f[:,59] = f[:,29], 1.0 #-- eps2, Z0

        u[:,0], u[:,1], u[:,2] = 0.0, 0.0, 0.0 #-- Sa, Ssa, Mm
        u[:,3] = (2.0*xi - 2.0*nu)/dtr #-- MSf
        u[:,4] = -2.0*xi/dtr #-- Mf
        u[:,7] = (2.0*xi - nu)/dtr #-- 2Q1
        u[:,8], u[:,9] = u[:,7], u[:,7] #-- sigma1, q1
        u[:,10], u[:,11] = u[:,7], u[:,7] #-- rho1, O1
        u[:,13] = np.arctan2(Mtmp2,Mtmp1)/dtr #-- M1
        u[:,14] = -nu/dtr #-- chi1
        u[:,15], u[:,16], u[:,17] = 0.0, 0.0, 0.0 #-- pi1, P1, S1
        u[:,18] = -nu_prime/dtr #-- K1
        u[:,19], u[:,20] = 0.0, 0.0 #-- psi1, phi1
        u[:,21] = -nu/dtr #-- theta1
        u[:,22] = u[:,21] #-- J1
        u[:,23] = (-2.0*xi - nu)/dtr #-- OO1
        u[:,24] = (2.0*xi - 2.0*nu)/dtr #-- 2N2
        u[:,25], u[:,26] = u[:,24], u[:,24] #-- mu2, N2
        u[:,27], u[:,29] = u[:,24], u[:,24] #-- nu2, M2
        u[:,31] = (2.0*xi - 2.0*nu)/dtr #-- lambda2
        u[:,32] = (2.0*xi - 2.0*nu - R)/dtr #-- L2
        u[:,33], u[:,34], u[:,35] = 0.0, 0.0, 0.0 #-- T2, S2, R2
        u[:,36] = -2.0*nu_sec/dtr #-- K2
        u[:,37] = -2.0*nu/dtr #-- eta2
        u[:,38] = (4.0*xi - 4.0*nu)/dtr #-- mns2
        u[:,39] = (2.0*xi - 2.0*nu)/dtr #-- 2SM2
        u[:,40] = (3.0*xi - 3.0*nu)/dtr #-- M3
        u[:,41] = (2.0*xi - 2.0*nu - 2.0*nu_prime)/dtr #-- MK3
        u[:,42] = 0.0 #-- S3
        u[:,43] = (4.0*xi - 4.0*nu)/dtr #-- MN4
        u[:,44] = (4.0*xi - 4.0*nu)/dtr #-- M4
        u[:,45] = (2.0*xi - 2.0*nu)/dtr  #-- MS4
        u[:,46] = (2.0*xi - 2.0*nu - 2.0*nu_sec)/dtr #-- MK4
        u[:,47], u[:,48] = 0.0, 0.0 #-- S4, S5
        u[:,49] = (6.0*xi - 6.0*nu)/dtr #-- M6
        u[:,50], u[:,51], u[:,52] = 0.0, 0.0, 0.0 #-- S6, S7, S8
        u[:,53] = (8.0*xi - 8.0*nu)/dtr #-- m8
        u[:,54] = (2.0*xi - 2.0*nu - 2.0*nu_sec)/dtr #-- mks2
        u[:,55], u[:,56] = u[:,4], u[:,4] #-- msqm, mtm
        u[:,57] = (4.0*xi - 4.0*nu)/dtr #-- n4
        u[:,58], u[:,59] = u[:,29], 0.0 #-- eps2, Z0

    elif CORRECTIONS in ('GOT',):
        f[:,9] = 1.009 + 0.187*cosn - 0.015*cos2n#-- Q1
        f[:,11] = f[:,9]#-- O1
        f[:,16], f[:,17] = 1.0, 1.0 #-- P1, S1
        f[:,18] = 1.006 + 0.115*cosn - 0.009*cos2n#-- K1
        f[:,26] = 1.000 - 0.037*cosn#-- N2
        f[:,29], f[:,34] = f[:,26], 1.0 #-- M2, S2
        f[:,36] = 1.024 + 0.286*cosn + 0.008*cos2n#-- K2
        f[:,44] = f[:,29]**2#-- M4

        u[:,9] = 10.8*sinn - 1.3*sin2n#-- Q1
        u[:,11] = u[:,9]#-- O1
        u[:,16], u[:,17] = 0.0, 0.0 #-- P1, S1
        u[:,18] = -8.9*sinn + 0.7*sin2n#-- K1
        u[:,26] = -2.1*sinn#-- N2
        u[:,29], u[:,34] = u[:,26], 0.0 #-- M2, S2
        u[:,36] = -17.7*sinn + 0.7*sin2n#-- K2
        u[:,44] = -4.2*sinn#-- M4

    nc = len(constituents)
    pu, pf, G = np.zeros((nt,nc)), np.zeros((nt,nc)), np.zeros((nt,nc))
    for i,c in enumerate(constituents):
        j, = [j for j,val in enumerate(cindex) if (val == c)]
        pu[:,i], pf[:,i], G[:,i] = u[:,j]*dtr, f[:,j], arg[:,j]
    return (pu,pf,G)

def predict_tide(t,hc,constituents,DELTAT=0.0,CORRECTIONS='OTIS'):
    npts,nc = np.shape(hc)
    pu,pf,G = load_nodal_corrections(t + 48622.0, constituents,
        DELTAT=DELTAT, CORRECTIONS=CORRECTIONS)
    ht = np.ma.zeros((npts))
    ht.mask = np.zeros((npts),dtype=bool)
    for k,c in enumerate(constituents):
        if CORRECTIONS in ('OTIS','ATLAS','netcdf'):
            amp,ph,omega,alpha,species = load_constituent(c)
            th = omega*t*86400.0 + ph + pu[0,k]
        elif CORRECTIONS in ('GOT','FES'):
            th = G[0,k]*np.pi/180.0 + pu[0,k]
        ht.data[:] += pf[0,k]*hc.real[:,k]*np.cos(th) - \
            pf[0,k]*hc.imag[:,k]*np.sin(th)
        ht.mask[:] |= (hc.real.mask[:,k] | hc.imag.mask[:,k])
    return np.squeeze(ht)

def predict_tide_drift(t,hc,constituents,DELTAT=0.0,CORRECTIONS='OTIS'):
    nt = len(t)
    pu,pf,G = load_nodal_corrections(t + 48622.0, constituents,
        DELTAT=DELTAT, CORRECTIONS=CORRECTIONS)
    ht = np.ma.zeros((nt))
    ht.mask = np.zeros((nt),dtype=bool)
    for k,c in enumerate(constituents):
        if CORRECTIONS in ('OTIS','ATLAS','netcdf'):
            amp,ph,omega,alpha,species = load_constituent(c)
            th = omega*t*86400.0 + ph + pu[:,k]
        elif CORRECTIONS in ('GOT','FES'):
            th = G[:,k]*np.pi/180.0 + pu[:,k]
        ht.data[:] += pf[:,k]*hc.real[:,k]*np.cos(th) - \
            pf[:,k]*hc.imag[:,k]*np.sin(th)
        ht.mask[:] |= (hc.real.mask[:,k] | hc.imag.mask[:,k])
    return ht

def read_atlas_grid(input_file):
    fd = os.open(os.path.expanduser(input_file),os.O_RDONLY)
    file_info = os.fstat(fd)
    fid = os.fdopen(fd, 'rb')
    fid.seek(4,0)
    nx, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    ny, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    lats = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
    lons = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
    dt, = np.fromfile(fid, dtype=np.dtype('>f4'), count=1)
    dlon, dlat = (lons[1] - lons[0])/nx, (lats[1] - lats[0])/ny
    x = np.linspace(lons[0]+dlon/2.0,lons[1]-dlon/2.0,nx)
    y = np.linspace(lats[0]+dlat/2.0,lats[1]-dlat/2.0,ny)
    nob, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    if (nob == 0):
        fid.seek(20,1)
        iob = []
    else:
        fid.seek(8,1)
        iob=np.fromfile(fid, dtype=np.dtype('>i4'), count=2*nob).reshape(nob,2)
        fid.seek(8,1)
    hz = np.fromfile(fid, dtype=np.dtype('>f4'), count=nx*ny).reshape(ny,nx)
    fid.seek(8,1)
    mz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nx*ny).reshape(ny,nx)
    fid.seek(8,1)
    pmask = np.fromfile(fid, dtype=np.dtype('>i4'), count=nx*ny).reshape(ny,nx)
    fid.seek(4,1)
    nmod, local = 0, {}
    while (fid.tell() < file_info.st_size):
        fid.seek(4,1)
        nmod += 1
        nx1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        ny1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nd, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        lt1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        ln1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        name = fid.read(20).strip()
        fid.seek(8,1)
        iz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nd)
        jz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nd)
        fid.seek(8,1)
        depth = np.ma.zeros((ny1,nx1))
        depth.mask = np.ones((ny1,nx1),dtype=bool)
        depth.data[jz-1,iz-1] = np.fromfile(fid,dtype=np.dtype('>f4'),count=nd)
        depth.mask[jz-1,iz-1] = False
        fid.seek(4,1)
        local[name] = dict(lon=ln1,lat=lt1,depth=depth)
    fid.close()
    return (x,y,hz,mz,iob,dt,pmask,local)

def combine_atlas_model(xi,yi,zi,pmask,local,VARIABLE=None):
    d30 = 1.0/30.0
    x30 = np.arange(d30/2.0, 360.0+d30/2.0, d30)
    y30 = np.arange(-90.0+d30/2.0, 90.0+d30/2.0, d30)
    z30 = np.ma.zeros((len(y30),len(x30)),dtype=zi.dtype)
    z30.mask = np.zeros((len(y30),len(x30)),dtype=bool)
    if np.iscomplexobj(z30):
        f1 = scipy.interpolate.RectBivariateSpline(xi,yi,zi.real.T,kx=1,ky=1)
        f2 = scipy.interpolate.RectBivariateSpline(xi,yi,zi.imag.T,kx=1,ky=1)
        z30.data.real[:,:], z30.data.imag[:,:] = f1(x30,y30).T, f2(x30,y30).T
    else:
        f = scipy.interpolate.RectBivariateSpline(xi, yi, zi.T, kx=1,ky=1)
        z30.data[:,:] = f(x30,y30).T
    for key,val in local.items():
        zlocal = val[VARIABLE][:]
        validy,validx = np.nonzero(~zlocal.mask)
        ilon = np.arange(val['lon'][0]+d30/2.0,val['lon'][1]+d30/2.0,d30)
        ilat = np.arange(val['lat'][0]+d30/2.0,val['lat'][1]+d30/2.0,d30)
        X,Y = np.meshgrid(ilon,ilat)
        for indy,indx in zip(validy,validx):
            lon30 = (X[indy,indx]+360.) if (X[indy,indx]<=0.0) else X[indy,indx]
            ii, jj = int((lon30-x30[0])/d30), int((Y[indy,indx]-y30[0])/d30)
            z30.data[jj,ii] = zlocal[indy,indx]
    return (x30,y30,z30)

def create_atlas_mask(xi,yi,mz,local,VARIABLE=None):
    d30 = 1.0/30.0
    x30 = np.arange(d30/2.0, 360.0+d30/2.0, d30)
    y30 = np.arange(-90.0+d30/2.0, 90.0+d30/2.0, d30)
    xcoords=np.clip((len(xi)-1)*(x30-xi[0])/(xi[-1]-xi[0]),0,len(xi)-1)
    ycoords=np.clip((len(yi)-1)*(y30-yi[0])/(yi[-1]-yi[0]),0,len(yi)-1)
    gridy,gridx=np.meshgrid(np.around(ycoords),np.around(xcoords),indexing='ij')
    m30 = np.ma.zeros((len(y30),len(x30)),dtype=np.int8,fill_value=0)
    m30.data[:,:] = mz[gridy.astype(np.int32),gridx.astype(np.int32)]
    for key,val in local.items():
        ilon = np.arange(val['lon'][0]+d30/2.0,val['lon'][1]+d30/2.0,d30)
        ilat = np.arange(val['lat'][0]+d30/2.0,val['lat'][1]+d30/2.0,d30)
        X,Y = np.meshgrid(ilon,ilat)
        validy,validx = np.nonzero(~val[VARIABLE].mask)
        for indy,indx in zip(validy,validx):
            lon30 = (X[indy,indx]+360.) if (X[indy,indx]<=0.0) else X[indy,indx]
            ii, jj = int((lon30 - x30[0])/d30), int((Y[indy,indx] - y30[0])/d30)
            m30[jj,ii] = 1
    m30.mask = (m30.data == m30.fill_value)
    return m30

def xy_ll_EPSG3031(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_user_input({'proj':'stere','lat_0':-90,'lat_ts':-71,
        'lon_0':0,'x_0':0.,'y_0':0.,'ellps': 'WGS84','datum': 'WGS84',
        'units':'km'})
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
    return transformer.transform(i1, i2, direction=direction)

def xy_ll_EPSG3413(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_user_input({'proj':'stere','lat_0':90,'lat_ts':70,
        'lon_0':-45,'x_0':0.,'y_0':0.,'ellps': 'WGS84','datum': 'WGS84',
        'units':'km'})
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
    return transformer.transform(i1, i2, direction=direction)

def xy_ll_CATS2008(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_user_input({'proj':'stere','lat_0':-90,'lat_ts':-71,
        'lon_0':-70,'x_0':0.,'y_0':0.,'ellps': 'WGS84','datum': 'WGS84',
        'units':'km'})
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
    return transformer.transform(i1, i2, direction=direction)

def xy_ll_EPSG3976(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_user_input({'proj':'stere','lat_0':-90,'lat_ts':-70,
        'lon_0':0,'x_0':0.,'y_0':0.,'ellps': 'WGS84','datum': 'WGS84',
        'units':'km'})
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
    return transformer.transform(i1, i2, direction=direction)

def xy_ll_PSNorth(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
        lon,lat = transformer.transform(i1, i2, direction=direction)
        o1 = (90.0-lat)*111.7*np.cos(lon/180.0*np.pi)
        o2 = (90.0-lat)*111.7*np.sin(lon/180.0*np.pi)
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
        lon = np.arctan2(i2,i1)*180.0/np.pi
        lat = 90.0 - np.sqrt(i1**2+i2**2)/111.7
        ii, = np.nonzero(lon < 0)
        lon[ii] += 360.0
        o1,o2 = transformer.transform(lon, lat, direction=direction)
    return (o1,o2)

def xy_ll_EPSG4326(i1,i2,BF,EPSG=4326):
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if (BF.upper() == 'F'):
        direction = pyproj.enums.TransformDirection.FORWARD
    elif (BF.upper() == 'B'):
        direction = pyproj.enums.TransformDirection.INVERSE
    return transformer.transform(i1, i2, direction=direction)

def convert_ll_xy(i1,i2,PROJ,BF,EPSG=4326):
    conversion_functions = {}
    conversion_functions['3031'] = xy_ll_EPSG3031
    conversion_functions['3413'] = xy_ll_EPSG3413
    conversion_functions['CATS2008'] = xy_ll_CATS2008
    conversion_functions['3976'] = xy_ll_EPSG3976
    conversion_functions['PSNorth'] = xy_ll_PSNorth
    conversion_functions['4326'] = xy_ll_EPSG4326
    if PROJ not in conversion_functions.keys():
        raise Exception('PROJ:{0} conversion function not found'.format(PROJ))
    o1,o2 = conversion_functions[PROJ](i1,i2,BF,EPSG=EPSG)
    return (o1,o2)

def read_tide_grid(input_file):
    fid = open(os.path.expanduser(input_file),'rb')
    fid.seek(4,0)
    nx, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    ny, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    ylim = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
    xlim = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
    dt, = np.fromfile(fid, dtype=np.dtype('>f4'), count=1)
    if (xlim[0] < 0) & (xlim[1] < 0) & (dt > 0):
        xlim += 360.0
    dx, dy = (xlim[1] - xlim[0])/nx, (ylim[1] - ylim[0])/ny
    x = np.linspace(xlim[0]+dx/2.0,xlim[1]-dx/2.0,nx)
    y = np.linspace(ylim[0]+dy/2.0,ylim[1]-dy/2.0,ny)
    nob, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    if (nob == 0):
        fid.seek(20,1)
        iob = []
    else:
        fid.seek(8,1)
        iob=np.fromfile(fid, dtype=np.dtype('>i4'), count=2*nob).reshape(nob,2)
        fid.seek(8,1)
    hz = np.fromfile(fid, dtype=np.dtype('>f4'), count=nx*ny).reshape(ny,nx)
    fid.seek(8,1)
    mz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nx*ny).reshape(ny,nx)
    fid.close()
    return (x,y,hz,mz,iob,dt)

def Muv(hz):
    ny,nx = np.shape(hz)
    mz = (hz > 0).astype(int)
    indx, indy = np.zeros((nx),dtype=int), np.zeros((ny),dtype=int)
    indx[:-1], indy[:-1] = np.arange(1,nx), np.arange(1,ny)
    indx[-1], indy[-1] = 0, 0
    mu, mv = np.zeros((ny,nx),dtype=int), np.zeros((ny,nx),dtype=int)
    mu[indy,:], mv[:,indx] = mz*mz[indy,:], mz*mz[:,indx]
    return (mu,mv,mz)

def Huv(hz):
    ny,nx = np.shape(hz)
    mu,mv,mz = Muv(hz)
    indx, indy = np.zeros((nx),dtype=int), np.zeros((ny),dtype=int)
    indx[0], indy[0] = nx-1, ny-1
    indx[1:], indy[1:] = np.arange(1,nx), np.arange(1,ny)
    hu, hv = mu*(hz + hz[indy,:])/2.0, mv*(hz + hz[:,indx])/2.0
    return (hu,hv)

def extend_array(input_array,step_size):
    n = len(input_array)
    temp = np.zeros((n+2),dtype=input_array.dtype)
    temp[0] = input_array[0] - step_size
    temp[1:-1] = input_array[:]
    temp[-1] = input_array[-1] + step_size
    return temp

def extend_matrix(input_matrix):
    ny,nx = np.shape(input_matrix)
    temp = np.ma.zeros((ny,nx+2),dtype=input_matrix.dtype)
    temp[:,0] = input_matrix[:,-1]
    temp[:,1:-1] = input_matrix[:,:]
    temp[:,-1] = input_matrix[:,0]
    return temp

def bilinear_interp(ilon,ilat,idata,lon,lat,fill_value=np.nan,dtype=np.float64):
    valid, = np.nonzero((lon >= ilon.min()) & (lon <= ilon.max()) &
        (lat > ilat.min()) & (lat < ilat.max()))
    npts = len(lon)
    data = np.ma.zeros((npts),dtype=dtype,fill_value=fill_value)
    data.mask = np.ones((npts),dtype=bool)
    data.data[:] = data.fill_value
    for i in valid:
        ix, = np.nonzero((ilon[0:-1] <= lon[i]) & (ilon[1:] > lon[i]))
        iy, = np.nonzero((ilat[0:-1] <= lat[i]) & (ilat[1:] > lat[i]))
        IM = np.ma.zeros((4),fill_value=fill_value,dtype=dtype)
        IM.mask = np.ones((4),dtype=bool)
        WM = np.zeros((4))
        for j,XI,YI in zip([0,1,2,3],[ix,ix+1,ix,ix+1],[iy,iy,iy+1,iy+1]):
            IM.data[j], = idata.data[YI,XI]
            IM.mask[j], = idata.mask[YI,XI]
            WM[3-j], = np.abs(lon[i]-ilon[XI])*np.abs(lat[i]-ilat[YI])
        if (np.isclose(lat[i],ilat[iy]) & np.isclose(lon[i],ilon[ix])):
            data.data[i] = idata.data[iy,ix]
            data.mask[i] = idata.mask[iy,ix]
        elif (np.isclose(lat[i],ilat[iy+1]) & np.isclose(lon[i],ilon[ix])):
            data.data[i] = idata.data[iy+1,ix]
            data.mask[i] = idata.mask[iy+1,ix]
        elif (np.isclose(lat[i],ilat[iy]) & np.isclose(lon[i],ilon[ix+1])):
            data.data[i] = idata.data[iy,ix+1]
            data.mask[i] = idata.mask[iy,ix+1]
        elif (np.isclose(lat[i],ilat[iy+1]) & np.isclose(lon[i],ilon[ix+1])):
            data.data[i] = idata.data[iy+1,ix+1]
            data.mask[i] = idata.mask[iy+1,ix+1]
        elif np.any(np.isfinite(IM) & (~IM.mask)):
            ii, = np.nonzero(np.isfinite(IM) & (~IM.mask))
            data.data[i] = np.sum(WM[ii]*IM[ii])/np.sum(WM[ii])
            data.mask[i] = np.all(IM.mask[ii])
    return data

def read_constituents(input_file):
    fid = open(os.path.expanduser(input_file),'rb')
    ll, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    nx,ny,nc = np.fromfile(fid, dtype=np.dtype('>i4'), count=3)
    fid.seek(16,1)
    constituents = [c.decode("utf-8").rstrip() for c in fid.read(nc*4).split()]
    fid.close()
    return (constituents,nc)

def read_atlas_elevation(input_file,ic,constituent):
    fd = os.open(os.path.expanduser(input_file),os.O_RDONLY)
    file_info, fid = os.fstat(fd), os.fdopen(fd, 'rb')
    ll, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    nx,ny,nc = np.fromfile(fid, dtype=np.dtype('>i4'), count=3)
    nskip = 8 + nc*4 + ic*(nx*ny*8 + 8)
    fid.seek(nskip,1)
    h = np.ma.zeros((ny,nx),dtype=np.complex64)
    h.mask = np.zeros((ny,nx),dtype=bool)
    for i in range(ny):
        temp = np.fromfile(fid, dtype=np.dtype('>f4'), count=2*nx)
        h.data.real[i,:] = temp[0:2*nx-1:2]
        h.data.imag[i,:] = temp[1:2*nx:2]
    nskip = (nc-ic-1)*(nx*ny*8 + 8) + 4
    fid.seek(nskip,1)
    nmod, local = 0, {}
    while (fid.tell() < file_info.st_size):
        fid.seek(4,1)
        nmod += 1
        nx1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        ny1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nc1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nz, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        lt1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        ln1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        cons = fid.read(nc1*4).strip().split()
        if (constituent in cons):
            ic1, = [i for i,c in enumerate(cons) if (c == constituent)]
            name = fid.read(20).strip()
            fid.seek(8,1)
            iz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nz)
            jz = np.fromfile(fid, dtype=np.dtype('>i4'), count=nz)
            nskip = 8 + ic1*(8*nz + 8)
            fid.seek(nskip,1)
            h1 = np.ma.zeros((ny1,nx1),fill_value=np.nan,dtype=np.complex64)
            h1.mask = np.ones((ny1,nx1),dtype=bool)
            temp = np.fromfile(fid, dtype=np.dtype('>f4'), count=2*nz)
            h1.data.real[jz-1,iz-1] = temp[0:2*nz-1:2]
            h1.data.imag[jz-1,iz-1] = temp[1:2*nz:2]
            h1.mask[jz-1,iz-1] = False
            local[name] = dict(lon=ln1,lat=lt1,z=h1)
            nskip = (nc1-ic1-1)*(8*nz + 8) + 4
            fid.seek(nskip,1)
        else:
            nskip = 40 + 16*nz + (nc1-1)*(8*nz + 8)
            fid.seek(nskip,1)
    fid.close()
    return (h,local)

def read_elevation_file(input_file,ic):
    fid = open(os.path.expanduser(input_file),'rb')
    ll, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    nx,ny,nc = np.fromfile(fid, dtype=np.dtype('>i4'), count=3)
    #nskip = ic*(nx*ny*8+8) + 8 + ll - 28
    nskip = ic*(int(nx)*int(ny)*8 + 8) + 8 + int(ll) - 28
    fid.seek(nskip,1)
    h = np.ma.zeros((ny,nx),dtype=np.complex64)
    h.mask = np.zeros((ny,nx),dtype=bool)
    for i in range(ny):
        temp = np.fromfile(fid, dtype=np.dtype('>f4'), count=2*nx)
        h.data.real[i,:] = temp[0:2*nx-1:2]
        h.data.imag[i,:] = temp[1:2*nx:2]
    h.mask[np.isnan(h.data)] = True
    h.data[h.mask] = h.fill_value
    fid.close()
    return h

def to_cartesian(lon,lat,h=0.0,a_axis=6378137.0,flat=1.0/298.257223563):
    lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
    count = np.count_nonzero(lon < 0)
    if (count != 0):
        lt0, = np.nonzero(lon < 0)
        lon[lt0] += 360.0
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1, dtr = lin_ecc/a_axis, np.pi/180.0
    latitude_geodetic_rad = lat*dtr
    N = a_axis/np.sqrt(1.0 - ecc1**2.0*np.sin(latitude_geodetic_rad)**2.0)
    X = (N + h) * np.cos(latitude_geodetic_rad) * np.cos(lon*dtr)
    Y = (N + h) * np.cos(latitude_geodetic_rad) * np.sin(lon*dtr)
    Z = (N * (1.0 - ecc1**2.0) + h) * np.sin(latitude_geodetic_rad)
    return (X,Y,Z)

def nearest_extrap(ilon,ilat,idata,lon,lat,fill_value=np.nan,
    dtype=np.float64,cutoff=np.inf,EPSG='4326'):
    dlon, dlat = np.abs(ilon[1] - ilon[0]), np.abs(ilat[1] - ilat[0])
    lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
    npts = len(lon)
    if (npts == 0):
        return
    data = np.ma.zeros((npts),dtype=dtype,fill_value=fill_value)
    data.mask = np.ones((npts),dtype=bool)
    data.data[:] = data.fill_value
    xmin,xmax = (np.min(lon),np.max(lon))
    ymin,ymax = (np.min(lat),np.max(lat))
    gridlon,gridlat = np.meshgrid(ilon,ilat)
    valid_bounds = (~idata.mask) & np.isfinite(idata.data)
    valid_bounds &= (gridlon >= (xmin-2.0*dlon))
    valid_bounds &= (gridlon <= (xmax+2.0*dlon))
    valid_bounds &= (gridlat >= (ymin-2.0*dlat))
    valid_bounds &= (gridlat <= (ymax+2.0*dlat))
    if not np.any(valid_bounds):
        return data
    indy,indx = np.nonzero(valid_bounds)
    iflat = idata.data[indy,indx]

    if (EPSG == '4326'):
        xflat,yflat,zflat = to_cartesian(gridlon[indy,indx],
                            gridlat[indy,indx],a_axis=6378.137)
        tree = scipy.spatial.cKDTree(np.c_[xflat,yflat,zflat])
        xs,ys,zs = to_cartesian(lon,lat,a_axis=6378.137)
        points = np.c_[xs,ys,zs]
    else:
        tree = scipy.spatial.cKDTree(np.c_[gridlon[indy,indx],
            gridlat[indy,indx]])
        points = np.c_[lon,lat]

    dd,ii = tree.query(points,k=1,distance_upper_bound=cutoff)
    ind, = np.nonzero(np.isfinite(dd))
    data.data[ind] = iflat[ii[ind]]
    data.mask[ind] = False
    return data

def read_atlas_transport(input_file,ic,constituent):
    fd = os.open(os.path.expanduser(input_file),os.O_RDONLY)
    file_info, fid = os.fstat(fd), os.fdopen(fd, 'rb')
    ll, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    nx,ny,nc = np.fromfile(fid, dtype=np.dtype('>i4'), count=3)
    nskip = 8 + nc*4 + ic*(nx*ny*16 + 8)
    fid.seek(nskip,1)
    u = np.ma.zeros((ny,nx),dtype=np.complex64)
    u.mask = np.zeros((ny,nx),dtype=bool)
    v = np.ma.zeros((ny,nx),dtype=np.complex64)
    v.mask = np.zeros((ny,nx),dtype=bool)
    for i in range(ny):
        temp = np.fromfile(fid, dtype=np.dtype('>f4'), count=4*nx)
        u.data.real[i,:] = temp[0:4*nx-3:4]
        u.data.imag[i,:] = temp[1:4*nx-2:4]
        v.data.real[i,:] = temp[2:4*nx-1:4]
        v.data.imag[i,:] = temp[3:4*nx:4]
    nskip = (nc-ic-1)*(nx*ny*16 + 8) + 4
    fid.seek(nskip,1)
    nmod, local = 0, {}
    while (fid.tell() < file_info.st_size):
        fid.seek(4,1)
        nmod += 1
        nx1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        ny1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nc1, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nu, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        nv, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
        lt1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        ln1 = np.fromfile(fid, dtype=np.dtype('>f4'), count=2)
        cons = fid.read(nc1*4).strip().split()
        if (constituent in cons):
            ic1, = [i for i,c in enumerate(cons) if (c == constituent)]
            name = fid.read(20).strip()
            fid.seek(8,1)
            iu = np.fromfile(fid, dtype=np.dtype('>i4'), count=nu)
            ju = np.fromfile(fid, dtype=np.dtype('>i4'), count=nu)
            fid.seek(8,1)
            iv = np.fromfile(fid, dtype=np.dtype('>i4'), count=nv)
            jv = np.fromfile(fid, dtype=np.dtype('>i4'), count=nv)
            nskip = 8 + ic1*(8*nu + 8*nv + 16)
            fid.seek(nskip,1)
            u1 = np.ma.zeros((ny1,nx1),fill_value=np.nan,dtype=np.complex64)
            u1.mask = np.ones((ny1,nx1),dtype=bool)
            tmpu = np.fromfile(fid, dtype=np.dtype('>f4'), count=2*nu)
            u1.data.real[ju-1,iu-1] = tmpu[0:2*nu-1:2]
            u1.data.imag[ju-1,iu-1] = tmpu[1:2*nu:2]
            u1.mask[ju-1,iu-1] = False
            fid.seek(8,1)
            v1 = np.ma.zeros((ny1,nx1),fill_value=np.nan,dtype=np.complex64)
            v1.mask = np.ones((ny1,nx1),dtype=bool)
            tmpv = np.fromfile(fid, dtype=np.dtype('>f4'), count=2*nv)
            v1.data.real[jv-1,iv-1] = tmpv[0:2*nv-1:2]
            v1.data.imag[jv-1,iv-1] = tmpv[1:2*nv:2]
            v1.mask[jv-1,iv-1] = False
            local[name] = dict(lon=ln1,lat=lt1,u=u1,v=v1)
            nskip = (nc1-ic1-1)*(8*nu + 8*nv + 16) + 4
            fid.seek(nskip,1)
        else:
            nskip = 56 + 16*nu + 16*nv + (nc1-1)*(8*nu + 8*nv + 16)
            fid.seek(nskip,1)
    fid.close()
    return (u,v,local)

def read_transport_file(input_file,ic):
    fid = open(os.path.expanduser(input_file),'rb')
    ll, = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
    nx,ny,nc = np.fromfile(fid, dtype=np.dtype('>i4'), count=3)
    nskip = ic*(nx*ny*16+8) + 8 + ll - 28
    fid.seek(nskip,1)
    u = np.ma.zeros((ny,nx),dtype=np.complex64)
    u.mask = np.zeros((ny,nx),dtype=bool)
    v = np.ma.zeros((ny,nx),dtype=np.complex64)
    v.mask = np.zeros((ny,nx),dtype=bool)
    for i in range(ny):
        temp = np.fromfile(fid, dtype=np.dtype('>f4'), count=4*nx)
        u.data.real[i,:] = temp[0:4*nx-3:4]
        u.data.imag[i,:] = temp[1:4*nx-2:4]
        v.data.real[i,:] = temp[2:4*nx-1:4]
        v.data.imag[i,:] = temp[3:4*nx:4]
    u.mask[np.isnan(u.data)], v.mask[np.isnan(v.data)] = True, True
    u.data[u.mask], v.data[v.mask] = u.fill_value, v.fill_value
    fid.close()
    return (u,v)

def find_closest_coords(lat, lon, grid_file, EPSG, model_EPSG):
    x,y,hz,mz,iob,dt=read_tide_grid(grid_file)
    coords = []
    for i in range(len(x)):
        for j in range(len(y)):
            coords.append([x[i],y[j]])
    coords = np.array(coords)
    points=convert_ll_xy(coords[:,1],coords[:,0],EPSG,'B',EPSG=model_EPSG)
    points = np.array(points)
    indexes = np.unique(points[0], return_index=True)[1]
    Y = [points[0][index] for index in sorted(indexes)]
    indexes = np.unique(points[1], return_index=True)[1]
    X = [points[1][index] for index in sorted(indexes)]
    tempLat, tempLon = [], []
    for k in range(len(lat)):
        idx = (np.abs(X - lon[k])).argmin()
        idy = (np.abs(Y - lat[k])).argmin()
        newidy, newidx = idy,idx
        i, j = 0,0
        if mz[idy,idx] == 0:
            minDist = 1e15
            for i in range(idy-19, idy+20):
                for j in range(idx-19, idx+20):
                    if not i == idy and not j == idx:
                        if mz[i,j] == 1:
                            dist = ((idy-i)**2 + (idx-j)**2)**0.5
                            if dist < minDist: 
                                minDist = dist
                                newidy, newidx = i, j
        tempLat.append(Y[newidy])
        tempLon.append(X[newidx])
    return np.array(tempLon), np.array(tempLat)

def extract_tidal_constants(ilon, ilat, grid_file, model_file, EPSG, TYPE='z',
                METHOD='spline', EXTRAPOLATE=False, CUTOFF=10.0, GRID='OTIS'):
    if (GRID == 'ATLAS'):
        x0,y0,hz0,mz0,iob,dt,pmask,local = read_atlas_grid(grid_file)
        xi,yi,hz = combine_atlas_model(x0,y0,hz0,pmask,local,VARIABLE='depth')
        mz = create_atlas_mask(x0,y0,mz0,local,VARIABLE='depth')
    else:
        xi,yi,hz,mz,iob,dt = read_tide_grid(grid_file)
    x,y = convert_ll_xy(np.atleast_1d(ilon),np.atleast_1d(ilat),EPSG,'F')
    dx, dy = xi[1] - xi[0], yi[1] - yi[0]

    if (TYPE != 'z'):
        mz,mu,mv = Muv(hz)
        hu,hv = Huv(hz)

    GLOBAL = False
    if ((xi[-1] - xi[0]) == (360.0 - dx)) & (EPSG == '4326'):
        xi = extend_array(xi, dx)
        hz = extend_matrix(hz)
        mz = extend_matrix(mz)
        GLOBAL = True

    if (np.min(x) < np.min(xi)) & (EPSG == '4326'):
        lt0, = np.nonzero(x < 0)
        x[lt0] += 360.0
    if (np.max(x) > np.max(xi)) & (EPSG == '4326'):
        gt180, = np.nonzero(x > 180)
        x[gt180] -= 360.0
    invalid = (x < xi.min()) | (x > xi.max()) | (y < yi.min()) | (y > yi.max())

    hz = np.ma.array(hz,mask=(hz==0))
    if (TYPE != 'z'):
        if GLOBAL:
            hu, hv = extend_matrix(hu), extend_matrix(hv)
            mu, mv = extend_matrix(mu), extend_matrix(mv)
        hu = np.ma.array(hu,mask=(hu==0))
        hv = np.ma.array(hv,mask=(hv==0))

    if (METHOD == 'bilinear'):
        D = bilinear_interp(xi,yi,hz,x,y)
        mz1 = bilinear_interp(xi,yi,mz,x,y)
        mz1 = np.floor(mz1).astype(mz.dtype)
        if (TYPE != 'z'):
            mu1 = bilinear_interp(xi,yi,mu,x,y)
            mu1 = np.floor(mu1).astype(mu.dtype)
            mv1 = bilinear_interp(xi,yi,mv,x,y)
            mv1 = np.floor(mv1).astype(mz.dtype)
    elif (METHOD == 'spline'):
        f1=scipy.interpolate.RectBivariateSpline(xi,yi,hz.T,kx=1,ky=1)
        f2=scipy.interpolate.RectBivariateSpline(xi,yi,mz.T,kx=1,ky=1)
        D = f1.ev(x,y)
        mz1 = np.floor(f2.ev(x,y)).astype(mz.dtype)
        if (TYPE != 'z'):
            f3=scipy.interpolate.RectBivariateSpline(xi,yi,mu.T,kx=1,ky=1)
            f4=scipy.interpolate.RectBivariateSpline(xi,yi,mv.T,kx=1,ky=1)
            mu1 = np.floor(f3.ev(x,y)).astype(mu.dtype)
            mv1 = np.floor(f4.ev(x,y)).astype(mv.dtype)
    else:
        r1 = scipy.interpolate.RegularGridInterpolator((yi,xi),hz,
            method=METHOD,bounds_error=False)
        r2 = scipy.interpolate.RegularGridInterpolator((yi,xi),mz,
            method=METHOD,bounds_error=False,fill_value=0)
        D = r1.__call__(np.c_[y,x])
        mz1 = np.floor(r2.__call__(np.c_[y,x])).astype(mz.dtype)
        if (TYPE != 'z'):
            r3 = scipy.interpolate.RegularGridInterpolator((yi,xi),mu,
                method=METHOD,bounds_error=False,fill_value=0)
            r4 = scipy.interpolate.RegularGridInterpolator((yi,xi),mv,
                method=METHOD,bounds_error=False,fill_value=0)
            mu1 = np.floor(r3.__call__(np.c_[y,x])).astype(mu.dtype)
            mv1 = np.floor(r4.__call__(np.c_[y,x])).astype(mv.dtype)

    if TYPE in ('v','u'):
        unit_conv = (D/100.0)
    elif TYPE in ('V','U'):
        unit_conv = 1.0

    if isinstance(model_file,list):
        constituents = [read_constituents(m)[0].pop() for m in model_file]
        nc = len(constituents)
    else:
        constituents,nc = read_constituents(model_file)
    npts = len(D)
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    for i,c in enumerate(constituents):
        if (TYPE == 'z'):
            if (GRID == 'ATLAS'):
                z0,zlocal = read_atlas_elevation(model_file,i,c)
                xi,yi,z=combine_atlas_model(x0,y0,z0,pmask,zlocal,VARIABLE='z')
            elif isinstance(model_file,list):
                z = read_elevation_file(model_file[i],0)
            else:
                z = read_elevation_file(model_file,i)
            if GLOBAL:
                z = extend_matrix(z)
            z1 = np.ma.zeros((npts),dtype=z.dtype)
            if (METHOD == 'bilinear'):
                z[z==0] = np.nan
                z1.data[:] = bilinear_interp(xi,yi,z,x,y,dtype=np.complex128)
                z1.mask = (np.isnan(z1.data) | (~mz1.astype(bool)))
                z1.data[z1.mask] = z1.fill_value
            elif (METHOD == 'spline'):
                f1 = scipy.interpolate.RectBivariateSpline(xi,yi,
                    z.real.T,kx=1,ky=1)
                f2 = scipy.interpolate.RectBivariateSpline(xi,yi,
                    z.imag.T,kx=1,ky=1)
                z1.data.real = f1.ev(x,y)
                z1.data.imag = f2.ev(x,y)
                z1.mask = (~mz1.astype(bool))
                z1.data[z1.mask] = z1.fill_value
            else:
                r1 = scipy.interpolate.RegularGridInterpolator((yi,xi),z,
                    method=METHOD,bounds_error=False,fill_value=z1.fill_value)
                z1 = np.ma.zeros((npts),dtype=z.dtype)
                z1.data[:] = r1.__call__(np.c_[y,x])
                z1.mask = (z1.data == z1.fill_value) | (~mz1.astype(bool))
                z1.data[z1.mask] = z1.fill_value
            if EXTRAPOLATE and np.any(z1.mask):
                inv, = np.nonzero(z1.mask)
                z[z==0] = np.nan
                z1.data[inv] = nearest_extrap(xi,yi,z,x[inv],y[inv],
                    dtype=np.complex128,cutoff=CUTOFF,EPSG=EPSG)
                z1.mask[inv] = np.isnan(z1.data[inv])
                z1.data[z1.mask] = z1.fill_value
            amplitude.data[:,i] = np.abs(z1.data)
            amplitude.mask[:,i] = np.copy(z1.mask)
            ph.data[:,i] = np.arctan2(-np.imag(z1.data),np.real(z1.data))
            ph.mask[:,i] = np.copy(z1.mask)
        elif TYPE in ('U','u'):
            if (GRID == 'ATLAS'):
                u0,v0,uvlocal = read_atlas_transport(model_file,i,c)
                xi,yi,u=combine_atlas_model(x0,y0,u0,pmask,uvlocal,VARIABLE='u')
            elif isinstance(model_file,list):
                u,v = read_transport_file(model_file[i],0)
            else:
                u,v = read_transport_file(model_file,i)
            if GLOBAL:
                u = extend_matrix(u)
            xu, u1 = xi - dx/2.0, np.ma.zeros((npts),dtype=u.dtype)
            if (METHOD == 'bilinear'):
                u[u==0] = np.nan
                u1.data[:] = bilinear_interp(xu,yi,u,x,y,dtype=np.complex128)
                u1.mask = (np.isnan(u1.data) | (~mu1.astype(bool)))
                u1.data[u1.mask] = u1.fill_value
            elif (METHOD == 'spline'):
                f1 = scipy.interpolate.RectBivariateSpline(xu,yi,
                    u.real.T,kx=1,ky=1)
                f2 = scipy.interpolate.RectBivariateSpline(xu,yi,
                    u.imag.T,kx=1,ky=1)
                u1.data.real = f1.ev(x,y)
                u1.data.imag = f2.ev(x,y)
                u1.mask = (~mu1.astype(bool))
                u1.data[u1.mask] = u1.fill_value
            else:
                r1 = scipy.interpolate.RegularGridInterpolator((yi,xu),u,
                    method=METHOD,bounds_error=False,fill_value=u1.fill_value)
                u1.data[:] = r1.__call__(np.c_[y,x])
                u1.mask = (u1.data == u1.fill_value) | (~mu1.astype(bool))
                u1.data[u1.mask] = u1.fill_value
            if EXTRAPOLATE and np.any(u1.mask):
                inv, = np.nonzero(u1.mask)
                u[u==0] = np.nan
                u1.data[inv] = nearest_extrap(xu,yi,u,x[inv],y[inv],
                    dtype=np.complex128,cutoff=CUTOFF,EPSG=EPSG)
                u1.mask[inv] = np.isnan(u1.data[inv])
                u1.data[u1.mask] = u1.fill_value
            amplitude.data[:,i] = np.abs(u1.data)/unit_conv
            amplitude.mask[:,i] = np.copy(u1.mask)
            ph.data[:,i] = np.arctan2(-np.imag(u1),np.real(u1))
            ph.mask[:,i] = np.copy(u1.mask)
        elif TYPE in ('V','v'):
            if (GRID == 'ATLAS'):
                u0,v0,uvlocal = read_atlas_transport(model_file,i,c)
                xi,yi,v = combine_atlas_model(x0,y0,v0,pmask,local,VARIABLE='v')
            elif isinstance(model_file,list):
                u,v = read_transport_file(model_file[i],0)
            else:
                u,v = read_transport_file(model_file,i)
            if GLOBAL:
                v = extend_matrix(v)
            yv, v1 = yi - dy/2.0, np.ma.zeros((npts),dtype=v.dtype)
            if (METHOD == 'bilinear'):
                v[v==0] = np.nan
                v1.data[:] = bilinear_interp(xi,yv,v,x,y,dtype=np.complex128)
                v1.mask = (np.isnan(v1.data) | (~mv1.astype(bool)))
                v1.data[v1.mask] = v1.fill_value
            elif (METHOD == 'spline'):
                f1 = scipy.interpolate.RectBivariateSpline(xi,yv,
                    v.real.T,kx=1,ky=1)
                f2 = scipy.interpolate.RectBivariateSpline(xi,yv,
                    v.imag.T,kx=1,ky=1)
                v1.data.real = f1.ev(x,y)
                v1.data.imag = f2.ev(x,y)
                v1.mask = (~mv1.astype(bool))
                v1.data[v1.mask] = v1.fill_value
            else:
                r1 = scipy.interpolate.RegularGridInterpolator((yv,xi),v,
                    method=METHOD,bounds_error=False,fill_value=v1.fill_value)
                v1.data[:] = r1.__call__(np.c_[y,x])
                v1.mask = (v1.data == v1.fill_value) | (~mv1.astype(bool))
                v1.data[v1.mask] = v1.fill_value
            if EXTRAPOLATE and np.any(v1.mask):
                inv, = np.nonzero(v1.mask)
                v[z==v] = np.nan
                v1.data[inv] = nearest_extrap(x,yv,v,x[inv],y[inv],
                    dtype=np.complex128,cutoff=CUTOFF,EPSG=EPSG)
                v1.mask[inv] = np.isnan(v1.data[inv])
                v1.data[v1.mask] = v1.fill_value
            amplitude.data[:,i] = np.abs(v1.data)/unit_conv
            amplitude.mask[:,i] = np.copy(v1.mask)
            ph.data[:,i] = np.arctan2(-np.imag(v1),np.real(v1))
            ph.mask[:,i] = np.copy(v1.mask)
        ph.mask[:,i] |= invalid
        amplitude.mask[:,i] |= invalid

    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    return (amplitude,phase,D,constituents)

def read_netcdf_grid(input_file, TYPE, GZIP=False):
    if GZIP:
        f = gzip.open(os.path.expanduser(input_file),'rb')
        fileID=netCDF4.Dataset(uuid.uuid4().hex,'r',memory=f.read())
    else:
        fileID=netCDF4.Dataset(os.path.expanduser(input_file),'r')
    nx, ny = fileID.dimensions['nx'].size, fileID.dimensions['ny'].size
    bathymetry = np.ma.zeros((ny,nx))
    if (TYPE == 'z'):
        bathymetry.data[:,:] = fileID.variables['hz'][:,:].T
        lon = fileID.variables['lon_z'][:].copy()
        lat = fileID.variables['lat_z'][:].copy()
    elif TYPE in ('U','u'):
        bathymetry.data[:,:] = fileID.variables['hu'][:,:].T
        lon = fileID.variables['lon_u'][:].copy()
        lat = fileID.variables['lat_u'][:].copy()
    elif TYPE in ('V','v'):
        bathymetry.data[:,:] = fileID.variables['hv'][:,:].T
        lon = fileID.variables['lon_v'][:].copy()
        lat = fileID.variables['lat_v'][:].copy()
    bathymetry.mask = (bathymetry.data == 0.0)
    fileID.close()
    f.close() if GZIP else None
    return (lon,lat,bathymetry)

def extract_netcdf_constants(ilon, ilat, grid_file, model_files, TYPE='z',
    METHOD='spline', EXTRAPOLATE=False, CUTOFF=10.0, GZIP=True, SCALE=1.0):
    if isinstance(model_files,str):
        print("Tide model is entered as a string")
        model_files = [model_files]
    lon,lat,bathymetry = read_netcdf_grid(grid_file, TYPE, GZIP=GZIP)
    dlon = lon[1] - lon[0]
    lon, bathymetry = extend_array(lon, dlon), extend_matrix(bathymetry)
    bathymetry.mask = (bathymetry.data == 0)

    ilon, ilat = np.atleast_1d(ilon), np.atleast_1d(ilat)
    lt0, = np.nonzero(ilon < 0)
    ilon[lt0] += 360.0
    npts = len(ilon)

    D = np.ma.zeros((npts))
    D.mask = np.zeros((npts),dtype=bool)
    if (METHOD == 'bilinear'):
        bathymetry[bathymetry.mask] = np.nan
        D.data[:] = bilinear_interp(lon,lat,bathymetry,ilon,ilat)
        D.mask[:] = np.isnan(D.data)
        D.data[D.mask] = D.fill_value
    elif (METHOD == 'spline'):
        f1 = scipy.interpolate.RectBivariateSpline(lon,lat,
            bathymetry.data.T,kx=1,ky=1)
        f2 = scipy.interpolate.RectBivariateSpline(lon,lat,
            bathymetry.mask.T,kx=1,ky=1)
        D.data[:] = f1.ev(ilon,ilat)
        D.mask[:] = np.ceil(f2.ev(ilon,ilat).astype(bool))
    else:
        r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
            bathymetry.data, method=METHOD, bounds_error=False)
        r2 = scipy.interpolate.RegularGridInterpolator((lat,lon),
            bathymetry.mask, method=METHOD, bounds_error=False, fill_value=1)
        D.data[:] = r1.__call__(np.c_[ilat,ilon])
        D.mask[:] = np.ceil(r2.__call__(np.c_[ilat,ilon])).astype(bool)

    if TYPE in ('v','u'):
        unit_conv = (D.data/100.0)
    elif TYPE in ('V','U'):
        unit_conv = 1.0

    nc, constituents = len(model_files), []
    ampl = np.ma.zeros((npts,nc))
    ampl.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    for i,model_file in enumerate(model_files):
        if (TYPE == 'z'):
            z,con = read_elevation_file(model_file, GZIP=GZIP)
            constituents.append(con)
            z = extend_matrix(z)
            z1 = np.ma.zeros((npts),dtype=z.dtype)
            z1.mask = np.zeros((npts),dtype=bool)
            if (METHOD == 'bilinear'):
                z[z.mask] = np.nan
                z1.data[:] = bilinear_interp(lon,lat,z,ilon,ilat,dtype=z.dtype)
                z1.mask[:] |= np.copy(D.mask)
                z1.data[z1.mask] = z1.fill_value
            elif (METHOD == 'spline'):
                f1 = scipy.interpolate.RectBivariateSpline(lon,lat,
                    z.data.real.T,kx=1,ky=1)
                f2 = scipy.interpolate.RectBivariateSpline(lon,lat,
                    z.data.imag.T,kx=1,ky=1)
                z1.data.real = f1.ev(ilon,ilat)
                z1.data.imag = f2.ev(ilon,ilat)
                z1.mask[:] |= np.copy(D.mask)
                z1.data[z1.mask] = z1.fill_value
            else:
                r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                    z.data, method=METHOD, bounds_error=False,
                    fill_value=z1.fill_value)
                z1.data[:] = r1.__call__(np.c_[ilat,ilon])
                z1.mask[:] |= np.copy(D.mask)
                z1.data[z1.mask] = z1.fill_value
            if EXTRAPOLATE and np.any(z1.mask):
                inv, = np.nonzero(z1.mask)
                z[z.mask] = np.nan
                z1.data[inv] = nearest_extrap(lon,lat,z,ilon[inv],ilat[inv],
                    dtype=z.dtype,cutoff=CUTOFF)
                z1.mask[inv] = np.isnan(z1.data[inv])
                z1.data[z1.mask] = z1.fill_value
            ampl.data[:,i] = np.abs(z1.data)
            ampl.mask[:,i] = np.copy(z1.mask)
            ph.data[:,i] = np.arctan2(-np.imag(z1.data),np.real(z1.data))
            ph.mask[:,i] = np.copy(z1.mask)
        elif TYPE in ('U','u','V','v'):
            tr,con = read_transport_file(model_file, TYPE, GZIP=GZIP)
            constituents.append(con)
            tr = extend_matrix(tr)
            tr1 = np.ma.zeros((npts),dtype=tr.dtype)
            tr1.mask = np.zeros((npts),dtype=bool)
            if (METHOD == 'bilinear'):
                tr1.data[:]=bilinear_interp(lon,lat,tr,ilon,ilat,dtype=tr.dtype)
                tr1.mask[:] |= np.copy(D.mask)
                tr1.data[tr1.mask] = tr1.fill_value
            elif (METHOD == 'spline'):
                f1 = scipy.interpolate.RectBivariateSpline(lon,lat,
                    tr.data.real.T,kx=1,ky=1)
                f2 = scipy.interpolate.RectBivariateSpline(lon,lat,
                    tr.data.imag.T,kx=1,ky=1)
                tr1.data.real = f1.ev(ilon,ilat)
                tr1.data.imag = f2.ev(ilon,ilat)
                tr1.mask[:] |= np.copy(D.mask)
                tr1.data[tr1.mask] = z1.fill_value
            else:
                r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                    tr.data, method=METHOD, bounds_error=False,
                    fill_value=tr1.fill_value)
                tr1.data[:] = r1.__call__(np.c_[ilat,ilon])
                tr1.mask[:] |= np.copy(D.mask)
                tr1.data[tr1.mask] = tr1.fill_value
            if EXTRAPOLATE and np.any(tr1.mask):
                inv, = np.nonzero(tr1.mask)
                tr[tr.mask] = np.nan
                tr1.data[inv] = nearest_extrap(lon,lat,tr,ilon[inv],ilat[inv],
                    dtype=tr.dtype,cutoff=CUTOFF)
                tr1.mask[inv] = np.isnan(tr1.data[inv])
                tr1.data[tr1.mask] = tr1.fill_value
            ampl.data[:,i] = np.abs(tr1.data)/unit_conv
            ampl.mask[:,i] = np.copy(tr1.mask)
            ph.data[:,i] = np.arctan2(-np.imag(tr1.data),np.real(tr1.data))
            ph.mask[:,i] = np.copy(tr1.mask)

    amplitude, phase = ampl*SCALE, ph*180.0/np.pi
    phase[phase < 0] += 360.0
    return (amplitude,phase,D,constituents)

def read_GOT_grid(input_file, GZIP=False):
    if GZIP:
        with gzip.open(os.path.expanduser(input_file),'rb') as f:
            file_contents = f.read().decode('utf-8').splitlines()
    else:
        with open(os.path.expanduser(input_file),'r') as f:
            file_contents = f.read().splitlines()
    constituent_list = ['Q1','O1','P1','K1','N2','M2','S2','K2','S1','M4']
    regex = re.compile(r'|'.join(constituent_list), re.IGNORECASE)
    cons = regex.findall(file_contents[0]).pop().lower()
    nlat,nlon = np.array(file_contents[2].split(), dtype=int)
    ilat = np.array(file_contents[3].split(), dtype=np.float64)
    ilon = np.array(file_contents[4].split(), dtype=np.float64)
    fill_value = np.array(file_contents[5].split(), dtype=np.float64)
    lat = np.linspace(ilat[0],ilat[1],nlat)
    lon = np.linspace(ilon[0],ilon[1],nlon)
    amp = np.ma.zeros((nlat,nlon),fill_value=fill_value[0],dtype=np.float32)
    ph = np.ma.zeros((nlat,nlon),fill_value=fill_value[0],dtype=np.float32)
    amp.mask = np.zeros((nlat,nlon),dtype=bool)
    ph.mask = np.zeros((nlat,nlon),dtype=bool)
    l1, l2 = 7, 14 + int(nlon//11)*nlat + nlat
    for i in range(nlat):
        for j in range(nlon//11):
            j1 = j*11
            amp.data[i,j1:j1+11] = np.array(file_contents[l1].split(),dtype='f')
            ph.data[i,j1:j1+11] = np.array(file_contents[l2].split(),dtype='f')
            l1 += 1
            l2 += 1
        j1 = (j+1)*11; j2 = nlon % 11
        amp.data[i,j1:j1+j2] = np.array(file_contents[l1].split(),dtype='f')
        ph.data[i,j1:j1+j2] = np.array(file_contents[l2].split(),dtype='f')
        l1 += 1
        l2 += 1
    hc = amp*np.exp(-1j*ph*np.pi/180.0)
    hc.mask = (amp.data == amp.fill_value) | (ph.data == ph.fill_value)
    return (hc,lon,lat,cons)

def extract_GOT_constants(ilon, ilat, model_files, METHOD=None,
    EXTRAPOLATE=False, CUTOFF=10.0, GZIP=True, SCALE=1.0):
    if isinstance(model_files,str):
        print("Tide model is entered as a string")
        model_files = [model_files]
    ilon, ilat = np.atleast_1d(ilon), np.atleast_1d(ilat)
    if (np.min(ilon) < 0.0):
        lt0, = np.nonzero(ilon < 0)
        ilon[lt0] += 360.0

    npts, constituents = len(ilon), []
    nc = len(model_files)
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    for i,model_file in enumerate(model_files):
        hc,lon,lat,cons = read_GOT_grid(os.path.expanduser(model_file),
            GZIP=GZIP)
        constituents.append(cons)
        dlon = np.abs(lon[1] - lon[0])
        lon = extend_array(lon,dlon)
        hc = extend_matrix(hc)
        hci = np.ma.zeros((npts),dtype=hc.dtype,fill_value=hc.fill_value)
        hci.mask = np.zeros((npts),dtype=bool)
        if (METHOD == 'bilinear'):
            hc[hc.mask] = np.nan
            hci.data[:] = bilinear_interp(lon,lat,hc,ilon,ilat,dtype=hc.dtype)
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif (METHOD == 'spline'):
            f1=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.data.real.T,kx=1,ky=1)
            f2=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.data.imag.T,kx=1,ky=1)
            f3=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.mask.T,kx=1,ky=1)
            hci.data.real[:] = f1.ev(ilon,ilat)
            hci.data.imag[:] = f2.ev(ilon,ilat)
            hci.mask[:] = f3.ev(ilon,ilat).astype(bool)
            hci.data[hci.mask] = hci.fill_value
        else:
            r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.data, method=METHOD, bounds_error=False,
                fill_value=hci.fill_value)
            r2 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.mask, method=METHOD, bounds_error=False, fill_value=1)
            hci.data[:] = r1.__call__(np.c_[ilat,ilon])
            hci.mask[:] = np.ceil(r2.__call__(np.c_[ilat,ilon])).astype(bool)
            hci.mask[:] |= (hci.data == hci.fill_value)
            hci.data[hci.mask] = hci.fill_value
        if EXTRAPOLATE and np.any(hci.mask):
            inv, = np.nonzero(hci.mask)
            hc[hc.mask] = np.nan
            hci.data[inv] = nearest_extrap(lon,lat,hc,ilon[inv],ilat[inv],
                dtype=hc.dtype,cutoff=CUTOFF)
            hci.mask[inv] = np.isnan(hci.data[inv])
            hci.data[hci.mask] = hci.fill_value
        amplitude.data[:,i] = np.abs(hci.data)*SCALE
        amplitude.mask[:,i] = np.copy(hci.mask)
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)

    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    return (amplitude,phase,constituents)

def read_ascii_file(input_file,GZIP=False,TYPE=None,VERSION=None):
    if GZIP:
        with gzip.open(os.path.expanduser(input_file),'rb') as f:
            file_contents = f.read().splitlines()
    else:
        with open(os.path.expanduser(input_file),'r') as f:
            file_contents = f.read().splitlines()
    lonmin,lonmax = np.array(file_contents[0].split(), dtype=np.float64)
    latmin,latmax = np.array(file_contents[1].split(), dtype=np.float64)
    dlon,dlat = np.array(file_contents[2].split(), dtype=np.float64)
    nlon,nlat = np.array(file_contents[3].split(), dtype=int)
    masked_values = file_contents[4].split()
    fill_value = np.float64(masked_values[0])
    lat = np.linspace(latmin, latmax, nlat)
    lon = np.linspace(lonmin,lonmax,nlon)
    amp = np.ma.zeros((nlat,nlon),fill_value=fill_value,dtype=np.float32)
    ph = np.ma.zeros((nlat,nlon),fill_value=fill_value,dtype=np.float32)
    amp.mask = np.zeros((nlat,nlon),dtype=bool)
    ph.mask, i1 = np.zeros((nlat,nlon),dtype=bool), 5
    for i in range(nlat):
        for j in range(nlon//30):
            j1 = j*30
            amp.data[i,j1:j1+30]=np.array(file_contents[i1].split(),dtype='f')
            ph.data[i,j1:j1+30]=np.array(file_contents[i1+1].split(),dtype='f')
            i1 += 2
        j1, j2 = (j+1)*30, nlon % 30
        amp.data[i,j1:j1+j2] = np.array(file_contents[i1].split(),dtype='f')
        ph.data[i,j1:j1+j2] = np.array(file_contents[i1+1].split(),dtype='f')
        i1 += 2
    hc = amp*np.exp(-1j*ph*np.pi/180.0)
    hc.mask = (amp.data == amp.fill_value) | (ph.data == ph.fill_value)
    return (hc,lon,lat)

def read_netcdf_file(input_file,GZIP=False,TYPE=None,VERSION=None):
    if GZIP:
        f = gzip.open(os.path.expanduser(input_file),'rb')
        fileID = netCDF4.Dataset(uuid.uuid4().hex,'r',memory=f.read())
    else:
        fileID = netCDF4.Dataset(os.path.expanduser(input_file),'r')
    if (VERSION == 'FES2012'):
        lon = fileID.variables['longitude'][:]
        lat = fileID.variables['latitude'][:]
    elif (VERSION == 'FES2014'):
        lon, lat = fileID.variables['lon'][:], fileID.variables['lat'][:]
    if (TYPE == 'z'):
        amp = fileID.variables['amplitude'][:]
        ph = fileID.variables['phase'][:]
    elif (TYPE == 'u'):
        amp, ph = fileID.variables['Ua'][:], fileID.variables['Ug'][:]
    elif (TYPE == 'v'):
        amp, ph = fileID.variables['Va'][:], fileID.variables['Vg'][:]
    fileID.close()
    f.close() if GZIP else None
    hc = amp*np.exp(-1j*ph*np.pi/180.0)
    hc.mask = (amp.data == amp.fill_value) | (ph.data == ph.fill_value)
    return (hc,lon,lat)

def extract_FES_constants(ilon, ilat, model_files, TYPE='z', VERSION=None,
    METHOD='spline', EXTRAPOLATE=False, CUTOFF=10.0, GZIP=True, SCALE=1.0):
    if isinstance(model_files,str):
        print("Tide model is entered as a string")
        model_files = [model_files]
    ilon, ilat = np.atleast_1d(ilon), np.atleast_1d(ilat)
    if (np.min(ilon) < 0.0):
        lt0, = np.nonzero(ilon < 0)
        ilon[lt0] += 360.0

    npts, nc = len(ilon), len(model_files)
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    for i,fi in enumerate(model_files):
        if VERSION in ('FES1999','FES2004'):
            hc,lon,lat = read_ascii_file(os.path.expanduser(fi),
                GZIP=GZIP, TYPE=TYPE, VERSION=VERSION)
        elif VERSION in ('FES2012','FES2014'):
            hc,lon,lat = read_netcdf_file(os.path.expanduser(fi),
                GZIP=GZIP, TYPE=TYPE, VERSION=VERSION)
        hci = np.ma.zeros((npts),dtype=hc.dtype,fill_value=hc.fill_value)
        hci.mask = np.zeros((npts),dtype=bool)
        if (METHOD == 'bilinear'):
            hc[hc.mask] = np.nan
            hci.data[:] = bilinear_interp(lon,lat,hc,ilon,ilat,dtype=hc.dtype)
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif (METHOD == 'spline'):
            f1=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.data.real.T,kx=1,ky=1)
            f2=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.data.imag.T,kx=1,ky=1)
            f3=scipy.interpolate.RectBivariateSpline(lon,lat,
                hc.mask.T,kx=1,ky=1)
            hci.data.real[:] = f1.ev(ilon,ilat)
            hci.data.imag[:] = f2.ev(ilon,ilat)
            hci.mask[:] = f3.ev(ilon,ilat).astype(bool)
            hci.data[hci.mask] = hci.fill_value
        else:
            r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.data, method=METHOD, bounds_error=False,
                fill_value=hci.fill_value)
            r2 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.mask, method=METHOD, bounds_error=False, fill_value=1)
            hci.data[:] = r1.__call__(np.c_[ilat,ilon])
            hci.mask[:] = np.ceil(r2.__call__(np.c_[ilat,ilon])).astype(bool)
            hci.mask[:] |= (hci.data == hci.fill_value)
            hci.data[hci.mask] = hci.fill_value
        if EXTRAPOLATE and np.any(hci.mask):
            inv, = np.nonzero(hci.mask)
            hc[hc.mask] = np.nan
            hci.data[inv] = nearest_extrap(lon,lat,hc,ilon[inv],ilat[inv],
                dtype=hc.dtype,cutoff=CUTOFF)
            hci.mask[inv] = np.isnan(hci.data[inv])
            hci.data[hci.mask] = hci.fill_value
        amplitude.data[:,i] = np.abs(hci.data)*SCALE
        amplitude.mask[:,i] = np.copy(hci.mask)
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)

    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    return (amplitude,phase)

def compute_tide_corrections(x, y, delta_time, DIRECTORY=None, MODEL=None,
    EPSG=4326, EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='UTC',
    METHOD='spline', EXTRAPOLATE=True, CUTOFF=10.0, FILL_VALUE=np.nan,
    NEAREST=False):

    fileList = os.listdir(DIRECTORY)
    modelFile = [x for x in fileList if 'Model' in x][0]
    with open(DIRECTORY + '\\' + modelFile, 'r') as f:
        content = f.readlines()
    grid_file = DIRECTORY + content[2].replace('\n','')
    model_file = DIRECTORY + content[0].replace('\n','')
    model_format = 'OTIS'
    model_EPSG = '3413'
    model_type = 'z'
    if NEAREST:
        x, y = find_closest_coords(y, x, grid_file, str(EPSG), int(model_EPSG))
    """
    if (MODEL == 'CATS0201'):
        grid_file = os.path.join(DIRECTORY,'cats0201_tmd','grid_CATS')
        model_file = os.path.join(DIRECTORY,'cats0201_tmd','h0_CATS02_01')
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'CATS2008'):
        grid_file = os.path.join(DIRECTORY,'CATS2008','grid_CATS2008')
        model_file = os.path.join(DIRECTORY,'CATS2008','hf.CATS2008.out')
        model_format, model_EPSG, model_type = 'OTIS', 'CATS2008', 'z'
    elif (MODEL == 'CATS2008_load'):
        grid_file = os.path.join(DIRECTORY,'CATS2008a_SPOTL_Load','grid_CATS2008a_opt')
        model_file = os.path.join(DIRECTORY,'CATS2008a_SPOTL_Load','h_CATS2008a_SPOTL_load')
        model_format, model_EPSG, model_type = 'OTIS', 'CATS2008', 'z'
    elif (MODEL == 'TPXO9-atlas'):
        model_directory = os.path.join(DIRECTORY,'TPXO9_atlas')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30.nc.gz','h_o1_tpxo9_atlas_30.nc.gz',
            'h_p1_tpxo9_atlas_30.nc.gz','h_k1_tpxo9_atlas_30.nc.gz',
            'h_n2_tpxo9_atlas_30.nc.gz','h_m2_tpxo9_atlas_30.nc.gz',
            'h_s2_tpxo9_atlas_30.nc.gz','h_k2_tpxo9_atlas_30.nc.gz',
            'h_m4_tpxo9_atlas_30.nc.gz','h_ms4_tpxo9_atlas_30.nc.gz',
            'h_mn4_tpxo9_atlas_30.nc.gz','h_2n2_tpxo9_atlas_30.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, model_type = 'netcdf', 'z'
        SCALE, GZIP = 1.0/1000.0, True
    elif (MODEL == 'TPXO9-atlas-v2'):
        model_directory = os.path.join(DIRECTORY,'TPXO9_atlas_v2')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v2.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30_v2.nc.gz','h_o1_tpxo9_atlas_30_v2.nc.gz',
            'h_p1_tpxo9_atlas_30_v2.nc.gz','h_k1_tpxo9_atlas_30_v2.nc.gz',
            'h_n2_tpxo9_atlas_30_v2.nc.gz','h_m2_tpxo9_atlas_30_v2.nc.gz',
            'h_s2_tpxo9_atlas_30_v2.nc.gz','h_k2_tpxo9_atlas_30_v2.nc.gz',
            'h_m4_tpxo9_atlas_30_v2.nc.gz','h_ms4_tpxo9_atlas_30_v2.nc.gz',
            'h_mn4_tpxo9_atlas_30_v2.nc.gz','h_2n2_tpxo9_atlas_30_v2.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, model_type = 'netcdf', 'z'
        SCALE, GZIP = 1.0/1000.0, True
    elif (MODEL == 'TPXO9-atlas-v3'):
        model_directory = os.path.join(DIRECTORY,'TPXO9_atlas_v3')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v3.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30_v3.nc.gz','h_o1_tpxo9_atlas_30_v3.nc.gz',
            'h_p1_tpxo9_atlas_30_v3.nc.gz','h_k1_tpxo9_atlas_30_v3.nc.gz',
            'h_n2_tpxo9_atlas_30_v3.nc.gz','h_m2_tpxo9_atlas_30_v3.nc.gz',
            'h_s2_tpxo9_atlas_30_v3.nc.gz','h_k2_tpxo9_atlas_30_v3.nc.gz',
            'h_m4_tpxo9_atlas_30_v3.nc.gz','h_ms4_tpxo9_atlas_30_v3.nc.gz',
            'h_mn4_tpxo9_atlas_30_v3.nc.gz','h_2n2_tpxo9_atlas_30_v3.nc.gz',
            'h_mf_tpxo9_atlas_30_v3.nc.gz','h_mm_tpxo9_atlas_30_v3.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, model_type = 'netcdf', 'z'
        SCALE, GZIP = 1.0/1000.0, True
    elif (MODEL == 'TPXO9-atlas-v4'):
        model_directory = os.path.join(DIRECTORY,'TPXO9_atlas_v4')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v4')
        model_files = ['h_q1_tpxo9_atlas_30_v4','h_o1_tpxo9_atlas_30_v4',
            'h_p1_tpxo9_atlas_30_v4','h_k1_tpxo9_atlas_30_v4',
            'h_n2_tpxo9_atlas_30_v4','h_m2_tpxo9_atlas_30_v4',
            'h_s2_tpxo9_atlas_30_v4','h_k2_tpxo9_atlas_30_v4',
            'h_m4_tpxo9_atlas_30_v4','h_ms4_tpxo9_atlas_30_v4',
            'h_mn4_tpxo9_atlas_30_v4','h_2n2_tpxo9_atlas_30_v4',
            'h_mf_tpxo9_atlas_30_v4','h_mm_tpxo9_atlas_30_v4']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'TPXO9.1'):
        grid_file = os.path.join(DIRECTORY,'TPXO9.1','DATA','grid_tpxo9')
        model_file = os.path.join(DIRECTORY,'TPXO9.1','DATA','h_tpxo9.v1')
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'TPXO8-atlas'):
        grid_file = os.path.join(DIRECTORY,'tpxo8_atlas','grid_tpxo8atlas_30_v1')
        model_file = os.path.join(DIRECTORY,'tpxo8_atlas','hf.tpxo8_atlas_30_v1')
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'TPXO7.2'):
        grid_file = os.path.join(DIRECTORY,'TPXO7.2_tmd','grid_tpxo7.2')
        model_file = os.path.join(DIRECTORY,'TPXO7.2_tmd','h_tpxo7.2')
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'TPXO7.2_load'):
        grid_file = os.path.join(DIRECTORY,'TPXO7.2_load','grid_tpxo6.2')
        model_file = os.path.join(DIRECTORY,'TPXO7.2_load','h_tpxo7.2_load')
        model_format, model_EPSG, model_type = 'OTIS', '4326', 'z'
    elif (MODEL == 'AODTM-5'):
        grid_file = os.path.join(DIRECTORY,'aodtm5_tmd','grid_Arc5km')
        model_file = os.path.join(DIRECTORY,'aodtm5_tmd','h0_Arc5km.oce')
        model_format, model_EPSG, model_type = 'OTIS', 'PSNorth', 'z'
    elif (MODEL == 'AOTIM-5'):
        grid_file = os.path.join(DIRECTORY,'aotim5_tmd','grid_Arc5km')
        model_file = os.path.join(DIRECTORY,'aotim5_tmd','h_Arc5km.oce')
        model_format, model_EPSG, model_type = 'OTIS', 'PSNorth', 'z'
    elif (MODEL == 'AOTIM-5-2018'):
        grid_file = os.path.join(DIRECTORY,'Arc5km2018','grid_Arc5km2018')
        model_file = os.path.join(DIRECTORY,'Arc5km2018','h_Arc5km2018')
        model_format, model_EPSG, model_type = 'OTIS', 'PSNorth', 'z'
    elif (MODEL == 'Gr1km-v2'):
        grid_file = os.path.join(DIRECTORY,'greenlandTMD_v2','grid_Greenland8.v2')
        model_file = os.path.join(DIRECTORY,'greenlandTMD_v2','h_Greenland8.v2')
        model_format, model_EPSG, model_type = 'OTIS', '3413', 'z'
    elif (MODEL == 'GOT4.7'):
        model_directory = os.path.join(DIRECTORY,'GOT4.7','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'GOT4.7_load'):
        model_directory = os.path.join(DIRECTORY,'GOT4.7','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'GOT4.8'):
        model_directory = os.path.join(DIRECTORY,'got4.8','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'GOT4.8_load'):
        model_directory = os.path.join(DIRECTORY,'got4.8','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'GOT4.10'):
        model_directory = os.path.join(DIRECTORY,'GOT4.10c','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'GOT4.10_load'):
        model_directory = os.path.join(DIRECTORY,'GOT4.10c','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        model_format, SCALE, GZIP = 'GOT', 1.0/100.0, True
    elif (MODEL == 'FES2014'):
        model_directory = os.path.join(DIRECTORY,'fes2014','ocean_tide')
        model_files = ['2n2.nc.gz','eps2.nc.gz','j1.nc.gz','k1.nc.gz',
            'k2.nc.gz','l2.nc.gz','la2.nc.gz','m2.nc.gz','m3.nc.gz','m4.nc.gz',
            'm6.nc.gz','m8.nc.gz','mf.nc.gz','mks2.nc.gz','mm.nc.gz',
            'mn4.nc.gz','ms4.nc.gz','msf.nc.gz','msqm.nc.gz','mtm.nc.gz',
            'mu2.nc.gz','n2.nc.gz','n4.nc.gz','nu2.nc.gz','o1.nc.gz','p1.nc.gz',
            'q1.nc.gz','r2.nc.gz','s1.nc.gz','s2.nc.gz','s4.nc.gz','sa.nc.gz',
            'ssa.nc.gz','t2.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        c = ['2n2','eps2','j1','k1','k2','l2','lambda2','m2','m3','m4','m6','m8',
            'mf','mks2','mm','mn4','ms4','msf','msqm','mtm','mu2','n2','n4',
            'nu2','o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
        model_format, TYPE = 'FES', 'z'
        SCALE, GZIP = 1.0/100.0, True
    elif (MODEL == 'FES2014_load'):
        model_directory = os.path.join(DIRECTORY,'fes2014','load_tide')
        model_files = ['2n2.nc.gz','eps2.nc.gz','j1.nc.gz','k1.nc.gz',
            'k2.nc.gz','l2.nc.gz','la2.nc.gz','m2.nc.gz','m3.nc.gz','m4.nc.gz',
            'm6.nc.gz','m8.nc.gz','mf.nc.gz','mks2.nc.gz','mm.nc.gz',
            'mn4.nc.gz','ms4.nc.gz','msf.nc.gz','msqm.nc.gz','mtm.nc.gz',
            'mu2.nc.gz','n2.nc.gz','n4.nc.gz','nu2.nc.gz','o1.nc.gz','p1.nc.gz',
            'q1.nc.gz','r2.nc.gz','s1.nc.gz','s2.nc.gz','s4.nc.gz','sa.nc.gz',
            'ssa.nc.gz','t2.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        c = ['2n2','eps2','j1','k1','k2','l2','lambda2','m2','m3','m4','m6',
            'm8','mf','mks2','mm','mn4','ms4','msf','msqm','mtm','mu2','n2',
            'n4','nu2','o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
        model_format, model_type = 'FES', 'z'
        SCALE, GZIP = 1.0/100.0, True
    else:
        raise Exception("Unlisted tide model")
    """

    try:
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(int(EPSG)))
    except (ValueError,pyproj.exceptions.CRSError):
        crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    if type(x)==float:
        #lon,lat = x,y
        lon,lat = transformer.transform(x, y)
        #lon,lat = lon/1000,lat/1000
    else:
        lon,lat = transformer.transform(x.flatten(), y.flatten())

    delta_time = np.atleast_1d(delta_time)
    if (TIME.upper() == 'GPS'):
        GPS_Epoch_Time = convert_delta_time(0, epoch1=EPOCH,
            epoch2=(1980,1,6,0,0,0), scale=1.0)
        GPS_Time = convert_delta_time(delta_time, epoch1=EPOCH,
            epoch2=(1980,1,6,0,0,0), scale=1.0)
        leap_seconds = count_leap_seconds(GPS_Time) - \
                       count_leap_seconds(np.atleast_1d(GPS_Epoch_Time))
    elif (TIME.upper() == 'TAI'):
        GPS_Epoch_Time = convert_delta_time(-19.0, epoch1=EPOCH,
            epoch2=(1980,1,6,0,0,0), scale=1.0)
        GPS_Time = convert_delta_time(delta_time-19.0, epoch1=EPOCH,
            epoch2=(1980,1,6,0,0,0), scale=1.0)
        leap_seconds = count_leap_seconds(GPS_Time) - \
                       count_leap_seconds(np.atleast_1d(GPS_Epoch_Time))
    else:
        leap_seconds = 0.0

    t = convert_delta_time(delta_time - leap_seconds, epoch1=EPOCH,
        epoch2=(1992,1,1,0,0,0), scale=(1.0/86400.0))
    #delta_file = get_data_path(['data','merged_deltat.data'])

    if model_format in ('OTIS','ATLAS'):
        amp,ph,D,c = extract_tidal_constants(lon, lat, grid_file, model_file,
            model_EPSG, TYPE=model_type, METHOD=METHOD, EXTRAPOLATE=True,
            CUTOFF=np.inf, GRID=model_format)
        deltat = np.zeros_like(t)
    """
    elif (model_format == 'netcdf'):
        amp,ph,D,c = extract_netcdf_constants(lon, lat, grid_file, model_file,
            TYPE=model_type, METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE,
            CUTOFF=CUTOFF, SCALE=SCALE, GZIP=GZIP)
        deltat = np.zeros_like(t)
    elif (model_format == 'GOT'):
        amp,ph,c = extract_GOT_constants(lon, lat, model_file, METHOD=METHOD,
            EXTRAPOLATE=EXTRAPOLATE, CUTOFF=CUTOFF, SCALE=SCALE, GZIP=GZIP)
        deltat = calc_delta_time(delta_file, t)
    elif (model_format == 'FES'):
        amp,ph = extract_FES_constants(lon, lat, model_file, TYPE=model_type,
            VERSION=MODEL, METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE,
            CUTOFF=CUTOFF, SCALE=SCALE, GZIP=GZIP)
        deltat = calc_delta_time(delta_file, t)
    """

    cph = -1j*ph*np.pi/180.0
    hc = amp*np.exp(cph)

    if (TYPE.lower() == 'grid'):
        ny,nx = np.shape(x); nt = len(t)
        tide = np.ma.zeros((ny,nx,nt),fill_value=FILL_VALUE)
        tide.mask = np.zeros((ny,nx,nt),dtype=bool)
        for i in range(nt):
            TIDE = predict_tide(t[i], hc, c,
                DELTAT=deltat[i], CORRECTIONS=model_format)
            MINOR = infer_minor_corrections(t[i], hc, c,
                DELTAT=deltat[i], CORRECTIONS=model_format)
            tide[:,:,i] = np.reshape((TIDE+MINOR), (ny,nx))
            tide.mask[:,:,i] = np.reshape((TIDE.mask | MINOR.mask), (ny,nx))
    else:
        npts = len(t)
        tide = np.ma.zeros((npts), fill_value=FILL_VALUE)
        tide.mask = np.any(hc.mask,axis=1)
        tide.data[:] = predict_tide_drift(t, hc, c,
            DELTAT=deltat, CORRECTIONS=model_format)
        minor = infer_minor_corrections(t, hc, c,
            DELTAT=deltat, CORRECTIONS=model_format)
        tide.data[:] += minor.data[:]
    tide.data[tide.mask] = tide.fill_value

    return tide