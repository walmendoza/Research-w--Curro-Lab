
###define functions and objects

import qutip as qt
import numpy as np
import scipy.fftpack as fftpack
from detecta import detect_peaks
from scipy.optimize import curve_fit
import pickle
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import scipy.stats as stats

#this function returns the single NV center hamiltonian
def h_NV_Single(h0, d_param=2870, gamma = 28025):
    return d_param*2*np.pi*qt.spin_Jz(1)**2 + 2*np.pi*(gamma*h0)*qt.spin_Jz(1)

#this function outputs a sign wave.
# takes in some arguments as a dictionary
# h1 is the power (amplitude) of the sin wave
# wrf is the frequency (use that to select transition)
# phi is the phase (good for phase cycling)
# t_start and t_duration are for the start and stop times of the pulse, i use a heaviside step function to make this work
# --- play with these parameters and plot them to see how it works----
def mw_sin_pulse_coeff(t, args):
    h1 = args['h1']
    wrf = args['wrf']
    phi = args['phi']
    t_start = args['t_start']
    t_duration = args['t_duration']
    return h1* np.sin(2*np.pi*wrf* (t-t_start) + phi)* (np.heaviside(t - t_start, 1)*(1-np.heaviside(t-(t_start+t_duration),1)))

#implenting the mw_sin_pulse_coeff function with specific args so that we have a "ramsey pulse" -- just 2 t90 pulses separated by soem spacing
def ramsey_pulse_coef(t, args):
    h1 = args['h1']
    wrf = args["wrf"]
    t_spacing = args["t_spacing"]
    phi1 = args['phi1']
    phi2 = args['phi2']
    #t90 = args['t90']
    t90 = np.pi/(np.sqrt(2)*h1) #if you know the power of your pulse then this is your theoretical t90
    pulse1 = {'h1':h1,'wrf':wrf,"phi":phi1,"t_duration":t90,"t_start":0}
    pulse2 = {'h1':h1,'wrf':wrf,"phi":phi2,"t_duration":t90,"t_start":t90+t_spacing}

    return mw_sin_pulse_coeff(t, pulse1) + mw_sin_pulse_coeff(t, pulse2)

#same as mw_sin_pulse_coeff but for two frequencies together
def mw_sin_twotone_pulse_coeff(t, args):
    h1 = args['h1']
    wrf1 = args['wrf1']
    wrf2 = args['wrf2']
    phi1 = args['phi1']
    phi2 = args['phi2']
    t_start = args['t_start']
    t_duration = args['t_duration']
    return h1* (np.sin(wrf1* (t-t_start) + phi1) + np.sin(wrf2* (t-t_start) + phi2))* (np.heaviside(t - t_start, 1)*(1-np.heaviside(t-(t_start+t_duration),1)))

#same as remasey_pulse_coef but for two frequencies
def ramsey_twotone_pulse_coeff(t, args):
    h1 = args['h1']
    wrf1 = args["wrf1"]
    wrf2 = args["wrf2"]
    t_spacing = args["t_spacing"]
    phi11 = args['phi11'] #phase of pulse channel 1 first
    phi12 = args['phi12'] #phase of pulse channel 1 second
    phi21 = args['phi21'] #phase of pulse channel 2 first
    phi22 = args['phi22'] #phase of pulse channel 2 second
    t90 = np.pi/(h1)
    pulse11 = {'h1':h1,'wrf':wrf1,"phi":phi11,"t_duration":t90,"t_start":0}
    pulse12 = {'h1':h1,'wrf':wrf1,"phi":phi12,"t_duration":t90,"t_start":t90+t_spacing}
    pulse21 = {'h1':h1,'wrf':wrf2,"phi":phi21,"t_duration":t90,"t_start":0}
    pulse22 = {'h1':h1,'wrf':wrf2,"phi":phi22,"t_duration":t90,"t_start":t90+t_spacing}
    return mw_sin_pulse_coeff(t, pulse11) + mw_sin_pulse_coeff(t, pulse12) + mw_sin_pulse_coeff(t, pulse21) + mw_sin_pulse_coeff(t, pulse22)

def sq_echo_pulse(t, args):
    h1 = args['h1']
    wrf = args['wrf']
    phi1 = args['phi']
    echo_t1 = args['echo_t1']
    echo_t2 = args['echo_t2']
    # t_start = args['t_start']
    # t_duration = args['t_duration']
    t90 = args['t90']
        
    pulse1 = {'h1':h1, 'wrf':wrf, 'phi':phi1, 't_start':0, 't_duration':t90}
    pulse2 = {'h1':h1, 'wrf':wrf, 'phi':phi1, 't_start':t90+echo_t1, 't_duration':2*t90}
    pulse3 = {'h1':h1, 'wrf':wrf, 'phi':phi1, 't_start':3*t90+echo_t1+echo_t2, 't_duration':t90}
    return mw_sin_pulse_coeff(t, pulse1) + mw_sin_pulse_coeff(t, pulse2) + mw_sin_pulse_coeff(t, pulse3)
   
def dq_echo_pulse(t, args): #should be similar to sq_echo_pulse function above, just with mw_twotone_pulse_coeff() being used.
    h1 = args['h1']
    wrf1 = args['wrf1']
    wrf2 = args['wrf2']
    phi1 = args['phi1']
    phi2 = args['phi2']
    echo_t1 = args['echo_t1']
    echo_t2 = args['echo_t2']
    t90 = args['t90']
    #still will have 3 pulses, but the pulses wll be using a different function
    pulse1 = {'h1':h1, 'wrf1':wrf1, 'wrf2':wrf2, 'phi1':phi1, 'phi2':phi2, 't_start':0, 't_duration':t90}
    pulse2 = {'h1':h1, 'wrf1':wrf1, 'wrf2':wrf2, 'phi1':phi1, 'phi2':phi2, 't_start':t90+echo_t1, 't_duration':2*t90}
    pulse3 = {'h1':h1, 'wrf1':wrf1, 'wrf2':wrf2, 'phi1':phi1, 'phi2':phi2, 't_start':3*t90+echo_t1+echo_t2, 't_duration':t90}
    #return the final pulse, sum of all three twotone pulses
    return mw_sin_twotone_pulse_coeff(t, pulse1) + mw_sin_twotone_pulse_coeff(t, pulse2) + mw_sin_twotone_pulse_coeff(t, pulse3)



"""some helper functions below that do specific things"""
def complexabs(input): #just a complex abs value function
    return np.conjugate(input)*input

def fourier_transform(xlist, ylist):
    yf = fft(ylist)
    xf = fftfreq(len(xlist), xlist[1]-xlist[0])
    yffinal = np.fft.fftshift(yf)
    xffinal = np.fft.fftshift(xf)
    # return xffinal, np.abs(yffinal)
    return np.array([xffinal, yffinal])


### hyperfine term defined below

def h_NV_full(h0, d_param=2870 , gamma=28025): #both in MHz
    
    Apar = 214 #MHz, was originally 2.14 MHz
    Aperp = 0 #2.7 #MHz
    gamma_e = gamma #see above param gamma
    electron_spin_ham = d_param*2*np.pi * qt.tensor(qt.spin_Jz(1)**2, qt.qeye(3)) + 2*np.pi*gamma*h0 * qt.tensor(qt.spin_Jz(1), qt.qeye(3))
    #nuclear_spin_ham = return to this with a time-dependent hamiltonian
    spin_interaction_ham = 2*np.pi*(Apar * qt.tensor(qt.spin_Jz(1), qt.spin_Jz(1))) #+ Aperp*(qt.tensor(qt.spin_Jx(1), qt.spin_Jx(1)) + qt.tensor(qt.spin_Jy(1), qt.spin_Jy(1)))
    
    return electron_spin_ham + spin_interaction_ham
    
#print('shape of ham is: ', np.shape(h_NV_full(870/28025)))
#print(h_NV_full(870/28025))
