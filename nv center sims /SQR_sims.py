from sim_setup import *

## perform a single quantum ramsey sim with the Hamiltonian defined in 
## hyperfine_term.py

""" 
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
"""

# The loop below runs the mesolve function in qutip and appends a list with a result
# for each point in time

sqr_args = {'h1':47, 'wrf':2000, 't_spacing':0.25, 'phi1':0, 'phi2':np.pi/2}
sqr_list = []
sqr_time = np.linspace(0, 1, 256)
h1=47
t90 = np.pi/(np.sqrt(2)*h1)
count = 0
psi_NV = qt.basis(3, 1)
psi_N = qt.Qobj([[1],[1],[1]])/np.sqrt(3)

for i in sqr_time:
    sqr_args = {'h1':h1, 'wrf':2000, 't_spacing':i, 'phi1':np.pi/2, 'phi2':-np.pi/2}
    tlist = np.arange(0, 2*t90+i, 0.00001)
    Hamtot = [h_NV_full(870/28025, d_param=2870, gamma=28025), [qt.tensor(qt.spin_Jx(1), qt.qeye(3)), ramsey_pulse_coef]]
    print(type(Hamtot))
    res = qt.mesolve(Hamtot, qt.tensor(psi_NV, psi_N), tlist, [], [], args=sqr_args, progress_bar=True)
    print('step '+ str(count)+' out of '+str(len(sqr_time))+' complete')
    midel = (res.states[-1].dag()*qt.tensor(psi_NV, psi_N))[0][0][0]
    sqr_list.append(np.conjugate(midel)*midel)
    count += 1

## the list can be plotted vs time to get an representation of the
## signal, on which we can perform an fft 


