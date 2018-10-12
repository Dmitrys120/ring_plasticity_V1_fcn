# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:55:48 2015

@author: Pavel Esir
"""
from __future__ import division
from matplotlib.gridspec import GridSpec
from matplotlib.pylab import *
from numpy import *
seed(0)

# if CalcMode is equal to 0 then a single realisation is calculated and then printed

# if CalcMode is equal to 1 then realisation for different U is calculated and saved into files

# if CalcMode is equal to 2 then U is set from argv and then single realisation is calculated
# can be used for parallel calculation with GNU parallel

# if CalcMode is equal to 3 then I0 vs U curve is calculated by bisection method
CalcMode = 0


SimTime = 10.0        # seconds
h = 0.002             # seconds
pltSampl = 0.002       # variable save interval in s
Tsim = int(SimTime/h)

# to load empirical calculated U vs I0 dependence
Mm = 0.5
Urange = arange(0.05, 0.951, 0.05)

Ierange = [-1.170, -0.917, -0.669, -0.485, -0.485, -0.542, -0.711,
           -0.934, -1.154, -1.398, -1.690, -1.950, -2.256, -2.389,
           -2.543, -2.717, -2.855, -3.033, -3.150]
#%%
D = 2.0
J0 = -12
J1 = 30

# duration of events in sec
T = 0.25
# amplitude of events
C = 20.0
# input poisson events rate, Hz
freq = 4

tau_r = 0.01

tau = 0.01
tau_n = 0.1
U = 0.3
tau_rec = 0.3
tau_facil = 1.5
N = 180
I0 = Ierange[int(U/0.05) - 1]

folderName = 'res_h_0.0020_D_{}_freq_{:.1f}_T_{:.2f}_m_{:.1f}/'.format(D, freq, T, Mm)
#Ierange = load('U_Iex_SimTime_20.0_h_0.0020_D_{}_N_200_eps_0.010_m_{:.1f}.npy'.format(D, Mm))

stime = arange(0, SimTime, h)
m = zeros(N)

ActM = zeros((int(SimTime/pltSampl), N), dtype='float32')
ActX = zeros((int(SimTime/pltSampl), N), dtype='float32')
ActU = zeros((int(SimTime/pltSampl), N), dtype='float32')
x = ones(N)
u = np.array([U]*N)
Inoise = zeros(N)
Iex = zeros(Tsim)
IexTest = zeros(Tsim)
ThetaEs = zeros(Tsim)
#ThetaEsTest = zeros(Tsim)

th = linspace(-pi, pi, N, endpoint=False)
testStimulRecepients = ones(N)
alpha, beta = meshgrid(th, th)
W = (J0 + J1*cos(2*(alpha - beta)))/N
del alpha, beta
#%%
# number of input events, nearly roughly poisson rate*time(колличество входных событий, примерно равных частоте пуассона * время)
Nev = int(freq*SimTime)

# times of Poisson events have exponential distribution(времена событий пуассона имеют эксп распределение)
#inpTimes = (exponential(1/freq, Nev) + T).cumsum()/h
#inpTimes = arange(0, 1, 0.1).cumsum()/h
startStim = 0. # sec
stopStim = 3. # sec
tInterStim = 0.3  # sec
inpTimes = arange(startStim, stopStim, tInterStim)/h
inpTimes = array(inpTimes, dtype='int')
#inpTimes = concatenate((inpTimes, [4/h]))
#inpTimes=append(inpTimes,2000)
inpTimesTest= array([2000, 3000, 4000])
Nev = len(inpTimes)
inpTheta = uniform(pi/4, pi/4, Nev)

for t, theta in zip(inpTimes, inpTheta):
    Iex[t:int(t + T/h)] = C
    ThetaEs[t:int(t + T/h)] = theta
for t, theta in zip(inpTimesTest, inpTheta):
    IexTest[t:int(t + T/h)] = C/2.
    #ThetaEsTest[t:int(t + T/h)] = theta

Nreads = linspace(int(N/10), N, 10, dtype='int')
R = zeros((Tsim, len(Nreads)), dtype='complex')
exactR = zeros(Tsim, dtype='complex')

choises = [[]]*len(Nreads)
for idx, n in enumerate(Nreads):
    choises[idx] = choice(range(N), n, replace=None)
#%%
def integrate():
    global Inoise, x, m, u
    for t in range(0, Tsim - 1):

        #Inoise = Inoise + (-Inoise*h/tau_n + D*sqrt(2*h/tau_n)*randn(N))
        u = u + ((U - u)/ tau_facil + U*(1-u)*m)*h
        x = x + ((1 - x)/tau_rec - u*x*m)*h
        Irec1 = Iex[t]*cos(th - ThetaEs[t])
        Itest1 = IexTest[t]*testStimulRecepients
        m = m + (- m + log(1 + exp( dot(W, u*x*m) + I0 + Irec1 + Itest1)))*(h/tau)
        #m = m + (- m + log(1 + exp((dot(W, u*x*m) + I0 + Iex[t]*cos(th - ThetaEs[t]) + Inoise))))*(h/tau)

        if t % int(pltSampl/h) == 0:
            ActM[int(t/(pltSampl/h))] = m
            ActX[int(t/(pltSampl/h))] = x
            ActU[int(t/(pltSampl/h))] = u
        #exactR[t+1] = sum(exp(1j*th)*m)/N
        #if CalcMode != 3:
            #for idx, Nread in enumerate(Nreads):
                #R[t+1, idx] = R[t, idx] + (-R[t, idx] + (1/Nread)*sum(exp(1j*th[choises[idx]])*poisson(m[choises[idx]]*h, Nread)))*h/tau_r

# error estimation function(функция оценивающая ошибку)
def estimErrDiffNread(lag=0):
    errEsR = zeros((len(Nreads), Nev))

    for i, (t, theta) in enumerate(zip(inpTimes, inpTheta)):
        for idx, Nread in enumerate(Nreads):
            nev = abs(angle(R[t + lag:int(t + lag + T/h), idx]) - ThetaEs[t:int(t + T/h)])
            anglDiff = amin([nev, 2*pi - nev], axis=0)
            errEsR[idx, i] = mean(anglDiff)

    return errEsR

def estimErrExactR(lag=0):
    errEsR = zeros(Nev)
    for i, (t, theta) in enumerate(zip(inpTimes, inpTheta)):
        nev = abs(angle(exactR[t + lag:int(t + lag + T/h)]) - ThetaEs[t:int(t + T/h)])
        anglDiff = amin([nev, 2*pi - nev], axis=0)
        errEsR[i] = mean(anglDiff)
    return errEsR

def calcErrDiffNread():
    lags = arange(0, int(T/h), 1, dtype='int')

    errR = zeros((len(lags), len(Nreads), Nev))
    errExactR = zeros((len(lags), Nev))

    for j, lag in enumerate(lags):
        errR[j] = estimErrDiffNread(lag)
        errExactR[j] = estimErrExactR(lag)
    return errR, errExactR

if CalcMode == 0:
    integrate()
#%%
    # plotting results(графики)
    figure(figsize=(4*2.5, 3*2.5))
#    gs = GridSpec(4, 2, height_ratios=[1, 1, 2, 1./3], width_ratios=[30, 1])
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[30, 1])
    gs.update(wspace=0.1, hspace=0.1)
    axM = subplot(gs[0, 0])
    axSpec = subplot(gs[1, 0], sharex=axM)
    axSpec1 = subplot(gs[2, 0], sharex=axM)
    axAngle = subplot(gs[3, 0], sharex=axM)
#    axEx = subplot(gs[3, 0], sharex=axM)
    axCbar = subplot(gs[:, 1])

    egg = axM.pcolormesh(arange(0, SimTime, pltSampl), th*360/(4*pi), ActM.T)
    spam = colorbar(egg, cax=axCbar)
    axCbar.set_title("m[Hz]")
   # axM.set_ylim([-90, 90])
    axM.set_ylabel(r"$\theta$")
    setp(axM.get_xticklabels(), visible=False)
    axM.set_title('$U={}\quad I_0={:.2f}$'.format(U, I0))

    #actxx=axSpec.pcolormesh(arange(0, SimTime, pltSampl),th*360/(4*pi), ActX.T,)
    #spam = colorbar(actxx, cax=axCbar)
    axSpec.plot(arange(0, SimTime, pltSampl), ActX[:, (0,80,135)])
   # axSpec.plot(stime, abs(exactR), label='exact readout')
   # axSpec.plot(stime, abs(R[:, 3]*Nreads[3]), label='sparse readout')
    #axSpec.set_ylim([0, 16])
#    axSpec.plot(arange(0, SimTime, pltSampl), mean(ActM, axis=1))
    setp(axSpec.get_xticklabels(), visible=False)
    axSpec.set_ylabel(r"$X$")
    #axSpec.set_ylabel(r"$|R|$")
    axSpec.legend(fontsize=16., loc='upper right')

    #actuu=axSpec1.pcolormesh(arange(0, SimTime, pltSampl),th*360/(4*pi), ActU.T,)
    axSpec1.plot(arange(0, SimTime, pltSampl), ActU[:, (0,80,135)])
    #spam = colorbar(actuu, cax=axCbar)
    setp(axSpec1.get_xticklabels(), visible=False)
    axSpec1.set_ylabel(r"$U$")
    axSpec1.legend(fontsize=16., loc='upper right')
    

    #actux=axAngle.pcolormesh(arange(0, SimTime, pltSampl),th*360/(4*pi), ActX.T*ActU.T)
    axAngle.plot(arange(0, SimTime, pltSampl), ActX[:, (0,80,135)]*ActU[:, (0,80,135)])
    #axAngle.plot(stime, angle(exactR)*360/(4*pi), label='exact readout')
    #axAngle.plot(stime, angle(R[:, 3])*360/(4*pi), label='sparse readout')
    #axAngle.hlines(inpTheta*360/(4*pi), inpTimes*h, inpTimes*h + T, 'C3', lw=5.)
    #axAngle.set_ylim([-55, 55])
    axAngle.legend(fontsize=16., loc='lower right')
    axAngle.set_ylabel(r"$SynapticVar$")
    axAngle.plot([0.,SimTime], [U, U], lw = 0.5, c = 'k', label = r'$Ux_0$')

    #axAngle.set_ylabel(r"$angle(R)$")
    axAngle.set_xlabel('Time[s]')
#    axEx.set_xlim([0, SimTime])
    #axAngle.set_xlim((4.59, 5.81))
    savefig('U_{}.png'.format(U), dpi=260.)
    show()
    #%%
    #  figure(figsize=(4*2.3, 3*2.3))
    #  plot(stime, angle(exactR)*360/(4*pi), label='exact readout')
    #  plot(stime, angle(R[:, 3])*360/(4*pi),  label='sparse readout')
    #  hlines(inpTheta*360/(4*pi), inpTimes*h, inpTimes*h + T, 'C3', linewidth=2.)
   #  # ylim([-90, 90])
    #  legend(fontsize=16., loc='upper right')
   #  # xlim((5.56, 5.8))
   #  # ylim((-11.0, 31.0))
    #  xlabel('Time[s]')
    #  ylabel('angle(R)')
#    savefig('angle_U_{}.png'.format(U), dpi=260.)
    #%%
#    figure(figsize=(4*2.5, 3*2.5))
#    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[30, 1])
#    gs.update(wspace=0.1, hspace=0.1)
#    axM = subplot(gs[0, 0])
#    axSpec = subplot(gs[1, 0], sharex=axM)
#    axCbar = subplot(gs[:, 1])
#    axM.set_title('$U={}\quad I_0={:.2f}$'.format(U, I0))
#
#    egg = axM.pcolormesh(arange(0, SimTime, pltSampl), th, ActM.T)
#    spam =colorbar(egg, cax=axCbar)
#    axCbar.set_title("m[Hz]")
#    axM.set_ylim([-pi, pi - pi/N])
#    axM.set_ylabel(r"$\theta$")
#    setp(axM.get_xticklabels(), visible=False)
#    axSpec.plot(arange(0, SimTime, pltSampl), mean(ActM, axis=1))
#    axSpec.set_ylabel(r"$<m>$")
#    axSpec.set_xlabel(r"$Time [s]$")
#    axSpec.set_xlim((0, SimTime))
#    axSpec.set_ylim((0, 0.75))
#    savefig('no_stim_U_{}.png'.format(U), dpi=260.)
#%%
    #errR, errExactR = calcErrDiffNread()

    ## errR shape lags, Nreads, Number of stimulus
    #minLags = argmin(mean(errR, axis=2), axis=0)*h*1000
    #minLagErr = amin(mean(errR, axis=2), axis=0)*360/(4*np.pi)

    #print("Lags for minimal error {} ms".format(minLags))
    #print("Minimal errors {}".format(minLagErr))

    #minLag = argmin(mean(errExactR, axis=-1), axis=0)*h*1000
    #minErr = amin(mean(errExactR, axis=-1), axis=0)*360/(4*np.pi)

    #print("Lag for minimal error {} ms (exact PV)".format(minLag))
    #print("Minimal error {} (exact PV)".format(minErr))

    #df = mean(ma.array(abs(R[:, -1]), mask=Iex))
##    print("Mean value between intervals {}".format(mean(df)))

    #df = mean(ma.array(abs(R[:, -1]), mask=~array(Iex, dtype='bool')))
##    print("Mean value when stimulus apply {}".format(mean(df)))
#%%
elif CalcMode == 1:
    for U, I0 in zip(Urange, Ierange):
        integrate()
        print("Calculating for U: {}".format(U))

        errR, errExactR = calcErrDiffNread()
        save(folderName + 'U_{:.2f}_C_{:.1f}_N_{:n}_SimTime_{:n}.npy'.format(U, C, N, SimTime), errR)
        save(folderName + 'U_{:.2f}_C_{:.1f}_N_{:n}_SimTime_{:n}_exactPV.npy'.format(U, C, N, SimTime), errExactR)
elif CalcMode == 2:
    import sys
    U = float(sys.argv[1])
    I0 = Ierange[int(U/0.05) - 1]
    integrate()

    errR, errExactR = calcErrDiffNread()
    save(folderName + 'U_{:.2f}_C_{:.1f}_N_{:n}_SimTime_{:n}.npy'.format(U, C, N, SimTime), errR)
    save(folderName + 'U_{:.2f}_C_{:.1f}_N_{:n}_SimTime_{:n}_exactPV.npy'.format(U, C, N, SimTime), errExactR)
elif CalcMode == 3:
    Iex[:] = 0
    fname = 'U_Iex_SimTime_{:.1f}_h_{:.4f}_D_{:.1f}_N_{:n}_eps_{:.3f}_m_{:.1f}.npy'

    Eps = .01
    meanAct = 2.0

    Imax = 15.0
    Imin = -15.0
    Ilow, Ihigh = Imin, Imax

    Ierange = zeros_like(Urange) + nan

    for idx, U in enumerate(Urange):
        for j in xrange(100):
            m = zeros(N)
            x = ones(N)
            Inoise = zeros(N)

            seed(1)

            integrate()
            cmeanAct = mean(ActM)

            if abs(meanAct - cmeanAct) < Eps:
                Ilow, Ihigh = Imin, Imax
                Ierange[idx] = I0
                break
            if (cmeanAct - meanAct) < 0:
                Ilow = I0
            else:
                Ihigh = I0
            I0 = (Ihigh + Ilow)*0.5
        print (U, I0, mean(ActM))
    save(fname.format(SimTime, h, D, N, Eps, meanAct), Ierange)
