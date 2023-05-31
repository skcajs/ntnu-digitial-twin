'''In this code, we have 6 states, 2 inputs, and 6 outputs. The outputs were chosen in accordance with the paper 
'Flatness-based MPC for underactuated surface vessels in confined areas' as y = [x,y,psi]'''

from vessel import Vessel
from matplotlib import pyplot as plt
import numpy as np

def predict(t):
    vs = Vessel()
    vs_exact = Vessel()
    ti = 0
    dt = 0.01
    Sk = np.eye(2) # depends on number of inputs (inputsxinputs)
    Pplus = np.eye(6) # covariance matrix, states x states 
    Upsilon = np.zeros((6,2)) # 6 rows and 2 columns, states x inputs
    theta = np.array([[0],[0]]) # fault in any of the control inputs so same shape as inputs
    Qf = 0.01*np.eye(6) # shape proportional to number of states
    Rf =  0.01*np.eye(6) # shape proportional to number of outputs
    a = 0.995 # random factor that they do not explain anyways


    while ti < t:
        vs_exact.Update(vs_exact.A @ vs_exact.X + dt*vs_exact.F() + dt*vs_exact.B @ vs_exact.u_input) # calcualtes x
        y = vs_exact.Cobvs @ vs_exact.X
        Pminus = vs.Fk(dt) @ Pplus @ vs.Fk(dt).T + Qf
        Sigma = vs.Cobvs @ Pminus @ vs.Cobvs.T + Rf
        K = Pminus @ vs.Cobvs.T @ np.linalg.inv(Sigma)
        Pplus = (np.eye(6) - (K @ vs.Cobvs)) @ Pminus
        Omega = vs.Cobvs @ vs.Fk(dt) @ Upsilon + vs.Cobvs @ vs.phi()
        Upsilon =  ((np.eye(6) - K @ vs.Cobvs) @ vs.Fk(dt) @ Upsilon) + (np.eye(6) - K @ vs.Cobvs) @ vs.phi()
        Lambda = np.linalg.inv((vs.llambda * Sigma) + (Omega@Sk@Omega.T))
        Tau = Sk @ Omega.T @ Lambda
        Sk = (1/vs.llambda)*Sk - (1/vs.llambda)*Omega.T@Lambda@Omega@Sk 
        thetak = theta # saving the previous theta
        theta = theta + Tau@vs.ytilde(y)
        Qf = a*Qf + (1-a) * (K@vs.ytilde(y)@vs.ytilde(y).T@K)
        Rf = a*Rf + (1-a) * (vs.ytilde(y)@vs.ytilde(y).T + vs.Cobvs @ Pplus @ vs.Cobvs.T)
        vs.Update(vs.A @ vs.X + dt*vs_exact.F() + vs.B @ vs.u_input + vs.phi() @ thetak + K @ vs.ytilde(y) + Upsilon @ (theta - thetak)) # calcualtes x_hat

        xtab_FE.append(vs_exact.X)
        ttab_FE.append(ti)
        xhatt.append(vs.X)
        ti += dt 
        print('running t=', ti)


def plot_results():

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'

    xtab_plot = np.array(xtab_FE)
    ttab_plot = np.array(ttab_FE)
    xhatt_plot = np.array(xhatt)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(xtab_plot[:,0,:].flatten(), xtab_plot[:,1,:].flatten(), label = 'x', color = CB91_Blue)
    ax1.plot(xhatt_plot[:,0,:].flatten(), xhatt_plot[:,1,:].flatten(), label = 'x_hat', color = 'black', linestyle = 'dotted')
    ax1.grid()
    ax1.legend()
    ax1.set_title('Vessel\'s XY plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.plot(ttab_plot, xtab_plot[:,3,:].flatten(), label = 'u', color = CB91_Purple)
    ax2.plot(ttab_plot, xtab_plot[:,4,:].flatten(), label = 'v', color = CB91_Amber)
    ax2.plot(ttab_plot, xtab_plot[:,5,:].flatten(), label = 'r', color = CB91_Green)
    ax2.plot(ttab_plot, xhatt_plot[:,3,:].flatten(), label = 'u KE', color = CB91_Purple,  linestyle = 'dotted')
    ax2.plot(ttab_plot, xhatt_plot[:,4,:].flatten(), label = 'v KE', color = CB91_Amber,  linestyle = 'dotted')
    ax2.plot(ttab_plot, xhatt_plot[:,5,:].flatten(), label = 'r KE', color = CB91_Green,  linestyle = 'dotted')
    ax2.set_title('Vessel u,v and r rates')
    ax2.set_xlabel('Time [s]')
    ax2.legend()

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.plot(ttab_plot, xtab_plot[:,2,:].flatten(),label = '$\psi$', color = CB91_Amber)
    ax3.plot(ttab_plot, xhatt_plot[:,2,:].flatten(),label = '$\psi KE$', color = CB91_Amber,  linestyle = 'dotted')
    ax3.set_title('Vessel\'s $\psi$')
    ax3.set_xlabel('Time [s]')
    ax3.legend()
    ax3.set_ylabel('$\psi$')

    fig.tight_layout()
    fig.suptitle('Vessel Dynamics')
    plt.show()

def theta_plot():
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(2, 2, 1)
    

    return

if __name__ == '__main__':
    xtab_FE = []
    ttab_FE = []
    xhatt = []
    predict(20)
    plot_results()
