'''In this code, we have 6 states, 2 inputs, and 6 outputs. The outputs were chosen in accordance with the paper 
'Flatness-based MPC for underactuated surface vessels in confined areas' as y = [x,y,psi]'''

from vessel import Vessel
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp

def predict(t):
    vs = Vessel()
    ti = 0
    dt = 1
    Sk = sp.eye(2) # depends on number of inputs (inputsxinputs)
    Pplus = sp.eye(6) # covariance matrix, states x states 
    Upsilon = sp.zeros(6,2) # 6 rows and 2 columns, states x inputs
    theta = sp.Matrix([[0],[0]]) # fault in any of the control inputs so same shape as inputs
    Qf = 0.0001*sp.eye(6) # shape proportional to number of states
    Rf =  0.0001*sp.eye(6) # shape proportional to number of outputs
    a = 0.2 # random factor that they do not explain anyways

    while ti < t:

        Pminus = vs.Fk() * Pplus * vs.Fk() + Qf
        Sigma = vs.Cobvs * Pminus * vs.Cobvs.T + Rf
        K = Pminus * vs.Cobvs.T*Sigma
        Pplus = (sp.eye(6) - K*vs.Cobvs)*Pplus
        Upsilon =  (sp.eye(6) - K * vs.Cobvs) * vs.Fk()*Upsilon + (sp.eye(6) - K * vs.Cobvs) * vs.phi()
        Omega = vs.Cobvs * vs.Fk() * Upsilon + vs.Cobvs * vs.phi()
        Lambda = (vs.llambda * Sigma + Omega*Sk*Omega.T).inv()
        Tau = Sk * Omega.T * Lambda
        Sk = (1/vs.llambda)*Sk - (1/vs.llambda)*Omega.T*Lambda*Omega*Sk 
        thetak = theta # saving the previous theta
        theta = theta + Tau*vs.ytilde(Rf)
        Qf = a*Qf + (1-a) * (K*vs.ytilde(Rf)*vs.ytilde(Rf).T*K)
        Rf = a*Rf + (1-a) * (vs.ytilde(Rf)*vs.ytilde(Rf).T + vs.Cobvs*Pplus*vs.Cobvs.T)

        # make F numerical 
        
        vs.Update(vs.A * (vs.x)+ vs.F() + vs.B*vs.u_input + vs.phi()*theta + K*vs.ytilde(Rf) + Upsilon*(theta - thetak))

        # vs.Update(vs.A * (vs.x)+ dt*vs.F() + dt*vs.B*vs.u)
        xtab_FE.append(vs.x)
        ttab_FE.append(ti)
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
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(xtab_plot[:,0,:].flatten(), xtab_plot[:,1,:].flatten(), color = CB91_Blue)
    ax1.grid()
    ax1.set_title('Vessel\'s XY plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('X')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.plot(ttab_plot, xtab_plot[:,3,:].flatten(), label = 'u', color = CB91_Purple)
    ax2.plot(ttab_plot, xtab_plot[:,4,:].flatten(), label = 'v', color = CB91_Amber)
    ax2.plot(ttab_plot, xtab_plot[:,5,:].flatten(), label = 'r', color = CB91_Green)
    ax2.set_title('Vessel u,v and r rates')
    ax2.set_xlabel('Time [s]')
    ax2.legend()

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.plot(ttab_plot, xtab_plot[:,2,:].flatten(),label = '$\psi$', color = CB91_Violet)
    ax3.set_title('Vessel\'s $\psi$')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('$\psi$')

    fig.tight_layout()
    fig.suptitle('Vessel Dynamics')
    plt.show()

if __name__ == '__main__':
    xtab_FE = []
    ttab_FE = []
    predict(20)
    plot_results()
