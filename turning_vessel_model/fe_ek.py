'''In this code, we have 6 states, 2 inputs, and 6 outputs. The outputs were chosen in accordance with the paper 
'Flatness-based MPC for underactuated surface vessels in confined areas' as y = [x,y,psi]'''

from vessel import Vessel
from matplotlib import pyplot as plt
import numpy as np

def predict(t_tot, ti, dt, u_input):

    timestamp = []
    x_state = []
    x_hat = []

    vs = Vessel(u_input)
    vs_exact = Vessel(u_input)
    Sk = np.eye(2) # depends on number of inputs (inputsxinputs)
    Pplus = np.eye(6) # covariance matrix, states x states 
    Upsilon = np.zeros((6,2)) # 6 rows and 2 columns, states x inputs
    theta = np.array([[0],[0]]) # fault in any of the control inputs so same shape as inputs
    Qf = 0.0001*np.eye(6) # shape proportional to number of states
    Rf =  0.0001*np.eye(6) # shape proportional to number of outputs
    a = 1 # random factor that they do not explain

    time_range = t_tot / dt
    i = 0

    while ti < t_tot:

        if(i == int(time_range / 8)):
            u_input = np.array([[u_input[0][0] + 2], [min(u_input[1][0] - 0.1, 0.2)]], dtype=float)
            vs_exact.updateInput(u_input)
            theta = np.array([[0.3],[0.25]])
        if(i == int(time_range / 4)):
            u_input = np.array([[u_input[0][0] - 2], [min(u_input[1][0] + 0.3, 0.2)]], dtype=float)
            vs_exact.updateInput(u_input)
            theta = np.array([[0.2],[0.45]])
        if(i == int(time_range / 2)):
            u_input = np.array([[u_input[0][0]], [min(u_input[1][0] - 0.2, 0.2)]], dtype=float)
            vs_exact.updateInput(u_input)
            theta = np.array([[0.3],[0.65]])
        if(i == int(time_range * 3 / 4)):
            u_input = np.array([[u_input[0][0]], [min(u_input[1][0] + 0.4, 0.2)]], dtype=float)
            vs_exact.updateInput(u_input)

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

        timestamp.append(ti)
        x_state.append(vs_exact.X)
        x_hat.append(vs.X)

        i += 1

        ti += dt 

    return np.array(timestamp).round(2), np.array(x_state), np.array(x_hat)


def plot_results():

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Purple = '#9D2EC5'
    CB91_Amber = '#F5B14C'

    xtab_plot = x_state
    xhatt_plot = x_hat
    ttab_plot = timestamp
    

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

if __name__ == '__main__':
    timestamp, x_state, x_hat = predict(
        t_tot=40,
        ti=0,
        dt=0.01,
        u_input=np.array([[1], [-0.02]], dtype=float))
    plot_results()
