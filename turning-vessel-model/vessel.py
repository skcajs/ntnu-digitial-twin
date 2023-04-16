import numpy as np
import sympy as sp
import random
random.seed(10)
class Vessel:

    m11, m22, m23, m32, m33  = 25.8, 33.8, 6.2, 6.2, 2.76
    Xu, Xuu, Yv, Yvv, Yr, Nv, Nr, Nrr = -12, -2.1,-17, -4.5, -0.2, -0.5, -0.5,-0.1
    A = sp.eye(6)
    Btau = sp.Matrix([[1,0],[0,0],[0,1]])
    M = sp.Matrix([[m11, 0, 0],[0, m22, m23],[0, m32, m33]])
    M_inv = M.inv() # inverse of M 
    B1 = sp.zeros(3,2)
    B2 = M_inv * Btau
    B = sp.Matrix(np.concatenate((B1,B2)))
    Cobvs = sp.Matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]) # C has dimensions outputs x inputs
    llambda = 0.998 # updated

    def __init__(self):
        self._x = sp.Matrix([[0.1], [0.1], [0.1],[0.1],[0.1],[0.1]])
        self.u_input = sp.Matrix([[1],[-1]], dtype=float) # input
        self.x_, self.y, self.psi, self.u, self.v, self.r = sp.symbols('x y psi u v r')
        self.x_jac = sp.Matrix([self.x_,self.y,self.psi,self.u,self.v,self.r])
           
    def D(self):
        return sp.Matrix([[self.Xu+self.Xuu*sp.Abs(self.u), 0, 0], 
            [0, self.Yv+self.Yvv*sp.Abs(self.v), self.Yr],
            [0, self.Nv, self.Nr+self.Nrr*sp.Abs(self.r)]])

    def C(self):
        c13 = -self.m22* self.v - ((self.m23+self.m32)/2) * self.r
        c23 = self.m11*self.u
        return sp.Matrix([[0,0,c13],
            [0,0, c23],
            [-c13,-c23,0]])

    def R(self):
        return sp.Matrix([[sp.cos((self.psi)), -sp.sin((self.psi)), 0],
                    [sp.sin((self.psi)), sp.cos((self.psi)), 0],
                    [0,0,1]])

    def F_symbols(self):
        gnu = sp.Matrix([[self.u],[self.v],[self.r]])
        F1 = self.R() * gnu
        F2 = -self.M_inv * ((self.C() - self.D()) * gnu)        
        return sp.Matrix(np.concatenate((F1,F2))) # F matrix in terms of sympy symbols
    
    def F(self):
        gnu = sp.Matrix([[self._x[3]],[self._x[4]],[self._x[5]]])
        F1 = self.R() * gnu
        F2 = -self.M_inv * ((self.C() - self.D()) * gnu) 
        F_halfsymbols = sp.Matrix(np.concatenate((F1,F2)))
        Jf = sp.lambdify(self.x_jac, F_halfsymbols,'numpy')       
        return  sp.Matrix(Jf(float(self._x[0]),float(self._x[1]),float(self._x[2]),float(self._x[3]),float(self._x[4]),float(self._x[5]))) # F matrix in terms of sympy symbols
       
    def phi(self):
        return -self.B * sp.diag(*self.u_input) 
    
    def Fk(self):
        "Jacobian of F"
        jcb = self.F_symbols().jacobian(self.x_jac)
        Jf = sp.lambdify(self.x_jac, jcb,'numpy') # this is to make it possible to evaluate jacobian @ xk, but calculates
                                                                             # jacobian based on symbols
        dfdx_atx = sp.Matrix(Jf(float(self._x[0]),float(self._x[1]),float(self._x[2]),float(self._x[3]),float(self._x[4]),float(self._x[5]))) # numerical jacobian @ xk
        return self.A + dfdx_atx

    def ytilde(self, Rf):
        mean = np.zeros(6)
        covariance = Rf
        Vk = sp.Matrix(np.random.multivariate_normal(mean, np.array(covariance)))
        yk = self.Cobvs * self._x + Vk
        return yk - self.Cobvs * self._x

    def Update(self, x):
        self._x = x

    @property
    def x(self):
        return self._x
    
