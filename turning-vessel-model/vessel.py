import numpy as np

class Vessel:

    m11, m22, m23, m32, m33  = 25.8, 33.8, 6.2, 6.2, 2.76
    Xu, Xuu, Yv, Yvv, Yr, Nv, Nr, Nrr = -12, -2.1,-17, -4.5, -0.2, -0.5, -0.5,-0.1
    A = np.eye(6);
    Btau = np.array([[1,0],[0,0],[0,1]])
    M = np.array([[m11, 0, 0],[0, m22, m23],[0, m32, m33]])
    M_inv = np.linalg.inv(M) # inverse of M
    B1 = np.zeros((3,2))
    B2 = M_inv.dot(Btau)
    B = np.concatenate((B1,B2))

    def __init__(self):
        self._x = np.array([[0], [0], [0], [0], [0], [0]]) # x,y,psi,u,v,r
        
    def D(self):
        return np.array([[self.Xu+self.Xuu*np.abs(float(self._x[3])), 0, 0], 
            [0, self.Yv+self.Yvv*np.abs(float(self._x[4])), self.Yr],
            [0, self.Nv, self.Nr+self.Nrr*np.abs(float(self._x[5]))]])


    def C(self):
        c13 = -self.m22* self._x[4] - ((self.m23+self.m32)/2) * self._x[5]
        c23 = self.m11*self._x[3]
        return np.array([[0,0,c13],
            [0,0, c23],
            [-c13,-c23,0]],  dtype=float)

    def R(self):
        return np.array([[np.cos(float(self._x[2])), -np.sin(float(self._x[2])), 0],
                    [np.sin(float(self._x[2])), np.cos(float(self._x[2])), 0],
                    [0,0,1]],  dtype=float)

    def F(self):
        v = np.array([[float(self.x[3])], 
                 [float(self.x[4])], 
                 [float(self.x[5])]],  dtype=float)
        F1 = self.R().dot(v)
        F2 = -self.M_inv.dot((self.C() - self.D()).dot(v))
        return np.concatenate((F1,F2))
    
    def Update(self, x):
        self._x = x

    @property
    def x(self):
        return self._x
    
