# type: ignore
from typing import Literal, TypedDict
import scipy as sp 
import numpy as np 

class TransformParameters(TypedDict):
    P: float
    R: float
    S: float
    T: float
    W: float
    D: float
    J: float
    
    @staticmethod
    def kT(p1=.8, p2=.4):
        z = lambda x: -np.log(-sp.special.lambertw(-x/np.e, k=0).real )
        f = lambda x:  np.exp(1.-x-np.exp(-x))
        b, m = TransformParameters.linreg(f, (z(p1), z(p2)))
        return b/m
    
    @staticmethod
    def kP(p1=.9, p2=.1):
        z = lambda x: np.sqrt(-np.log(x))
        f = lambda x: np.exp(- x**2 )
        b, m = TransformParameters.linreg(f, (z(p1), z(p2)))
        return b/m
        
    @staticmethod
    def kJ():
        return -np.log((3+np.sqrt(5))/2.)

    @staticmethod
    def linreg(f, lims):
        q = lambda i,j: sp.integrate.quad(lambda z: z**i*f(z)**j, *lims)[0]

        q00, q01 = q(0, 0), q(0, 1)
        q10, q11 = q(1, 0), q(1, 1)
        q20 = q(2, 0)

        b = (q11*q10 - q01*q20) / (q00*q20 - q10**2)
        m = (q11 + b*q10) / q20
        return  b, m
    
class TransformModel:
    def __init__(self, K: TransformParameters, model) -> None:
        """
        temporal features in [rad]
        temporal measurements in [rad]
        """

        self.F = TransformModel._F(K)
        self.M = TransformModel._M()
        self.f = model # model(x, θ)
        Ft = TransformModel._Ft(K, self.M)
        self.invFt = np.linalg.inv(Ft)

    def forward(self, θ):
        _, θt = self.split(θ, 'features')
        ϕ = self.F @ θt
        ϕa, ϕt = self.split(ϕ, 'fiducials')
        ψt = self.M @ ϕt
        ψa = self.f(ϕa, θ)
        return self.join(ψa, ψt, 'measurements') 

    def inverse(self, ψ):
        ψa, ψt = self.split(ψ, 'measurements') 
        b = np.append(ψt, values=(0,0,0,np.pi))
        θt = np.ravel(self.invFt @ b)
        θa = self.g(θt, x0=ψa)
        return self.join(θa, θt, 'features')

    def g(self, θt, x0):
        f = lambda x: self.f(θt[::2], self.join(x, θt, type='features')) - x0
        return sp.optimize.broyden1(f, x0)

    def split(self, v: np.ndarray, type: Literal['features', 'fiducials', 'measurements']):
        if type == 'measurements':
            v_a, v_t = v[[4,6,7,8]], v[:4] # 5 is Qpeak
        return v_a, v_t
    
    def join(self, v_a: np.ndarray, v_t: np.ndarray, type: Literal['features', 'fiducials', 'measurements']):
        if type == 'features':
            v = np.insert(v_t, [0, 2, 4, 6], values=v_a)
        return v
        
    @staticmethod
    def _F(K: TransformParameters):
        """
        temporal features to fiducial points

        uP sP uR sR uS sS uT sT -> Pon Ppeak Poff RSon Rpeak Speak J Tpeak Toff

        """
        T = sp.sparse.dok_array((9, 8))
        T[0,0], T[0,1] = 1, -K['P']
        T[1,0] = 1
        T[2,0], T[2,1] = 1, K['P']
        T[3,2], T[3,3] = 1, -K['R']
        T[4,2] = 1
        T[5,4] = 1
        T[6,4], T[6,5] = 1, K['S']
        T[7,6] = 1
        T[8,6], T[8,7] = 1, K['T']
        return T

    @staticmethod
    def _M():
        """
        temporal fiducial points to temporal measurements

        Pon Poff RSon J Toff -> P duration, PR, RS complex, QT

        """
        T = sp.sparse.dok_array((4, 5))
        T[0, 0], T[0, 1] = -1, 1
        T[1, 0], T[1, 2] = -1, 1
        T[2, 2], T[2, 3] = -1, 1
        T[3, 2], T[3, 4] = -1, 1
        return T

    @staticmethod
    def _SF(K: TransformParameters):
        """
        remove rows of F to select temporal fiducials from fiducials

        uP sP uR sR uS sS uT sT -> Pon Poff RSon J Toff

        """
        T = sp.sparse.dok_array((5, 8))
        T[0,0], T[0,1] = 1, -K['P']
        T[1,0], T[1,1] = 1, K['P']
        T[2,2], T[2,3] = 1, -K['R']
        T[3,4], T[3,5] = 1, K['S']
        T[4,6], T[4,7] = 1, K['T']
        return T

    @staticmethod
    def _R(K: TransformParameters):
        """
        uP sP uR sR uS sS uT sT -> 0, 0, 0, pi
        """
        T = sp.sparse.dok_array((4, 8))
        T[0,3], T[0,5] = 1, -K['W']
        T[1,2], T[1,3], T[1,4] = 1, K['D'], -1
        T[2,4], T[2,5], T[2,6], T[2,7] = 1, K['S'], -1, -K['J']
        T[3,2] = 1
        return T 

    @staticmethod
    def _Ft(K: TransformParameters, M):
        SF = TransformModel._SF(K)
        R = TransformModel._R(K)
        return sp.sparse.vstack([M @ SF, R]).todense()