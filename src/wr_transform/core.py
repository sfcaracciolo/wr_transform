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
    
class TransformModel:
    def __init__(self, K: TransformParameters, model) -> None:
        """
        temporal features in [rad]
        temporal measurements in [rad]
        """

        self.F1 = TransformModel._F1(K)
        self.F2 = TransformModel._F2()
        self.F3 = model # model(x, fea)
        self.TF1 = TransformModel._TF1(K)
        self.Ft = self.F2 @ self.TF1 # temporal fiducials to temporal measurements. shape (4, 8)
        self.R = TransformModel._R(K)
        self.FtR = sp.sparse.vstack([self.Ft, self.R]).todense() # uP sP uR sR uS sS uT sT -> P duration, PR, RS complex, QT, 0, 0, 0, pi

    def forward(self, fea):
        _, fea_t = self.split(fea, 'features')
        fid = self.F1 @ fea_t
        fid_a, fid_t = self.split(fid, 'fiducials')
        mea_t = self.F2 @ fid_t
        mea_a = self.F3(fid_a, fea)
        return self.join(mea_a, mea_t, 'measurements') 

    def inverse(self, mea):
        mea_a, mea_t = self.split(mea, 'measurements') 
        fea_t = self.invFtR(mea_t)
        fea_a = self.invF3(fea_t, x0=mea_a)
        return self.join(fea_a, fea_t, 'features')

    def invFtR(self, mea_t):
        b = np.append(mea_t, values=(0,0,0,np.pi))
        return sp.linalg.solve(self.FtR, b)
    
    def invF3(self, fea_t, x0):
        f = lambda fea_a: self.F3(fea_t[::2], self.join(fea_a, fea_t, type='features')) - x0
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
    def _F1(K: TransformParameters):
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
    def _F2():
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
    def _TF1(K: TransformParameters):
        """
        remove rows of F1 to select temporal fiducials from fiducials

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
        T[2,4], T[2,5], T[2,6], T[2,7] = 1, K['S'], -1, (3-np.sqrt(5))/2.
        T[3,2] = 1
        return T 
