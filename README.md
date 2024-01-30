# WR Transforms
Class to compute features of the ECGSYN model (adapted for Wistar rat) from ECG measurements.

## Usage

```python
from ecg_models import Rat, utils 
from src.wr_transform import TransformModel, TransformParameters
import numpy as np 

# set the transform parameters, ki.
K = TransformParameters(
    P=TransformParameters.kP(.95, .05),
    R=3.0,
    S=2.5,
    T=TransformParameters.kT(.8, .4),
    W=1.0,
    D=2.0,
    J=TransformParameters.kJ()
)

# set the rat model
model = lambda x, fea: Rat.f(x, utils.modelize([0]+fea.tolist(), Rat.Waves))

# build the transform
tr = TransformModel(K, model)

fs = ... # sampling frequency in Hz
window = ... # size of beat
measurements = ... # set the measurements: temporal values (P duration, PR, RS complex, QT) in secs and amplitudes (P, R, S, T) in arbitrary unit.
measurements[:4] *= fs * 2*np.pi / window # time to rad conversion
features = tr.inverse(measurements) # get the features in rad unit for (a_i, u_i, s_i) for i = P, R, S, T.
```
