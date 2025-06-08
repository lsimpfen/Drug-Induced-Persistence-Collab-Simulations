import sys
import numpy as np
sys.path.append("./")
from odeModelClass import ODEModel

# LINEAR MODEL

class EinarPersistorModelType3L(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelType3L"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Linear 3',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'k': 0.0004,
                         'm': 0.0004,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))

        u = dic['u0'] + dic['k'] * c
        v = dic['v0'] - dic['m'] * c

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec

#########################################

class EinarPersistorModelType2L(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelType2L"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Linear 2',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'k': 0.0004,
                         'm': 0.0004,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))

        u = dic['u0']
        v = dic['v0'] - dic['m'] * c

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec

#########################################

class EinarPersistorModelType1L(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelType1L"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Linear 1',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'k': 0.0004,
                         'm': 0.0004,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))

        u = dic['u0'] + dic['k'] * c
        v = dic['v0']

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec

# UNIFORM MODEL

class EinarPersistorModelTypeU(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelTypeU"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Uniform',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08,
                         'delta_u': 0.004,
                         'delta_v': 0.003
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))
        
        u_max = dic['u0'] + dic['delta_u']
        v_min = dic['v0'] - dic['delta_v']

        if c > 0:
            u = u_max
            v = v_min
        else:
            u = dic['u0']
            v = dic['v0']

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec

#######################################
    
class EinarPersistorModelTypeU1(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelTypeU"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Uniform',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08,
                         'delta_u': 0.004,
                         'delta_v': 0.003
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))
        
        u_max = dic['u0'] + dic['delta_u']
        v_min = dic['v0'] - dic['delta_v']

        if c > 0:
            u = u_max
            v = dic['v0']
        else:
            u = dic['u0']
            v = dic['v0']

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec
    
######################################
    
class EinarPersistorModelTypeU2(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "EinarPersistorModelTypeU"
        self.paramDic = {**self.paramDic,
                         'CASE': 'Uniform',
                         'n': 1500,
                         'fracRes': 0.01,
                         'Cmax': 10,
                         'u0': 0.0004,
                         'v0': 0.004,
                         'lambda0': 0.04,
                         'lambda1': 0.001,
                         'delta_d0': 0.08,
                         'delta_u': 0.004,
                         'delta_v': 0.003
                        }
        self.stateVars = ['S', 'R'] # State variables of the model. These are the variables that will be solved for in the ODE solver (Note: by default the solver will add an extra state variable for the drug concentration).

    def ModelEqns(self, t, uVec):
        dudtVec = np.zeros_like(uVec)
        dic = self.paramDic.copy()
        S, R, cfrac = uVec
        
        c = cfrac * dic['Cmax']
        lamb = dic['lambda0'] - dic['delta_d0'] * (c / (c + 1))
        
        u_max = dic['u0'] + dic['delta_u']
        v_min = dic['v0'] - dic['delta_v']

        if c > 0:
            u = dic['u0']
            v = v_min
        else:
            u = dic['u0']
            v = dic['v0']

        dudtVec[0] = (lamb - u) * S + v * R
        dudtVec[1] = (dic['lambda1'] - v) * R + u * S
        dudtVec[2] = 0

        return dudtVec