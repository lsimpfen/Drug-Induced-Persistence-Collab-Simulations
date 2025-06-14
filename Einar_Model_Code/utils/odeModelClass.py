# ====================================================================================
# Abstract model class
# ====================================================================================
import numpy as np
import scipy.integrate
import pandas as pd
import os
import sys
import contextlib
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils

class ODEModel():
    def __init__(self, **kwargs):
        # Initialise parameters
        self.paramDic = {'DMax':100}
        self.stateVars = ['P1']
        self.resultsDf = None

        # Set the parameters
        self.SetParams(**kwargs)

        # Configure the solver
        self.dt = kwargs.get('dt', 1e-3)  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', 1.0e-8)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', 1.0e-6)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', 'DOP853')  # ODE solver used
        self.max_step = kwargs.get('max_step', np.inf) # Maximum step size permitted by solver
        self.numericalStabilisationB = kwargs.get('numericalStabilisationB', False)  # Whether to apply numerical stabilisation
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          False)  # If true, suppress output of ODE solver (including warning messages)
        self.successB = False  # Indicate successful solution of the ODE system

    # =========================================================================================
    # Function to set the parameters
    def SetParams(self, **kwargs):
        if len(self.paramDic.keys()) > 1:
            for key in self.paramDic.keys():
                self.paramDic[key] = float(kwargs.get(key, self.paramDic[key]))
            self.initialStateList = [self.paramDic[var + "0"] for var in self.stateVars]

    # =========================================================================================
    # Function to simulate the model
    def Simulate(self, treatmentScheduleList, **kwargs):
        # Allow configuring the solver at this point as well
        self.dt = float(kwargs.get('dt', self.dt))  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', self.absErr)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', self.relErr)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', self.solverMethod)  # ODE solver used
        self.max_step = kwargs.get('max_step', self.max_step) # Maximum step size permitted by solver
        self.successB = False  # Indicate successful solution of the ODE system
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          self.suppressOutputB)  # If true, suppress output of ODE solver (including warning messages)
        self.numericalStabilisationB = kwargs.get('numericalStabilisationB', self.numericalStabilisationB)  # Whether to apply numerical stabilisation

        # Solve
        self.treatmentScheduleList = treatmentScheduleList
        if self.resultsDf is None or treatmentScheduleList[0][0] == 0:
            currStateVec = self.initialStateList + [0]
            self.resultsDf = None
        else:
            currStateVec = [self.resultsDf[var].iloc[-1] for var in self.stateVars] + [self.resultsDf['DrugConcentration'].iloc[-1]]
        resultsDFList = []
        encounteredProblemB = False
        for intervalId, interval in enumerate(treatmentScheduleList):
            tVec = np.arange(interval[0], interval[1], self.dt)
            if intervalId == (len(treatmentScheduleList) - 1):
                tVec = np.arange(interval[0], interval[1] + self.dt, self.dt)
                # Floating point inaccuracies mean that it can happen that the 
                # final time point is cut off. To ensure it's included in tVec,
                # manually insert it, if this happens.
                if tVec[-1] <= interval[1]: tVec[-1] = interval[1]
            currStateVec[-1] = interval[2]
            if self.suppressOutputB:
                with stdout_redirected():
                    solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                       t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                       method=self.solverMethod,
                                                       atol=self.absErr, rtol=self.relErr,
                                                       max_step=self.max_step)
            else:
                solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                   t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                   method=self.solverMethod,
                                                   atol=self.absErr, rtol=self.relErr,
                                                   max_step=self.max_step)
            # Check that the solver converged
            self.errMessage = ""
            self.solObj = solObj
            if not solObj.success:
                encounteredProblemB = True
                self.errMessage = solObj.message
            elif np.any(solObj.y < 0):
                self.errMessage = "Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver."
                if self.numericalStabilisationB:
                    solObj.y[solObj.y < 0] = 0
                    self.errMessage += "... Applying numerical stabilisation."
                else:
                    encounteredProblemB = True
            
            # Output error message if required
            if not self.suppressOutputB and len(self.errMessage) > 0:
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print(self.errMessage)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            if encounteredProblemB: break

            # Save results
            resultsDFList.append(
                pd.DataFrame({"Time": tVec, "DrugConcentration": solObj.y[-1, :],
                              **dict(zip(self.stateVars,solObj.y))}))
            currStateVec = solObj.y[:, -1]
        # If the solver diverges in the first interval, it can't return any solution. Catch this here, and in this case
        # replace the solution with all zeros.
        if len(resultsDFList) > 0:
            resultsDf = pd.concat(resultsDFList)
        else:
            resultsDf = pd.DataFrame({"Time": tVec, "DrugConcentration": np.zeros_like(tVec),
                                     **dict(zip(self.stateVars,np.zeros_like(tVec)))})
        # Compute the fluorescent area that we'll see
        resultsDf['TumourSize'] = pd.Series(self.RunCellCountToTumourSizeModel(resultsDf),
                                            index=resultsDf.index)
        if self.resultsDf is not None:
            resultsDf = pd.concat([self.resultsDf, resultsDf])
        self.resultsDf = resultsDf
        self.successB = True if not encounteredProblemB else False

    # =========================================================================================
    # Define the model mapping cell counts to observed fluorescent area
    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        # Note: default scaleFactor value assumes a cell radius of 10uM. Volume is given in mm^3 -> r^3 = (10^-2 mm)^3 = 10^-6
        theta = self.paramDic.get('scaleFactor', 1)
        return theta * (np.sum(popModelSolDf[self.stateVars].values,axis=1))

    # =========================================================================================
    # Simulate adaptive therapy (dose modulation strategy)
    def Simulate_AT1(self, atThreshold=0.2, doseAdjustFac=0.5, D0=None, v_min=0, intervalLength=1.,
                     mode="original", t_end=1000, nCycles=np.inf, t_span=None, solver_kws={}):
        t_span = t_span if t_span is not None else (0, t_end)
        currInterval = [t_span[0], t_span[0] + intervalLength]
        refSize = self.paramDic.get('scaleFactor', 1) * np.sum(self.initialStateList)
        dose = self.paramDic['DMax'] if D0 is None else D0
        lastNonZeroDose = dose # Remember the last non-zero dose if withdraw drug
        currCycleId = 0
        while (currInterval[1] <= t_end + intervalLength) and (currCycleId < nCycles):
            # Simulate
            # print(currInterval,refSize)
            self.Simulate([[currInterval[0], currInterval[1], dose]], **solver_kws)

            # Update dose
            # print(self.resultsDf.TumourSize.iat[-1],(1+atThreshold)*refSize)
            dose = lastNonZeroDose if dose == 0 else dose
            if self.resultsDf.TumourSize.iat[-1] < v_min: # Withdraw treatment below a certain size
                lastNonZeroDose = dose
                dose = 0
            elif self.resultsDf.TumourSize.iat[-1] < (1 - atThreshold) * refSize: # Reduce dose if sufficient shrinkage
                if mode == "original":
                    dose = max((1 - doseAdjustFac) * dose, 0) # Adjustment as proposed in Enriquez-Navas et al (2015)
                else:
                    dose = max(1/doseAdjustFac * dose, 0)
            elif self.resultsDf.TumourSize.iat[-1] > (1 + atThreshold) * refSize: # Increase dose if excessive growth
                if mode == "original":
                    dose = min((1+doseAdjustFac) * dose, self.paramDic['DMax']) # Adjustment as proposed in Enriquez-Navas et al (2015)
                else:
                    dose = min(doseAdjustFac * dose, self.paramDic['DMax'])
            else: # If size remains within a window of +- atThreshold, keep the same dose
                dose = dose

            # Update interval
            refSize = self.resultsDf.TumourSize.iloc[-1]
            currInterval = [x + intervalLength for x in currInterval]

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)

    # =========================================================================================
    # Simulate adaptive therapy (dose skipping strategy)
    def Simulate_AT2(self, atThreshold=0.2, D_star=None, D0=None, DMin=0, intervalLength=1., n_days_lookback=2, t_end=1000,
                     nCycles=np.inf, t_span=None, solver_kws={}):
        t_span = t_span if t_span is not None else (0, t_end)
        currInterval = [t_span[0], t_span[0] + intervalLength]
        refSize = self.paramDic.get('scaleFactor', 1) * np.sum(self.initialStateList)
        prevSizesList = [refSize]*n_days_lookback
        dose = self.paramDic['DMax'] if D0 is None else D0
        D_star = self.paramDic['DMax'] if D_star is None else D_star
        currCycleId = 0
        while (currInterval[1] <= t_end + intervalLength) and (currCycleId < nCycles):
            # Simulate
            # print(currInterval,refSize)
            self.Simulate([[currInterval[0], currInterval[1], dose]], **solver_kws)

            # Update dose
            if self.resultsDf.TumourSize.iat[-1] > (1 + atThreshold) * refSize:  # Treat if excessive growth
                dose = D_star #min(D_star, self.paramDic['DMax'])
            else:  # Otherwise treat at minimum dose (=0 in Enriquez-Navas et al)
                dose = DMin

            # Update interval
            refSize = prevSizesList[0]
            prevSizesList[:-1] = prevSizesList[1:]
            prevSizesList[-1] = self.resultsDf.TumourSize.iloc[-1] # AT2 uses the 2 time steps to decide dose
            # print(prevSizesList, refSize, dose)
            # currSize = self.resultsDf.TumourSize.iloc[-1] # AT2 uses the 2 time steps to decide dose
            # refSize = prevSize if currCycleId!=0 else refSize # For the initial step
            # prevSize = currSize
            # print(currSize,refSize, dose)
            currInterval = [x + intervalLength for x in currInterval]
            currCycleId += 1

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)

    # =========================================================================================
    # Simulate adaptive therapy (Zhang et al algorithm)
    def Simulate_AT50(self, refSize=None, atThreshold=0.5, D_Star=None, D_min=0,
                           intervalLength_on=1., intervalLength_off=None,
                           t_end=1000, nCycles=np.inf, t_span=None, solver_kws={}):
        intervalLength_off = intervalLength_on if intervalLength_off is None else intervalLength_off
        intervalLength = intervalLength_on
        t_span = t_span if t_span is not None else (0, t_end)
        currInterval = [t_span[0], t_span[0] + intervalLength]
        refSize = self.paramDic.get('scaleFactor', 1) * np.sum(
            self.initialStateList) if refSize is None else refSize
        dose = self.paramDic['DMax']
        D_Star = self.paramDic['DMax'] if D_Star is None else D_Star
        currCycleId = 0
        while (currInterval[1] <= t_end + intervalLength) and (currCycleId < nCycles):
            # Simulate
            # print(currInterval,refSize)
            self.Simulate([[currInterval[0], currInterval[1], dose]], **solver_kws)

            # Update dose
            # print(self.resultsDf.TumourSize.iat[-1],(1-atThreshold)*refSize)
            if self.resultsDf.TumourSize.iat[-1] < (
                    1 - atThreshold) * refSize:  # Withdraw treatment below a certain size
                dose = D_min
                intervalLength = intervalLength_off
            elif self.resultsDf.TumourSize.iat[-1] > refSize:
                dose = D_Star
                intervalLength = intervalLength_on
            else:  # If size remains within a window of +- atThreshold, keep the same dose
                dose = dose
            # print(dose, intervalLength)

            # Update interval
            currInterval = [x + intervalLength for x in currInterval]
            currCycleId += 1

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)

    # =========================================================================================
    # Simulate a long term assay, where we passage the cells at every interval
    def Simulate_LongTermAssay(self, treatment_schedule, seeding_density=5e5, solver_kws={}):
        '''
        Simulate a long term assay, where we passage the cells at regular intervals, reseeding them at 
        a given density.
        treatmentSchedule: List of treatment intervals
        seeding_density: Density at which to reseed the cells
        solver_kws: Keyword arguments to pass to the solver
        '''
        # Initialise the model
        for cycle_id, cycle in enumerate(treatment_schedule):
            # Seed the cells
            if cycle_id > 0:
                # Read out the current seeding density
                if type(seeding_density) is not list or len(seeding_density) == 1:
                    # If a single value is given, assume that the seeding density is constant
                    curr_seeding_density = seeding_density
                else:
                    # Otherwise, assume that the seeding density is given as a list of values
                    curr_seeding_density = seeding_density[cycle_id]
                curr_resistance_fraction = self.resultsDf['R'].iloc[-1]/self.resultsDf['TumourSize'].iloc[-1]
                self.resultsDf['S'].iloc[-1] = curr_seeding_density * (1-curr_resistance_fraction)
                self.resultsDf['R'].iloc[-1] = curr_seeding_density * curr_resistance_fraction
                self.resultsDf['TumourSize'].iloc[-1] = self.resultsDf['S'].iloc[-1] + self.resultsDf['R'].iloc[-1]

            # Simulate the current cycle
            self.Simulate([cycle], **solver_kws)

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)

    # =========================================================================================
    # Simulate a long term assay, where we passage the cells at every interval
    def Simulate_LongTermAssay_PKill(self, treatment_schedule, seeding_density=5e5, passaging_loss=0.8, solver_kws={}):
        '''
        Simulate a long term assay, where we passage the cells at regular intervals, reseeding them at 
        a given density.
        treatmentSchedule: List of treatment intervals
        seeding_density: Density at which to reseed the cells
        solver_kws: Keyword arguments to pass to the solver
        '''
        # Initialise the model
        previous_cycle_on_drug = False
        for cycle_id, cycle in enumerate(treatment_schedule):
            # Seed the cells
            if cycle_id > 0:
                # Read out the current seeding density
                if type(seeding_density) is not list or len(seeding_density) == 1:
                    # If a single value is given, assume that the seeding density is constant
                    curr_seeding_density = seeding_density
                else:
                    # Otherwise, assume that the seeding density is given as a list of values
                    curr_seeding_density = seeding_density[cycle_id]
                curr_resistance_fraction = self.resultsDf['R'].iloc[-1]/self.resultsDf['TumourSize'].iloc[-1]
                self.resultsDf['S'].iloc[-1] = curr_seeding_density * (1-curr_resistance_fraction)
                if previous_cycle_on_drug: # Add additional kill from passaging
                    self.resultsDf['S'].iloc[-1] *= (1-passaging_loss)
                self.resultsDf['R'].iloc[-1] = curr_seeding_density * curr_resistance_fraction
                self.resultsDf['TumourSize'].iloc[-1] = self.resultsDf['S'].iloc[-1] + self.resultsDf['R'].iloc[-1]

            # Simulate the current cycle
            self.Simulate([cycle], **solver_kws)
            previous_cycle_on_drug = cycle[-1] > 0

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)

    # =========================================================================================
    # Interpolate to specific time resolution (e.g. for plotting)
    def Trim(self, t_eval=None, dt=1):
        t_eval = np.arange(0, self.resultsDf.Time.max(), dt) if t_eval is None else t_eval
        tmpDfList = []
        trimmedResultsDic = {'Time': t_eval}
        for variable in [*self.stateVars, 'TumourSize', 'DrugConcentration']:
            f = scipy.interpolate.interp1d(self.resultsDf.Time, self.resultsDf[variable])
            trimmedResultsDic = {**trimmedResultsDic, variable: f(t_eval)}
        tmpDfList.append(pd.DataFrame(trimmedResultsDic))
        self.resultsDf = pd.concat(tmpDfList)

    # =========================================================================================
    # Function to plot the model predictions
    def Plot(self, plotPops=True, n_boot=1,
             xmin=0, xlim=None, ymin=0, ylim=None, y2lim=1, palette=None,
             decorateAxes=True, legend=False,
             drugBarPosition=0.85, drugBarColour="black", 
             decoratey2=True, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots(1, 1)
        varsToPlotList = ["TumourSize"]
        if plotPops: varsToPlotList += self.stateVars
        currModelPredictionDf = pd.melt(self.resultsDf, id_vars=['Time'], value_vars=varsToPlotList)

        # 1. Plot the model predictions
        sns.lineplot(x="Time", y="value", hue="variable", style="variable", n_boot=n_boot,
                     lw=5, palette=palette,
                     legend=legend,
                     data=currModelPredictionDf, ax=ax)

        # 2. Add the drug bars
        ax2 = ax.twinx()  # instantiate a second axis that shares the same x-axis
        drug_data_df = self.resultsDf
        drugConcentrationVec = utils.TreatmentListToTS(
            treatmentList=utils.ExtractTreatmentFromDf(drug_data_df, timeColumn="Time",
                                                    treatmentColumn="DrugConcentration",
                                                    mode="post"),
            tVec=drug_data_df["Time"])
        drugConcentrationVec[drugConcentrationVec < 0] = 0
        drugConcentrationVec = np.array([x / (np.max(drugConcentrationVec) + 1e-12) for x in drugConcentrationVec])
        drugConcentrationVec = drugConcentrationVec / (1 - drugBarPosition) + drugBarPosition
        ax2.fill_between(drug_data_df["Time"], drugBarPosition, drugConcentrationVec,
                        step="post", color=drugBarColour, alpha=1., label="Drug Concentration")
        ax2.axis("off")

        # Format the plot
        if xlim is not None: ax.set_xlim([xmin, xlim])
        if ylim is not None: ax.set_ylim([ymin, ylim])
        if y2lim is not None: ax2.set_ylim([0, y2lim])
        ax.set_xlabel("Time in Hours" if decorateAxes else "")
        ax.set_ylabel("Confluence" if decorateAxes else "")
        ax2.set_ylabel(r"Drug Concentration in $\mu M$" if decorateAxes else "")
        ax.set_title(kwargs.get('title', ''))
        plt.tight_layout()
        if kwargs.get('saveFigB', False):
            plt.savefig(kwargs.get('outName', 'modelPrediction.png'), orientation='portrait', format='png')
            plt.close()

# ====================================================================================
# Functions used to suppress output from odeint
# Taken from: https://stackoverflow.com/questions/31681946/disable-warnings-originating-from-scipy
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied