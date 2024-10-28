################################################
# Libraries
################################################
# Load packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sammhelper as sh
import scipy as sp

# ################################################
# %% Task d)
# ################################################
# Parameters: Time
STARTTIME =       
STOPTIME =        
DT =              

# Time span
time = np.arange(STARTTIME, STOPTIME, DT)

# Parameters: Process


############### 1 CSTR ################
# Parameters: Process


# Parameters: Initial


# Define ODE
def model(var, t, param):

    = var
    = param

    C_in = 
    dCdt = 

    return dCdt

# Solve ODE
C_CSTR = sh.sol_ode(...)

plt.figure('Task d)')
plt.grid()
plt.plot(time,C_CSTR)

############### 6 CSTR #################
# Parameters: Process


# Parameters: Initial


# Define ODE
def model(var, t, param):

    = var
    = param

    C_in = 

    ...
    
    return dCdt

# Solve ODE
C_Cascade = sh.sol_ode(...)

plt.figure('Task d)')
plt.plot(time,C_Cascade[:,-1],linestyle='dashed')

############### PFR ################
# Parameters: Process
 

# Parameters: Initial
C_in_delay = 

initC = C_in_delay

# Define ODE
def model(var, t, param):

    var
    param

    ...
    
    return dCdt

# Solve ODE
PFR = sh.sol_ode(...)
C_PFR = PFR[1,:]

plt.figure('Task d)')
plt.plot(time, C_PFR,linestyle='dotted')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.legend(['CSTR','Cascade','PFR'])
plt.show()

###############################################
# %% Task III f)
###############################################
# Parameters: Time
STARTTIME =
STOPTIME =
DT =       

# Time span
time = np.arange(STARTTIME, STOPTIME, DT)

# Parameters: Process


# Parameters: Initial


def model(var, t, param):
    = var
    = param

    rX = 
    rS = 

    dXdt = 
    dSdt = 

    return dXdt,dSdt

X,S  = sh.sol_ode(...)

avg, std = sh.curve_fit_ode(model, [...], time, xdata=, ydata=, param=[...], guess0=[...], x_ind=0)

plt.figure('curve fit')
plt.xlabel('xlabel')
plt.ylabel('ylabel')

################################################
# %% Task III g)
################################################
# Parameters: Time
STARTTIME =
STOPTIME =
DT =       

# Time span
time = np.arange(STARTTIME, STOPTIME, DT)

# Parameters: Process


# Parameters: Initial


######## Scenario 1 ###########
def model(var, t, param):
    = Var
    = param

    Q =                                             
    S_in =

    ...

    return dXdt, dSdt

X1, S1= sh.sol_ode(...)

L1_in =                                             # calculate the load based on the concentration
L1_out =                                            # calculate the load based on the concentration

# plot results
plt.figure('Task g) 1')
plt.grid()
plt.plot(time, L1_in)

plt.figure('Task g) 2')
plt.grid()
plt.plot(time, L1_out)

plt.figure('Task g) 3')
plt.grid()
plt.plot(time, S_in)

plt.figure('Task g) 4')
plt.grid()
plt.plot(time, S1)

######## Scenario 2 ###########
def model(var, t, param):
    = var
    = param

    ...

    return dXdt, dSdt

X2, S2 = sh.sol_ode(...)

L2_in = 
L2_out = 

# plot results
plt.figure('Task g) 1')
plt.plot(time, L2_in, linestyle='dashed')

plt.figure('Task g) 2')
plt.plot(time, L2_out, linestyle='dashed')

plt.figure('Task g) 3')
plt.plot(time, S_in, linestyle='dashed')

plt.figure('Task g) 4')
plt.plot(time, S2, linestyle='dashed')

######## Scenario 3 ###########
def model(var, t, param):
    = var
    = param

    ...

    return dXdt, dSdt

X3, S3 = sh.sol_ode(...)

L3_in = 
L3_out =    

# plot results
plt.figure('Task g) 1')
plt.plot(time, L3_in, linestyle='dotted')

plt.figure('Task g) 2')
plt.plot(time, L3_out, linestyle='dotted')

plt.figure('Task g) 3')
plt.plot(time, S_in, linestyle='dotted')

plt.figure('Task g) 4')
plt.plot(time, S3, linestyle='dotted')

######## Scenario 4 ###########
def model(var, t, param):
    = var
    = param

    ...

    return dXdt, dSdt

X4, S4 = sh.sol_ode(...)

L4_in = 
L4_out = 

# plot results
plt.figure('Task g) 1')
plt.plot(time, L4_in, linestyle='dashdot')

plt.figure('Task g) 2')
plt.plot(time, L4_out, linestyle='dashdot')

plt.figure('Task g) 3')
plt.plot(time, S_in, linestyle='dashdot')

plt.figure('Task g) 4')
plt.plot(time, S4, linestyle='dashdot')

plt.figure('Task g) 1')
plt.legend(['In 1', 'In 2', 'In 3', 'In 4'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.xlim(xmin=8, xmax=10)
plt.show()

plt.figure('Task g) 2')
plt.legend(['Eff 1', 'Eff 2', 'Eff 3', 'Eff 4'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.xlim(xmin=8, xmax=10)
plt.show()

plt.figure('Task g) 3')
plt.legend(['S1$_{in}$', 'S2$_{in}$', 'S3$_{in}$', 'S4$_{in}$'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.xlim(xmin=8, xmax=10)
plt.show()

plt.figure('Task g) 4')
plt.legend(['S1', 'S2', 'S3', 'S4'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.xlim(xmin=8, xmax=10)
plt.show()

################################################
# %% Task III h)
################################################
# Parameters: Time
STARTTIME =
STOPTIME =
DT =       

# Time span
time = np.arange(STARTTIME, STOPTIME, DT)

# Time span


# Parameters: Process


# Parameters: Initial


######## Constant inflow ###########
def model(var, t, param):
    = var
    = param

    Qex = 

    Q = 
    S_in =

    dXdt = 
    dSdt = 

    dXdt[0] = 
    dSdt[0] = 

    dXdt[1:n] = 
    dSdt[1:n] = 
    return dXdt,dSdt

X,S = sh.sol_ode(...)

plt.figure('Task h) 1')
plt.plot(time,X)
plt.legend(['CSTR 1', 'CSTR 2', 'CSTR 3', 'CSTR 4', 'CSTR 5', 'CSTR 6'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

plt.figure('Task h) 2')
plt.plot(time,S)
plt.legend(['CSTR 1', 'CSTR 2', 'CSTR 3', 'CSTR 4', 'CSTR 5', 'CSTR 6'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

######## Scenario 3 ###########
def model(var, t, param):
    = var
    = param

    Qex = 

    Q = 
    S_in =

    dXdt = 
    dSdt = 

    dXdt[0] = 
    dSdt[0] = 

    dXdt[1:n] = 
    dSdt[1:n] = 
    return dXdt,dSdt

X,S = sh.sol_ode(...)

plt.figure('Task h) 3')
plt.plot(time,X)
plt.legend(['CSTR 1', 'CSTR 2', 'CSTR 3', 'CSTR 4', 'CSTR 5', 'CSTR 6'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

plt.figure('Task h) 4')
plt.plot(time,S)
plt.legend(['CSTR 1', 'CSTR 2', 'CSTR 3', 'CSTR 4', 'CSTR 5', 'CSTR 6'])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()