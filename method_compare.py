import numpy as np
import matplotlib.pyplot as plt

# Import regression data
data = np.loadtxt("Eccentricity-Data.csv", skiprows=1, delimiter=",")
# column 0 = time
# column 1 = pos
# column 2 = delta
x = data[:,1] *np.pi/180
b = data[:,2]
n=7

# function to calculate regression based on data
def calculate_regression(n, method):
# calculate regression
    A = np.zeros((data.shape[0],2*n+1))
    A[:,0]=np.ones((data.shape[0],))
    if method=="cos_phase":
        for i in range(0,n):
            A[:,1+2*i]=np.cos(x+np.pi/(2*i+1))
            A[:,1+2*i+1]=np.cos(x+np.pi/(2*i+2))
    if method=="power":
        for i in range(0,n):
            A[:,1+2*i]=np.cos(x)**(i+1)
            A[:,1+2*i+1]=np.sin(x)**(i+1)
    elif method=="frequency":
        for i in range(0,n):
            A[:,1+2*i]=np.cos(x*(i+1))
            A[:,1+2*i+1]=np.sin(x*(i+1))
    # A z = b
    # z ~= (A^T @ A)^{-1} A^T b
    try:
        z = np.linalg.solve(A.T @ A, A.T @ b) # coefficients of our regression equation
    except:
        z = np.ones(2*n+1)
    return z

# function to correct signal at any measured delta and position
def clean_signal(delta, pos, coeffs, method):
    # n = len(coeffs)
    a = np.zeros(len(coeffs))
    a[0] = 1
    # np.cos(x+np.pi()/(2*i+2))
    if method=="cos_phase":
        for i in range(0,(len(coeffs)-1)//2):
            a[1+2*i]=np.cos(pos+np.pi/(2*i+1))
            a[1+2*i+1]=np.cos(pos+np.pi/(2*i+2))
    if method=="power":
        for i in range(0,(len(coeffs)-1)//2):
            a[1+2*i]=np.cos(pos)**(i+1)
            a[1+2*i+1]=np.sin(pos)**(i+1)
    elif method=="frequency":
        for i in range(0,(len(coeffs)-1)//2):
            a[1+2*i]=np.cos(pos*(i+1))
            a[1+2*i+1]=np.sin(pos*(i+1))
    
    corr = a.dot(coeffs)

    return delta-corr

# compare the methods of regression

stepSizes = np.arange(1,11)
errs      = np.zeros ([3,10])

for n in stepSizes:
    
    errPower     = np.zeros(len(x))
    errFrequency = np.zeros(len(x))
    errPhase = np.zeros(len(x))

    coeffsPower     = calculate_regression(n, "power")
    coeffsFrequency = calculate_regression(n, "frequency")
    coeffsPhase     = calculate_regression(n, "cos_phase")
    
    for t in range(len(x)):
        errPower[t]     = (clean_signal(b[t], x[t], coeffsPower, "power"))
        errFrequency[t] = (clean_signal(b[t], x[t], coeffsFrequency, "frequency"))
        errPhase[t]     = (clean_signal(b[t], x[t], coeffsPhase, "cos_phase"))
    errs[:,n-1] = [np.linalg.norm(errPower), np.linalg.norm(errFrequency), np.linalg.norm(errPhase)]

## FREQUENCY CONVERGES FASTER FOR THIS DATA SET
print("Power, Frequency, Phase (Phase is broken)")
print(errs)
    


