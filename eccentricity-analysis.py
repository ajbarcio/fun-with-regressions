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

# calculate regression
A = np.zeros((data.shape[0],2*n+1))
A[:,0]=np.ones((data.shape[0],))
for i in range(0,n):
    A[:,1+2*i]=np.cos(x*(i+1))
    A[:,1+2*i+1]=np.sin(x*(i+1))

# A z = b
# z ~= (A^T @ A)^{-1} A^T b
z = np.linalg.solve(A.T @ A, A.T @ b) # coefficients of our regression equation

# function to correct signal at any measured delta and position
def clean_signal(delta, pos):
    a = np.zeros(2*n+1)
    a[0] = 1
    for i in range(0,n):
        a[1+2*i]=np.cos(pos*(i+1))
        a[1+2*i+1]=np.sin(pos*(i+1))
    
    corr = a.dot(z)#*(np.linalg.norm(a))

    return delta-corr



#################################### Tests that prove it works
# plt.figure(1)
# plt.scatter(x, b)
# plt.scatter(x, A @ z)
# plt.figure(2)
# plt.scatter(x, b-A @ z)

# err = np.zeros(len(x))
# for i in range(len(x)):
#     err[i] = (clean_signal(b[i], x[i]))
# assert(len(err)==len(x))
# print(len(err))
# print(len(x))
# plt.figure(2)
# plt.scatter(x, err)

# # plt.plot(x, err, "--")

# plt.show()


