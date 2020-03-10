# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.integrate import solve_ivp#,odeint, ode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import sqrt


#Lotka Volterra Paramaters:
a = 1.0
b = 1.0
c = 1.0
d = 1.0
f_min = 0.25
r_min = 0.25
x_0 = [1.5,1.5,0]  #Initial Conditions


#p = np.random.random_sample(9)-0.5
p = np.zeros(9)
#p = [0.00014501, -0.0001194,  -0.00016883,  0.00019789,  0.00026289,  0.00046044, 0.00041951, -0.00034702,  0.00043658]
#p=[0.00,0.00,0.00,0.00]
#p = 0.001*p
print("p_initial:", p)

t_max = 1000 #duration of simulation
#num_slices = 20000 # number of points to calculate
#t = np.linspace(0, t_max,num_slices)
#t = np.linspace(0, t_max)
t=(0,t_max)
    
def mat_to_vec(p_matrix):
    p_vector = p_matrix.reshape(p_matrix.shape[0])
    return(p_vector)
    
def vec_to_mat(p_vector):
    dim = int(sqrt(p_vector.shape[0]))
    p_matrix = p_vector.reshape([dim,dim])
    return(p_matrix)
    
#Harvest function 
def H(x, p_vec): #x=[r,f] , p=parameter matrix for polynomial approximation (square)
    #if x[0]<r_min or x[1] < f_min:
        #h=0.0
        #return(h)
    xispositive = np.greater(x, [0,0,0])
    p_mat = vec_to_mat(p_vec)
    n = p_mat.shape[0] 
    e = np.arange(n)
    #print("P in function H = ", p)
    #sys.exit()
    r = x[0]*np.ones(n)
    Rv = np.power(r,e)
    
    f = x[1]*np.ones(n)
    Fv = np.power(f,e)

    M = np.outer(Rv , Fv)
    #print("M = ", M)
    #print("MdotP" , np.tensordot(M, p) )
    #sys.exit()
    #h = np.linalg.norm(np.multiply(M,p))
    h = abs(np.tensordot(M,p_mat))
    h = h*xispositive[0]*xispositive[1] #h is zero unless r, f > 0
    return(h)
    
#Lotka Volterra differential equations with harvest 
def dx(t, x):
    r = x[0]
    f = x[1]
    #TotH = x[2]
    
    h = H(x, p)
        
    dr = a*r - b*r*f - h
    df = -c*f + d*r*f
        
    dTotH = h
    #print(t,x[2],dTotH)
    return(dr,df,dTotH)
    
def get_x(p_):
    global p 
    p = p_
    sol = solve_ivp(dx, t, x_0)
    #print("y_shape = ", sol.y.shape )
    return(sol.y)
 
def get_M(p_):
   
    x = get_x(p_)
    #plt.close()
    l = plt.plot(x[0,0], x[1,0], 'ro')
    #plt.show()
    
    input("Press Enter to continue...")
    min_r = np.amin(x[0,:])
    min_f = np.amin(x[1,:])
    
    if min_r < r_min or min_f < f_min:
        print("Max H_tot = 100")
        return(100)
    #print ("Xlength= ", x.shape)
    #M = 1/(1+np.amax(x[:,2]))
    print("Max H_tot: ", np.amax(x[2,:]))
    return(1/(1+np.amax(x[2,:])))
    
#x = np.zeros(num_slices)
#M = get_M(p)
#print("MaxH = ", 1/M)
p0 = p
min_res = minimize(get_M, p0, method='Nelder-Mead')
print("result: ", min_res.x)

x = get_x(p)  
locmax = np.argmax(x[:,2])
k = plt.plot(t[0:locmax], x[:,2][0:locmax], 'bo')
m = plt.plot(t[0:locmax], x[:,1][0:locmax], 'ro')
o = plt.plot(t[0:locmax], x[:,0][0:locmax], 'go')
l = plt.plot(x[:,0], x[:,1], 'ro')
print("Tot_H: ", np.amax(x[2,:]))


#@1plt.show()
