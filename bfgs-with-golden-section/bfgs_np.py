from golden_section import Golden_Section_Search
import numpy as np
import scipy
import time
from math import sin, cos, sqrt
import cProfile
import sys
"""
Functions to be optimized and their gradient vector
"""
func1 = lambda x : x[0]**2 + x[1]**2 - x[0]*x[1] - 4*x[0]- x[1]
func1g_vec = [lambda x : 2*x[0] - x[1] - 4, lambda x : -x[0] + 2*x[1] - 1]

func2 = lambda x : (1-x[0])**2 + (-(x[0]**2)+x[1])**2
func2g_vec = [lambda x : -4*x[0]*(-x[0]**2 + x[1]) + 2*x[0] - 2, lambda x : -2*x[0]**2 + 2*x[1]]

func3 = lambda x : (x[0]+3*x[1]+x[2])**2 + 4*(x[0]-x[1])**2 + x[0]*sin(x[2])
func3g_vec = [lambda x : 10*x[0] - 2*x[1] + 2*x[2] + sin(x[2]), lambda x : -2*x[0] + 26*x[1]+ 6*x[2], lambda x : x[0]*cos(x[2]) + 2*x[0] + 6*x[1] + 2*x[2]]

"""
Functions optimized using sciy to obtain ground truth
"""
#print(scipy.optimize.minimize(func1, np.array([4,4]).reshape(2,1), method='BFGS'))
#print(scipy.optimize.minimize(func2, np.array([6/5,5/4]).reshape(2,1), method='BFGS'))
#print(scipy.optimize.minimize(func3, np.array([-1,-1,-1]).reshape(3,1), method='BFGS'))

def get_stop_condition(vec, alpha):
  """
  Calculates |d_vec*alpha| for the BFGS stopping criteria.

  Parameters
  ----------
  vec : vector
      Direction Vector.
  alpha : float
      Step size.
  """
  square_sum = 0
  for term in vec:
    square_sum+=(alpha*term[0])**2
  return sqrt(square_sum)

def BFGS(func, g_func_vec, point_vec, tolerance=1e-8, maxIter=100, verbose=False):
  """
  Perform a BFGS optimization of a function in the form of a Sympy expression.

  Parameters
  ----------
  func : Function handle
      The objective function.
  g_func_vec : List of function handles
      Gradient vector.
  point_vec : column array
      The initial starting points for minimum.
  tolerance : float.
      Stopping criteria for optimum.
  maxIter : int.
      Max number of iterations.
  verbose : bool.
      If set to True, print out point, function value, and alpha every iteration.
  """
  print("\nBegin processing...")

  alphaList = []
  numParam = len(point_vec)                                 # Number of parameters in objective function
  iterations = 0
  prev_point_vec = point_vec
  point_list = list(point_vec.reshape(1,-1).tolist())       # Save new point every iteration
  func_value = None                                         # Function Value
  alpha = 1                                                 # Initial Alpha guess
  alpha_ub = 10                                             # Initial Alpha search upper bound
  alpha_lb = 0                                              # Initial Alpha search lower bound
  inv_hessian = np.linalg.inv(np.identity(numParam))        # Initial Inverse Identity matrix for inverse Hessian

  prev_g_vec = np.array([g_func(point_vec) for g_func in g_func_vec]).astype(np.float64)  # Substitute point vector into partial derivatives
  start_time = time.time()

  while True:
    if verbose:
      print(f"Point: {point_vec}, FVal: {func_value} Alpha: {alpha}")
      print(f"Inverse Hessian:\n{inv_hessian}\n")
      print(f"Gradient Vector:\n{prev_g_vec}\n")
      
    d_vec = -1*inv_hessian@prev_g_vec                                                     # Search direction vector    
    min_alpha_func = lambda a: func([point + a*d for point, d in zip(point_vec, d_vec)])  # Alpha substituted into objective function


    # Perform golden section search on min_alpha_func while warning if one of the limits is hit
    alpha = Golden_Section_Search(alpha_lb, alpha_ub, 0.00000001, min_alpha_func)
    alphaList.append(alpha)
    if abs(alpha-alpha_ub) < 0.01:
      print(f"!!!!!!!!      Alpha {alpha} hitting upper limit {alpha_ub} in iteration {iterations}\n")
    elif abs(alpha-alpha_lb) < 0.01:
      print(f"!!!!!!!!      Alpha {alpha} hitting lower limit {alpha_lb} in iteration {iterations}\n")


    prev_point_vec = point_vec                                                              # Save current point vector
    point_vec = prev_point_vec + alpha*d_vec                                                # Calculate new point vector
    point_list.append(point_vec.reshape(1,-1)[0].tolist())                                  # Add new point to list
    g_vec = np.array([g_func(point_vec) for g_func in g_func_vec]).astype(np.float64)       # Get new gradient vector 

    delta_g = g_vec - prev_g_vec                # Calculate delta gradient
    delta_point = point_vec - prev_point_vec    # Calculate delta point vector
    
    # Perform BFGS Method to obtain new inverse Hessian
    bfgs_term_1 = 1 + (np.transpose(delta_g)@inv_hessian@delta_g)/(np.transpose(delta_point)@delta_g)
    bfgs_term_2 = (delta_point@np.transpose(delta_point))/(np.transpose(delta_point)@delta_g)
    bfgs_term_3 = ((inv_hessian@delta_g@np.transpose(delta_point))+(delta_point@np.transpose(delta_g)@inv_hessian))/(np.transpose(delta_point)@delta_g)

    inv_hessian = inv_hessian + bfgs_term_1*bfgs_term_2 - bfgs_term_3


    prev_g_vec = g_vec                   # Save gradient vector
    func_value = func(point_vec)         # Calculate function value 
    iterations+=1                        # increment iteration

    # Evaluate Stopping criteria
    if get_stop_condition(d_vec, alpha) < tolerance or iterations == maxIter:
      print(f"Time elapsed: {time.time()-start_time}")
      print(f"Function: {func}")
      print(f"Function Value: {func_value}, Iterations: {iterations}")
      print(alphaList)
      [print(f"x{i}: {point_vec[i]}") for i in range(0,len(point_vec))]

      return point_list

if __name__ == '__main__':
  try:
    if sys.argv[1] == "-v":
      verboseFlag = True
  except:
    verboseFlag = False

  BFGS(func1, func1g_vec, np.array([[4],[4]]), 0.00000001, verbose=verboseFlag)
  BFGS(func2, func2g_vec, np.array([[6/5],[5/4]]), 0.00000001, verbose=verboseFlag)
  BFGS(func3, func3g_vec, np.array([[-1],[-1],[-1]]), 0.00000001, 1000, verbose=verboseFlag)
