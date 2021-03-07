import math
import numpy as np

# Functions defined in Coding Assignment 1
def func1(x):
  return x*math.cos(math.pi*(x**2))

def func2(x):
  return 4*x**2 - 12*x + 9

def func3(x):
  return 3*(x**5) - 7*(x**3) -54*x + 21

def Golden_Section_Search(lb, ub, tolerance, funcHandle):
  """ 
  Perform golden section search to find point with minimum gradient. 

  Parameters
  ----------
  lb : float
      Lower search bound.
  ub : float
      Upper search bound.
  tolerance : float.
      Stopping criteria for optimum.
  funcHandle : Function handle.
      The objective function.
  """
  iterations = 0
  max_iterations = 100
  x_min = 0
  f_min = 0
  lowerBoundList = [lb]
  upperBoundList = [ub]
  k_gold = 2/(1+math.sqrt(5)) # Golden ratio

  search_interval = k_gold*(ub - lb)
  x_l = ub - search_interval # Initial x_lower
  x_u = lb + search_interval # Initial x_upper

  f_xl = funcHandle(x_l) # Iniital f(x_lower)
  f_xu = funcHandle(x_u) # Initial f(x_upper)

  # While loop executes as long as abs(upper bound - lower bound) is less than tolerance
  # and number of iterations is less than max iterations
  while abs(ub - lb) >= tolerance and iterations <= max_iterations:
    if f_xl <= f_xu:
      search_interval = k_gold*(x_u - lb)
      ub = x_u
      x_u = x_l
      x_l = ub - search_interval
      f_xu = f_xl
      f_xl = funcHandle(x_l)
      lowerBoundList.append(lb)
      upperBoundList.append(ub)

    else:
      lb = x_l
      search_interval = k_gold*(ub - lb)
      x_l = x_u
      x_u = lb + search_interval
      f_xl = f_xu
      f_xu = funcHandle(x_u)
      lowerBoundList.append(lb)
      upperBoundList.append(ub)

    iterations+=1

  x_min = x_l
  f_min = f_xl

  #print(f"x_l = ({x_l}, {funcHandle(x_l)}), x_u = ({x_u}, {funcHandle(x_u)}), iterations = {iterations}")
  return (x_l+x_u)/2
        
if __name__=='__main__':
  Golden_Section_Search(0, 0.7, 0.0001, func1)


