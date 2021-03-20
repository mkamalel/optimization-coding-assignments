import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from statistics import mean 
import time
import seaborn as sns
import sys

sns.set(rc={'figure.figsize':(10, 6)})


func1 = lambda x : x[0]**2 + x[1]**2 - x[0]*x[1] - 4*x[0]- x[1]
func2 = lambda x : (1-x[0])**2 + (-(x[0]**2)+x[1])**2
func3_unc = lambda x : (1-x[0])**2 + (-(x[0]**2)+x[1])**2
func3_constraint = lambda x : -10*x[0]-3*x[1]+25

def func3(x, r):
  if -10*x[0]-3*x[1]+25 >= 0:
    return (1-x[0])**2 + (-(x[0]**2)+x[1])**2 + r*(-10*x[0]-3*x[1]+25)**2
  else:
    return (1-x[0])**2 + (-(x[0]**2)+x[1])**2

class Particle():
  def __init__(self, x_vec, velocity_vec, max_velocity, func):
    self.x_vec = x_vec
    self.velocity_vec = velocity_vec
    self.max_velocity = max_velocity
    self.func = func
    self.x_best = None
    self.func_best = None
    self.x_vec_history = [x_vec]

  def calculate_func_val(self):
    func_val = self.func(self.x_vec)

    try:
      if func_val < self.func_best:
        self.x_best = self.x_vec
        self.func_best = func_val
    except:
      self.x_best = self.x_vec
      self.func_best = func_val

    return func_val

  def update(self, g_best, c1, c2, w):
    self.velocity_vec = w*self.velocity_vec + c1*np.random.random()*(self.x_best - self.x_vec) + c2*np.random.random()*(g_best - self.x_vec)
    self.velocity_vec[self.velocity_vec>self.max_velocity] = self.max_velocity
    self.velocity_vec[self.velocity_vec<-self.max_velocity] = -self.max_velocity

    self.x_vec = self.x_vec + self.velocity_vec    
    self.x_vec_history.append(self.x_vec)



def PSO(func, num_vars, x_limits=[-100.0,100.0], velocity_limits=[-1000,1000], num_particles=30,
        max_velocity=1000, w=0.7, c1=2, c2=2, tol=1e-6, max_iter=400, stopping_criteria='Function Value',
        stall_iterations_limit=10, verbose=False):
  """
  Perform a PSO optimization of a function.

  Parameters
  ----------
  func : Function handle
      The objective function.
  num_vars: int
      Number of variables in cost function
  x_limits: list
      Upper and lower bounds of x vector
  velocity_limits: list
      Upper and lower bounds of velocity
  num_particles: int
      Number of particles in population
  max_velocity: number
      Max velocity of each particle
  w: float
      Inertia factor
  c1: float
      Learning factor 1
  c2: float
      Learning factor 2
  tol : float
      Stopping criteria evaluation for optimum.
  max_iter : int
      Max number of iterations.
  stopping_criteria: string
      Stopping criteria, either Function Value or Velocity
  stall_iterations_limit: int
      Max stall iterations
  verbose : bool.
      If set to True, print out point, function value every iteration.
  """

  x_init_vectors = np.random.uniform(low=x_limits[0], high=x_limits[1], size=(num_particles, num_vars))

  velocity_init_vectors = np.random.uniform(low=x_limits[0], high=x_limits[1], size=(num_particles, num_vars))

  #print(x_init_vectors[0])
  population = [Particle(x_init_vectors[i], velocity_init_vectors[i], max_velocity, func) for i in range(0, num_particles)]

  iterations = 0
  stall_iterations = 0
  g_best_list = []
  g_func_best_list = []
  g_best = None
  g_func_best = None
  g_func_prev_best = 0
  initial_time = time.time()
  while iterations < max_iter:
    for particle in population:
      func_val = particle.calculate_func_val()
      
      try:
        if func_val < g_func_best:
          g_best = particle.x_vec
          g_func_prev_best = g_func_best
          g_func_best = func_val
          if verbose == True:
            print(f"stall_iterations:{stall_iterations}|g_best:{g_best}|g_func_best:{g_func_best}")
          stall_iterations = 0
      except Exception as e:
          g_best = particle.x_vec
          g_func_best = func_val
          g_func_prev_best = g_func_best+10

    g_best_list.append(g_best)
    g_func_best_list.append(g_func_best)

    if stopping_criteria == "Velocity":
      if mean([np.linalg.norm(particle.velocity_vec) for particle in population]) < tol and stall_iterations > stall_iterations_limit:
        total_time = time.time() - initial_time
        print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
        return population, g_best_list, g_func_best_list, iterations, total_time
    else:
      if abs(g_func_best-g_func_prev_best) < tol and stall_iterations > stall_iterations_limit:
        total_time = time.time() - initial_time
        print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
        return population, g_best_list, g_func_best_list, iterations, total_time

    for particle in population:
      particle.update(g_best, c1, c2, w)

    iterations+=1
    stall_iterations+=1

  total_time = time.time() - initial_time
  #print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
  return population, g_best_list, g_func_best_list, iterations, total_time

def penaltyPSO(func, num_vars, constraint_func, x_limits=[-100.0,100.0], velocity_limits=[-1000,1000], num_particles=30,
        max_velocity=1000, w=0.7, c1=2, c2=2, tol=1e-6, max_iter=400, stopping_criteria='Function Value',
        stall_iterations_limit=10, penalty_iter=500):
  """
  Perform a PSO optimization of a function.

  Parameters
  ----------
  func : Function handle
      The objective function.
  num_vars: int
      Number of variables in cost function
  constraint_func: Function handle
      Constraint function
  x_limits: list
      Upper and lower bounds of x vector
  velocity_limits: list
      Upper and lower bounds of velocity
  num_particles: int
      Number of particles in population
  max_velocity: number
      Max velocity of each particle
  w: float
      Inertia factor
  c1: float
      Learning factor 1
  c2: float
      Learning factor 2
  tol : float
      Stopping criteria evaluation for optimum.
  max_iter : int
      Max number of iterations.
  stopping_criteria: string
      Stopping criteria, either Function Value or Velocity
  stall_iterations_limit: int
      Max stall iterations
  penalty_iter: int
      Max evaluations of PSO
  """
  r_iterations = 0
  r = 0
  prev_pso_best = 1e20
  pso_best = [1e99]
  pso_x_best = None
  pso_pop_best = None
  iterations_best = None
  while r_iterations < penalty_iter:
    input_func = lambda x : func(x, r)
    pop, x_best, pso_val, iterations, _ = PSO(input_func, num_vars, tol=tol, stopping_criteria=stopping_criteria)

    if pso_val[-1] < pso_best[-1] and constraint_func(x_best[-1]) <= 0:
      pso_x_best = x_best
      pso_pop_best = pop
      iterations_best = iterations
      prev_pso_best = pso_best[-1]
      pso_best = pso_val
      r_best = r
      print(f"Penalty: x_best:{x_best[-1]}|func_best:{pso_best[-1]}|r:{r}")

    if abs(pso_best[-1]-prev_pso_best) < tol:
      print(f"Penalty Optimum Found after {r_iterations} iterations at x={pso_x_best[-1]}, f={pso_best[-1]}, r:{r_best}, constraint={constraint_func(pso_x_best[-1])}")
      return pso_pop_best, pso_x_best, pso_best, iterations_best, input_func,r_best
    else:
     print(f"Penalty: x_best:{x_best[-1]}|func_best:{pso_best[-1]}|r:{r}|constraint:{constraint_func(x_best[-1])}")

    r+=1 
    r_iterations+=1

  print(f"Penalty Optimum Found after {r_iterations} iterations at x={pso_x_best[-1]}, f={pso_best[-1]}, r:{r_best}, constraint={constraint_func(pso_x_best[-1])}")
  return pso_pop_best, pso_x_best, pso_best, iterations_best, input_func,r_best



def visualizeHistory2D(func=None, data=None, bounds=None, 
                       minima=None, g_best=None, g_func_best=None):
    """Visualize the process of optimizing
    # Arguments
        func: object function
        data: list, population of particles returned from PSO function
        bounds: list, bounds of each dimention
        minima: list, the exact minima to show in the plot
        g_best: list, global x best at each iteration
        g_func_best: list, global func value best at each iteration
    """
    print('## Visualizing optimizing')
    assert len(bounds)==2

    plot_list = [[list(particle.x_vec_history[j]) for particle in data] for j in range(0, iterations)]
    data = [[list(particle.x_vec_history[j]) for particle in data] for j in range(0, iterations)]
    # define meshgrid according to given boundaries
    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([x, y]) for x, y in zip(X, Y)])

    # initialize figure
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, facecolor='w')
    ax2 = fig.add_subplot(122, facecolor='w')

    # animation callback function
    def animate(frame, plot_lst):
        #print('current frame:',frame)
        ax1.cla()
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title(f'iter={frame}|g_best={g_best[frame]}')
        ax1.set_xlim(bounds[0], bounds[1])
        ax1.set_ylim(bounds[0], bounds[1])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title(f'g_func_best={g_func_best[frame]}')
        ax2.set_xlim(2,len(g_func_best))
        ax2.set_ylim(g_func_best[-1]*1.1,int(mean(g_func_best[0:10])*1.1))

        # contour and global minimum
        contour = ax1.contour(X,Y,Z, levels=50, cmap="magma")
        ax1.plot(minima[0], minima[1] ,marker='o', color='black')

        # plot particles
        ax1.scatter(*zip(*plot_list[frame]), marker='x', color='black')
        if frame > 1:
            for i in range(len(data[0])):
                ax1.plot([plot_list[frame-n][i][0] for n in range(2,-1,-1)],
                         [plot_list[frame-n][i][1] for n in range(2,-1,-1)])
        elif frame == 1:
            for i in range(len(data[0])):
                ax1.plot([plot_list[frame-n][i][0] for n in range(1,-1,-1)],
                         [plot_list[frame-n][i][1] for n in range(1,-1,-1)])
        

        x_range = np.arange(1, frame+2)
        ax2.plot(x_range, g_func_best[0:frame+1])

    ani = animation.FuncAnimation(fig, animate, fargs=(data,),
                    frames=len(data), interval=10, repeat=False, blit=False)



    # Uncomment to save as gif
    #ani.save('PSO_population.gif', writer="imagemagick")

    plt.show()

def plot_bar(x, y, xlabel, ylabel, title):
  plt.figure()
  plt.bar(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  for i, v in enumerate(y):
    plt.text(i - 0.1, v + 0.001, str(round(v,3)))

  plt.savefig(f'func2/{title}.png', bbox_inches='tight')

if __name__ == '__main__':
  # Function Select
  #"""
  func3Flag = False
  try:
    if sys.argv[1] == "3":
      eval_func = func2
      funcNum = 3
      func3Flag = True
    elif sys.argv[1] == "2":
      eval_func = func2
      funcNum = 2
    else:
      eval_func = func1
      funcNum = 1
  except:
    eval_func = func1
    funcNum = 1


  if func3Flag == False:
    # Run PSO 5 times
    func_best_list = []
    iterations_list = []
    runtime_list = []
    for i in range(0,5):
      pop, x_best, func_best, iterations, runtime = PSO(eval_func, 2, tol=1e-6, stopping_criteria="Function Value")
      func_best_list.append(func_best[-1])
      iterations_list.append(iterations)
      runtime_list.append(runtime)

    # Function Value
    plot_bar(range(len(func_best_list)), func_best_list, "Function Value", "Run", f"Function Value Best for Function {funcNum} over 5 Runs")

    # Num of Iterations
    #plt.figure()
    #plt.xlabel("Iterations")
    #plt.ylabel("Frequency")
    #plt.title(f"Function {funcNum} iterations disribution over 5000 runs")
    #plt.hist(iterations_list, bins=100)
    #plot_bar(range(len(iterations_list)), iterations_list, "# of Iterations", "Run", f"# of Iterations to Find Optimum for Function {funcNum} over 5 Runs")

    # Runtime
    plot_bar(range(len(runtime_list)), runtime_list, "Runtime(s)", "Run", f"Runtime to Find Optimum for Function {funcNum} over 5 Runs")

    # Change # of particles [10, 100, 1000]
    func_best_list = []
    iterations_list = []
    runtime_list = []
    popSize = ["10", "100", "1000"]
    for num in popSize:
      pop, x_best, func_best, iterations, runtime = PSO(eval_func, 2, tol=1e-6, stopping_criteria="Function Value", num_particles=int(num))
      func_best_list.append(func_best[-1])
      iterations_list.append(iterations)
      runtime_list.append(runtime)

    # Function Value
    plot_bar(popSize, func_best_list, "Function Value", "Run", f"Function Value Best for Function {funcNum} with Varying Population Sizes")

    # Num of Iterations
    plot_bar(popSize, iterations_list, "# of Iterations", "Run", f"# of Iterations to Find Optimum for Function {funcNum} with Varying Population Sizes")

    # Runtime
    plot_bar(popSize, runtime_list, "Runtime(s)", "Run", f"Runtime to Find Optimum for Function {funcNum} with Varying Population Sizes")    

    # Change c1,c2 [0.5, 5]
    func_best_list = []
    iterations_list = []
    runtime_list = []
    learning_factors = ["0.5", "5"]
    for c in learning_factors:
      pop, x_best, func_best, iterations, runtime = PSO(eval_func, 2, tol=1e-6, stopping_criteria="Function Value", c1=float(c), c2=float(c))
      func_best_list.append(func_best[-1])
      iterations_list.append(iterations)
      runtime_list.append(runtime)

    # Function Value
    plot_bar(learning_factors, func_best_list, "Function Value", "Run", f"Function Value Best for Function {funcNum} with Varying Learning Factors")

    # Num of Iterations
    plot_bar(learning_factors, iterations_list, "# of Iterations", "Run", f"# of Iterations to Find Optimum for Function {funcNum} with Varying Learning Factors")

    # Runtime
    plot_bar(learning_factors, runtime_list, "Runtime(s)", "Run", f"Runtime to Find Optimum for Function {funcNum} with Varying Learning Factors")    
    #"""
  else:
    #Function 3
    r_best_list = []
    func_best_list = []
    constraint_list = []
    iterations_list = []
    for i in range(0,5):
      pop, x_best, func_best, iterations, penalty_func, r_best = penaltyPSO(func3, 2, func3_constraint, tol=1e-6, stopping_criteria="Function Value")
      func_best_list.append(func_best[-1])
      iterations_list.append(iterations)
      r_best_list.append(r_best)
      constraint_list.append(func3_constraint(x_best[-1]))

    # Function Value
    plot_bar(range(len(func_best_list)), func_best_list, "Function Value", "Run", f"Constrained Optimum for Function {funcNum} over 5 Runs")

    # Num of Iterations
    plot_bar(range(len(func_best_list)), iterations_list, "# of Iterations", "Run", f"# of Iterations to Find Constrained Optimum for Function {funcNum} over 5 Runs")

    # Runtime
    plot_bar(range(len(func_best_list)), r_best_list, "Penalty Factor", "Run", f"Penalty Factor to Find Constrained Optimum for Function {funcNum} over 5 Runs")    
    #"""

    # Constraint Value
    plot_bar(range(len(func_best_list)), constraint_list, "Constraint Value", "Run", f"Constraint Value for Function {funcNum} over 5 Runs")


  # Uncomment to see PSO visualization
  #visualizeHistory2D(func=eval_func, data=pop, bounds=[-100,100], minima=x_best[-1], g_best=x_best, g_func_best=func_best)

  plt.show()