import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from statistics import mean 

func1 = lambda x : x[0]**2 + x[1]**2 - x[0]*x[1] - 4*x[0]- x[1]
func2 = lambda x : (1-x[0])**2 + (-(x[0]**2)+x[1])**2

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



def PSO(func, num_vars, x_limits=[-100.0,100.0], velocity_limits=[-100,100], num_particles=30,
        max_velocity=100, w=0.7, c1=2, c2=2, tol=1e-6, max_iter=400, stopping_criteria='Function Value',
        stall_iterations_limit=10):
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
  while iterations < max_iter:
    for particle in population:
      func_val = particle.calculate_func_val()
      
      try:
        if func_val < g_func_best:
          g_best = particle.x_vec
          g_func_prev_best = g_func_best
          g_func_best = func_val
          print(f"stall_iterations:{stall_iterations}|g_best:{g_best}|g_func_best:{g_func_best}")
          stall_iterations = 0
      except Exception as e:
          print(e)
          g_best = particle.x_vec
          g_func_best = func_val
          g_func_prev_best = g_func_best+10

    g_best_list.append(g_best)
    g_func_best_list.append(g_func_best)

    if stopping_criteria == "Velocity":
      if mean([np.linalg.norm(particle.velocity_vec) for particle in population]) < tol and stall_iterations > stall_iterations_limit:
        print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
        return population, g_best_list, g_func_best_list, iterations
    else:
      if abs(g_func_best-g_func_prev_best) < tol and stall_iterations > stall_iterations_limit:
        print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
        return population, g_best_list, g_func_best_list, iterations

    for particle in population:
      particle.update(g_best, c1, c2, w)

    iterations+=1
    stall_iterations+=1

  print(f"Optimum Found after {iterations} iterations at x={g_best}, f={g_func_best}")
  return population, g_best_list, g_func_best_list, iterations

def visualizeHistory2D(func=None, data=None, bounds=None, 
                       minima=None, g_best=None, g_func_best=None):
    """Visualize the process of optimizing
    # Arguments
        func: object function
        history: dict, object returned from pso above
        bounds: list, bounds of each dimention
        minima: list, the exact minima to show in the plot
        func_name: str, the name of the object function
        save2mp4: bool, whether to save as mp4 or not
    """

    print('## Visualizing optimizing')
    assert len(bounds)==2

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
        #ax2.set_yscale('log')

        # data to be plot
        #data = history['particles'][frame]
        #global_best = np.array(history['global_best_fitness'])

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




    ani.save('PSO_population.gif', writer="imagemagick")

    plt.show()

if __name__ == '__main__':

  pop, x_best, func_best, iterations = PSO(func2, 2, tol=1e-6)

  plot_list = [[list(particle.x_vec_history[j]) for particle in pop] for j in range(0, iterations)]

  visualizeHistory2D(func=func2, data=plot_list, bounds=[-100,100], minima=x_best[-1], g_best=x_best, g_func_best=func_best)
  #plt.scatter(*zip(*plot_list[0]))
  #plt.show()