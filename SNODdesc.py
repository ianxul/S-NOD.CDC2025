import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Single agent S-NOD
def SA_SNOD(t, x, params):
    # State variables: opinion is z, recovery variable is s
    z, s = x
    # Parameters: eps is timescale, d is leak, a is linear self-reinforcement, k is nonlinear self-reinforcement, ks is gain on recovery, mu0 is basal sensitiviy, b is input
    eps, d, a, k, ks, mu0, b = params

    dzdt = -d*z + np.tanh((k*z**2 + mu0 - s)*a*z + b)
    dsdt = eps*(-s + ks*z**4)

    return [dzdt, dsdt]

# Backward-time version 
def SA_SNOD_inv(t, x, params):
    dzdt, dsdt = SA_SNOD(t, x, params)
    return [-dzdt, -dsdt]

# Jacobian of the system
def SA_SNOD_jacobian(t, x, params):
    # State variables
    z, s = x
    # Parameters
    eps, d, a, k, ks, mu0, b = params

    # phi is the function inside the tanh
    phi = a*z*(k*z**2 + mu0 - s) + b

    # Fill in the Jacobian matrix with partial derivatives
    J = np.zeros((2, 2))
    J[0,0] = -d + a*(3*k*z**2 + mu0 - s)/(np.cosh(phi)**2)
    J[0,1] = -(a*z)/(np.cosh(phi)**2)
    J[1,0] = 4*eps*ks*z**3
    J[1,1] = -eps

    return J

# Determine stability of a fixed point. The tolerance tol is used to determine if an eigenvalue is close enough to zero to be considered zero.
def calc_stability(x, params, tol=1e-3):
    J = SA_SNOD_jacobian(0, x, params)
    eigvals = np.linalg.eigvals(J)

    # We consider four cases:
    if all([eig.real < -tol for eig in eigvals]): return "Stable"
    if all([eig.real > tol for eig in eigvals]): return "Unstable"
    if all([np.abs(eig.real) < tol for eig in eigvals]): return "Singular"
    else: return "Saddle"

# Escape condition. This is used to stop simulations that diverge.
def esc_event(t, x, params):
    return norm(x) - 500.0
esc_event.terminal = True # This line makes the event stop the integration

# Find all roots of a function in a given interval using the bisection method
def find_all_roots(f, interval, num_points=1000):
    x = np.linspace(interval[0], interval[1], num_points)
    roots = []
    for i in range(len(x) - 1):
        # Only check in intervals where the function changes sign, this indicates a root exists in the interval
        # Resolution should be high enough, otherwise we might miss roots
        if np.sign(f(x[i])) != np.sign(f(x[i + 1])):
            sol = root_scalar(f, bracket=[x[i], x[i + 1]])
            if sol.converged:
                roots.append(sol.root)
    return np.array(roots)

# Check if a trajectory has a limit cycle by looking for repeated crossings of a Poincare section, we assume transient behavior has been eliminated
def has_limit_cycle(y):
    # y is a 2D array where each column is a point in the trajectory
    # We look for points that are close to the point with the maximum norm
    max_y = y[:, np.argmax([norm(y[:, i]) for i in range(y.shape[1])])]
    min_y = y[:, np.argmin([norm(y[:, i]) for i in range(y.shape[1])])]
    # y_dist_to_max is an array of distances from each point to the max_y point
    y_dist_to_max = np.array([norm(y[:, i]-max_y) for i in range(y.shape[1])])
    # return_idx is an array of indices where the distance is less than 6% of the distance between max_y and min_y. The 6% is arbitrary, but seems to work well
    return_idx = np.where(y_dist_to_max < 0.06*norm(max_y-min_y))[0]

    # If there are less than 2 such points, we cannot have a limit cycle
    if len(return_idx) <= 1: return False
    # diff is an array of differences between consecutive indices in return_idx
    diff = np.diff(return_idx)
    # gaps is an array of indices where the difference is greater than 1, indicating a the trajectory has left and returned to the Poincare section
    gaps = np.where(diff > 1)[0]
    if len(gaps) == 0: return False # if the trajectory never left the Poincare section, we cannot have a limit cycle
    # min_diff = np.average(diff[gaps])
    # Returns all the periods found
    periods = diff[gaps]
    return periods

# Simulate the system from an initial condition x0 and determine if it converges to a fixed point, diverges, or converges to a limit cycle
def get_orbit_limit(f, x0, params):
    t_int = 100 # initial time interval, set to 100 to eliminate transient behavior
    orb_type = None # orbit limit, if found
    lim_pt = None # limit
    lim_per = 0 # limit cycle period
    itr = 0 # iteration counter, to make sure we don't get stuck in an infinite loop

    while orb_type is None and itr < 5: # limit to 5 iterations, since we were running many initial conditions
        itr += 1 
        t = np.linspace(0, t_int, t_int*200) 
        sol = solve_ivp(f, [0, t_int], x0, args=(params,), events=[esc_event], t_eval=t) # simulate the system with terminal divergence event
        x0 = sol.y[:, -1] # update the initial condition
        if itr <= 2: continue # skip the transient, first three iterations, so we assume a pretty long transient

        if sol.status == 1: # if the simulation ran into an escape event
            orb_type = "Diverges"
            lim_pt = np.inf
            lim_per = [0]
        elif max([norm(sol.y[:, i]-sol.y[:, -1]) for i in range(sol.y.shape[1])]) < 0.005: # if the solution converged to a fixed point, within a tolerance of 0.005
            orb_type = "Converges"
            lim_pt = sol.y[:, -1]
            lim_per = 0
        elif np.any(lim_per := has_limit_cycle(sol.y)): # if the solution converged to a limit cycle
            orb_type = "Limit cycle"
            # we take the point with the maximum norm as the representative point of the limit cycle, but the orbit is actually represented by its period
            lim_pt = sol.y[:, np.argmax([norm(sol.y[:, i]) for i in range(sol.y.shape[1])])] 
        else: # if the solution did not converge to anything, increase the time interval and try again, we only do this twice
            t_int += 50

    # If we exit the loop without finding an orbit type, we set it to unknown
    if orb_type is None: orb_type = "Unknown"; lim_pt = np.nan; lim_per = [0]

    return orb_type, lim_pt, lim_per

# Get a description of the system for given parameters, counting fixed points and limit cycles
def get_desc(params:list, samples:int = 3):
    z_zeros = find_all_roots(lambda z: SA_SNOD(0, [z, params[4]*z**4], params)[0], [-1/params[1], 1/params[1]])
    desc = {
        "Stable": [],
        "Unstable": [],
        "Saddle": [],
        "Singular": [],
        "Limit cycle": [],
        "Unknown": []
    }
    for z in z_zeros:

        # Determine stability of the fixed point
        z_type = calc_stability([z, params[4]*z**4], params)
        desc[z_type].append([z, params[4]*z**4])

        # Limit cycles encircle unstable fixed points, so we only search for limit cycles if the fixed point is unstable
        if z_type == "Unstable":
            for _ in range(samples):
                # We sample initial conditions around the unstable fixed point, adding a small amount of noise. This is most important when the limit cycles are small
                s = params[4]*z**4 + np.random.normal(0, 0.001)
                z = z + np.random.normal(0, 0.001)
                x0 = [z, s]

                orb_type, lim_pt, lim_per = get_orbit_limit(SA_SNOD, x0, params)
                if orb_type == "Limit cycle" and ((len(desc["Limit cycle"]) == 0) or (min([norm(lim_pt-pt[0]) for pt in desc["Limit cycle"]]) > 0.1)):
                    desc["Limit cycle"].append((lim_pt, lim_per))

    return desc
    
