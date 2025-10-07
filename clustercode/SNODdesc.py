import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# single agent S-NOD
def SA_SNOD(t, x, params):
    z, s = x
    eps, d, a, k, ks, mu0, I = params

    dzdt = -d*z + np.tanh((k*z**2 + mu0 - s)*a*z + I)
    dsdt = eps*(-s + ks*z**4)

    return [dzdt, dsdt]

def SA_SNOD_inv(t, x, params):
    dzdt, dsdt = SA_SNOD(t, x, params)
    return [-dzdt, -dsdt]

def SA_SNOD_jacobian(t, x, params):
    z, s = x
    eps, d, a, k, ks, mu0, I = params

    phi = a*z*(k*z**2 + mu0 - s) + I
    J = np.zeros((2, 2))

    J[0,0] = -d + a*(3*k*z**2 + mu0 - s)/(np.cosh(phi)**2)
    J[0,1] = -(a*z)/(np.cosh(phi)**2)
    J[1,0] = 4*eps*ks*z**3
    J[1,1] = -eps

    return J

def calc_stability(x, params, tol=1e-3):
    J = SA_SNOD_jacobian(0, x, params)
    eigvals = np.linalg.eigvals(J)
    if all([eig.real < -tol for eig in eigvals]): return "Stable"
    if all([eig.real > tol for eig in eigvals]): return "Unstable"
    if all([np.abs(eig.real) < tol for eig in eigvals]): return "Singular"
    else: return "Saddle"

# escape condition
def esc_event(t, x, params):
    return norm(x) - 500.0
esc_event.terminal = True

def find_all_roots(f, interval, num_points=1000):
    x = np.linspace(interval[0], interval[1], num_points)
    roots = []
    for i in range(len(x) - 1):
        if np.sign(f(x[i])) != np.sign(f(x[i + 1])):
            sol = root_scalar(f, bracket=[x[i], x[i + 1]])
            if sol.converged:
                roots.append(sol.root)
    return np.array(roots)

def has_limit_cycle(y):
    max_y = y[:, np.argmax([norm(y[:, i]) for i in range(y.shape[1])])]
    min_y = y[:, np.argmin([norm(y[:, i]) for i in range(y.shape[1])])]
    # max_y = [0, 0.5]
    y_dist_to_max = np.array([norm(y[:, i]-max_y) for i in range(y.shape[1])])
    return_idx = np.where(y_dist_to_max < 0.06*norm(max_y-min_y))[0]

    if len(return_idx) <= 1: return False
    diff = np.diff(return_idx)
    gaps = np.where(diff > 1)[0]
    if len(gaps) == 0: return False
    # min_diff = np.average(diff[gaps])
    min_diff = diff[gaps]
    return min_diff

def get_orbit_limit(f, x0, params):
    t_int = 100 # initial time interval, set to 10 to eliminate transient behavior
    orb_type = None # orbit limit if found
    lim_pt = None # limit
    lim_per = 0 # limit cycle period
    itr = 0 # iteration counter, to make sure we don't get stuck in an infinite loop

    while orb_type is None and itr < 5: 
        itr += 1 
        t = np.linspace(0, t_int, t_int*200) 
        sol = solve_ivp(f, [0, t_int], x0, args=(params,), events=[esc_event], t_eval=t) # simulate the system
        x0 = sol.y[:, -1] # update the initial condition
        if itr <= 2: continue # skip the transient

        if sol.status == 1: # if the simulation ran into an escape event
            orb_type = "Diverges"
            lim_pt = np.inf
            lim_per = [0]
        elif max([norm(sol.y[:, i]-sol.y[:, -1]) for i in range(sol.y.shape[1])]) < 0.005: # if the solution converged to a fixed point
            orb_type = "Converges"
            lim_pt = sol.y[:, -1]
            lim_per = 0
        elif np.any(lim_per := has_limit_cycle(sol.y)): # if the solution converged to a limit cycle
            orb_type = "Limit cycle"
            lim_pt = sol.y[:, np.argmax([norm(sol.y[:, i]) for i in range(sol.y.shape[1])])]
        else: # if the solution did not converge to anything, increase the time interval and try again
            t_int += 50

    if orb_type is None: orb_type = "Unknown"; lim_pt = np.nan; lim_per = [0]

    return orb_type, lim_pt, lim_per


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

        z_type = calc_stability([z, params[4]*z**4], params)
        desc[z_type].append([z, params[4]*z**4])

        if z_type == "Unstable":
            for _ in range(samples):
                s = params[4]*z**4 + np.random.normal(0, 0.001)
                z = z + np.random.normal(0, 0.001)
                x0 = [z, s]

                orb_type, lim_pt, lim_per = get_orbit_limit(SA_SNOD, x0, params)
                if orb_type == "Limit cycle" and ((len(desc["Limit cycle"]) == 0) or (min([norm(lim_pt-pt[0]) for pt in desc["Limit cycle"]]) > 0.1)):
                    desc["Limit cycle"].append((lim_pt, lim_per))

    return desc
    
