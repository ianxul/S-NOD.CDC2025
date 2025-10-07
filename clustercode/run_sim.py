import numpy as np
import pickle
import os
import SNODdesc as snod
from itertools import product

if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    mu0 = list(np.linspace(0.7, 1.1, 500))[idx]
    Is = list(np.linspace(0.0, 0.1, 500))

    save_obj = []
        
    for i, I in enumerate(Is):
        snod_params = [1/10, 1.0, 1.0, 2.3, 16.0, mu0, I]
        desc = snod.get_desc(snod_params)
        
        save_obj.append({"mu0": mu0, "I": I, "desc": desc})

    # save the results as pickle
    with open("/scratch/network/ib4602/SNOD/pickles/desc_adaptivefinebig_{}.pkl".format(idx), "wb") as f:
        pickle.dump(save_obj, f)
    print(f"Saved desc")
    