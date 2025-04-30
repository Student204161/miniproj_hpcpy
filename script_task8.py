from os.path import join
import sys
import numpy as np
from numba import jit
from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask



def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

#5.69074 vs 8.7491
@cuda.jit
def jacobi(u, u_new, interior_mask):
    x, y = cuda.grid(2)
    #interior_mask: 512x512
    #u:  514x514
    if 0 <= x <= interior_mask.shape[0] - 1 and 0 <= y <= interior_mask.shape[1] - 1:
        if interior_mask[x, y]:
            x += 1
            y += 1
            u_new[x, y] = 0.25 * (u[x, y-1] + u[x, y+1] + u[x-1, y] + u[x+1, y])

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

#@profile
def main():
        # Load data
    LOAD_DIR = 'data'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = 4
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 2000 #20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)

    threadsperblock = (16, 16)  # common optimal size for 2D kernels

    for i, (u, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        
        u = cuda.to_device(u)
        u_new = cuda.device_array_like(u)
        #u_new is just zeros


        interior_mask = cuda.to_device(interior_mask)
        blockspergrid_x = (interior_mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (interior_mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        print(blockspergrid_x*threadsperblock[0],  blockspergrid_y * threadsperblock[1])
        for k in range(MAX_ITER):
            jacobi[blockspergrid, threadsperblock](u, u_new, interior_mask)
            #u = cuda.device_array_like(u_new)
            #u.copy_to_device(u_new)
            u, u_new = u_new, u
            
        u = u.copy_to_host()
        
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

if __name__ == '__main__':
    main()
    
