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


#5.69074 vs 8.7491
@cuda.jit
def jacobi(u, u_new, interior_mask, batch_size, height, width):
    batch_idx, x, y = cuda.grid(3)

    if batch_idx < batch_size and x < height and y < width:
        if interior_mask[batch_idx, x, y]:
            x_u = x + 1
            y_u = y + 1
            u_new[batch_idx, x_u, y_u] = 0.25 * (
                u[batch_idx, x_u, y_u - 1] +
                u[batch_idx, x_u, y_u + 1] +
                u[batch_idx, x_u - 1, y_u] +
                u[batch_idx, x_u + 1, y_u]
            )


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

def helper(all_u0, all_interior_mask, MAX_ITER, ABS_TOL):
    batch_size = all_u0.shape[0]
    threadsperblock = (4, 16, 16)  # batch, x, y threads

    u = cuda.to_device(np.copy(all_u0))
    u_new = cuda.to_device(np.copy(all_u0))
    interior_mask_d = cuda.to_device(all_interior_mask)

    blockspergrid_b = (batch_size + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_x = (512 + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_y = (512 + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_b, blockspergrid_x, blockspergrid_y)

    for k in range(MAX_ITER):
        jacobi[blockspergrid, threadsperblock](u, u_new, interior_mask_d, batch_size, 512, 512)
        u, u_new = u_new, u

        if k % 100 == 0:
            u_host = u.copy_to_host()
            u_new_host = u_new.copy_to_host()
            u_center = u_host[:, 1:-1, 1:-1]
            u_new_center = u_new_host[:, 1:-1, 1:-1]

            # Masked deltas: set non-interior positions to 0 (won't affect max)
            delta_field = np.abs(u_center - u_new_center)
            masked_delta = np.where(all_interior_mask, delta_field, 0)
            max_delta = masked_delta.max()

            if max_delta < ABS_TOL:
                print("k at break", k)
                break

                

    return u.copy_to_host()


#@profile
def main():
        # Load data
    LOAD_DIR = 'data'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    batch_size = 4
    #building_ids = building_ids[:n_batches]

    # Load floor plans
    all_u0 = np.empty((batch_size, 514, 514))
    all_interior_mask = np.empty((batch_size, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20000  #20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)


    #for i, (u, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        
    all_u = helper(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)


    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

if __name__ == '__main__':
    main()
    
