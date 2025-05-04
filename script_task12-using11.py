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
def jacobi(u, u_new, interior_mask, height, width):
    x, y = cuda.grid(2)
    batch_idx = cuda.blockIdx.z  # batch dimension is the 3rd grid dim

    if batch_idx < u.shape[0] and x < height and y < width:
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
    height = 512
    width = 512

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y, batch_size)

    u = cuda.to_device(np.copy(all_u0))
    u_new = cuda.to_device(np.copy(all_u0))
    interior_mask_d = cuda.to_device(all_interior_mask)

    for k in range(MAX_ITER):
        jacobi[blockspergrid, threadsperblock](u, u_new, interior_mask_d, height, width)
        u, u_new = u_new, u

        if k % 100 == 0:
            u_host = u.copy_to_host()
            u_new_host = u_new.copy_to_host()
            u_center = u_host[:, 1:-1, 1:-1]
            u_new_center = u_new_host[:, 1:-1, 1:-1]

            delta_field = np.abs(u_center - u_new_center)
            masked_delta = np.where(all_interior_mask, delta_field, 0)
            max_delta = masked_delta.max()

            if max_delta < ABS_TOL:
                print("k at break", k)
                break

    return u.copy_to_host()

def main():
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    batch_size = 20
    MAX_ITER = 20000
    ABS_TOL = 1e-4

    all_building_ids = []
    all_u_list = []
    all_mask_list = []

    for batch_start in range(0, len(building_ids), batch_size):
        batch_ids = building_ids[batch_start:batch_start + batch_size]
        actual_batch_size = len(batch_ids)

        batch_u0 = np.empty((actual_batch_size, 514, 514))
        batch_mask = np.empty((actual_batch_size, 512, 512), dtype=bool)

        for i, bid in enumerate(batch_ids):
            u0, mask = load_data(LOAD_DIR, bid)
            batch_u0[i] = u0
            batch_mask[i] = mask

        batch_u = helper(batch_u0, batch_mask, MAX_ITER, ABS_TOL)

        all_building_ids.extend(batch_ids)
        all_u_list.append(batch_u)
        all_mask_list.append(batch_mask)

    # Concatenate results
    all_u = np.concatenate(all_u_list, axis=0)
    all_mask = np.concatenate(all_mask_list, axis=0)

    # Print stats for everything at once
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))

    for bid, u, mask in zip(all_building_ids, all_u, all_mask):
        stats = summary_stats(u, mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

if __name__ == '__main__':
    main()
    
