from os.path import join
import sys
import numpy as np
import time

def load_data(load_dir, bid):
    start = time.time()
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    print(f"[{bid}] Data loaded in {time.time() - start:.4f} seconds")
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    start = time.time()
    u = np.copy(u)

    loop_times = []
    for i in range(max_iter):
        t_loop = time.time()
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        loop_duration = time.time() - t_loop
        if i < 5:  # Print only first 5 for brevity
            print(f"  Iter {i} took {loop_duration:.6f} sec, delta={delta:.6e}")
        loop_times.append(loop_duration)
        if delta < atol:
            break
    print(f"Jacobi converged in {i+1} iterations, total time: {time.time() - start:.4f} seconds")
    return u


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

if __name__ == '__main__':
    total_start = time.time()

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    all_u = np.empty_like(all_u0)

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        print(f"\nRunning jacobi for building {building_ids[i]}...")
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nbuilding_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    print(f"\nTotal script time: {time.time() - total_start:.2f} seconds")
