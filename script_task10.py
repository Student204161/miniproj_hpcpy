from os.path import join
import sys
import cupy as cp
import time


def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def batched_jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)  # shape: (B, 514, 514)

    for i in range(max_iter):
        # Compute new values: average of 4 neighbors
        u_new = 0.25 * (
            u[:, 1:-1, :-2] +  # left
            u[:, 1:-1, 2:]  +  # right
            u[:, :-2, 1:-1] +  # top
            u[:, 2:, 1:-1]     # bottom
        )  # shape: (B, 512, 512)

        if i % 100 == 0:
            delta = cp.abs(u[:, 1:-1, 1:-1] - u_new)
            delta = cp.where(interior_mask, delta, 0.0)
            max_delta = cp.max(delta)
            if max_delta < atol:
                break

        # Update interior values only
        u[:, 1:-1, 1:-1] = cp.where(interior_mask, u_new, u[:, 1:-1, 1:-1])

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
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
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20000  #20_000
    ABS_TOL = 1e-4

    start_time = time.time()

    all_u = batched_jacobi(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)

    end_time = time.time()
    print(f"Jacobi loop time: {end_time - start_time:.4f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))


if __name__ == '__main__':
    main()
