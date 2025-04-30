import sys
import time
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


def process_building(args):
    bid, load_dir, max_iter, abs_tol = args
    u0, interior_mask = load_data(load_dir, bid)
    u = jacobi(u0, interior_mask, max_iter, abs_tol)
    stats = summary_stats(u, interior_mask)
    return bid, stats


def run_parallel(building_ids, load_dir, max_iter, abs_tol, num_workers):
    args = [(bid, load_dir, max_iter, abs_tol) for bid in building_ids]
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(process_building, args)
    duration = time.time() - start_time
    return duration

def run_parallel_dynamic(building_ids, load_dir, max_iter, abs_tol, num_workers):
    args = [(bid, load_dir, max_iter, abs_tol) for bid in building_ids]
    start_time = time.time()
    #chunksize = max(1, len(args) // (4 * num_workers)) 
    with Pool(num_workers) as pool:
        results = list(pool.imap(process_building, args, chunksize=1)) #imap unordered is possible to optimize a bit. 
    duration = time.time() - start_time
    return duration


def main():
    
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    if len(sys.argv) < 3:
        print("Usage: python simulate_ex5.py <N_BUILDINGS> <WORKERS_LIST_COMMA_SEPARATED>")
        sys.exit(1)

    N = int(sys.argv[1])
    workers_list = [int(w) for w in sys.argv[2].split(",")]

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    timings = {}
    for num_workers in workers_list:
        print(f"\nRunning with {num_workers} workers...")
        duration = run_parallel_dynamic(building_ids, LOAD_DIR, MAX_ITER, ABS_TOL, num_workers)
        print(f"Time taken with {num_workers} workers: {duration:.2f} seconds")
        timings[num_workers] = duration
  
    
    # Plot speedup and time taken
    worker_counts = sorted(timings.keys())
    times = [timings[w] for w in worker_counts]
    baseline_time = timings[min(worker_counts)]
    speedups = [baseline_time / t for t in times]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot speedup
    line1, = ax1.plot(worker_counts, speedups, marker='o', color='blue', label='Speedup')
    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Speedup', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot time on secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(worker_counts, times, marker='s', color='red', label='Time Taken')
    ax2.set_ylabel('Time Taken (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Speedup and Time Taken vs Number of Workers')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("speedup_and_time_plot_50floorsdynamo.png")
    print("\nSaved plot as 'speedup_and_time_plot.png'")



if __name__ == '__main__':
    main()
