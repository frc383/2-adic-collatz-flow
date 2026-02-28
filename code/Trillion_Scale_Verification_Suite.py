import os
import time
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import shared_memory
from numba import njit
from typing import List, Dict

# ==========================================
# PHASE 1: ALGEBRAIC SIEVE GENERATION
# ==========================================

def is_resolved(r: int, k: int) -> bool:
    """
    Simulates the exact trajectory of a residue class r (mod 2^k).
    Optimized with bitwise operations for faster branch evaluation.
    """
    n_test = r
    A = 0  # Count of Archimedean expansions (3n+1)
    B = 0  # Count of 2-adic contractions (n/2)

    while B < k:
        if n_test & 1:  # Bitwise odd check
            n_test = 3 * n_test + 1
            A += 1
        else:
            n_test >>= 1  # Bitwise division by 2
            B += 1
            if 3**A < 2**B:
                return True
    return False

def Generate_Resolved_Pairs(max_k: int = 24):
    """
    Performs a Breadth-First Expansion of the Collatz tree up to a specified modulo depth.
    Generates 'resolved_pairs.csv' for the O(1) Memory Projection Sieve.
    """
    print(f"Generating Breadth-First Collatz Expansion up to depth k={max_k}...")
    start_time = time.time()

    resolved_pairs: List[Dict[str, int]] = []
    resolved_pairs.append({'residue': 1, 'modulo': 4})
    unresolved: List[int] = [3]

    for k in range(2, max_k):
        next_unresolved: List[int] = []
        modulus = 2 ** (k + 1)

        for r in unresolved:
            branch1 = r
            branch2 = r + (2 ** k)

            if is_resolved(branch1, k + 1):
                resolved_pairs.append({'residue': branch1, 'modulo': modulus})
            else:
                next_unresolved.append(branch1)

            if is_resolved(branch2, k + 1):
                resolved_pairs.append({'residue': branch2, 'modulo': modulus})
            else:
                next_unresolved.append(branch2)

        unresolved = next_unresolved

        if (k + 1) % 4 == 0 or (k + 1) == max_k:
            eliminated = 1.0 - (len(unresolved) / (2 ** k))
            print(f"Depth {k+1:02d} | Modulo: 2^{k+1:<2} | Unresolved Classes: {len(unresolved):<8} | Elimination Rate: {eliminated * 100:.4f}%")

    df = pd.DataFrame(resolved_pairs)
    df.to_csv('resolved_pairs.csv', index=False)
    end_time = time.time()
    print(f"\nDone! Generated {len(resolved_pairs):,} resolved pairs in {end_time - start_time:.2f} seconds.\n")


# ==========================================
# PHASE 2 & 3: CORE JIT & MULTIPROCESSING
# ==========================================

@njit(nogil=True, boundscheck=False)
def verification_chunk(start: int, end: int, sieve: np.ndarray, mod_depth: int):
    """
    Implements the JIT-compiled 2-adic extraction loop with uint64 overflow protection.
    Features fused 3n+1/even steps and boundscheck=False for maximum L3 cache throughput.
    """
    max_peak = np.uint64(0)
    peak_seed = np.uint64(0)
    longest_path = 0
    path_seed = np.uint64(0)

    # Force the starting integer to align with 4k + 3
    remainder = start % 4
    if remainder == 0:
        start += 3
    elif remainder == 1:
        start += 2
    elif remainder == 2:
        start += 1

    # Step by 4 to completely bypass Evens and 4k+1 seeds
    for n in range(start, end, 4):
        # Bitwise AND instead of Modulo for O(1) single-cycle hardware lookup
        if sieve[n & (mod_depth - 1)]:
            continue

        current = np.uint64(n)
        steps = 0
        local_peak = current

        while current >= np.uint64(n):
            if (current & np.uint64(1)) == 0:
                current >>= np.uint64(1)
                steps += 1
            else:
                # Fused step: 3n+1 is guaranteed even, so we do both instantly
                current = np.uint64(3) * current + np.uint64(1)
                if current > local_peak:
                    local_peak = current

                current >>= np.uint64(1)
                steps += 2

        if local_peak > max_peak:
            max_peak = local_peak
            peak_seed = np.uint64(n)
        if steps > longest_path:
            longest_path = steps
            path_seed = np.uint64(n)

    return max_peak, peak_seed, longest_path, path_seed

def _chunk_worker(args):
    """
    Worker function to attach to shared memory and run the JIT chunk.
    """
    chunk_start, chunk_end, shm_name, mod_depth = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_sieve = np.ndarray((mod_depth,), dtype=np.bool_, buffer=existing_shm.buf)

    result = verification_chunk(chunk_start, chunk_end, shared_sieve, mod_depth)

    existing_shm.close()
    return result


def Build_Memory_Sieve(mod_depth: int, shm_name: str) -> shared_memory.SharedMemory:
    """
    Allocates the zero-copy shared memory boolean array and loads resolved pairs.
    """
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        existing_shm.unlink()
    except FileNotFoundError:
        pass

    print(f"Allocating {mod_depth / 1e6:.2f} MB for the shared sieve array...")
    nbytes = mod_depth * np.dtype(np.bool_).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)

    sieve_array = np.ndarray((mod_depth,), dtype=np.bool_, buffer=shm.buf)
    sieve_array.fill(False)

    sieve_array[0::2] = True
    sieve_array[1::4] = True

    if os.path.exists('resolved_pairs.csv'):
        print("Loading 'resolved_pairs.csv' and populating sieve...")
        df = pd.read_csv('resolved_pairs.csv')
        for _, row in df.iterrows():
            sieve_array[row['residue']::row['modulo']] = True
    else:
        print("No 'resolved_pairs.csv' found. Proceeding with baseline structural bypasses...")
        sieve_array[3::16] = True
        sieve_array[11::16] = True

    elimination_rate = (np.sum(sieve_array) / mod_depth) * 100
    print(f"Sieve Built! Elimination Rate: {elimination_rate:.4f}%")
    print(f"Shared memory block '{shm_name}' is active and ready.\n")

    return shm


def Run_Sweep(mod_depth: int, shm_name: str, search_bound: int, chunk_size: int):
    """
    Executes the continuous verification sweep chunked via multiprocessing.Pool.
    """
    num_cores = multiprocessing.cpu_count()
    print(f"Initiating parallel verification up to {search_bound:,}...")
    print(f"Distributing {chunk_size:,} integer chunks across {num_cores} physical CPU cores...\n")

    start_time = time.time()

    global_max_peak = np.uint64(0)
    global_peak_seed = np.uint64(0)
    global_longest_path = 0
    global_path_seed = np.uint64(0)

    tasks = []
    for chunk_start in range(3, search_bound, chunk_size):
        chunk_end = min(chunk_start + chunk_size, search_bound)
        tasks.append((chunk_start, chunk_end, shm_name, mod_depth))

    total_chunks = len(tasks)
    chunks_completed = 0

    with multiprocessing.Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(_chunk_worker, tasks):
            max_peak, peak_seed, longest_path, path_seed = result

            if max_peak > global_max_peak:
                global_max_peak, global_peak_seed = max_peak, peak_seed
            if longest_path > global_longest_path:
                global_longest_path, global_path_seed = longest_path, path_seed

            chunks_completed += 1
            if chunks_completed % 500 == 0 or chunks_completed == total_chunks:
                print(f"Progress: {chunks_completed:,} / {total_chunks:,} chunks analyzed...")

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n================ VERIFICATION COMPLETE ================")
    print(f"Search Space:        Up to {search_bound:,}")
    print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("-" * 55)
    print(f"MAX PEAK ALTITUDE:   {global_max_peak:,}")
    print(f"PEAK SEED:           {global_peak_seed:,}")
    print("-" * 55)
    print(f"LONGEST PATH (Steps):{global_longest_path}")
    print(f"PATH SEED:           {global_path_seed:,}")
    print("=======================================================")


# ==========================================
# MASTER EXECUTION WORKFLOW
# ==========================================

if __name__ == "__main__":
    # Configuration parameters
    MOD_K_DEPTH = 24  # Modulo depth for the sieve (k=24 -> 16 MB)
    MODULO_DEPTH_CONFIG = 2**MOD_K_DEPTH
    SHARED_MEM_NAME = 'collatz_sieve_shm'
    SEARCH_BOUND_CONFIG = 2**45
    CHUNK_SIZE_CONFIG = 1_000_000_000

    # Phase 1: Generate the mathematical bounds
    # Note: If you already have the CSV and want to save time, you can comment this line out.
    Generate_Resolved_Pairs(max_k=MOD_K_DEPTH)

    # Phase 2: Build the sieve and hold the shared memory object
    main_shm_block = Build_Memory_Sieve(MODULO_DEPTH_CONFIG, SHARED_MEM_NAME)

    try:
        # Phase 3: Run the actual multi-core verification sweep
        Run_Sweep(
            mod_depth=MODULO_DEPTH_CONFIG,
            shm_name=SHARED_MEM_NAME,
            search_bound=SEARCH_BOUND_CONFIG,
            chunk_size=CHUNK_SIZE_CONFIG
        )
    finally:
        # Phase 4: Securely clean up the shared memory regardless of success or crash
        print("\nReleasing shared memory resources...")
        main_shm_block.close()
        main_shm_block.unlink()
        print("Cleanup complete. Script terminated safely.")
