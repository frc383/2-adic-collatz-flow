"""
2-Adic Collatz Flow: High-Performance Verification Suite

This script executes the trillion-scale empirical verification.
Phase 1: Generates the O(1) Memory Projection Sieve via Breadth-First Expansion.
Phase 2: Executes a JIT-compiled, multi-threaded 2-adic extraction sweep to 2^45.
"""

import pandas as pd
import numpy as np
import time
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit

# ==========================================
# PHASE 1: TREE EXPANSION (Cell 4.5)
# ==========================================
def is_resolved(r: int, k: int) -> bool:
    n_test = r
    A, B = 0, 0
    while B < k:
        if n_test % 2 == 1:
            n_test = 3 * n_test + 1
            A += 1
        else:
            n_test //= 2
            B += 1
            if 3**A < 2**B: return True
    return False

def Generate_Resolved_Pairs(max_k: int = 24) -> str:
    csv_filename = 'resolved_pairs.csv'
    if os.path.exists(csv_filename):
        print(f"'{csv_filename}' already exists. Skipping tree expansion.")
        return csv_filename
        
    print(f"Generating Breadth-First Collatz Expansion up to depth k={max_k}...")
    start_time = time.time()
    resolved_pairs = [{'residue': 1, 'modulo': 4}]
    unresolved = [3]
    
    for k in range(2, max_k):
        next_unresolved = []
        modulus = 2 ** (k + 1)
        for r in unresolved:
            branch1, branch2 = r, r + (2 ** k)
            if is_resolved(branch1, k + 1): resolved_pairs.append({'residue': branch1, 'modulo': modulus})
            else: next_unresolved.append(branch1)
            
            if is_resolved(branch2, k + 1): resolved_pairs.append({'residue': branch2, 'modulo': modulus})
            else: next_unresolved.append(branch2)
        unresolved = next_unresolved
        
    pd.DataFrame(resolved_pairs).to_csv(csv_filename, index=False)
    print(f"Generated {len(resolved_pairs):,} resolved pairs in {time.time() - start_time:.2f}s.")
    return csv_filename

# ==========================================
# PHASE 2: MEMORY PROJECTION SIEVE (Cell 5)
# ==========================================
def Build_Memory_Sieve(csv_filename: str):
    MODULO_DEPTH = 2**24 
    print(f"\nAllocating {MODULO_DEPTH / 1e6} MB for the O(1) sieve array...")
    sieve_array = np.zeros(MODULO_DEPTH, dtype=np.bool_)
    
    # Structural limits
    sieve_array[0::2] = True # Evens
    sieve_array[1::4] = True # 1 mod 4
    
    df = pd.read_csv(csv_filename)
    for _, row in df.iterrows():
        sieve_array[row['residue']::row['modulo']] = True
        
    elimination_rate = (np.sum(sieve_array) / MODULO_DEPTH) * 100
    print(f"Sieve Built! Elimination Rate: {elimination_rate:.4f}%")
    return sieve_array, MODULO_DEPTH

# ==========================================
# PHASE 3: JIT COMPILATION & SWEEP (Cell 6)
# ==========================================
@njit(nogil=True)
def verification_chunk(start: int, end: int, sieve: np.ndarray, mod_depth: int):
    max_peak, peak_seed, path_seed = np.uint64(0), np.uint64(0), np.uint64(0)
    longest_path = 0
    for n in range(start, end):
        if sieve[n % mod_depth]: continue
        
        current = np.uint64(n)
        steps = 0
        local_peak = current
        
        while current >= np.uint64(n):
            if current % np.uint64(2) == 0:
                current >>= np.uint64(1)
            else:
                current = np.uint64(3) * current + np.uint64(1)
                if current > local_peak: local_peak = current
            steps += 1
            
        if local_peak > max_peak: max_peak, peak_seed = local_peak, np.uint64(n)
        if steps > longest_path: longest_path, path_seed = steps, np.uint64(n)
            
    return max_peak, peak_seed, longest_path, path_seed

def Run_Sweep(sieve_array, MODULO_DEPTH, SEARCH_BOUND=2**45):
    CHUNK_SIZE = 1_000_000_000 
    num_cores = multiprocessing.cpu_count()
    print(f"\nInitiating parallel verification up to {SEARCH_BOUND:,}...")
    print(f"Distributing 1-Billion integer chunks across {num_cores} physical CPU cores...\n")
    
    start_time = time.time()
    global_max_peak, global_peak_seed, global_path_seed = np.uint64(0), np.uint64(0), np.uint64(0)
    global_longest_path = 0
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for chunk_start in range(3, SEARCH_BOUND, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, SEARCH_BOUND)
            futures.append(executor.submit(verification_chunk, chunk_start, chunk_end, sieve_array, MODULO_DEPTH))
        
        for idx, future in enumerate(as_completed(futures), 1):
            m_peak, p_seed, l_path, pa_seed = future.result()
            if m_peak > global_max_peak: global_max_peak, global_peak_seed = m_peak, p_seed
            if l_path > global_longest_path: global_longest_path, global_path_seed = l_path, pa_seed
            if idx % 500 == 0: print(f"Progress: {idx:,} / {len(futures):,} chunks analyzed...")
    
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*55 + "\nVERIFICATION COMPLETE\n" + "="*55)
    print(f"Search Space:        Up to {SEARCH_BOUND:,}")
    print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("-" * 55)
    print(f"MAX PEAK ALTITUDE:   {global_max_peak:,} (Seed: {global_peak_seed:,})")
    print(f"LONGEST PATH:        {global_longest_path} steps (Seed: {global_path_seed:,})")
    print("=" * 55)

if __name__ == "__main__":
    # Execute the entire High-Performance pipeline sequentially
    csv_file = Generate_Resolved_Pairs(max_k=24)
    sieve, depth = Build_Memory_Sieve(csv_file)
    Run_Sweep(sieve, depth, SEARCH_BOUND=2**45)
