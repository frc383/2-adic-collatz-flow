import os
import time
import numpy as np
import pandas as pd
from numba import cuda, uint64

# ==========================================
# PHASE 1: DEEP SIEVE GENERATOR
# ==========================================

def is_resolved(r, k):
    n_test, A, B = r, 0, 0
    while B < k:
        if n_test & 1:
            n_test = 3 * n_test + 1
            A += 1
        else:
            n_test >>= 1
            B += 1
            if 3**A < 2**B: return True
    return False

def build_deep_sieve(max_k=30, filename="resolved_pairs_deep.csv"):
    print(f"\n[PHASE 1] Initializing Deep Sieve for depth k={max_k}...")
    start_t = time.time()
    
    with open(filename, "w") as f:
        f.write("residue,modulo\n")
        f.write("1,4\n") 
        
    unresolved = [3]
    
    for k in range(2, max_k):
        next_un = []
        mod = 2**(k+1)
        resolved_this_level = []
        
        for r in unresolved:
            b1 = r
            b2 = r + (2**k)
            
            if is_resolved(b1, k+1): resolved_this_level.append(f"{b1},{mod}\n")
            else: next_un.append(b1)
                
            if is_resolved(b2, k+1): resolved_this_level.append(f"{b2},{mod}\n")
            else: next_un.append(b2)
        
        with open(filename, "a") as f:
            f.writelines(resolved_this_level)
            
        unresolved = next_un
        print(f"Depth k={k:02d} | Saved: {len(resolved_this_level):10,} | Unresolved branches carrying over: {len(unresolved):10,}")
        
    dur = time.time() - start_t
    print(f"Deep Sieve Generation Complete! Saved to {filename}")
    print(f"Total Computation Time: {dur:.2f} seconds\n")

# ==========================================
# PHASE 2: VRAM ALLOCATION & CUDA KERNEL
# ==========================================

def Build_Dense_Residue_Array(mod_depth, filename="resolved_pairs_deep.csv"):
    print(f"[PHASE 2] Allocating {mod_depth / 10**9:.2f} GB in RAM for Deep Sieve Construction...")
    sieve = np.zeros(mod_depth, dtype=bool)
    sieve[0::2] = True
    sieve[1::4] = True
    
    print("Applying deep residue elimination patterns from disk...")
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        sieve[row['residue']::row['modulo']] = True
        
    targets = np.where(sieve == False)[0].astype(np.uint64)
    elimination_rate = 100 * (1 - len(targets) / mod_depth)
    
    print(f"Deep Sieve complete! Eliminated {elimination_rate:.4f}% of bounds.")
    print(f"Remaining hard targets per chunk: {len(targets):,}\n")
    return targets

@cuda.jit
def verification_kernel_gpu(base_val, residues, d_max_p, d_peak_s, d_long_c, d_seed_c, d_long_s, d_seed_s, d_overflows, d_ov_idx):
    idx = cuda.grid(1)
    
    if idx >= residues.size:
        return
        
    n_seed = uint64(base_val + residues[idx])
    current = n_seed
    c_steps = 0
    s_steps = 0
    local_peak = n_seed
    
    SAFE_LIMIT = uint64(6148914691236517205) # (2^64 - 1) // 3
    
    while current >= n_seed:
        if current > SAFE_LIMIT:
            pos = cuda.atomic.add(d_ov_idx, 0, 1)
            if pos < d_overflows.size:
                d_overflows[pos] = n_seed
            break 
            
        next_odd = uint64(3) * current + uint64(1)
        if next_odd > local_peak: 
            local_peak = next_odd
            
        s_steps += 1
        current = next_odd >> 1
        c_steps += 2
        
        # Branchless Trailing Zeros Optimization
        tz = cuda.popc(current ^ (current - uint64(1))) - 1
        current >>= tz
        c_steps += tz
        
    if current < n_seed:
        if local_peak > d_max_p[0]:
            old_p = cuda.atomic.max(d_max_p, 0, local_peak)
            if old_p < local_peak: d_peak_s[0] = n_seed
            
        if c_steps > d_long_c[0]:
            old_c = cuda.atomic.max(d_long_c, 0, c_steps)
            if old_c < c_steps: d_seed_c[0] = n_seed
            
        if s_steps > d_long_s[0]:
            old_s = cuda.atomic.max(d_long_s, 0, s_steps)
            if old_s < s_steps: d_seed_s[0] = n_seed

# ==========================================
# PHASE 3: MASTER CONTROLLER & LOGGER
# ==========================================

def py_verify_infinite(n_seed):
    current = int(n_seed)
    c_steps, s_steps = 0, 0
    local_peak = current
    
    while True:
        if current & 1:
            current = 3 * current + 1
            s_steps += 1
            if current > local_peak: local_peak = current
        else:
            current >>= 1
        c_steps += 1
        if current < n_seed: break
            
    return int(local_peak), c_steps, s_steps

if __name__ == "__main__":
    K = 30
    BOUND = 2**50
    MOD = 2**K  
    SIEVE_FILE = "resolved_pairs_deep.csv"
    LOG_FILE = "collatz_cuda_deep_checkpoint.csv"
    
    # --- SMART ORCHESTRATION ---
    if not os.path.exists(SIEVE_FILE):
        build_deep_sieve(max_k=K, filename=SIEVE_FILE)
    else:
        print(f"\n[PHASE 1] Sieve file '{SIEVE_FILE}' found. Bypassing generation.")
        
    targets = Build_Dense_Residue_Array(MOD, SIEVE_FILE)
    d_residues = cuda.to_device(targets)
    
    threads_per_block = 256
    blocks_per_grid = (targets.size + (threads_per_block - 1)) // threads_per_block
    
    completed_chunks = set()
    
    if os.path.exists(LOG_FILE):
        print(f"Parsing '{LOG_FILE}' for resume logic...")
        try:
            df_log = pd.read_csv(LOG_FILE)
            if not df_log.empty:
                completed_chunks = set(df_log['Chunk_Base'].values)
                print(f"Skipping {len(completed_chunks)} completed bounds.")
        except Exception as e: pass
    else:
        with open(LOG_FILE, "w") as f:
            f.write("Chunk_Base,Max_Peak,Peak_Seed,Longest_Collatz,Collatz_Seed,Longest_Syracuse,Syracuse_Seed\n")

    bases_to_process = [b for b in range(0, BOUND, MOD) if b not in completed_chunks]
    total_tasks = len(bases_to_process)
    
    if total_tasks == 0:
        print("\nAll bounds processed!")
        import sys; sys.exit()
        
    print(f"Initiating Deep CUDA Sweep across {blocks_per_grid * threads_per_block:,} threads per grid...")
    start_t = time.time()
    completed = 0
    
    gm, gp = 0, 0
    gl_c, gs_c = 0, 0
    gl_s, gs_s = 0, 0

    for base in bases_to_process:
        d_max_p = cuda.to_device(np.zeros(1, dtype=np.uint64))
        d_peak_s = cuda.to_device(np.zeros(1, dtype=np.uint64))
        d_long_c = cuda.to_device(np.zeros(1, dtype=np.int32))
        d_seed_c = cuda.to_device(np.zeros(1, dtype=np.uint64))
        d_long_s = cuda.to_device(np.zeros(1, dtype=np.int32))
        d_seed_s = cuda.to_device(np.zeros(1, dtype=np.uint64))
        d_overflows = cuda.to_device(np.zeros(100, dtype=np.uint64))
        d_ov_idx = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        verification_kernel_gpu[blocks_per_grid, threads_per_block](
            base, d_residues, 
            d_max_p, d_peak_s, d_long_c, d_seed_c, d_long_s, d_seed_s, 
            d_overflows, d_ov_idx
        )
        
        cuda.synchronize()
        
        l_m_p = int(d_max_p.copy_to_host()[0])
        l_p_s = int(d_peak_s.copy_to_host()[0])
        l_l_c = int(d_long_c.copy_to_host()[0])
        l_s_c = int(d_seed_c.copy_to_host()[0])
        l_l_s = int(d_long_s.copy_to_host()[0])
        l_s_s = int(d_seed_s.copy_to_host()[0])
        
        ov_count = d_ov_idx.copy_to_host()[0]
        if ov_count > 0:
            ov_arr = d_overflows.copy_to_host()[:ov_count]
            for o in ov_arr:
                pm, pc, ps = py_verify_infinite(o)
                if pm > l_m_p: l_m_p, l_p_s = pm, int(o)
                if pc > l_l_c: l_l_c, l_s_c = pc, int(o)
                if ps > l_l_s: l_l_s, l_s_s = ps, int(o)
        
        if l_m_p > gm: gm, gp = l_m_p, l_p_s
        if l_l_c > gl_c: gl_c, gs_c = l_l_c, l_s_c
        if l_l_s > gl_s: gl_s, gs_s = l_l_s, l_s_s
            
        with open(LOG_FILE, "a") as f:
            f.write(f"{base},{l_m_p},{l_p_s},{l_l_c},{l_s_c},{l_l_s},{l_s_s}\n")
            
        completed += 1
        percent = (completed / total_tasks) * 100
        elapsed = time.time() - start_t
        rate = completed / elapsed if elapsed > 0 else 0.001
        rem = (total_tasks - completed) / rate
        h, rem = divmod(int(rem), 3600)
        m, s = divmod(rem, 60)
        
        print(f"\rCUDA Sweep: {percent:.2f}% | ETA: {h}h {m:02d}m {s:02d}s | Peak: {gm:,}", end="", flush=True)

    print("\n\n================ CUDA VERIFIED RESULTS ================")
    print(f"TRUE MAX PEAK:         {gm:,}")
    print(f"PEAK SEED:             {int(gp):,}")
    print(f"LONGEST COLLATZ PATH:  {gl_c} steps (Seed: {int(gs_c):,})")
    print(f"LONGEST SYRACUSE PATH: {gl_s} steps (Seed: {int(gs_s):,})")
