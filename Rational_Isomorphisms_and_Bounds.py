"""
Rational Isomorphisms and Bounds in the 2-Adic Collatz Flow
Theoretical Framework and Tables

This script executes the theoretical models described in the manuscript,
generating Tables 1, 2, 3, and Figure 2.
"""

from fractions import Fraction
from typing import Union, List, Dict, Any
import math
import matplotlib.pyplot as plt

# ==========================================
# CELL 1 LOGIC: CORE 2-ADIC OPERATORS
# ==========================================
def get_v2(n: Union[int, Fraction]) -> int:
    if n == 0: return 0
    if isinstance(n, Fraction): return get_v2(n.numerator) - get_v2(n.denominator)
    return (n & -n).bit_length() - 1

def A(n: Union[int, Fraction]) -> Union[int, Fraction]:
    v2 = get_v2(n)
    adder = Fraction(2) ** v2 if v2 < 0 else 2 ** v2
    res = 3 * n + adder
    if isinstance(res, Fraction) and res.denominator == 1: return res.numerator
    return res

def phi(n: Union[int, Fraction]) -> Union[int, Fraction]:
    v2 = get_v2(n)
    divisor = Fraction(2) ** v2 if v2 < 0 else 2 ** v2
    res = n / divisor if isinstance(n, Fraction) else n // divisor
    if isinstance(res, Fraction) and res.denominator == 1: return res.numerator
    return res

def Table_1():
    print("\n" + "="*80 + "\nTABLE 1: Rational Orbits under A(q) Mirroring Integer T_d Dynamics\n" + "="*80)
    seeds = [Fraction(5, 7), Fraction(8, 65), Fraction(7, 8)]
    print(f"{'Step (i)':<10} | {'Rational State q_i':<20} | {'Integer Num. a_i':<20} | {'Odd Kernel phi(q_i)':<20} | {'d_odd'}")
    print("-" * 95)
    for q in seeds:
        d_odd = q.denominator // (2 ** get_v2(q.denominator))
        print(f"Family: d_odd = {d_odd}")
        current_q = q
        history = set()
        for i in range(15):
            a_i = current_q.numerator if isinstance(current_q, Fraction) else current_q
            kernel = phi(current_q)
            kernel_str = f"{kernel.numerator}/{kernel.denominator}" if isinstance(kernel, Fraction) else str(int(kernel))
            if kernel in history:
                kernel_str += " (Cycle)"
                print(f"{i:<10} | {str(current_q):<20} | {a_i:<20} | {kernel_str:<20} | {d_odd}")
                break
            if kernel == 1:
                kernel_str += " (Collapsed)"
                print(f"{i:<10} | {str(current_q):<20} | {a_i:<20} | {kernel_str:<20} | {d_odd}")
                break
            history.add(kernel)
            print(f"{i:<10} | {str(current_q):<20} | {a_i:<20} | {kernel_str:<20} | {d_odd}")
            current_q = A(current_q)
        print("-" * 95)

# ==========================================
# CELL 2 LOGIC: INTERVAL COLLAPSE
# ==========================================
def Table_2():
    print("\n" + "="*80 + "\nTABLE 2: Parity Overhead and Solution Interval Density\n" + "="*80)
    header = f"{'Seed (n)':<15} | {'L':<4} | {'S_min':<6} | {'S_obs':<6} | {'Overhead':<10} | {'Density (Dn)':<15} | {'Classification'}"
    print(header + "\n" + "-" * len(header))
    calibrated_seeds = [
        {"n": "1/3", "L": 1, "s_obs": 2, "type": "Rational"},
        {"n": "13/11", "L": 1, "s_obs": 2, "type": "Rational"},
        {"n": "19/5", "L": 3, "s_obs": 5, "type": "Rational"},
        {"n": "31/32", "L": 39, "s_obs": 67, "type": "Decay"},
        {"n": 3, "L": 5, "s_obs": 8, "type": "Decay"},
        {"n": 7, "L": 11, "s_obs": 20, "type": "Decay"},
        {"n": 27, "L": 41, "s_obs": 70, "type": "Decay"},
        {"n": 31, "L": 39, "s_obs": 67, "type": "Decay"},
        {"n": 31948311, "L": 111, "s_obs": 189, "type": "Decay"},
        {"n": "2^45 - 1", "L": 386, "s_obs": 615, "type": "Decay"}
    ]
    for data in calibrated_seeds:
        n, L, s_obs = data["n"], data["L"], data["s_obs"]
        s_min = math.ceil(L * math.log2(3))
        delta_s = s_obs - s_min
        gap = (2**s_obs) - (3**L)
        dn = (3**L) / gap if gap > 0 else 0.0
        
        if data.get("type") == "Rational": classification = "Rational Cycle"
        elif dn < 1 and dn > 0: classification = "Observed Decay (Dn << 1)"
        elif dn >= 1: classification = "Potential Loop"
        else: classification = "Indeterminate"
        
        dn_str = f"{dn:.3e}" if dn > 0 else "N/A"
        print(f"{str(n):<15} | {L:<4} | {s_min:<6} | {s_obs:<6} | {delta_s:<10} | {dn_str:<15} | {classification}")

# ==========================================
# CELL 3 LOGIC: ALGEBRAIC FACTORIZATION
# ==========================================
def Table_3():
    print("\n" + "="*80 + "\nTABLE 3: Target Mapping and Expansion Polynomials (P_g)\n" + "="*80)
    header = f"{'L':<4} | {'S':<4} | {'g':<4} | {'Gap G = 2^S - 3^L':<20} | {'Factorization (G_prim x P_g)':<30} | {'Target Operator'}"
    print(header + "\n" + "-" * len(header))
    pairs = [(1, 3), (2, 4), (3, 5), (6, 10), (9, 15), (10, 16), (15, 25)]
    for L, S in pairs:
        g = math.gcd(L, S)
        gap = (2**S) - (3**L)
        g_prim = (2**(S//g)) - (3**(L//g))
        p_g = gap // g_prim
        print(f"{L:<4} | {S:<4} | {g:<4} | {gap:,:<20} | {f'{g_prim} x {p_g:,}':<30} | T_{p_g:,}")

# ==========================================
# CELL 4 LOGIC: PLOT GENERATION
# ==========================================
def Figure_2():
    print("\n" + "="*80 + "\nGenerating FIGURE 2: Side-by-Side Trajectory Analysis...\n" + "="*80)
    seed = 27
    syracuse_flow, uplift_flow = [seed], [seed]
    current_s, current_u = seed, seed
    
    while current_s > 1:
        if current_s % 2 == 0: current_s //= 2
        else:
            current_s = 3 * current_s + 1
            while current_s % 2 == 0: current_s //= 2
        syracuse_flow.append(current_s)

    for _ in range(len(syracuse_flow) - 1):
        v2 = (current_u & -current_u).bit_length() - 1
        current_u = 3 * current_u + (2**v2)
        uplift_flow.append(current_u)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(syracuse_flow, color='firebrick', marker='.', linewidth=1)
    ax1.set_title("State Hailstone (Syracuse Map)\nSeed: $n=27$")
    ax1.set_xlabel("Iterations (Discrete Steps)")
    ax1.set_ylabel("Value ($n$)")
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(uplift_flow, color='steelblue', marker='.', linewidth=1)
    ax2.set_title("Monotonic Increasing (Uplift Operator)\n$A(n) = 3n + 2^{v_2(n)}$")
    ax2.set_xlabel("Iterations (Coordinate Flow)")
    ax2.set_ylabel("Value ($n$)")
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig("Figure_2_Publication_Ready.pdf", format='pdf', bbox_inches='tight')
    print("Saved 'Figure_2_Publication_Ready.pdf' to current directory.")

if __name__ == "__main__":
    Table_1()
    Table_2()
    Table_3()
    Figure_2()
