Computational Bounds and Rational Isomorphisms in the 2-Adic Collatz Flow

Author: Frank Cortese

Paper: [Insert Link to arXiv or Published Journal Here]

Overview

This repository contains the computational architecture and verification suite for the paper "Computational Bounds and Rational Isomorphisms in the 2-Adic Collatz Flow." Traditional approaches to the Collatz conjecture rely on stochastic "hailstone" models. This project introduces the lifted operator $A(n) = 3n + 2^{v_2(n)}$, effectively re-indexing the sequence to track the strictly accumulating 2-adic valuation. By expanding the domain to the dyadic ring and the rational field, we establish a dynamical isomorphism between non-integer rational orbits and the integer cycles of generalized $3n+d$ variants.

This codebase empirically verifies these algebraic structures up to $2^{45}$ (approximately 35.1 trillion) using a highly optimized, hardware-bound sieving architecture.

Repository Structure

The codebase is separated into core mathematical logic, visualization, and the execution environment:
'''text
├── Collatz_Verification_Suite.ipynb  # The main 7-cell execution environment
├── computation_engine.py             # Theoretical JIT-compiled logic and L3 cache sieving
├── visualization_tools.py            # Visualization and data-formatting
├── requirements.txt                  # Standardized dependencies
└── README.md                         # Project documentation


Computational Architecture & Hardware Requirements

Standard floating-point architecture (IEEE 754) is fundamentally incompatible with tracking 2-adic flows across thousands of iterations. This project leverages Python's native arbitrary-precision integers to guarantee exact mathematical preservation of the denominator.

Performance bottlenecks are bypassed using a two-pronged approach:

The $\mathcal{O}(1)$ Memory Projection Sieve: We pre-compute a breadth-first expansion of the Collatz tree. Millions of resolved residue pairs are projected into a flat, 16 MB boolean array. Hardware Note: This array is sized strictly to align with modern CPU L3 cache specifications, allowing for a memory-bound look-up latency of ~1 nanosecond and eliminating 98.29% of the search space.

JIT Compilation & 2-Adic Extraction: The remaining indeterminate seeds are subjected to explicit simulation. The core loop uses the LLVM-based numba library with the @njit(nogil=True) decorator to bypass Python's Global Interpreter Lock, achieving 100% CPU utilization.

Quick Start / Execution

This verification suite was designed for execution within a Google Colab Pro environment to handle the memory-intensive caching of massive trajectory histories.

Clone this repository:

git clone [https://github.com/frc383/2-adic-collatz-flow.git](https://github.com/frc383/2-adic-collatz-flow.git)


Open the Notebook: Load Collatz_Verification_Suite.ipynb in Google Colab or your local Jupyter server.

Hardware Check: Ensure your runtime has sufficient RAM for the 16 MB L3 cache array.

Install dependencies:

pip install -r requirements.txt


Execute: Run the notebook cells sequentially to compile the LLVM backend and initialize the parallel sweep.

Empirical Discoveries ($n = 2^{45}$)

Executing this parallelized sweep yielded the following bounding data for the entire domain:

Maximum Peak Altitude: 9,223,371,995,968,455,082 (Originating seed: 11,390,726,016,927)

Longest Path (Stopping Time): 949 steps to drop below initial value (Originating seed: 14,616,588,676,251)

Citation

If you use this code or the theoretical concepts in your own research, please cite the associated paper:

BibTeX:

@article{cortese_collatz_2026,
  title={Computational Bounds and Rational Isomorphisms in the 2-Adic Collatz Flow},
  author={Frank L. Cortese},
  year={2026},
  journal={Pre-print},
  url={[https://github.com/frc383/2-adic-collatz-flow](https://github.com/frc383/2-adic-collatz-flow)},
  note={Adjunct Mathematics Faculty, Shasta Community College}
}


License

This project is licensed under the MIT License - see the LICENSE file for details.
