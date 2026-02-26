# Computational Bounds and Rational Isomorphisms in the 2-Adic Collatz Flow

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)

**Author:** Frank Cortese  
**Paper:** [Insert Link to arXiv or Published Journal Here]

## Overview
This repository contains the computational architecture and verification suite for the paper *"Computational Bounds and Rational Isomorphisms in the 2-Adic Collatz Flow."* Traditional approaches to the Collatz conjecture rely on stochastic "hailstone" models. This project introduces the lifted operator $A(n) = 3n + 2^{v_2(n)}$, effectively re-indexing the sequence to track the strictly accumulating 2-adic valuation. By expanding the domain to the dyadic ring and the rational field, we establish a dynamical isomorphism between non-integer rational orbits and the integer cycles of generalized $3n+d$ variants.

This codebase empirically verifies these algebraic structures up to $2^{45}$ (approximately 35.1 trillion) using a highly optimized, hardware-bound sieving architecture.

## Repository Structure
The codebase is separated into core mathematical logic, visualization, and the execution environment:

```text
├── [Your_Notebook_Name].ipynb    # The main 7-cell execution environment (optimized for Google Colab Pro)
├── [Heavy_Computation].py        # Theoretical JIT-compiled logic and L3 cache sieving
├── [Tables_and_Charts].py        # Visualization, data-formatting, and matplotlib generation
├── requirements.txt              # Standardized dependencies
└── README.md                     # Project documentation
