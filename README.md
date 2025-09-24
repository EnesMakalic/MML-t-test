# Minimum Message Length (MML) t-test in MATLAB

This repository contains a MATLAB implementation of the **Minimum Message Length (MML) t-test** from:

**Enes Makalic, Daniel F. Schmidt, _Minimum Message Length t-test_, AJCAI 2025: Australasian Joint Conference on Artificial Intelligence, 2025.**

## Overview

The code implements a statistical test for discriminating between two groups based on a numerical target/outcome variable. The observed data are two independent samples:

- $$ y_1 \sim \mathcal{N}(\mu_1, \sigma_1^2), \quad y_1 \in \mathbb{R}^{n_1 \times 1} $$
- $$ y_2 \sim \mathcal{N}(\mu_2, \sigma_2^2), \quad y_2 \in \mathbb{R}^{n_2 \times 1} $$


We consider four competing models:

1. **Model 1:** Common mean, common standard deviation  
   $$ \mu_1 = \mu_2, \quad \sigma_1 = \sigma_2 $$
2. **Model 2:** Common mean, different standard deviations  
   $$ \mu_1 = \mu_2, \quad \sigma_1 \neq \sigma_2 $$
3. **Model 3:** Different means, common standard deviation  
   $$ \mu_1 \neq \mu_2, \quad \sigma_1 = \sigma_2 $$
4. **Model 4:** Different means, different standard deviations  
   $$ \mu_1 \neq \mu_2, \quad \sigma_1 \neq \sigma_2 $$


MML requires prior distributions on all parameters:

- **Grand mean**:  
  $$ \mu \sim \text{Uniform (location-invariant)} $$
- **Standard deviations**:  
  $$ \sigma \sim \text{Half-Cauchy}(0, 1) $$
- **Effect size**:  
  $$ \delta \sim \text{Cauchy}(0, 1) $$


The function returns:

- MML estimates of all model parameters for each of the four models  
- Codelengths of each model  

The model with the **shortest codelength** is deemed optimal.  

> **Note:** The Bayesian Information Criterion (BIC) can be seen as an asymptotic approximation to the MML codelength.

---

## Usage

The main function implementing the MML t-test is **`mmlttest.m`**.  To re-create Table 1 in the paper, use **`run_model_selection_experiment.m`**. 

Example run:

```matlab
y1 = normrnd(0, 2, 25, 1);
y2 = normrnd(5, 3, 15, 1);

[codelengths, theta] = mmlttest(y1, y2, verbose=true);
