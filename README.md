# Inventory Optimization Under Demand Uncertainty

## Overview

This project applies a classical **newsvendor framework** and Monte Carlo simulation to determine optimal daily production levels under uncertain demand and perishable inventory. The objective is to quantify the tradeoff between stockouts (lost sales) and excess inventory (waste) and identify the production policy that maximizes expected annual profit.

## Problem Definition

A pizzeria produces fresh dough daily. Because dough is perishable and cannot be carried forward, overproduction results in waste, while underproduction results in lost sales and reduced service levels.

The key decision problem:

_What daily production offset relative to expected demand maximizes expected annual profit while balancing service performance and waste?_

Key characteristics/assumptions of the setting:

- One year of historical transactional demand data
- Demand approximately normally distributed
- No meaningful autocorrelation (independence assumption supported)
- Statistically significant weekday differences → weekday-specific modeling
- Zero salvage value for unsold inventory

## Approach

### 1. Demand Modeling

- Historical daily demand analyzed for distributional fit and independence.
- Separate demand distributions estimated by weekday.
- Mean and variance estimated from historical data.

### 2. Policy Design

Production policies were defined as offsets from expected demand:

- Offset values evaluated: -20, -10, 0, +10, +20 units

Negative offsets prioritize waste reduction.

Positive offsets prioritize service level.

### 3. Simulation Framework

- Monte Carlo simulation of one full year of daily demand
- 2,000 annual replications per policy
- Performance metrics calculated for each offset:
  - Annual profit
  - Stockout rate
  - Fill rate
  - Average leftover inventory per day

All simulation logic is implemented in:

`src/newsvendor_simulation.py`

The full analytical workflow, including parameter configuration, data preparation, and result generation, is available in the `notebooks/ directory.

## Results

The optimal production policy was:

Offset = 0

This policy produced:

- Highest expected annual profit
- Balanced tradeoff between stockout risk and waste
- Service level above 97%

As offset increases:

- Stockout rate decreases monotonically
- Fill rate increases and saturates
- Waste increases substantially
- Profit declines beyond the optimal point

This confirms the economic intuition of the newsvendor model that increasing service beyond the profit-maximizing level introduces diminishing returns.

Detailed results are available in:

`results/simulation_summary.xlsx`

## Key Takeaways

- Weekday-specific modeling materially improves policy evaluation.
- The classical newsvendor solution provides a strong baseline for perishable inventory optimization.
- Simulation enables full distributional understanding of profit and service tradeoffs.
- Marginal service improvements beyond the optimal offset reduce expected profitability.

## Tools

- Python
- NumPy
- pandas
- SciPy

Dependencies are listed in `requirements.txt`

## Potential Extensions

- Multi-product inventory interactions
- Price elasticity and substitution effects
- Non-normal demand distributions
- Explicit risk constraints (e.g., service level targets)

## Reproducibility

The simulation results were generated programmatically using Jupyter notebook. The Excel workbook in `results/` is a decision-ready summary of the Monte Carlo outputs.
