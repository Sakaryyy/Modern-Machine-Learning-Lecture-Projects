# Project 4: Reinforcement Learning (Energy Budgeting)

This project implements the energy budgeting environment described in the project PDF using Gymnasium.

## Project Structure

```
Project_4_Reinforcement_Learning/
├── README.md
└── src/
    ├── config/
    │   └── env_config.py
    ├── diagnostics/
    │   └── interactive_checks.py
    ├── env/
    │   └── energy_budget_env.py
    ├── policies/
    │   └── baseline_policy.py
    ├── utils/
    │   └── logging.py
    ├── main.py
    └── __init__.py
```

## Usage

> **Note**: Ensure dependencies are installed (e.g., `gymnasium`, `numpy`).

### Run the baseline simulation

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode baseline --steps 24
```

### Run interactive diagnostics

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode diagnostics --steps 24 --interactive
```

## Environment Summary

The environment models:

- Hourly time steps (t) with time-of-day `τ(t) = t mod 24`.
- Solar irradiation and stochastic cloud cover.
- A stochastic market price with demand-driven structure and solar-dependent discounts.
- A battery with bounded capacity and discrete energy units.
- Demand that must be fully covered each time step.

For more details, see `src/env/energy_budget_env.py`.