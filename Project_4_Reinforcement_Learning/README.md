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

### Run PPO training with a hyperparameter sweep

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode train --total-timesteps 40000 --n-envs 4 --eval-episodes 12
```

Optionally provide a JSON grid file with hyperparameter values:

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode train --grid-json hyperparameter_grid.json
```

### Long-horizon episodes (optional)

By default, episodes are 24 hours long. If you want to encourage long-horizon degradation behavior
within a single episode, you can extend the episode length to a multi-day horizon:

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode train --long-horizon-days 365
```

You can also manually specify the episode length in hours:

```bash
python -m Project_4_Reinforcement_Learning.src.main --mode train --episode-length 168
```

## Environment Summary

The environment models:

- Hourly time steps (t) with time-of-day `τ(t) = t mod 24`.
- Solar irradiation and stochastic cloud cover.
- A stochastic market price with demand-driven structure and solar-dependent discounts.
- A battery with bounded capacity and discrete energy units.
- Demand that must be fully covered each time step.

For more details, see `src/env/energy_budget_env.py`.

## Algorithm overview (supported options)

This project currently supports two Stable Baselines3 algorithms:

- **PPO (Proximal Policy Optimization)**: A robust, clipped-policy gradient method that performs
  well in noisy environments with continuous stochasticity. PPO balances stable updates with good
  sample efficiency and is resilient to reward variance, making it the default choice for this
  energy budgeting task.
- **A2C (Advantage Actor-Critic)**: A simpler on-policy method with faster iterations and lower
  computational overhead. A2C is helpful for quick baselines or when you want a lighter-weight
  training loop at the cost of potentially noisier updates.

**Why PPO is the default here:** the environment has stochastic solar/demand/price dynamics and a
MultiDiscrete action space. PPO’s clipped objective tends to learn more reliably under that noise,
while still supporting the discrete action space without extra wrappers. A2C remains useful for
fast experiments and comparisons, but PPO typically yields steadier performance for longer-horizon
training runs.

## Visual diagnostics

Training runs now produce additional plots:

- **Evaluation reward distribution** (agent vs baseline).
- **Strategy performance comparison** (avg reward, battery energy, health).
- **Energy allocation share comparison** (solar/battery/grid usage splits).