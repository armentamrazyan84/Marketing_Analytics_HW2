# Homework 2: Multi-Armed Bandits (Epsilon-Greedy vs. Thompson Sampling)

## 📊 Empirical Results & Performance Analysis
Simulations were run across **20,000 trials** with $4$ bandit arms ($Bandit\_Reward = [1, 2, 3, 4]$) and a known reward precision of $\tau = 1.0$. Below is the performance summary extracted from the experiment logs:

| Algorithm | Total Cumulative Reward | Total Cumulative Regret | Average Reward / Trial | Average Regret / Trial |
| :--- | :--- | :--- | :--- | :--- |
| **Epsilon-Greedy** | 79,900.74 | 57.00 | 3.9950 | 0.0029 |
| **Thompson Sampling** | **79,979.81** | **26.00** | **3.9990** | **0.0013** |

* **Key Observation:** Thompson Sampling achieved significantly lower cumulative regret ($26.00$ vs $57.00$), demonstrating superior sample efficiency. Because Thompson Sampling uses posterior distribution sampling rather than a forced $1/t$ exploration decay schedule, it converges onto the optimal arm (reward = $4$) much faster and avoids suboptimal arm pulls later in the simulation.

---

## 🛠️ Architectural Critique & Better Implementation Plan

To scale this simulation for production reinforcement learning pipelines or high-throughput optimization tasks, the following architectural enhancements are proposed:

### 1. Vectorized NumPy Simulation (Performance Optimization)
* **Current Limitation:** Sequential Python `for` loops spanning 20,000 iterations incur interpreter overhead.
* **Proposed Enhancement:** Transition trial rollouts to fully vectorized NumPy matrix operations or tensor-based batch sampling (via JAX or PyTorch). This enables parallel environment execution and reduces runtime complexity by over an order of magnitude.

### 2. Object-Oriented Reward Distribution Classes
* **Current Limitation:** Environment rewards are coupled directly to inline `np.random.normal()` calls mapped to global variables.
* **Proposed Enhancement:** Implement a polymorphic `RewardDistribution` base class with concrete subclasses (e.g., `GaussianDistribution`, `BernoulliDistribution`) linked dynamically to individual bandit arms. This decouples environment dynamics from policy logic, facilitating non-stationary or adversarial reward experimentation.

### 3. Asynchronous Experiment Telemetry & Logging
* **Current Limitation:** Performance metrics are stored in transient dictionaries and flushed to static CSV files at termination.
* **Proposed Enhancement:** Integrate experiment tracking platforms (such as MLflow or Weights & Biases) to stream trial metrics in real-time, facilitating hyperparameter optimization and automated regression testing.
