# Team 18 Final Project
## MetaHold'em

This repository demonstrates the implementation code of **MetaHold'em**.

This project studies Meta Reinforcement Learning (Meta-RL) for Heads-Up Limit Texas Hold’em, aiming to train an agent that can adapt rapidly to different opponent behaviors from limited interaction data. We decompose the learning pipeline into three progressive phases, combining game-theoretic baselines, representation learning, and Meta-Learning.

**Phase 1: CFR**

We first train a CFR-based agent as a strong and stable baseline, and then deviate from the baseline models through reward shaping with PPO to obtain four models with different playing styles.

**Phase 2: Autoencoder**

We train an autoencoder to compress high-dimensional poker states into a compact latent representation. This latent space serves as an efficient and task-agnostic observation for downstream learning.

**Phase 3: Meta-RL**

Finally, we apply Meta-RL (RL^2-style) with a recurrent policy to enable fast online adaptation across opponents, using interaction history to adjust strategy within a few episodes.

## Environments

### Build the Python Environment (Mandatory)

Install dependencies using pip3 on Python 3.10.

```bash
pip3 install -r requirements.txt
```

## Execution for Phase 1 - CFR

### 1. Preprocess/Data Collection

### 2. Training

### 3. Evaluation

## Execution for Phase 2 - AutoEncoder

### 1. Preprocess/Data Collection

### 2. Training

### 3. Evaluation

## Execution for Phase 3 - Meta-RL
### 1. Preprocess/Data Collection
* **CFR Opponent Models:** Place `aggressive.zip`, `passive.zip`, `tight.zip`, `loose.zip`, and `baseline.zip` into `agent/cfr/models/`.

* **AutoEncoder Model:** Place `autoencoder.pt` into `preprocess/models/`.

### 2. Training
Execute the following command.
```bash
python3 train.py
```
You can change your model name in `train.py`.

### 3. Evaluation
First add your model's name to the `MODELS` list within `eval.py`.

Execute the following command.

```bash
python3 eval.py --agent {your_model_name} --opponent baseline --fast --plot
```

#### Supported Agent Types
* **Trained Agents:** `discrete`, `continuous`, `mix`, (Your Meta-RL models)
* **CFR-based Models:** `baseline`, `aggressive`, `passive`, `tight`, `loose`
* **Naive Models:** `call`, `fold`, `rand`
* **Parametric Models:** `param`
* **Human Interface:** `human`

#### Valid Evaluation Pairs
| Agent Under Test | Opponent Types | Description |
| :--- | :--- | :--- |
| **Trained Models** | CFR, Naive, Param, Human | Main performance evaluation |
| **CFR / Naive** | CFR, Naive, Param | Benchmarking VPIP and Agg |
| **Human** | CFR, Naive | Qualitative testing |

#### Execution Parameters
* **Standard Evaluation:** Runs for 100,000 episodes by default. For `param`, it evaluates 1,000 agents for 1,000 episodes each.
* **Fast Evaluation** `(--fast)`**:** Reduces the total episode count by 10x (e.g., 10,000 episodes total).
* **Human Mode:** Episode constraints do not apply. Press `Ctrl+C` to terminate the session.
* **Visualization** `(--plot)`**:** Generates plots illustrating Win Rate and Mean Reward relative to episodes.

#### Distribution Analysis
To visualize the distribution (VPIP and Agg) of `param`, execute the following command.
```bash
python3 dist.py
```
