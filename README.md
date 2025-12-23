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
All shell commands in this session should be run in the `task_generation` directory

### 1. Preprocess/Data Collection
First run and save CFR with
```bash
python3 cfr/cfr.py
```
and
```bash
python3 dataset/data.py
```
### 2. Training
First train the neural network with
```bash
python3 base_nn/train.py
```
and
```bash
python3 finetune/from_nn.py
```
to prepare the base model

Lastly, run
```bash
python3 finetune/from_ppo.py
```

with the following combinations of global variables 
```python
# first combination
AGGRESSIVE=10.0
TIGHT=0.0

#second combination
AGGRESSIVE=0.0
TIGHT=10.0

#third combination
AGGRESSIVE=-10.0
TIGHT=0.0

#fourth combination
AGGRESSIVE=0.0
TIGHT=-10.0
```
### 3. Evaluation
(This step is optional)
To evaluate the trained model, run
```bash
python3 bench.py --model1 <path_to_first_model> --model2 <path_to_second_model>
```
The models can be any `*.zip` or `*.pt` file in the `nn_models/` directory.
## Execution for Phase 2 - AutoEncoder

### 1. Preprocess/Data Collection
After executing all the scripts in phase 1, there should be a file named `cfr_bc_dataset.npz` in the `/task_generation/nn_models/base` directory.

We will use this dataset generated in phase1 to train the AutoEncoder for phase 2.

### 2. Training
Run the following command to return to the root directory of the repository.
```bash
cd ..
```

Execute the following script to train the AutoEncoder. 
```bash
python3 -m preprocess.autoencoder --mode train
```
When training is complete, a t-SNE visualization figure will be displayed to show the training results.

The encoder's parameter will be automatically saved to `/preprocess/models/autoencoder.pt`

### 3. Evaluation
You can execute the following command to evaluate the AutoEncoder.
```bash
python3 -m preprocess.autoencoder --mode evaluate
```
This will display a t-SNE visualization figure if the encoder parameters exists at `/preprocess/models/autoencoder.pt`

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
