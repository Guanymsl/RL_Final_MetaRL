import rlcard
import pickle
from rlcard.agents import CFRAgent

env = rlcard.make("limit-holdem", config={"allow_step_back": True})

env.game.allowed_raise_num = 2

agent = CFRAgent(env)
print(env.game.allowed_raise_num)

for i in range(30000):
    agent.train()
    if i + 1 % 5000 == 0:
        with open(f"cfr_model_reg_{i}.pkl", "wb") as f:
            pickle.dump(agent.regrets, f)

        with open(f"cfr_model_pol_{i}.pkl", "wb") as f:
            pickle.dump(agent.policy, f)

        with open(f"cfr_model_avg_pol_{i}.pkl", "wb") as f:
            pickle.dump(agent.average_policy, f)
