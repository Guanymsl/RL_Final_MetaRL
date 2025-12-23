import rlcard
import pickle
from rlcard.agents import CFRAgent

env = rlcard.make("limit-holdem", config={"allow_step_back": True})

env.game.allowed_raise_num = 2

agent = CFRAgent(env)
print(env.game.allowed_raise_num)
print(env.state_shape)

for i in range(2):
    agent.train()
    print("Iteration", i)


with open("cfr_models/cfr_model_reg.pkl", "wb") as f:
    pickle.dump(agent.regrets, f)

with open("cfr_models/cfr_model_pol.pkl", "wb") as f:
    pickle.dump(agent.policy, f)

with open("cfr_models/cfr_model_avg_pol.pkl", "wb") as f:
    pickle.dump(agent.average_policy, f)
