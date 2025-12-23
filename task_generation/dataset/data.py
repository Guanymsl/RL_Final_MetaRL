import numpy as np
import pickle

# Load CFR raw policy sums
with open("cfr_models/cfr_model_avg_pol.pkl", "rb") as f:
    avg_pol = pickle.load(f)

obs_list = []
probs_list = []

ACTION_DIM = 4

first_key = next(iter(avg_pol.keys()))
print("len(bytes):", len(first_key))


for obs_key, raw_vec in avg_pol.items():

    obs = np.frombuffer(obs_key, dtype=np.float64).astype(np.float32)

    prob_vec = raw_vec.astype(np.float32)

    total = prob_vec.sum()
    if total > 0:
        prob_vec = prob_vec / total
    else:
        # rare edge case: no probability → uniform among legal actions
        prob_vec = np.ones(ACTION_DIM, dtype=np.float32) / ACTION_DIM

    obs_list.append(obs)
    probs_list.append(prob_vec)

obs_arr = np.array(obs_list, dtype=np.float32)
probs_arr = np.array(probs_list, dtype=np.float32)

print("Dataset size:", obs_arr.shape, probs_arr.shape)
print("Example:", probs_arr[0], "sum:", probs_arr[0].sum())

np.savez_compressed("nn_models/base/cfr_bc_dataset.npz", obs=obs_arr, probs=probs_arr)
print("Saved dataset.")
