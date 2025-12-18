import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WinRateCallback(BaseCallback):
    def __init__(self, batch_size=1000, verbose=1):
        super().__init__(verbose)

        self.total_tasks = 0
        self.task_rewards = []

        self.batch_size = batch_size
        self.batch_counts = 0
        self.batch_wins = 0
        self.batch_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if not info.get("hand_done", False):
                continue

            task_reward = info["reward"]

            self.total_tasks += 1
            self.task_rewards.append(task_reward)

            self.batch_counts += 1
            self.batch_rewards.append(task_reward)

            if task_reward > 0:
                self.batch_wins += 1

            if self.batch_counts >= self.batch_size:
                self._log_batch()

        return True

    def _log_batch(self):
        batch_win_rate = self.batch_wins / self.batch_counts
        batch_mean_reward = float(np.mean(self.batch_rewards))

        wandb.log({
            "Win Rate": batch_win_rate,
            "Reward": batch_mean_reward,
        })

        if self.verbose > 0:
            width = 25
            print(
                "\n" + "=" * width + "\n"
                f"| {'WinRateCallback':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {batch_win_rate:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {batch_mean_reward:<10.2f}|\n"
                f"| {'Tasks':<10}| {self.total_tasks:<10d}|\n"
                + "=" * width
            )

        self.batch_counts = 0
        self.batch_wins = 0
        self.batch_rewards = []
