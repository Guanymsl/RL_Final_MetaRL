import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WinRateCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

        self.episode_rewards = []
        self.episode_wins = 0
        self.episode_counts = 0

        self.task_counts = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [])

        if not info[0].get("hand_done", False):
            return True

        episode_reward = info[0]["reward"]
        self.episode_rewards.append(episode_reward)
        self.episode_counts += 1

        if episode_reward > 0:
            self.episode_wins += 1

        if info[0].get("episode", False):
            self.task_counts += 1
            if self.task_counts % 10 == 0:
                self._log_batch()

        return True

    def _log_batch(self):
        task_win_rate = self.episode_wins / self.episode_counts
        task_mean_reward = float(np.mean(self.episode_rewards))

        wandb.log({
            "Win Rate": task_win_rate,
            "Reward": task_mean_reward,
        })

        if self.verbose > 0:
            width = 25
            print(
                "\n" + "=" * width + "\n"
                f"| {'WinRateCallback':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {task_win_rate:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {task_mean_reward:<10.2f}|\n"
                f"| {'Tasks':<10}| {self.task_counts:<10d}|\n"
                + "=" * width
            )

        self.episode_rewards = []
        self.episode_wins = 0
        self.episode_counts = 0
