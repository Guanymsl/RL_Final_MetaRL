import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WinRateCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

        self.episode_rewards = []
        self.episode_wins = 0

        self.batch_tasks = []

        self.task_counts = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [])

        if not info[0].get("hand_done", False):
            return True

        episode_reward = info[0]["reward"]
        self.episode_rewards.append(episode_reward)

        if episode_reward > 0:
            self.episode_wins += 1

        if info[0].get("episode", False):
            self.task_counts += 1

            self.batch_tasks.append(np.array(self.episode_rewards, dtype=np.float32))
            self.episode_rewards = []

            if self.task_counts % 10 == 0:
                self._log_batch()

        return True

    def _log_batch(self):
        def win_rate(x):
            return np.mean(x > 0)
        def draw_rate(x):
            return np.mean(x == 0)

        wr_100, wr_75, wr_50, wr_25 = [], [], [], []
        dr_100, dr_75, dr_50, dr_25 = [], [], [], []
        mr_100, mr_75, mr_50, mr_25 = [], [], [], []

        for r in self.batch_tasks:
            wr_100.append(win_rate(r))
            wr_75.append(win_rate(r[-75:]))
            wr_50.append(win_rate(r[-50:]))
            wr_25.append(win_rate(r[-25:]))

            dr_100.append(draw_rate(r))
            dr_75.append(draw_rate(r[-75:]))
            dr_50.append(draw_rate(r[-50:]))
            dr_25.append(draw_rate(r[-25:]))

            mr_100.append(np.mean(r))
            mr_75.append(np.mean(r[-75:]))
            mr_50.append(np.mean(r[-50:]))
            mr_25.append(np.mean(r[-25:]))

        stats = {
            "Win Rate/100%": float(np.mean(wr_100)),
            "Win Rate/75%": float(np.mean(wr_75)),
            "Win Rate/50%": float(np.mean(wr_50)),
            "Win Rate/25%": float(np.mean(wr_25)),
            "Draw Rate/100%": float(np.mean(dr_100)),
            "Draw Rate/75%": float(np.mean(dr_75)),
            "Draw Rate/50%": float(np.mean(dr_50)),
            "Draw Rate/25%": float(np.mean(dr_25)),
            "Mean Reward/100%": float(np.mean(mr_100)),
            "Mean Reward/75%": float(np.mean(mr_75)),
            "Mean Reward/50%": float(np.mean(mr_50)),
            "Mean Reward/25%": float(np.mean(mr_25)),
        }

        wandb.log(stats)

        if self.verbose > 0:
            width = 25
            print(
                "\n" + "=" * width + "\n"
                f"| {'WinRateCallback':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Last 100 Hands':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {stats['Win Rate/100%']:<10.3f}|\n"
                f"| {'Draw Rate':<10}| {stats['Draw Rate/100%']:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {stats['Mean Reward/100%']:<10.3f}|\n"
                + "-" * width + "\n"
                f"| {'Last 75 Hands':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {stats['Win Rate/75%']:<10.3f}|\n"
                f"| {'Draw Rate':<10}| {stats['Draw Rate/75%']:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {stats['Mean Reward/75%']:<10.3f}|\n"
                + "-" * width + "\n"
                f"| {'Last 50 Hands':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {stats['Win Rate/50%']:<10.3f}|\n"
                f"| {'Draw Rate':<10}| {stats['Draw Rate/50%']:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {stats['Mean Reward/50%']:<10.3f}|\n"
                + "-" * width + "\n"
                f"| {'Last 25 Hands':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {stats['Win Rate/25%']:<10.3f}|\n"
                f"| {'Draw Rate':<10}| {stats['Draw Rate/25%']:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {stats['Mean Reward/25%']:<10.3f}|\n"
                + "-" * width + "\n"
                f"| {'Tasks':<10}| {self.task_counts:<10d}|\n"
                + "=" * width
            )

        self.batch_tasks = []
