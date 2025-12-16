from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class WinRateCallback(BaseCallback):
    """
    Callback to track and log win rate during training.
    A win is defined as an episode with reward > 0.
    """
    
    def __init__(self, batch_size=1000, verbose=1):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.episode_rewards = []
        self.total_episodes = 0
        # Track current batch statistics
        self.current_batch_wins = 0
        self.current_batch_episodes = 0
        self.current_batch_rewards = []
        # Track current episode rewards for each environment
        self.current_episode_rewards = {}
        if self.verbose > 0:
            print(f"[Win Rate Callback] Initialized with batch_size={batch_size}")
        
    def _on_step(self) -> bool:
        """
        Called at each step. Manually tracks episode rewards by monitoring
        done flags and accumulating rewards.
        """
        # Get step information
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        
        # First, try SB3's standard episode logging
        for i, info in enumerate(infos):
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                self.episode_rewards.append(episode_reward)
                self.total_episodes += 1
                self.current_batch_episodes += 1
                self.current_batch_rewards.append(episode_reward)
                
                if episode_reward > 0:
                    self.current_batch_wins += 1
                
                # Log win rate every batch_size episodes
                if self.current_batch_episodes >= self.batch_size:
                    batch_win_rate = self.current_batch_wins / self.current_batch_episodes
                    batch_mean_reward = np.mean(self.current_batch_rewards)
                    
                    if self.verbose > 0:
                        print(f"\n[Win Rate Callback] Batch completed: {self.current_batch_episodes} episodes | "
                              f"Batch Win Rate: {batch_win_rate:.3f} | "
                              f"Batch Mean Reward: {batch_mean_reward:.2f} | "
                              f"Total Episodes: {self.total_episodes}")
                    
                    # Reset batch counters
                    self.current_batch_wins = 0
                    self.current_batch_episodes = 0
                    self.current_batch_rewards = []
        
        # Manual tracking: accumulate rewards and check for episode end
        for i in range(len(dones)):
            env_idx = i
            
            # Initialize tracking for this environment if needed
            if env_idx not in self.current_episode_rewards:
                self.current_episode_rewards[env_idx] = 0.0
            
            # Accumulate reward
            if i < len(rewards):
                self.current_episode_rewards[env_idx] += rewards[i]
            
            # Check if episode ended
            if dones[i]:
                episode_reward = self.current_episode_rewards[env_idx]
                
                # Only count if we haven't already counted it from SB3's episode info
                if len(infos) == 0 or (i < len(infos) and "episode" not in infos[i]):
                    self.episode_rewards.append(episode_reward)
                    self.total_episodes += 1
                    self.current_batch_episodes += 1
                    self.current_batch_rewards.append(episode_reward)
                    
                    if episode_reward > 0:
                        self.current_batch_wins += 1
                    
                    # Log win rate every batch_size episodes
                    if self.current_batch_episodes >= self.batch_size:
                        batch_win_rate = self.current_batch_wins / self.current_batch_episodes
                        batch_mean_reward = np.mean(self.current_batch_rewards)
                        
                        if self.verbose > 0:
                            print(f"\n[Win Rate Callback] Batch completed: {self.current_batch_episodes} episodes | "
                                  f"Batch Win Rate: {batch_win_rate:.3f} | "
                                  f"Batch Mean Reward: {batch_mean_reward:.2f} | "
                                  f"Total Episodes: {self.total_episodes}")
                        
                        # Reset batch counters
                        self.current_batch_wins = 0
                        self.current_batch_episodes = 0
                        self.current_batch_rewards = []
                
                # Reset for next episode
                self.current_episode_rewards[env_idx] = 0.0
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout collection. Logs win rate if batch threshold reached."""
        if self.current_batch_episodes >= self.batch_size:
            batch_win_rate = self.current_batch_wins / self.current_batch_episodes
            batch_mean_reward = np.mean(self.current_batch_rewards)
            
            if self.verbose > 0:
                print(f"\n[Win Rate Callback] Batch completed: {self.current_batch_episodes} episodes | "
                      f"Batch Win Rate: {batch_win_rate:.3f} | "
                      f"Batch Mean Reward: {batch_mean_reward:.2f} | "
                      f"Total Episodes: {self.total_episodes}")
            
            # Reset batch counters
            self.current_batch_wins = 0
            self.current_batch_episodes = 0
            self.current_batch_rewards = []
    
    def _on_training_end(self) -> None:
        """Log final statistics at the end of training."""
        if self.total_episodes > 0:
            # Log final batch if incomplete
            if self.current_batch_episodes > 0:
                batch_win_rate = self.current_batch_wins / self.current_batch_episodes
                batch_mean_reward = np.mean(self.current_batch_rewards)
                
                if self.verbose > 0:
                    print(f"\n[Win Rate Callback] Final incomplete batch: {self.current_batch_episodes} episodes | "
                          f"Batch Win Rate: {batch_win_rate:.3f} | "
                          f"Batch Mean Reward: {batch_mean_reward:.2f}")
            
            # Overall statistics
            total_wins = sum(1 for r in self.episode_rewards if r > 0)
            final_win_rate = total_wins / self.total_episodes
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            
            if self.verbose > 0:
                print(f"\n===== Training Summary =====")
                print(f"Total Episodes: {self.total_episodes}")
                print(f"Overall Win Rate: {final_win_rate:.3f}")
                print(f"Mean Reward: {mean_reward:.2f}")
                print(f"Std Reward: {std_reward:.2f}")
                print(f"============================\n")
        else:
            if self.verbose > 0:
                print(f"\n[Win Rate Callback] Warning: No episodes tracked. "
                      f"Total episodes: {self.total_episodes}")

