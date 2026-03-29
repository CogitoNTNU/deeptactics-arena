import torch
import numpy as np
import wandb

from src.environments.environment import build_environment
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet


def record_episode(
    model: AlphaZeroNet,
    env_name: str,
    episode: int,
    device: str,
    fps: int = 4,
) -> None:
    env = build_environment(env_name, render_mode="rgb_array")
    env.reset()

    frames = []
    while not env.is_done():
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs = env.observe(env.agent_selection)
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).to(device)
        with torch.no_grad():
            policy, _ = model(obs_tensor)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool).to(device)

        policy[~action_mask] = -float("inf")
        action = torch.argmax(policy).item()
        env.step(action)

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    final_reward = env.rewards.get(env.agents[0], 0) if env.agents else 0
    game_length = len(frames)

    if frames:
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        print(f"Logging episode {episode} video with {len(frames)} frames.")
        wandb.log(
            {
                "episode/game_video": wandb.Video(video, fps=fps, format="mp4"),
                "episode/final_reward": final_reward,
                "episode/game_length": game_length,
                "episode": episode,
            }
        )
