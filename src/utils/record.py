import random
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
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool).to(device)
        with torch.no_grad():
            policy, _ = model(obs_tensor, action_mask=action_mask)
        action = torch.argmax(policy).item()
        env.step(action)

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    if frames:
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        print(f"Logging episode {episode} video with {len(frames)} frames.")
        wandb.log(
            {
                "episode/game_video": wandb.Video(video, fps=fps, format="mp4"),
                "episode": episode,
            }
        )


def evaluate_vs_random(
    model: AlphaZeroNet,
    env_name: str,
    device: torch.device | str,
    num_games: int = 20,
) -> dict:
    """Play num_games against a random agent (alternating who goes first) and return win/draw/loss rates."""
    model.eval()
    wins, draws, losses = 0, 0, 0

    for game_idx in range(num_games):
        env = build_environment(env_name)
        env.reset()
        az_agent = env.possible_agents[game_idx % 2]  # Alternate first/second player

        while not env.is_done():
            current_agent = env.agent_selection
            obs = env.observe(current_agent)

            if current_agent == az_agent:
                obs_tensor = torch.tensor(
                    obs["observation"], dtype=torch.float32
                ).to(device)
                action_mask = torch.tensor(
                    obs["action_mask"], dtype=torch.bool
                ).to(device)
                with torch.no_grad():
                    policy, _ = model(obs_tensor, action_mask=action_mask)
                action = torch.argmax(policy).item()
            else:
                legal = [i for i, m in enumerate(obs["action_mask"]) if m]
                action = random.choice(legal)

            env.step(action)

        az_reward = env._cumulative_rewards[az_agent]
        if az_reward > 0:
            wins += 1
        elif az_reward == 0:
            draws += 1
        else:
            losses += 1

    model.train()
    return {
        "eval/win_rate_vs_random": wins / num_games,
        "eval/draw_rate_vs_random": draws / num_games,
        "eval/loss_rate_vs_random": losses / num_games,
    }
