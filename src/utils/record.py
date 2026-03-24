import random

import numpy as np
import wandb

from src.environment import build_environment


def record_episode(env_name: str, episode: int, fps: int = 4) -> None:
    env = build_environment(env_name, render_mode="rgb_array")
    env.reset()

    frames = []
    while not env.is_done():
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        env.step(random.choice(env.legal_moves()))

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    if frames:
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        wandb.log({"episode/game_video": wandb.Video(video, fps=fps, format="mp4"), "episode": episode})
