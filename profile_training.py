import cProfile
from main import training_loop
from src.configuration import Configuration, load_config
import wandb
from torch.profiler

if __name__ == "__main__":
    print("Profiling with cProfile...")
    wandb.init(mode="disabled")
    config = load_config("tic-tac-toe.yaml")
    cProfile.run("training_loop(config)", "profile_output.prof")
