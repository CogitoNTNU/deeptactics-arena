import copy
import optuna
import wandb

from configs import connect_four
from main import training_loop

def build_trial_config(base_config, trial):
    config = copy.deepcopy(base_config)

    config.train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    #config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    config.mcts.cpuct = trial.suggest_float("cpuct", 0.5, 10.0)
    config.mcts.pi_temp = trial.suggest_float("pi_temp", 0.5, 2.0)
    config.mcts.epsilon = trial.suggest_float("epsilon", 0.05, 0.4)

    # Make trials cheaper than full training
    config.train.num_episodes = 30
    config.train.num_epochs = 3
    config.train.num_batches = 5

    return config

def objective(trial):
    base_config = load_config("tic-tac-toe.yaml")
    config = build_trial_config(base_config, trial)

    run = wandb.init(
        project="AlphaZero deeptactics optuna",
        config=config.model_dump(),
        reinit=True,
        group="optuna",
        name=f"trial-{trial.number}",
    )

    metrics = training_loop(config)
    score = metrics["eval/win_rate_vs_random"]

    trial.set_user_attr("draw_rate", metrics["eval/draw_rate_vs_random"])
    trial.set_user_attr("loss_rate", metrics["eval/loss_rate_vs_random"])

    wandb.log({"optuna/objective": score, "trial_number": trial.number})
    run.finish()

    return score
