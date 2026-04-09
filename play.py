"""Play tic-tac-toe against a trained AlphaZero model in the terminal."""

import sys
import torch
from src.configuration import load_config
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet
from src.environments.environment import build_environment


def print_board(env):
    """Print the tic-tac-toe board in a human-readable format."""
    board = env.board.squares
    symbols = {0: ".", 1: "X", 2: "O"}
    print()
    print("  0 1 2")
    for row in range(3):
        print(f"  {symbols[board[row * 3]]} {symbols[board[row * 3 + 1]]} {symbols[board[row * 3 + 2]]}   {row * 3} {row * 3 + 1} {row * 3 + 2}")
    print()


def get_human_move(legal_actions):
    """Prompt the human player for a move."""
    while True:
        try:
            move = int(input(f"Your move (legal: {legal_actions}): "))
            if move in legal_actions:
                return move
            print(f"Illegal move. Choose from {legal_actions}")
        except (ValueError, EOFError):
            print("Enter a number.")


def get_ai_move(model, env, device):
    """Get the AI's move using the trained model."""
    obs = env.observe(env.agent_selection)
    obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).to(device)

    with torch.no_grad():
        policy, value = model(obs_tensor, action_mask=mask_tensor)

    action = torch.argmax(policy).item()
    print(f"AI plays: {action}  (value: {value.item():.3f})")
    print(f"AI policy: {['%.2f' % p for p in policy.tolist()]}")
    return action


def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet(config.network).to(device)

    # Load the best model (lowest loss)
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    while True:
        human_first = input("Do you want to go first? (y/n): ").strip().lower() == "y"

        env = build_environment("tic_tac_toe")
        env.reset()

        agents = env.possible_agents
        human_agent = agents[0] if human_first else agents[1]
        ai_agent = agents[1] if human_first else agents[0]
        print(f"You are {'X' if human_first else 'O'} ({human_agent}), AI is {'O' if human_first else 'X'} ({ai_agent})")

        print_board(env)

        while not env.is_done():
            current = env.agent_selection
            obs = env.observe(current)
            legal = [i for i, m in enumerate(obs["action_mask"]) if m]

            if current == human_agent:
                action = get_human_move(legal)
            else:
                action = get_ai_move(model, env, device)

            env.step(action)
            print_board(env)

        # Show result
        human_reward = env.rewards[human_agent]
        if human_reward > 0:
            print("You win!")
        elif human_reward < 0:
            print("AI wins!")
        else:
            print("Draw!")

        if input("\nPlay again? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
