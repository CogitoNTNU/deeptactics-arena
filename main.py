from src.nn_architecture.network_config import load_config, Configuration

def main():
    config = load_config("config.yaml")
    print(config)


if __name__ == "__main__":
    main()
