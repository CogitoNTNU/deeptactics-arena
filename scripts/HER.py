import gymnasium as gym

envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")

envs.close()