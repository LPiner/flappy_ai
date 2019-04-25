from flappy_ai.models.network_configs.dqn_config import DQNConfig
import configparser

config = configparser.ConfigParser()
config.read("config/config.ini")

dqn_config = DQNConfig(
    gamma=float(config["DQN_CONFIG"]["Gamma"]),
    start_epsilon=float(config["DQN_CONFIG"]["StartingEpsilon"]),
    epsilon_min=float(config["DQN_CONFIG"]["MinEpsilon"]),
    explore_rate=int(config["DQN_CONFIG"]["ExploreRate"]),
    observe_rate=int(config["DQN_CONFIG"]["ObserveRate"]),
    learning_rate=float(config["DQN_CONFIG"]["LearningRate"]),
    memory_size=int(config["DQN_CONFIG"]["MaxMemorySize"]),
    batch_size=int(config["DQN_CONFIG"]["BatchSize"]),
    save_location=str(config["DQN_CONFIG"]["ModelSaveLocation"])
)
