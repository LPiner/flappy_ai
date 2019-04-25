import attr


@attr.s(auto_attribs=True)
class DQNConfig:
    gamma: float
    start_epsilon: float
    epsilon_min: float
    explore_rate: int
    observe_rate: int
    learning_rate: float
    memory_size: int
    batch_size: int
    save_location: str

