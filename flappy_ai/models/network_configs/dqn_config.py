import attr


@attr.s(auto_attribs=True)
class DQNConfig:
    gamma: float
    start_epsilon: float
    epsilon_min: float
    anneal_epsilon_over_x_frames: int
    observe_frames_before_learning: int
    learning_rate: float
    memory_size: int
    batch_size: int
    save_location: str
