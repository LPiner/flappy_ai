from flappy_ai.config import dqn_config
from flappy_ai.types.network_types import NetworkTypes


def network_factory(network_type: NetworkTypes):
    # Networks are imported inside to prevent the loading of tensorflow until it is needed.
    from flappy_ai.models.networks.dqn_network import DQNNetwork

    if network_type is NetworkTypes.DQN:
        return DQNNetwork(config=dqn_config)
    else:
        raise NotImplementedError(f"Network type of {DQNNetwork} is not implemented.")
