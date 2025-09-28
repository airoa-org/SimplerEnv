from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config


@PreTrainedConfig.register_subclass("g3pi0")
@dataclass
class G3PI0Config(PI0Config):
    max_ft_dim : str = ""
    train_ft_proj : bool = True
    action_key : str = ""
    ft_key : str = ""
    encoder_type : str = "seq_cnn"
    finetune: bool = False
    finetune_model: str = ""
    multi_embodiment: bool = False
