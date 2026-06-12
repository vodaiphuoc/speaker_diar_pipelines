from omegaconf import OmegaConf
from omegaconf import errors as omega_error

from SDP.onnx.diarization.types import SortformerModuleConfig

def load_sortformer_modules_config(
        config_path: str = 'configs/pretrained_config.yaml'
    )->SortformerModuleConfig:

    try:
        base_conf = OmegaConf.load(config_path)
        schema = OmegaConf.structured(SortformerModuleConfig)

        merged = OmegaConf.merge(schema, base_conf.sortformer_modules)
        return OmegaConf.to_object(merged)
    
    except omega_error.ConfigKeyError as e:
        print(e)
        raise e
    except FileNotFoundError as e:
        print(e)
        raise e