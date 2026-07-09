import hydra 
from omegaconf import DictConfig

import logging

from pathlib import Path
import torch

from modules.sd15.main import run_sd15_pipeline
from modules.guided_sd15.main import run_guided_sd15_pipeline
from modules.qwen3_judge.main import run_qwen3_judge_pipeline
from modules.utils import choose_device


ROOT_DIR = Path(__file__).resolve().parent


logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.device is None:
        device = choose_device()
    else:
        device = cfg.device
    
    if cfg.pipeline.name == 'sd15':
        run_sd15_pipeline(cfg, ROOT_DIR, device)
        
    elif cfg.pipeline.name == 'guided_sd15':
        run_guided_sd15_pipeline(cfg, ROOT_DIR, device)
        
    elif cfg.pipeline.name == 'qwen3_judge':
        run_qwen3_judge_pipeline(cfg, ROOT_DIR, device)
    
    else:
        raise ValueError(f'Unknown pipeline name: {cfg.pipeline.name = }')
        
        
        
if __name__ == "__main__":
    main()
        








