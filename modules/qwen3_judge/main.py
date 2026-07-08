
from PIL import Image
from hydra.utils import instantiate
from pathlib import Path
import pandas as pd
import re
from ..utils import judgment_hash

from . import Qwen3Judge


def judge_if_not_yet(judge, image_path: Path, image_prompt: str, df: pd.DataFrame, setup: str, hash_fn: callable, explanation_max_words: int):
    image = Image.open(image_path).convert('RGB')
    
    hashed_repr = hash_fn(image_path, image_prompt, setup)
    
    if df['hash'].eq(hashed_repr).any(): return df
    
    judgement = judge(image, image_prompt, explanation_max_words)
    
    new_row = pd.DataFrame([{
        'setup': setup,
        'alignment': judgement['alignment_score'],
        'quality': judgement['quality_score'],
        'alignment_explanation': judgement['alignment_explanation'],
        'quality_explanation': judgement['quality_explanation'],
        'path': str(image_path),
        'prompt': image_prompt,
        'hash': hashed_repr
    }])
    
    return pd.concat([df, new_row], ignore_index=True)





def run_qwen3_judge_pipeline(cfg, ROOT_DIR: Path, device: str):
    
    qwen3judge = instantiate(cfg.pipeline.model, device=device)

    data_path = ROOT_DIR / cfg.pipeline.params.data_folder
    csv_path = ROOT_DIR / cfg.pipeline.params.judgment_results_file

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['setup', 'alignment', 'quality', 'alignment_explanation', 'quality_explanation', 'prompt', 'path', 'hash'])
    
    if cfg.pipeline.params.judge_type == 'single_prompt':
        image_folder = data_path / cfg.generation.output_folder
        
        image_path = next(image_folder.glob('*.png'), None)
        
        if image_path is None:
            raise ValueError(f'path: {image_folder} has no .png file for get single_prompt judgment done')
        
        with open(image_folder / 'prompt.txt') as f:
            image_prompt = f.read().strip()
            
        df = judge_if_not_yet(
            judge=qwen3judge,
            image_path=image_path,
            image_prompt=image_prompt,
            df=df,
            setup='single_prompt',
            hash_fn=judgment_hash,
            explanation_max_words=cfg.pipeline.params.explanation_max_words
        )
        
    elif cfg.pipeline.params.judge_type == 'multi_prompt':
        
        multi_prompt_folders_prefix = cfg.generation.multi_prompt_folders_prefix
        multi_prompt_folder_pattern = re.compile(rf'^{multi_prompt_folders_prefix}')
        
        for multi_prompt_dir in data_path.iterdir():
            if multi_prompt_dir.is_dir() and multi_prompt_folder_pattern.match(multi_prompt_dir.name):
                
                image_folder_pattern = re.compile(r'^\d{3}')
                
                for subdir in multi_prompt_dir.iterdir():
                    if subdir.is_dir() and image_folder_pattern.match(subdir.name):
                        
                        image_path = next(subdir.glob('*.png'), None)
                        
                        if image_path is None:
                            raise ValueError(f'path: {data_path} has no .png file for get multi_prompt judgment done')
                        
                        with open(subdir / 'prompt.txt') as f:
                            image_prompt = f.read().strip()
                            
                        df = judge_if_not_yet(
                            judge=qwen3judge,
                            image_path=image_path,
                            image_prompt=image_prompt,
                            df=df,
                            setup=multi_prompt_dir.name,
                            hash_fn=judgment_hash,
                            explanation_max_words=cfg.pipeline.params.explanation_max_words
                        )
                                        
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)                 
        





