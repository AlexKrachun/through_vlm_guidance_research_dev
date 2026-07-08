
from PIL import Image
from hydra.utils import instantiate
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import re
from ..utils import judgment_hash, generation_output_dir

from . import Qwen3Judge


def judge_if_not_yet(judge, image_path: Path, image_prompt: str, df: pd.DataFrame, setup: str, sample: str,  hash_fn: callable, explanation_max_words: int):
    image = Image.open(image_path).convert('RGB')
    
    hashed_repr = hash_fn(image_path, image_prompt, f'{setup}/{sample}')
    
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
        'sample': sample,
        'hash': hashed_repr
    }])
    
    return pd.concat([df, new_row], ignore_index=True)



def collect_multi_prompt_judge_tasks(data_path: Path, multi_prompt_prefix: str='multi_prompt'):
    tasks = []
    
    run_pattern = re.compile(rf'^{re.escape(multi_prompt_prefix)}_')
    sample_pattern = re.compile(r'^\d{3}')
    
    for run_dir in sorted(data_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        if not run_pattern.match(run_dir.name):
            continue
        
        for sample_dir in sorted(run_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            
            if not sample_pattern.match(sample_dir.name):
                continue
            
            prompt_path = sample_dir / 'prompt.txt'
            
            if not prompt_path.exists():
                continue
            
            image_paths = sorted(p for p in sample_dir.glob('*.png') if p.is_file())
            
            if len(image_paths) == 0:
                print(f"Warning, judge never found image in {sample_dir}")
                
            if len(image_paths) == 0:
                print(f"Warning, judge found {len(image_paths)} imags in {sample_dir}. Processing {image_paths[0]}")
            
            tasks.append({
                'setup': run_dir.name, 
                'sample': sample_dir.name, 
                'image_path': image_paths[0],
                'prompt_path': prompt_path,
            })
            
    return tasks


def run_qwen3_judge_pipeline(cfg, ROOT_DIR: Path, device: str):
    
    qwen3judge = instantiate(cfg.pipeline.model, device=device)

    csv_path = ROOT_DIR / cfg.pipeline.params.judgment_results_file

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['setup', 'alignment', 'quality', 'alignment_explanation', 'quality_explanation', 'prompt', 'path', 'sample', 'hash'])
    
    if cfg.pipeline.params.judge_type == 'single_prompt':
        image_folder = generation_output_dir(ROOT_DIR, cfg)
        
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
            sample=image_path.stem,
            hash_fn=judgment_hash,
            explanation_max_words=cfg.pipeline.params.explanation_max_words
        )
        
    elif cfg.pipeline.params.judge_type == 'multi_prompt':
        
        data_path = ROOT_DIR / cfg.paths.output_dir
        
        tasks = collect_multi_prompt_judge_tasks(
            data_path=data_path, 
            multi_prompt_prefix=cfg.generation.folder_prefix,
        )
        
        loader = tqdm(
            tasks, 
            total=len(tasks), 
            desc='judge-images', 
            position=0, 
            leave=True
        )
        
        for task in loader:
            setup = task['setup']
            sample = task['sample']
            image_path = task['image_path']
            prompt_path: Path = task['prompt_path']
            
            loader.set_postfix_str(f'{setup}/{sample}')
            
            image_prompt = prompt_path.read_text(encoding='utf-8').strip()
            
            
            df = judge_if_not_yet(
                judge=qwen3judge,
                image_path=image_path,
                image_prompt=image_prompt,
                df=df,
                setup=setup,
                sample=sample,
                hash_fn=judgment_hash,
                explanation_max_words=cfg.pipeline.params.explanation_max_words
            )
                                        
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)                 
        





