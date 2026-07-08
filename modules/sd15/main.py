from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from . import model_loader
from . import pipeline
from . import assets
from .. import utils

from transformers import CLIPTokenizer



def run_sd15_pipeline(cfg, ROOT_DIR, device):
    
    SOURCE_DIR = cfg.paths.source_dir
    CLIP_REPO = cfg.pipeline.assets.clip_repo_id
    SD15_REPO = cfg.pipeline.assets.sd_repo_id
    SD15_CKPT = cfg.pipeline.assets.sd_ckpt
    
    OUTPUT_DIR = ROOT_DIR / cfg.paths.output_dir
        
    vocab_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=CLIP_REPO, filename='vocab.json')
    merges_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=CLIP_REPO, filename='merges.txt')
    model_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=SD15_REPO, filename=SD15_CKPT)


    tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path))
    models = model_loader.preload_models_from_standart_weights(str(model_path), device)
    
    
    seed =  cfg.constants.seed
    n_inference_steps = cfg.pipeline.params.n_inference_steps
    do_cfg = cfg.pipeline.params.do_cfg
    cfg_scale = cfg.pipeline.params.cfg_scale
        
    if cfg.generation.mode == 'single_prompt':
    
        prompt = cfg.generation.prompt
        uncond_prompt = cfg.generation.uncond_prompt
        

        output_image = pipeline.generate(
            prompt=prompt, 
            do_cfg=do_cfg, 
            cfg_scale=cfg_scale, 
            n_inference_steps=n_inference_steps, 
            seed=seed,
            models=models, 
            device=device, 
            idle_device='cpu', 
            tokenizer=tokenizer,
            uncond_prompt=uncond_prompt,
            
            progress_position=0,
            progress_leave=True,
        )
        
        filename = utils.normalize_prompt(prompt, postfix='.png')
        output_path = OUTPUT_DIR / cfg.generation.output_folder / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(output_image).save(output_path)
        
        with open(OUTPUT_DIR / cfg.generation.output_folder / 'prompt.txt', 'w', encoding='utf-8') as f:
            print(prompt, file=f)
    
    
    elif cfg.generation.mode == 'multi_prompt':
        with open(cfg.generation.prompts_file, encoding='utf-8') as f:
            prompts = [prompt.replace('\n', '') for prompt in f.readlines()]
            
        uncond_prompt = cfg.generation.uncond_prompt
        
        loader = tqdm(
            enumerate(prompts),
            total=len(prompts),
            desc='prompts',
            position=0,
            leave=True,
        )
        
        for i, prompt in loader:
            loader.set_description(f'prompts {i + 1}/{len(prompts)}')
            loader.refresh()
            
            output_image = pipeline.generate(
                prompt=prompt, 
                do_cfg=do_cfg, 
                cfg_scale=cfg_scale, 
                n_inference_steps=n_inference_steps, 
                seed=seed,
                models=models, 
                device=device, 
                idle_device='cpu', 
                tokenizer=tokenizer,
                uncond_prompt=uncond_prompt,
                
                progress_position=1,
                progress_leave=False,
            )
            
            foldername = utils.normalize_prompt(prompt, prefix=f'{i+1:03}')
            output_path = OUTPUT_DIR / cfg.generation.output_folder / foldername / 'sd15.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(output_image).save(output_path)
            
            with open( OUTPUT_DIR / cfg.generation.output_folder / foldername / 'prompt.txt', 'w', encoding='utf-8') as f:
                print(prompt, file=f)
                
            

