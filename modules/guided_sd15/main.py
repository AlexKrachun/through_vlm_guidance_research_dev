from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from . import model_loader
from . import pipeline
from . import assets
from .. import utils

from transformers import CLIPTokenizer
import torch

from .. import utils

def run_guided_sd15_pipeline(cfg, ROOT_DIR, device):
    
    SOURCE_DIR = cfg.paths.source_dir
    CLIP_REPO = cfg.pipeline.assets.clip_repo_id
    SD15_REPO = cfg.pipeline.assets.sd_repo_id
    SD15_CKPT = cfg.pipeline.assets.sd_ckpt
    
    OUTPUT_DIR = utils.generation_output_dir(ROOT_DIR, cfg)
        
    vocab_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=CLIP_REPO, filename='vocab.json')
    merges_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=CLIP_REPO, filename='merges.txt')
    model_path = assets.hf_get(sources_dir=SOURCE_DIR, repo_id=SD15_REPO, filename=SD15_CKPT)


    tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path))
    models = model_loader.preload_models_from_standart_weights(str(model_path), 'cpu')
    
    
    seed =  cfg.constants.seed
    n_inference_steps = cfg.pipeline.params.n_inference_steps
    do_cfg = cfg.pipeline.params.do_cfg
    cfg_scale = cfg.pipeline.params.cfg_scale
    
    guidance_steps = list(cfg.pipeline.guidance.steps_to_guide)
    
    if not all(guidance_steps[i] >= guidance_steps[i-1] for i in range(1, len(guidance_steps))):
        raise ValueError(f'Error: steps_to_guide are to be non-decreasing')

    
        
    if cfg.generation.mode == 'single_prompt':
    
        prompt = cfg.generation.prompt
        uncond_prompt = cfg.generation.uncond_prompt
    

        filename = utils.normalize_prompt(prompt, postfix='.png')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_img_path = OUTPUT_DIR / filename
        (
            logging_save_intermediate_finals_path,
            logging_save_guidance_diffs_path,
            logging_save_latents_path,
            logging_save_nablas_path, 
        ) = utils.create_extra_pipeline_dirs(OUTPUT_DIR, cfg)
        

        

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
            
            guidance_steps=guidance_steps, 
            cfg=cfg,
            logging_save_intermediate_finals_path=logging_save_intermediate_finals_path,
            logging_save_guidance_diffs_path=logging_save_guidance_diffs_path,
            logging_save_latents_path=logging_save_latents_path,
            logging_save_nablas_path=logging_save_nablas_path,
            logging_save_general_path=OUTPUT_DIR,
        )
        
        

        
        
        with open(OUTPUT_DIR / 'prompt.txt', 'w', encoding='utf-8') as f:
            print(prompt, file=f)
        
        Image.fromarray(output_image).save(result_img_path)
        

    
    
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
            
            foldername = utils.normalize_prompt(prompt, prefix=f'{i+1:03}')
            output_path = OUTPUT_DIR / foldername
            output_path.mkdir(parents=True, exist_ok=True)
            result_img_path = output_path  / 'guided_sd15.png'
            
            (
                logging_save_intermediate_finals_path,
                logging_save_guidance_diffs_path,
                logging_save_latents_path,
                logging_save_nablas_path, 
            ) = utils.create_extra_pipeline_dirs(output_path, cfg)
        
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
                
                guidance_steps=guidance_steps, 
                cfg=cfg,
                logging_save_intermediate_finals_path=logging_save_intermediate_finals_path,
                logging_save_guidance_diffs_path=logging_save_guidance_diffs_path,
                logging_save_latents_path=logging_save_latents_path,
                logging_save_nablas_path=logging_save_nablas_path,
                logging_save_general_path=output_path,
            )
        
            with open(output_path / 'prompt.txt', 'w', encoding='utf-8') as f:
                print(prompt, file=f)
                
            Image.fromarray(output_image).save(result_img_path)
                
            

