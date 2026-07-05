import torch
from pathlib import Path
from PIL import Image
import re


def normalize_prompt(prompt, prefix='', postfix=''):
    name = prompt.lower().strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^_a-z0-9]+', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_-')
    name = name[:60]
    name = prefix + '_' + name + postfix
    return name 


def choose_device(allow_cuda: bool=True, allow_mps: bool=True) -> str:
    if torch.cuda.is_available() and allow_cuda:
        device = 'cuda'
        
    elif torch.backends.mps.is_available() and allow_mps:
        device = 'mps'
        
    print(f'{device = }')
    return device



def save_0_255_guidance_diffs(img_tensors: list[torch.tensor], path: Path):
    for i in range(len(img_tensors)):
        img_tensors[i] = img_tensors[i].detach().clone().to(device='cpu')
        img_tensors[i] = img_tensors[i].permute(0, 2, 3, 1)
        img_tensors[i] = img_tensors[i][0]
        
    for i in range(1 ,len(img_tensors)):
        
        diff = (img_tensors[i] - img_tensors[i - 1]).abs()
        diff = diff.clamp(0, 255)
        diff = diff.to(dtype=torch.uint8).numpy()

        result_path = path / f'diff{i}-{i-1}.png'
        Image.fromarray(diff).save(result_path)
        
    
        
        
        

def save_0_255_tensor_as_img(img_tensor: torch.tensor, path: str | Path):
    img_tensor = img_tensor.detach().clone().to(device='cpu')
    img_tensor = img_tensor.permute(0, 2, 3, 1)
    img_tensor = img_tensor.to(dtype=torch.uint8).numpy()
    img_tensor = img_tensor[0]
    Image.fromarray(img_tensor).save(path)
    


def yes_no_loss_(yes_logits, no_logits, target_yes=True):
    # -log(sigma(p_yes - p_no))

    yes_score = torch.logsumexp(yes_logits, dim=0)
    no_score = torch.logsumexp(no_logits, dim=0)
    
    margin = yes_score - no_score
    
    if target_yes:
        loss = torch.nn.functional.softplus(-margin)
    else:
        loss = torch.nn.functional.softplus(margin)
        
    return loss




def create_extra_pipeline_dirs(base_path: Path, cfg):

        logging_save_intermediate_finals_path = base_path / 'intermediate_finals'
        logging_save_guidance_diffs_path = base_path / 'guidance_diffs'
        logging_save_latents_path = base_path / 'latents'
        logging_save_nablas_path = base_path / 'nablas'
        
        if cfg.pipeline.logging.save_intermediate_finals: logging_save_intermediate_finals_path.mkdir(parents=True, exist_ok=True)

        if cfg.pipeline.logging.save_guidance_diffs: logging_save_guidance_diffs_path.mkdir(parents=True, exist_ok=True)

        if cfg.pipeline.logging.save_latents: logging_save_latents_path.mkdir(parents=True, exist_ok=True)

        if cfg.pipeline.logging.save_nablas: logging_save_nablas_path.mkdir(parents=True, exist_ok=True)

        return (
            logging_save_intermediate_finals_path,
            logging_save_guidance_diffs_path,
            logging_save_latents_path,
            logging_save_nablas_path,
        )







