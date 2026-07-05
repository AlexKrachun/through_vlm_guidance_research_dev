import torch
import numpy as np
from tqdm.auto import tqdm
from .ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512


LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(prompt: str,
             uncond_prompt: str,
             input_image=None,
             strength=0.8, 
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=67, 
             device="mps", 
             idle_device=None, 
             tokenizer=None,
             progress_position=0,
             progress_leave=False,
            ):
    
    with torch.no_grad():
        
        if not (0 < strength <= 1):
            raise ValueError(f"strength must be in (0, 1], given {strength}")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        
        clip = models['clip']
        clip.to(device)

        def tokenize(text):
            tokens = tokenizer(
                [text],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].to(device)

        if do_cfg:
            cond_tokens = tokenize(prompt)  # B Seq
            cond_context = clip(cond_tokens)  # B Seq Dim
            
            uncond_tokens = tokenize(uncond_prompt)  # B Seq
            uncond_context = clip(uncond_tokens)  # B Seq Dim
            
            context = torch.cat([cond_context, uncond_context]) # 2 Seq Dim
        
        else:
            cond_tokens = tokenize(prompt)  # B Seq
            context = clip(cond_tokens)  # B Seq Dim
        
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f'Unknown sampler {sampler_name}')
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            encoder = models['encoder']
            encoder.to(device)
            
            input_image_tensor = input_image.resize([WIDTH, HEIGHT])
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)  # B H W C
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)  # B C H W 
            input_image_tensor = input_image_tensor.to(device)
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
            
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
            
        diffusion = models['diffusion']
        diffusion.to(device)
        
            
        timesteps = tqdm(
            sampler.timesteps, 
            desc='steps', 
            position=progress_position, 
            leave=progress_leave
        )
        for i, timestep in enumerate(timesteps):
            timestep = timestep.item()
            # 1 320
            time_embedding = get_time_embedding(timestep).to(device)
            
            # B 4 latent_H latent_W
            model_input = latents
            
            if do_cfg:
                # 2*B 4 latent_H latent_W
                model_input = model_input.repeat(2, 1, 1, 1)
            
            
            model_output = diffusion(model_input, context, time_embedding)  # run unet
            
            if do_cfg:
                output_cond, output_uncod = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncod) + output_uncod
            
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)
        
        decoder = models['decoder']
        decoder.to(device)
        
        images = decoder(latents)
        to_idle(decoder)
        
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # B C H W -> B H W C
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()
        
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
        
    return x
        

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)  # 160
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]  # 1 160
    
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
