from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
from pathlib import Path
import gc
from hydra.utils import instantiate



class Qwen35Loss:
    def __init__(self, model_id: str, cache_dir: str, loss_fn: callable, device, dtype: torch.dtype | str=torch.float32):
        
        self.loss_fn = loss_fn
        
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
            
        self.device = device
        
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype='auto',
        ).to(self.device)


        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.yes_ids = self.single_token_ids_(['Yes', 'yes', 'YES', ' Yes', ' yes'])
        self.no_ids = self.single_token_ids_(['No', 'no', 'NO', ' No', ' no'])
        
        
        

    def __call__(self, image_rgb: torch.Tensor, img_prompt: str, target_yes: bool=True, give_me_distribution: bool=False):
        
        messages = self.build_messages_(image_rgb=image_rgb, img_prompt=img_prompt)
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        
        inputs = inputs.to(self.device)
        
        outputs = self.model(**inputs)
        
        last_logits = outputs.logits[0, -1, :].to(dtype=self.dtype)
        
        
        yes_logits = last_logits[self.yes_ids]
        no_logits = last_logits[self.no_ids]
                
        loss = self.loss_fn(yes_logits=yes_logits, no_logits=no_logits, target_yes=target_yes)
        
        if give_me_distribution:
            with torch.no_grad():
                probs = torch.softmax(last_logits.float(), dim=-1)
                
                yes_distribution = probs[self.yes_ids].detach().cpu().sum().item()
                no_distribution = probs[self.no_ids].detach().cpu().sum().item()
                return loss, {'yes_distribution': yes_distribution, 'no_distribution': no_distribution}

        return loss, {}
        


    def single_token_ids_(self, tokens):
        token_ids = []
        
        for token in tokens:
            ids = self.processor.tokenizer.encode(token, add_special_tokens=False)
            
            if len(ids) == 1:
                token_ids.append(ids[0])

        return torch.tensor(token_ids, device=self.device, dtype=torch.long)
    
    
    
    def build_messages_(self, image_rgb, img_prompt):
        messages = [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'You are a visual binary classifier. Answer with exactly one word: Yes or No.',
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_rgb},
                    {
                        'type': 'text',
                        'text': (
                            f'Question: Does this image show {img_prompt}?\n'
                            'Valid answers: Yes, No.\n'
                            'Answer:'
                        ),
                    },
                ],
            },
        ]
        return messages
    

    
        
    
    




    
