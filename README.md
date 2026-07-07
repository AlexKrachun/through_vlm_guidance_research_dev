# through_guidance

setup
```shell
git clone https://github.com/AlexKrachun/through_vlm_guidance_research_dev.git
cd through_vlm_guidance_research_dev
conda env create -f environment.yml
conda activate through_guidance
```


```shell
HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=single_prompt\
    pipeline.params.n_inference_steps=10\

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10\

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=single_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[3, 6, 9]'\

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[2]'\
    pipeline.optimizer.lr=1e-4






HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=single_prompt\
    'generation.prompt=A bear does a handstand in the park'\
    pipeline.params.n_inference_steps=50\
    'pipeline.guidance.steps_to_guide=[10, 10, 10, 10, 10, 10, 10, 10]'\
    pipeline.optimizer.lr=1e-1\
```





<!-- 
todo
- сохранять текст промпта рядом с картиной.
- сделать llm as a jundge и прогнать экспы на улучшение alignment
- запустить girdsearch:
    - пусть будет i in 0-50, steps_to_guide=[i, i, i, i]
    - lr: 1e-3, 1e-2, 1e-1





how guided pipeline works
- vlm guidance via vlm loss
- ddpm noise caching для того, чтобы сравнивать корректировку траектории, а не "разные сиды"


-->




