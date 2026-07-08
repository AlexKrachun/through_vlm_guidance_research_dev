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
    pipeline.params.n_inference_steps=10

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=single_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[3, 6, 9]'

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
    'pipeline.guidance.steps_to_guide=[20]'\
    pipeline.optimizer.lr=1e-1\
```









<!-- 
todo
- поэксперементировать с гайденсом и разным cfg коэффицентом
- сделать llm as a judge сделать
- прогнать экспы на улучшение alignment - girdsearch:
    - пусть будет i in range(0, 51, 10), steps_to_guide=[i, i, i, i, i]
    - lr: 1e-3, 1e-2, 1e-1
- построить графики
    - yes, no расределений и соотнести с картинками (возможно надо сделать vlm промпт строже)
    - норм градиентов
    - норм латентов





how guided pipeline works
- vlm guidance via vlm loss
- ddpm noise caching для того, чтобы сравнивать корректировку траектории, а не "разные сиды"


потребление vram:
- sd1.5 ~11gb
- guided_sd1.5 ~23gb
- judge ~17gb
-->




