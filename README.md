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
    'pipeline.guidance.steps_to_guide=[3]'

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

- дотестить потенциально интересные области grid search
- сравнить лучший найденный набор гиперпараметров с sd1.5 с лучшим cfg_scale

что я хочу узнать от графиков
- какие параметры оптимальны (возможно какие стоит допроверить) - 3d plot alignment / (lr, cfg, i)
- улучшают ли шаги гайденса финальный результат для лучшего сочетания парамтеров - p_yes / guidance_iteration ; выставить intermidiate finals в ряд
- насколько наш лучший пайплайн улучшает картинки - сравнение прироста alignment (pisitive/negative bars in barplot) и топ лучших и худших пар картинок (со ссылками на файлы картинок)









how guided pipeline works
- vlm guidance via vlm loss
- ddpm noise caching для того, чтобы сравнивать корректировку траектории, а не "разные сиды"


потребление vram:
- sd1.5 ~11gb
- guided_sd1.5 ~23gb
- judge ~17gb

Автор исполнял код на 
- RTX 5090 32gb vram
- RTX 4090 24gb vram

-->




