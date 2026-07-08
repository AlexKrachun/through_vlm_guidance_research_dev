
HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=50\
    generation.output_folder='sd15'


HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[2]'\
    pipeline.optimizer.lr=1e-4\
    generation.output_folder='guided_sd15-test'
