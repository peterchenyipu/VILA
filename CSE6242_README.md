# NVILA for Emotion Ratings

## Dataset Prep

See [EmoStimDS](https://github.com/peterchenyipu/EmoStimDS).


## Training

1. Set the desired epochs in `scripts/NVILA-Lite/sft.sh`, the option is --num_train_epochs
2. Set the experiment name in `ft.slurm`
3. Launch `ft.slurm`.

## Testing

Prepare a set of prompts and corresponding video paths for querying the VLM. An example file is `vlm_emotion_dataset_test_descriptive.json`.
Launch `test.slurm`. It will launch the `infer_on_test_set.py` script. The output files are `vila_results.json` and `vila_results_partial.json`.

## Inference

Prepare a set of prompts and corresponding video paths for querying the VLM. An example file is `inference.json`.
Launch `infer.slurm`. It will launch the `infer_on_trailers.py` script. The output files are `inference_results.json` and `inference_results_partial.json`.