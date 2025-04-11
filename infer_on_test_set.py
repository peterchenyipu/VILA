import os
import json
import shutil
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat

# === CONFIG ===
TMPDIR = os.environ.get("TMPDIR", "/tmp")

INPUT_JSON = "./vlm_emotion_dataset_test_descriptive.json"
FINAL_OUTPUT_JSON = "vila_results.json"
PARTIAL_OUTPUT_JSON = "vila_results_partial.json"

MODEL_PATHS = {
    "nvila-15b-sft": "./runs/train/nvila-15b-sft",
    "nvila-15b-sft2ep": "./runs/train/nvila-15b-sft2ep",
    "nvila-15b-sft3ep": "./runs/train/nvila-15b-sft3ep",
    "NVILA-Lite-15B-Video": "Efficient-Large-Model/NVILA-Lite-15B-Video",
}

response_format = None
conv_mode = 'auto'

VIDEO_SRC_DIR = "../EmoStimDS/videos"
VIDEO_TMP_DIR = os.path.join(TMPDIR, "videos")

HF_CACHE_SRC = os.path.abspath("../.hf_cache")
HF_CACHE_DST = os.path.join(TMPDIR, "hf_cache")

# reset HF_HOME to download model to compute node
os.environ["HF_HOME"] = HF_CACHE_DST

# === COPY finetuned Models to TMPDIR ===
print("Copying local models to TMPDIR...")
model_tmp_paths = {}
for name, path in MODEL_PATHS.items():
    if os.path.isdir(path):
        dst = os.path.join(TMPDIR, name)
        if not os.path.exists(dst):
            print(f"Copying {name} -> {dst}")
            shutil.copytree(path, dst)
        model_tmp_paths[name] = dst
    else:
        model_tmp_paths[name] = path  # HF model path used directly

# === COPY Videos to TMPDIR ===
if not os.path.exists(VIDEO_TMP_DIR):
    print("Copying videos to TMPDIR...")
    shutil.copytree(VIDEO_SRC_DIR, VIDEO_TMP_DIR)

# === LOAD TEST SET ===
with open(INPUT_JSON, "r") as f:
    test_data = json.load(f)

# === LOAD PARTIAL RESULTS (if any) ===
completed_set = set()
partial_results = []

if Path(PARTIAL_OUTPUT_JSON).exists():
    with open(PARTIAL_OUTPUT_JSON, "r") as f:
        partial_results = [json.loads(line) for line in f if line.strip()]
    for entry in partial_results:
        key = (entry["model"], entry["video"], entry["prompt"])
        completed_set.add(key)

print(f"Loaded {len(completed_set)} previously completed tasks.")


# setup model
model_name = "nvila-15b-sft2ep"
model = llava.load(model_tmp_paths[model_name])
clib.default_conversation = clib.conv_templates[conv_mode].copy()



# === FLATTEN TASKS ===
tasks = []

for item in test_data:
    video_file = os.path.basename(item["video"])
    video_path = os.path.join(VIDEO_TMP_DIR, video_file)

    for conv in item["conversations"]:
        if conv["from"] != "human":
            continue

        key = (model_name, video_file, conv["value"])
        if key in completed_set:
            continue
        tasks.append({
            "model_name": model_name,
            "video": video_file,
            "video_path": video_path,
            "prompt": conv["value"]
        })

print(f"Total new tasks to evaluate: {len(tasks)}")
# print(tasks[0])
# exit()
new_results = []
for idx, task in tqdm(enumerate(tasks)):
    prompt = []
    video_path = task["video_path"]
    prompt.append(Video(video_path))
    prompt.append(task["prompt"])
    response = model.generate_content(prompt, response_format=response_format)
    prediction = response
    result_entry = {
        "model": task["model_name"],
        "video": task["video"],
        "prompt": task["prompt"],
        "prediction": prediction,
    }
    with open(PARTIAL_OUTPUT_JSON, "a") as f:
        f.write(json.dumps(result_entry) + "\n")
    new_results.append(result_entry)

# === COMBINE OLD AND NEW RESULTS ===
all_results = partial_results + new_results

# === GROUP RESULTS ===
output_by_model = defaultdict(lambda: defaultdict(list))
for res in all_results:
    output_by_model[res["model"]][res["video"]].append({
        "prompt": res["prompt"],
        "prediction": res["prediction"]
    })

# === SAVE FINAL OUTPUT ===
with open(FINAL_OUTPUT_JSON, "w") as f:
    json.dump(output_by_model, f, indent=2)

print(f"\nAll results saved to {FINAL_OUTPUT_JSON}")
