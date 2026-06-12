# Script for runing pipeline on modal.com

import modal
import os
import json

dockerfile_image = modal.Image.from_dockerfile(
    "./Dockerfile",
    context_dir=os.path.dirname(__file__),
    ignore="./.dockerignore"
).add_local_dir(
    local_path="./data/part1/",
    remote_path="/data/part1/",
)

app = modal.App("pipeline-inference-v2")

@app.function(
        image=dockerfile_image,
        gpu = "T4",
        timeout = 60*20
)
def run():
    import subprocess
    import os
    subprocess.run(["pip", "install", "typing_extensions>=4.14.0", "--ignore-installed"], check=True)

    os.environ["LOG_LEVEL"] = "INFO"
    from SDP import Pipeline
    import glob

    SAMPLING_RATE = 16000

    assert os.path.isdir("/data/part1")
    audio_list = glob.glob("/data/part1/*.wav")

    p =  Pipeline()
    
    batch_results = []
    for audio_file in audio_list:
        assert os.path.isfile(audio_file)
        outputs = p.forward(audio_file, sampling_rate=SAMPLING_RATE)
        batch_results.append({
            "audio_file": audio_file,
            "results": outputs
        })

    return batch_results

@app.local_entrypoint()
def main():
    r = run.remote()
    
    with open("data_results/part1/results.json","w", encoding="utf-8") as fp:
        json.dump(r, fp, ensure_ascii=False, indent=4)
