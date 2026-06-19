## Release code: 
https://github.com/vodaiphuoc/speaker_diar_pipelines/releases/tag/v1.0.0 

## Description
Speaker diarization and ASR pipeline

## Task Status
|task id|Description|Notes|Status|
|---|---|---|---|
|Phase 1|Do inference speaker diarization given an audio| see implementation in [class ](../SDP/diarization.py#Speaker_Diarization_Offline)| Done |
|Phase 2|Do offline inference speaker diarization given | See results in [json file](../data_results/part1/results.json) |Done|
|Phase 3|Do streaming diarization and ASR | | 50% |
|Evaluation| Make auto tool to create label for diarization dataset (data/part1) | | 0%|

## Issues Status
|issue id|Description|Notes|Status|
|---|---|---|---|
|1| Export pretrained selected model in Nemo toolkit to onnx format (both offline and online pipeline) | diarization scope|Done|
|2| Add inference with onnx for production (both offline and online) | keep the pytorch base inference, diarization scope |Done|
|3| Add example generator to create chunk audio for streaming | diarization scope |Done |
|4| forward method for streaming pipeline | diarization scope| Done |
|5| add streaming ASR model of Nemo | ASR scope| 0% |
|6| add export tool fr nemotron ASR | ASR scope| 50% |


## WEEK 4 summarization
- done export preprocessor, model check point of `diar_streaming_sortformer_4spk-v2` model to onnx with custom patch function/correctness shape inputs. See [more](../exports/diar.py)
- done onnx inference from preprocessing, inference, post processing for diarization which mimic behavior from original implementation in Nemo-toolkit package, without have to use/install nemo-toolkit, cuda. See [implementation](../SDP/onnx/)
- Add scripts runing phase 2 and phase 3 (with missing ASR) where [run_phase_three.py](../run_phase_three.py) script show capability of intergrate with microphone/websocket since it able to take audio bytes input from the stream. See [results phase 2](../data_results/part1/results_phase2.json), [results phase 3](../data_results/part1/results_phase3.json)
- Current blocker: try to export Nemotron streaming ASR to onnx, require deepdive into massive parent classes, details of implementaion. See [more](./dev/asr_flow.md)
