## Task overview
Owner: Vo Dai Phuoc \
Report to: Luu Gia Thien

## Description
Speaker diarization and ASR pipeline

## Task Status
|task id|Description|Notes|Status|
|---|---|---|---|
|Phase 1|Do inference speaker diarization given an audio| see implementation in [class ](../SDP/diarization.py:Speaker_Diarization_Offline)| Done |
|Phase 2|Do offline inference speaker diarization given | See results in [json file](../data_results/part1/results.json) |Done|
|Phase 3|Do streaming diarization and ASR | | 50% |

## Issues Status
|issue id|Description|Notes|Status|
|---|---|---|---|
|1| Export pretrained selected model in Nemo toolkit to onnx format (both offline and online pipeline) | diarization scope|Done|
|2| Add inference with onnx for production (both offline and online) | keep the pytorch base inference, diarization scope |0%|
|3| Add example generator to create chunk audio for streaming | diarization scope |0% |
|4| forward method for streaming pipeline | diarization scope|0% |
|5| add streaming ASR mdoel of Nemo | ASR scope|0% |
