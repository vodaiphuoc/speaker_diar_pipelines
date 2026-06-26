## Release code: 
https://github.com/vodaiphuoc/speaker_diar_pipelines/releases/tag/v1.0.0 

## Description
Speaker diarization and ASR pipeline

## Task Status
|task id|Description|Notes|Status|
|---|---|---|---|
|Phase 1|Do inference speaker diarization given an audio| see implementation in [class ](../SDP/diarization.py#Speaker_Diarization_Offline)| Done |
|Phase 2|Do offline inference speaker diarization given | See results in [json file](../data_results/part1/results.json) |Done|
|Phase 3|Do streaming diarization and ASR | | Done |
|Evaluation| Make auto tool to create label for diarization dataset (data/part1) | | 0%|

## Issues Status
|issue id|Description|Notes|Status|
|---|---|---|---|
|1| Export pretrained selected model in Nemo toolkit to onnx format (both offline and online pipeline) | diarization scope|Done|
|2| Add inference with onnx for production (both offline and online) | keep the pytorch base inference, diarization scope |Done|
|3| Add example generator to create chunk audio for streaming | diarization scope |Done |
|4| forward method for streaming pipeline | diarization scope| Done |
|5| add streaming ASR model of Nemo | ASR scope| Done |
|6| add export tool fr nemotron ASR | ASR scope| Done |


## WEEK 4 summarization
- done export preprocessor, model check point of `diar_streaming_sortformer_4spk-v2` model to onnx with custom patch function/correctness shape inputs. See [more](../exports/diar.py)
- done onnx inference from preprocessing, inference, post processing for diarization which mimic behavior from original implementation in Nemo-toolkit package, without have to use/install nemo-toolkit, cuda. See [implementation](../SDP/onnx/)
- Add scripts runing phase 2 and phase 3 (with missing ASR) where [run_phase_three.py](../run_phase_three.py) script show capability of intergrate with microphone/websocket since it able to take audio bytes input from the stream. See [results phase 2](../data_results/part1/results_phase2.json), [results phase 3](../data_results/part1/results_phase3.json)
- Current blocker: try to export Nemotron streaming ASR to onnx, require deepdive into massive parent classes, details of implementaion. See [more](./dev/asr_flow.md)

## WEEK 5 summarization
- Implement onnx version of Nemotron 3.5 streaming and do calibration between native nemo-toolkit model with export onnx model in [PR #1](https://github.com/vodaiphuoc/speaker_diar_pipelines/pull/1)
- Implement merging logics from diarization events and asr events [PR #3](https://github.com/vodaiphuoc/speaker_diar_pipelines/pull/3):
  1) Aligment using darization events timeline
  2) Aligment using asr events timeline
- Full pipeline streaming with diarization and asr
- The [CI](https://github.com/vodaiphuoc/speaker_diar_pipelines/actions/runs/28217800649) run calibration pipelines (includes both diarization and asr) between native pipeline with nemo-toolkit and onnx models. Atifact outputs includes: 
  - pipeline results with logic 1): pipeline-calibration-asr_timeline-logs-28217800649.zip
    - output snippet:
```json
  "onnx_pipeline": {
    "full_text": "Bé nó cũng khèn như khoảng một tháng cứ như này với lại nó cứ mũi cứ mũi rất là đ này, lâu cứ như này cố tình hay là hay là thi thoảng, tức là bình thường nó vẫn thế này hay là gọi là mình nói đấy nó mới. Mình nhắc đến nó thì nó ho như này lúc nào cũng như này lúc nào cũng thế không phải lúc nào mà là em thấy là thỉnh thoảng thôi, ví dụ như là chiều em đón nó ở được thì không thấy mấy so với khi tắm không thấy, nhưng mà có khi ngồi xem hoạt hình ý thì lại thấy nó khè khẹt khẹt khẹc này ở lớp thì không thấy cô nào không thấy ì lại th à lớp, ở lớp cô các bạn nó khạc không ông thấy cô nói gì đúng không? Bởi vì khẹc thế này thì ảnh hưởng đến lớp với các bạn học đúng không nếu có là cô sẽ nói ngay đúng không việc mà con nó khậm khoẹp vừa rồi bác nghĩ nhiều đến tật hơn bệnh thực sự nước bình thường hỏng bình thường xen ké với âm thanh trẻ em vậy lứa nhiều hơn thực sự thì lúc nào nó cũng bị còn đây là đôi khi là nó chú ý hoặc là nó mải cái gì đấy thì nó lại quên đi hoặc đang ngồi học thì nó không bị c nào mình để ý đến nó hoặc là nó nhớ ra nó lại làm ? tật nhiều hơn là bệnh còn nếu mà nó ho thực sự thì ngồi trong lớp ho sao cấm được không ạ Bác cho kiểm tra thôi nhưng mà về bản sẽ là không vấn đâu nhá",
    "segments": [
      {
        "stream_id": "onnx",
        "sequence_id": 0,
        "speaker_id": 0,
        "start": 0.0,
        "end": 1.68,
        "text": "Bé nó cũng khèn như",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 1,
        "speaker_id": 0,
        "start": 3.52,
        "end": 4.88,
        "text": "khoảng một tháng",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 2,
        "speaker_id": 0,
        "start": 8.48,
        "end": 10.72,
        "text": "cứ như này với lại nó cứ mũi",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 3,
        "speaker_id": 0,
        "start": 11.2,
        "end": 12.56,
        "text": "cứ mũi rất là đ",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 4,
        "speaker_id": 1,
        "start": 1.68,
        "end": 2.64,
        "text": "này, lâu",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 5,
        "speaker_id": 1,
        "start": 5.12,
        "end": 5.36,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 6,
        "speaker_id": 1,
        "start": 7.36,
        "end": 7.92,
        "text": "cứ như này",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 7,
        "speaker_id": 1,
        "start": 12.64,
        "end": 12.96,
        "text": ""
      },
      {
        "stream_id": "onnx",
        "sequence_id": 8,
        "speaker_id": 1,
        "start": 13.84,
        "end": 21.44,
        "text": "cố tình hay là hay là thi thoảng, tức là bình thường nó vẫn thế này hay là gọi là mình nói đấy nó mới.",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 9,
        "speaker_id": 1,
        "start": 22.0,
        "end": 23.44,
        "text": "Mình nhắc đến nó thì nó ho như này",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 10,
        "speaker_id": 1,
        "start": 24.08,
        "end": 24.88,
        "text": "lúc nào cũng",
        "token_ids": [2, 66, 344, 73, 2, 113, 212, 46, 995, 13011],
        "token_times": [
          [24.24, 24.32],
          [24.24, 24.32],
          [24.32, 24.4],
          [24.32, 24.4],
          [24.48, 24.56],
          [24.48, 24.56],
          [24.56, 24.64],
          [24.56, 24.64],
          [24.64, 24.72],
          [24.72, 24.8]
        ]
      },
      {
        "stream_id": "onnx",
        "sequence_id": 11,
        "speaker_id": 1,
        "start": 28.56,
        "end": 28.8,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 12,
        "speaker_id": 0,
        "start": 23.12,
        "end": 38.16,
        "text": "như này lúc nào cũng thế không phải lúc nào mà là em thấy là thỉnh thoảng thôi, ví dụ như là chiều em đón nó ở được thì không thấy mấy so với khi tắm không thấy, nhưng mà có khi ngồi xem hoạt hình ý thì lại thấy nó khè khẹt khẹt khẹc này",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 13,
        "speaker_id": 0,
        "start": 41.44,
        "end": 42.8,
        "text": "ở lớp thì không thấy cô nào không thấy",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 14,
        "speaker_id": 1,
        "start": 36.72,
        "end": 36.88,
        "text": "ì lại th",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 15,
        "speaker_id": 1,
        "start": 38.4,
        "end": 38.8,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 16,
        "speaker_id": 1,
        "start": 38.88,
        "end": 40.96,
        "text": "à lớp, ở lớp cô các bạn nó khạc không",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 17,
        "speaker_id": 0,
        "start": 57.92,
        "end": 58.64,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 18,
        "speaker_id": 0,
        "start": 58.8,
        "end": 58.88,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 19,
        "speaker_id": 0,
        "start": 58.96,
        "end": 59.28,
        "text": "",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 20,
        "speaker_id": 1,
        "start": 42.56,
        "end": 48.32,
        "text": "ông thấy cô nói gì đúng không? Bởi vì khẹc thế này thì ảnh hưởng đến lớp với các bạn học đúng không nếu có là cô sẽ nói ngay đúng không",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 21,
        "speaker_id": 1,
        "start": 48.4,
        "end": 50.4,
        "text": "việc mà con nó khậm khoẹp vừa rồi",
      },
      {
        "stream_id": "onnx",
        "sequence_id": 22,
        "speaker_id": 1,
        "start": 50.48,
        "end": 51.76,
        "text": "bác nghĩ nhiều đến tật hơn",

```
  - pipeline results with logic 2): pipeline-calibration-diarization_timeline-logs-28217800649.zip
    - output snippet:
```json
{
  "onnx_pipeline": {
    "full_text": "Bé nó cũng khèn như khoảng một tháng cứ như này với lại nó cứ mũi cứ mũi rất là đ này, lâu cứ như này cố tình hay là hay là thi thoảng, tức là bình thường nó vẫn thế này hay là gọi là mình nói đấy nó mới. Mình nhắc đến nó thì nó ho như này lúc nào cũng như này lúc nào cũng thế không phải lúc nào mà là em thấy là thỉnh thoảng thôi, ví dụ như là chiều em đón nó ở được thì không thấy mấy so với khi tắm không thấy, nhưng mà có khi ngồi xem hoạt hình ý thì lại thấy nó khè khẹt khẹt khẹc này ở lớp thì không thấy cô nào không thấy ì lại th à lớp, ở lớp cô các bạn nó khạc không ông thấy cô nói gì đúng không? Bởi vì khẹc thế này thì ảnh hưởng đến lớp với các bạn học đúng không nếu có là cô sẽ nói ngay đúng không việc mà con nó khậm khoẹp vừa rồi bác nghĩ nhiều đến tật hơn bệnh thực sự nước bình thường hỏng bình thường xen ké với âm thanh trẻ em vậy lứa nhiều hơn thực sự thì lúc nào nó cũng bị còn đây là đôi khi là nó chú ý hoặc là nó mải cái gì đấy thì nó lại quên đi hoặc đang ngồi học thì nó không bị c nào mình để ý đến nó hoặc là nó nhớ ra nó lại làm ? tật nhiều hơn là bệnh còn nếu mà nó ho thực sự thì ngồi trong lớp ho sao cấm được không ạ Bác cho kiểm tra thôi nhưng mà về bản sẽ là không vấn đâu nhá",
    "segments": [
      {
        "stream_id": "onnx",
        "sequence_id": 0,
        "speaker_id": 0,
        "start": 0.0,
        "end": 1.68,
        "text": "Bé nó cũng khèn như"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 1,
        "speaker_id": 0,
        "start": 3.52,
        "end": 4.88,
        "text": "khoảng một tháng"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 2,
        "speaker_id": 0,
        "start": 8.48,
        "end": 10.72,
        "text": "cứ như này với lại nó cứ mũi"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 3,
        "speaker_id": 0,
        "start": 11.2,
        "end": 12.56,
        "text": "cứ mũi rất là đ"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 4,
        "speaker_id": 1,
        "start": 1.68,
        "end": 2.64,
        "text": "này, lâu"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 5,
        "speaker_id": 1,
        "start": 5.12,
        "end": 5.36,
        "text": ""
      },
      {
        "stream_id": "onnx",
        "sequence_id": 6,
        "speaker_id": 1,
        "start": 7.36,
        "end": 7.92,
        "text": "cứ như này"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 7,
        "speaker_id": 1,
        "start": 12.64,
        "end": 12.96,
        "text": ""
      },
      {
        "stream_id": "onnx",
        "sequence_id": 8,
        "speaker_id": 1,
        "start": 13.84,
        "end": 21.44,
        "text": "cố tình hay là hay là thi thoảng, tức là bình thường nó vẫn thế này hay là gọi là mình nói đấy nó mới."
      },
      {
        "stream_id": "onnx",
        "sequence_id": 9,
        "speaker_id": 1,
        "start": 22.0,
        "end": 23.44,
        "text": "Mình nhắc đến nó thì nó ho như này"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 10,
        "speaker_id": 1,
        "start": 24.08,
        "end": 24.88,
        "text": "lúc nào cũng"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 11,
        "speaker_id": 1,
        "start": 28.56,
        "end": 28.8,
        "text": ""
      },
      {
        "stream_id": "onnx",
        "sequence_id": 12,
        "speaker_id": 0,
        "start": 23.12,
        "end": 38.16,
        "text": "như này lúc nào cũng thế không phải lúc nào mà là em thấy là thỉnh thoảng thôi, ví dụ như là chiều em đón nó ở được thì không thấy mấy so với khi tắm không thấy, nhưng mà có khi ngồi xem hoạt hình ý thì lại thấy nó khè khẹt khẹt khẹc này"
      },
      {
        "stream_id": "onnx",
        "sequence_id": 13,
        "speaker_id": 0,
        "start": 41.44,
        "end": 42.8,
        "text": "ở lớp thì không thấy cô nào không thấy"
      }
    ]
  }
}
```
- Qualitative evaluation:
  - Onnx version nearly has quality output with native version.
  - The native pipeline (original speaker diarization and streaming asr) still get wrong time start/end of speaker and wrong words transcript due to overlap segments between speakers
  - Merging logic 1) and 2) have nearly equally quality outputs. However, the merging logic cause losing some words compare to full transcription
- Next steps:
  - Manually labeling diarization for audio then re-test native streaming diarization on Vietnamese long-and-high frequency overlap segments
  - Finetuning [multitalker ASR](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) on vietnamse dataset to solve ASR error confuse between 2 speakers at same moment
