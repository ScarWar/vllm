# Bug Report: `mm_processor_kwargs` with `use_audio_in_video` Crashes Server via OpenAI API

**Date:** 2026-01-21  
**vLLM Version:** 0.14.0  
**Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct  

## Summary

When using the OpenAI-compatible API endpoint (`/v1/chat/completions`) with `mm_processor_kwargs: {"use_audio_in_video": true}`, the vLLM server crashes with an index out of bounds error. This parameter works correctly in offline inference but fails via the HTTP API.

## Environment

- **vLLM Version:** 0.14.0
- **Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct
- **Served Model Name:** qwen3-omni-ltx
- **Deployment:** Docker container with 4x GPU (tensor parallel)
- **Python:** 3.12

## Steps to Reproduce

### 1. Start vLLM server with Qwen3-Omni model

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --served-model-name qwen3-omni-ltx \
    --tensor-parallel-size 4 \
    --allowed-local-media-path /media
```

### 2. Send request with `mm_processor_kwargs`

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-omni-ltx",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/videos/sample.mp4"}},
          {"type": "audio_url", "audio_url": {"url": "file:///media/audio/sample.wav"}},
          {"type": "text", "text": "Describe this video with audio."}
        ]
      }],
      "mm_processor_kwargs": {"use_audio_in_video": true},
      "temperature": 0.6,
      "max_tokens": 512
    }'
```

## Expected Behavior

The request should process successfully, similar to the offline inference example at:
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen3_omni/only_thinker.py

```python
# This works in offline inference:
return QueryResult(
    inputs={
        "prompt": prompt,
        "multi_modal_data": {
            "video": asset.np_ndarrays,
            "audio": audio,
        },
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    },
    limit_mm_per_prompt={"audio": 1, "video": 1},
)
```

## Actual Behavior

The server returns a 500 Internal Server Error and crashes:

### API Response

```json
{
    "error": {
        "message": "EngineCore encountered an issue. See stack trace (above) for the root cause.",
        "type": "Internal Server Error",
        "param": null,
        "code": 500
    }
}
```

### Server Logs

```
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 491, in step_with_batch_queue
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     exec_model_fut.result()
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 80, in result
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     return super().result()
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     return self.__get_result()
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     raise self._exception
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 84, in wait_for_response
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     response = self.aggregate(get_response())
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 342, in get_response
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938]     raise RuntimeError(
(EngineCore_DP0 pid=42) ERROR 01-21 04:15:38 [v1/engine/core.py:938] RuntimeError: Worker failed with error 'index 1 is out of bounds for dimension 0 with size 1', please check the stack trace above for the root cause

(Worker_TP2 pid=58) INFO 01-21 04:15:38 [v1/executor/multiproc_executor.py:707] Parent process exited, terminating worker
(Worker_TP1 pid=57) INFO 01-21 04:15:38 [v1/executor/multiproc_executor.py:707] Parent process exited, terminating worker
(Worker_TP0 pid=56) INFO 01-21 04:15:38 [v1/executor/multiproc_executor.py:707] Parent process exited, terminating worker
(Worker_TP3 pid=59) INFO 01-21 04:15:38 [v1/executor/multiproc_executor.py:707] Parent process exited, terminating worker

(APIServer pid=15) ERROR 01-21 04:15:38 [v1/engine/async_llm.py:546] AsyncLLM output_handler failed.
(APIServer pid=15) ERROR 01-21 04:15:38 [v1/engine/async_llm.py:546] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

### Root Cause

```
RuntimeError: Worker failed with error 'index 1 is out of bounds for dimension 0 with size 1'
```

### Exact Error Location

**File:** `vllm/model_executor/models/qwen3_omni_moe_thinker.py`  
**Line:** 1604  
**Function:** `get_mrope_input_positions`  
**Code:** `audio_feature_lengths[audio_idx]`

**Full Stack Trace:**
```
gpu_model_runner.py:3138  execute_model()
    └── gpu_model_runner.py:935   _update_states()
        └── gpu_model_runner.py:1119  _init_mrope_positions()
            └── qwen3_omni_moe_thinker.py:1604  get_mrope_input_positions()
                └── audio_feature_lengths[audio_idx]  ← IndexError
```

**Analysis:** The `audio_idx` variable is 1, but `audio_feature_lengths` tensor only has size 1 (valid index is 0). This suggests that when `use_audio_in_video=true` is passed via the API, the audio feature lengths array is not being populated correctly with both video-embedded audio and explicit audio inputs.

## Workaround

Using separate `video_url` and `audio_url` **without** `mm_processor_kwargs` works correctly:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-omni-ltx",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/videos/sample.mp4"}},
          {"type": "audio_url", "audio_url": {"url": "file:///media/audio/sample.wav"}},
          {"type": "text", "text": "Describe this video with audio."}
        ]
      }],
      "temperature": 0.6,
      "max_tokens": 512
    }'
```

### Successful Response (without mm_processor_kwargs)

```json
{
    "id": "chatcmpl-973669d91f20a3c7",
    "object": "chat.completion",
    "model": "qwen3-omni-ltx",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "An aerial camera slowly moves over a vast field of uniformly arranged, cube-shaped structures as a narrator states, \"也许我们还需要时间\" (yěxǔ wǒmen hái xūyào shíjiān)."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 2437,
        "total_tokens": 2485,
        "completion_tokens": 48
    }
}
```

## Analysis

The `mm_processor_kwargs` parameter appears to be supported for offline inference (as shown in the official vLLM examples) but is not properly handled when passed through the OpenAI-compatible HTTP API. The index out of bounds error suggests a mismatch in how the multimodal inputs are being indexed when this flag is set via the API.

## Tested Variations (All Failed)

1. **mm_processor_kwargs at request level:**
   ```json
   {"mm_processor_kwargs": {"use_audio_in_video": true}}
   ```

2. **use_audio_in_video inside video_url object:**
   ```json
   {"type": "video_url", "video_url": {"url": "...", "use_audio_in_video": true}}
   ```

3. **Via vllm bench serve --extra-body:**
   ```bash
   --extra-body '{"mm_processor_kwargs": {"use_audio_in_video": true}}'
   ```

## Impact

- Server crashes and requires restart
- All in-flight requests are lost
- Cannot use `use_audio_in_video` feature via the API

## Suggested Fix

The OpenAI API endpoint should properly handle `mm_processor_kwargs` and pass it to the multimodal processor in the same way offline inference does.

## Related Files

- **`vllm/model_executor/models/qwen3_omni_moe_thinker.py`** (line 1604) - **ROOT CAUSE**
- `vllm/v1/worker/gpu_model_runner.py` (lines 935, 1119, 3138)
- `vllm/v1/engine/core.py` (line 491, 938)
- `vllm/v1/executor/multiproc_executor.py` (lines 80, 84, 342, 817, 822)
- `vllm/entrypoints/openai/serving_chat.py` (API request handling)

## References

- Working offline inference example: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen3_omni/only_thinker.py
