# Bug Report: `mm_processor_kwargs` Silently Ignored via OpenAI-Compatible API

**Date:** 2026-01-21  
**vLLM Version:** 0.14.0  
**Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct  

## Summary

The `mm_processor_kwargs` parameter (including `use_audio_in_video`) is **silently ignored** when passed through the OpenAI-compatible API endpoint (`/v1/chat/completions`). This parameter works correctly in offline inference but has no effect via the HTTP API. The audio input is not processed and the response contains only visual descriptions.

## Environment

- **vLLM Version:** 0.14.0
- **Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct
- **Served Model Name:** qwen3-omni-ltx
- **Deployment:** Docker container with 4x GPU (tensor parallel)
- **Python:** 3.12

## Steps to Reproduce

### Request WITH `mm_processor_kwargs`:

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
          {"type": "text", "text": "Describe this video including any spoken content."}
        ]
      }],
      "mm_processor_kwargs": {"use_audio_in_video": true},
      "temperature": 0.6,
      "max_tokens": 512
    }'
```

### Request WITHOUT `mm_processor_kwargs`:

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
          {"type": "text", "text": "Describe this video including any spoken content."}
        ]
      }],
      "temperature": 0.6,
      "max_tokens": 512
    }'
```

## Expected Behavior

Both requests should produce similar results. The `mm_processor_kwargs` parameter should be passed to the multimodal processor, as it works in offline inference:

```python
# From vllm/examples/offline_inference/qwen3_omni/only_thinker.py
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

| Request Type | Audio Transcription | Response Quality |
|-------------|---------------------|------------------|
| **Without** `mm_processor_kwargs` | **Yes** - includes speech | Full multimodal response |
| **With** `mm_processor_kwargs` | **No** - visual only | Audio is ignored |

### Response WITHOUT `mm_processor_kwargs` (Correct):

```json
{
    "id": "chatcmpl-9d6c524b1c4f4af8",
    "object": "chat.completion",
    "model": "qwen3-omni-ltx",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "An aerial view shows a vast field of neatly arranged, cube-shaped stone structures, while a voiceover states, \"也许我们还需要时间\" (yěxǔ wǒmen hái xūyào shíjiān), which translates to \"Perhaps we still need time.\""
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 2437,
        "completion_tokens": 59,
        "total_tokens": 2496
    }
}
```

### Response WITH `mm_processor_kwargs` (Audio Ignored):

```json
{
    "id": "chatcmpl-b49f942baafd320a",
    "object": "chat.completion",
    "model": "qwen3-omni-ltx",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "A large number of white rectangular boxes are arranged in a grid on a vast, flat surface, with a person in a yellow safety vest walking among them."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 2437,
        "completion_tokens": 32,
        "total_tokens": 2469
    }
}
```

## Analysis

The `mm_processor_kwargs` parameter is completely ignored by the OpenAI-compatible API:

1. **No error is returned** - the parameter fails silently
2. **Audio input is not processed** - the response contains only visual descriptions
3. **No indication of failure** - users receive degraded results without knowing why

The API endpoint likely does not parse or forward the `mm_processor_kwargs` field from the request body to the underlying multimodal processor.

## Tested Variations (All Ignored)

1. **`mm_processor_kwargs` at request level:**
   ```json
   {"mm_processor_kwargs": {"use_audio_in_video": true}}
   ```

2. **`use_audio_in_video` inside `video_url` object:**
   ```json
   {"type": "video_url", "video_url": {"url": "...", "use_audio_in_video": true}}
   ```

3. **Via `vllm bench serve --extra-body`:**
   ```bash
   --extra-body '{"mm_processor_kwargs": {"use_audio_in_video": true}}'
   ```

## Workaround

Use separate `video_url` and `audio_url` **without** `mm_processor_kwargs`. The audio will be processed correctly:

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
          {"type": "text", "text": "Describe this video including any spoken content."}
        ]
      }],
      "temperature": 0.6,
      "max_tokens": 512
    }'
```

**Note:** You must extract the audio from the video file separately (e.g., using `ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav`) and provide it as a separate `audio_url`.

## Impact

1. **Silent Failure:** Users passing `mm_processor_kwargs` get degraded results with no indication of failure
2. **Feature Unavailable:** The `use_audio_in_video` feature (documented in offline inference) cannot be used via the API
3. **Workaround Required:** Users must manually extract audio and provide separate URLs

## Suggested Fix

The OpenAI-compatible API endpoint (`vllm/entrypoints/openai/serving_chat.py`) should:

1. **Parse `mm_processor_kwargs`** from the request body
2. **Pass it to the multimodal processor** in the same way offline inference does
3. **Return an error** if the parameter is not supported, rather than silently ignoring it

## Related Files

| File | Relevance |
|------|-----------|
| `vllm/entrypoints/openai/serving_chat.py` | API request handling - should parse `mm_processor_kwargs` |
| `vllm/entrypoints/openai/protocol.py` | Request schema - should include `mm_processor_kwargs` field |

## References

- [Working offline inference example](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen3_omni/only_thinker.py)
- [vLLM OpenAI-compatible API documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
