# Bug Report: `mm_processor_kwargs` Causes 500 Internal Server Error via OpenAI-Compatible API

**Date:** 2026-01-21  
**vLLM Version:** 0.14.0  
**Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct  

## Summary

When using the OpenAI-compatible API endpoint (`/v1/chat/completions`) with `mm_processor_kwargs: {"use_audio_in_video": true}` and a video-only request (no separate `audio_url`), the server returns a 500 Internal Server Error. This parameter works correctly in offline inference but fails via the HTTP API.

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

### 2. Send request with `mm_processor_kwargs` and video-only

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-omni-ltx",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/videos/sample.mp4"}},
          {"type": "text", "text": "Describe this video including audio."}
        ]
      }],
      "mm_processor_kwargs": {"use_audio_in_video": true},
      "temperature": 0.6,
      "max_tokens": 256
    }'
```

## Expected Behavior

The request should process successfully, extracting audio from the video file and including it in the response. The offline inference example shows this working:

```python
# From vllm/examples/offline_inference/qwen3_omni/only_thinker.py
asset = VideoAsset(name="baby_reading", num_frames=16)
audio = asset.get_audio(sampling_rate=16000)
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

The server returns a 500 Internal Server Error immediately (within ~1.4 seconds):

### API Response

```json
{
    "error": {
        "message": "",
        "type": "Internal Server Error",
        "param": null,
        "code": 500
    }
}
```

### Comparison

| Request Type | Result | Response Time |
|-------------|--------|---------------|
| Video-only **without** `mm_processor_kwargs` | ✅ Success | ~7.8 seconds |
| Video-only **with** `mm_processor_kwargs` | ❌ 500 Error | ~1.4 seconds |

### Successful Request (without mm_processor_kwargs)

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-omni-ltx",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/videos/sample.mp4"}},
          {"type": "text", "text": "Describe this video."}
        ]
      }],
      "temperature": 0.6,
      "max_tokens": 256
    }'
```

**Response:**
```json
{
    "id": "chatcmpl-ba01022502f73635",
    "object": "chat.completion",
    "model": "qwen3-omni-ltx",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The video presents an aerial view of a vast, organized array of identical, light-brown, cube-shaped objects arranged in neat rows and columns across a flat, sandy, and dry landscape..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 2318,
        "completion_tokens": 156,
        "total_tokens": 2474
    }
}
```

## Analysis

The `mm_processor_kwargs` parameter causes the API to fail immediately with a 500 error. The fast response time (~1.4 seconds vs ~7.8 seconds for successful requests) suggests the error occurs during request validation or early processing, before video processing begins.

The error message is empty, making debugging difficult. The server logs show:
```
(APIServer pid=21) INFO: 127.0.0.1:56576 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```

But no detailed error trace is visible in the API server logs, suggesting the error may be caught and suppressed before reaching the logging layer.

## Tested Variations

1. **`mm_processor_kwargs` with video-only:**
   ```json
   {"mm_processor_kwargs": {"use_audio_in_video": true}}
   ```
   Result: 500 Internal Server Error

2. **Video-only without `mm_processor_kwargs`:**
   Result: ✅ Success (visual description only, no audio)

3. **Video + separate audio_url without `mm_processor_kwargs`:**
   Result: ✅ Success (includes audio transcription)

## Workaround

Use separate `video_url` and `audio_url` **without** `mm_processor_kwargs`. Extract audio from video separately:

```bash
# Extract audio from video
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav

# Then use both in the request
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-omni-ltx",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/videos/sample.mp4"}},
          {"type": "audio_url", "audio_url": {"url": "file:///media/audio/sample.wav"}},
          {"type": "text", "text": "Describe this video including audio."}
        ]
      }],
      "temperature": 0.6,
      "max_tokens": 256
    }'
```

## Impact

1. **Feature Unavailable:** The `use_audio_in_video` feature cannot be used via the API
2. **Poor Error Handling:** Empty error message makes debugging difficult
3. **Workaround Required:** Users must manually extract audio and provide separate URLs
4. **Inconsistent Behavior:** Works in offline inference but fails via HTTP API

## Suggested Fix

The OpenAI-compatible API endpoint (`vllm/entrypoints/openai/serving_chat.py`) should:

1. **Properly parse and validate** `mm_processor_kwargs` from the request body
2. **Extract audio from video** when `use_audio_in_video: true` is set
3. **Pass `mm_processor_kwargs`** to the multimodal processor in the same way offline inference does
4. **Return detailed error messages** instead of empty 500 errors for debugging

## Related Files

| File | Relevance |
|------|-----------|
| `vllm/entrypoints/openai/serving_chat.py` | API request handling - should parse `mm_processor_kwargs` |
| `vllm/entrypoints/openai/protocol.py` | Request schema - should include `mm_processor_kwargs` field |
| `vllm/multimodal/` | Multimodal processing - should handle audio extraction from video |

## References

- [Working offline inference example](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen3_omni/only_thinker.py)
- [vLLM OpenAI-compatible API documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

## Related Bug Reports

- `bug_report_mm_processor_kwargs.md` - Server crash with video+audio+mm_processor_kwargs
- `bug_report_mm_processor_kwargs_ignored.md` - Parameter silently ignored with video+audio
