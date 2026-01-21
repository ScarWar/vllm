# Qwen3-Omni `use_audio_in_video` OpenAI API Fix

**Date:** 2026-01-21  
**Branch:** `fix/use-audio-in-video-qwen3-omni`  
**vLLM Version:** 0.14.0  

## Summary

This fix enables the `use_audio_in_video` feature for Qwen3-Omni via the OpenAI-compatible API (`/v1/chat/completions`). Previously, using this feature caused server crashes.

## The Bug

When users sent requests with both `video_url` and `audio_url` along with `mm_processor_kwargs: {"use_audio_in_video": true}`, the server crashed with:

```
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

**Location:** `vllm/model_executor/models/qwen3_omni_moe_thinker.py:1604`

### Root Cause

The chat parsing layer was generating **two separate placeholders**:
1. Video placeholder (with interleaved audio tokens)
2. Standalone audio placeholder

But the Qwen3 processor only expected **one audio segment** (embedded in video). When `get_mrope_input_positions` tried to access `audio_feature_lengths[1]`, it crashed because only index `[0]` existed.

## The Fix

### Files Modified

| File | Change |
|------|--------|
| `vllm/entrypoints/chat_utils.py` | Added `mm_processor_kwargs` parameter, omit standalone audio placeholder when `use_audio_in_video=True` |
| `vllm/entrypoints/llm.py` | Pass `mm_processor_kwargs` to chat parsing |
| `vllm/entrypoints/openai/serving_engine.py` | Pass `mm_processor_kwargs` from request to chat parsing |

### Logic

```python
# When use_audio_in_video=True with video present AND a separate audio_url,
# audio is embedded within the video placeholder, so we must omit standalone
# audio placeholders. If audio is extracted from video (no separate audio_url),
# we must keep the audio placeholder for validation.
omit_audio_placeholder = (
    mm_processor_kwargs is not None
    and mm_processor_kwargs.get("use_audio_in_video", False)
    and _messages_contain_video(messages)
    and _messages_contain_standalone_audio(messages)
)
```

## Usage Guide

### Correct Usage (OpenAI Client)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# You MUST provide BOTH video_url AND audio_url
video_url = "https://example.com/video.mp4"
audio_url = "https://example.com/audio.wav"  # Extract from video with ffmpeg

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the video and transcribe the speech."},
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ],
    }],
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    max_completion_tokens=256,
    extra_body={
        "mm_processor_kwargs": {"use_audio_in_video": True}
    }
)
```

### Extract Audio from Video

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav
```

## Important Limitations

### vLLM Does NOT Auto-Extract Audio from Video

Unlike the offline inference examples where audio is manually extracted:

```python
# Offline inference (works)
asset = VideoAsset(name="baby_reading", num_frames=16)
audio = asset.get_audio(sampling_rate=16000)  # Manual extraction!
```

The API does **NOT** automatically extract audio from video files. Users must provide audio separately.

### Behavior Matrix

| Scenario | Result |
|----------|--------|
| `use_audio_in_video=True` + `video_url` + `audio_url` | ✅ Works (our fix) |
| `use_audio_in_video=True` + `video_url` only | ❌ CRASH - No audio data |
| `use_audio_in_video=False` + `video_url` + `audio_url` | ✅ Works (processed separately) |
| `video_url` + `audio_url` (no flag) | ✅ Works (processed separately) |

### Why Video-Only with `use_audio_in_video=True` Crashes

The Qwen3 processor validates:

```python
num_videos = len(placeholders["video"])
num_audios = len(mm_prompt_updates.get("audio", []))
if num_audios != num_videos:
    raise ValueError(
        f"use_audio_in_video requires equal number of audio and video items, "
        f"got {num_audios=}, {num_videos=}"
    )
```

Since vLLM's video loader only extracts **frames** (not audio), `num_audios=0` while `num_videos=1`, causing the validation to fail.

## Token Interleaving

When `use_audio_in_video=True`, audio and video tokens are interleaved:

```
151669, <|video_pad|> * 576, <|audio_pad|> * 25, <|video_pad|> * 576, <|audio_pad|> * 25, ...
```

Structure:
- `151669` = `<|audio_start|>`
- Alternating blocks of `<|video_pad|>` and `<|audio_pad|>`
- `151670` = `<|audio_end|>`

All wrapped in `<|vision_start|>` ... `<|vision_end|>`.

## Tests

| Test | Purpose |
|------|---------|
| `test_openai_chat_use_audio_in_video_prevents_qwen3_mrope_audio_idx_oob` | Original crash fix |
| `test_video_only_with_use_audio_in_video_keeps_audio_placeholder` | Video-only doesn't regress |
| `test_use_audio_in_video_generates_interleaved_tokens` | Token interleaving verification |
| `test_openai_client_video_audio_use_audio_in_video_e2e` | OpenAI client pattern E2E |

Run tests:
```bash
pytest -v tests/entrypoints/test_qwen3_omni_use_audio_in_video_openai_mm_kwargs.py
```

## Applying the Patch

For vLLM 0.14.0 Docker:

```bash
# Inside the container
cd /path/to/vllm
git apply qwen3_omni_use_audio_in_video_hom_fix_v0.14.0.patch
```

## Future Work

To fully support `use_audio_in_video=True` with video-only input, vLLM would need:

1. **Auto audio extraction**: Modify `VideoMediaIO` to optionally extract audio from video files
2. **API parameter**: Add flag like `extract_audio_from_video=True`
3. **Dependency**: Ensure `librosa` or similar is available for audio extraction

## Related Files

| File | Description |
|------|-------------|
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | Qwen3-Omni model implementation |
| `vllm/multimodal/video.py` | Video loading (frames only, no audio) |
| `vllm/multimodal/audio.py` | Audio loading |
| `vllm/assets/video.py` | `get_audio()` helper for examples/tests only |
| `examples/offline_inference/qwen3_omni/only_thinker.py` | Offline inference example |
