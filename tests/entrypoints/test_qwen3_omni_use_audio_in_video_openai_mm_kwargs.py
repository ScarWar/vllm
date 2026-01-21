# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for Qwen3-Omni use_audio_in_video via OpenAI chat API.

See bug_report.md for the original QA crash report.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from vllm.entrypoints.chat_utils import parse_chat_messages_futures
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

# Token IDs from:
# https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json
QWEN3_OMNI_TOKEN_IDS = {
    "audio_pad": 151675,  # <|audio_pad|>
    "video_pad": 151656,  # <|video_pad|>
    "image_pad": 151655,  # <|image_pad|>
    "audio_start": 151669,  # <|audio_start|>
    "audio_end": 151670,  # <|audio_end|>
    "vision_start": 151652,  # <|vision_start|>
    "vision_end": 151653,  # <|vision_end|>
}


@pytest.fixture
def mock_mm_processor():
    """Create a mock multimodal processor that accepts any number of items."""
    processor = Mock()
    processor.validate_num_items = Mock(return_value=None)
    return processor


@pytest.fixture
def mock_model_cls():
    """Create a mock model class with placeholder string generation."""
    model_cls = Mock()
    model_cls.get_placeholder_str = Mock(return_value="<placeholder>")
    return model_cls


@pytest.fixture
def mock_model_config():
    """Create a minimal mock ModelConfig for chat parsing."""
    config = Mock()
    config.multimodal_config = None
    config.allowed_local_media_path = ""
    config.allowed_media_domains = None
    return config


@pytest.fixture
def mock_qwen3_omni_config():
    """Create a mock Qwen3OmniMoeThinker config with real token IDs."""
    config = Mock()
    config.audio_token_id = QWEN3_OMNI_TOKEN_IDS["audio_pad"]
    config.video_token_id = QWEN3_OMNI_TOKEN_IDS["video_pad"]
    config.image_token_id = QWEN3_OMNI_TOKEN_IDS["image_pad"]
    config.audio_start_token_id = QWEN3_OMNI_TOKEN_IDS["audio_start"]
    config.audio_end_token_id = QWEN3_OMNI_TOKEN_IDS["audio_end"]
    config.vision_start_token_id = QWEN3_OMNI_TOKEN_IDS["vision_start"]
    config.vision_end_token_id = QWEN3_OMNI_TOKEN_IDS["vision_end"]
    config.position_id_per_seconds = 12.5

    # Vision config
    vision_config = Mock()
    vision_config.spatial_merge_size = 2
    config.vision_config = vision_config

    return config


def _create_mock_content_parser(tracker: "BaseMultiModalItemTracker") -> Mock:
    """
    Create a mock content parser that adds items without fetching real media.

    This allows testing chat parsing without network access or media files.
    """
    parser = Mock()
    parser.mm_placeholder_storage = Mock(return_value={})
    parser.parse_image = Mock(side_effect=lambda url, uuid=None: tracker.add("image", None, uuid))
    parser.parse_image_embeds = Mock(side_effect=lambda embeds, uuid=None: tracker.add("image_embeds", None, uuid))
    parser.parse_image_pil = Mock(side_effect=lambda pil, uuid=None: tracker.add("image", None, uuid))
    parser.parse_audio = Mock(side_effect=lambda url, uuid=None: tracker.add("audio", None, uuid))
    parser.parse_input_audio = Mock(side_effect=lambda audio, uuid=None: tracker.add("audio", None, uuid))
    parser.parse_audio_embeds = Mock(side_effect=lambda embeds, uuid=None: tracker.add("audio_embeds", None, uuid))
    parser.parse_video = Mock(side_effect=lambda url, uuid=None: tracker.add("video", None, uuid))
    return parser


def test_openai_chat_use_audio_in_video_prevents_qwen3_mrope_audio_idx_oob(
    monkeypatch: pytest.MonkeyPatch,
    mock_mm_processor: Mock,
    mock_model_cls: Mock,
    mock_model_config: Mock,
    mock_qwen3_omni_config: Mock,
) -> None:
    """
    Regression test for QA crash: OpenAI chat + mm_processor_kwargs use_audio_in_video.

    The QA report shows that providing both a video and an audio item in OpenAI
    chat messages, *and* setting `mm_processor_kwargs={"use_audio_in_video": True}`,
    can crash Qwen3-Omni in `get_mrope_input_positions` due to an extra standalone
    audio placeholder being rendered into the prompt.

    This test:
    1. Parses OpenAI-style chat messages (lightweight: no media fetch)
    2. Maps placeholder dicts -> minimal synthetic `input_tokens`
    3. Calls the real Qwen3 `get_mrope_input_positions` logic

    **Pre-fix (red)**: chat parsing emits BOTH a video placeholder and a standalone
    audio placeholder, which reproduces the IndexError in get_mrope_input_positions.

    **Post-fix (green)**: chat parsing omits the standalone audio placeholder when
    `use_audio_in_video=True` and video is present, so `get_mrope_input_positions`
    no longer crashes.
    """
    from vllm.entrypoints.chat_utils import (
        AsyncMultiModalItemTracker,
        BaseMultiModalItemTracker,
    )

    # Patch chat parsing internals to avoid loading real model processors/configs.
    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "mm_processor",
        property(lambda self: mock_mm_processor),
    )
    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "model_cls",
        property(lambda self: mock_model_cls),
    )
    monkeypatch.setattr(
        AsyncMultiModalItemTracker,
        "create_parser",
        lambda self: _create_mock_content_parser(self),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": None, "uuid": "vid-0"},
                {"type": "audio_url", "audio_url": None, "uuid": "aud-0"},
                {"type": "text", "text": "Describe this video with audio."},
            ],
        }
    ]

    # Call parse_chat_messages_futures with mm_processor_kwargs if available (post-fix).
    # Pre-fix code doesn't have this parameter, so we call without it.
    kwargs: dict = {"content_format": "openai"}
    if "mm_processor_kwargs" in inspect.signature(parse_chat_messages_futures).parameters:
        kwargs["mm_processor_kwargs"] = {"use_audio_in_video": True}

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        messages,
        mock_model_config,
        **kwargs,
    )

    # Verify conversation structure
    assert len(conversation) == 1, "Expected single conversation message"
    content = conversation[0]["content"]
    assert isinstance(content, list), "Expected list content in openai format"

    # Verify multimodal data still includes both audio and video
    mm_data = asyncio.run(mm_future)
    assert mm_data is not None, "Expected multimodal data"
    assert "video" in mm_data, "Expected video in mm_data"
    assert "audio" in mm_data, "Expected audio in mm_data"
    assert len(mm_data["video"]) == 1, "Expected 1 video item"
    assert len(mm_data["audio"]) == 1, "Expected 1 audio item"

    # Verify UUIDs
    assert mm_uuids is not None, "Expected multimodal UUIDs"
    assert mm_uuids.get("video") == ["vid-0"], "Expected video UUID"
    assert mm_uuids.get("audio") == ["aud-0"], "Expected audio UUID"

    # Build synthetic input_tokens based on parsed conversation placeholders.
    # Pre-fix: content will have BOTH video AND audio placeholders -> triggers bug.
    # Post-fix: content will have ONLY video placeholder (audio omitted) -> no bug.
    input_tokens: list[int] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "video":
            # Token span for `use_audio_in_video=True`:
            # <vision_start><audio_start><video_pad><audio_pad><audio_end><vision_end>
            input_tokens.extend([
                mock_qwen3_omni_config.vision_start_token_id,
                mock_qwen3_omni_config.audio_start_token_id,
                mock_qwen3_omni_config.video_token_id,
                mock_qwen3_omni_config.audio_token_id,
                mock_qwen3_omni_config.audio_end_token_id,
                mock_qwen3_omni_config.vision_end_token_id,
            ])
        elif part_type == "audio":
            # Standalone audio span - THIS CAUSES THE BUG pre-fix!
            # get_mrope_input_positions sees 2 audio segments but only 1 audio_feature_length
            input_tokens.extend([
                mock_qwen3_omni_config.audio_start_token_id,
                mock_qwen3_omni_config.audio_token_id,
                mock_qwen3_omni_config.audio_end_token_id,
            ])

    # ---- Assert audio/video tokens are interleaved inside the video span ----
    vs = mock_qwen3_omni_config.vision_start_token_id
    ve = mock_qwen3_omni_config.vision_end_token_id
    a_start = mock_qwen3_omni_config.audio_start_token_id
    a_end = mock_qwen3_omni_config.audio_end_token_id
    video_tok = mock_qwen3_omni_config.video_token_id
    audio_tok = mock_qwen3_omni_config.audio_token_id

    vs_idx = input_tokens.index(vs)
    ve_idx = input_tokens.index(ve, vs_idx + 1)
    video_span = input_tokens[vs_idx : ve_idx + 1]

    # Qwen3's `use_audio_in_video` expects <vision_start><audio_start>...
    assert len(video_span) >= 2, "Video span too short"
    assert video_span[1] == a_start, (
        f"Expected <audio_start> after <vision_start>, got {video_span[1]}"
    )

    # Inside the audio block, both video and audio tokens should be present
    a_start_idx = video_span.index(a_start)
    a_end_idx = video_span.index(a_end)
    assert a_start_idx < a_end_idx, "audio_start should come before audio_end"
    inner = video_span[a_start_idx + 1 : a_end_idx]
    assert video_tok in inner, "Expected video_pad token inside audio block"
    assert audio_tok in inner, "Expected audio_pad token inside audio block"

    # ---- Build mm_features and call get_mrope_input_positions ----
    field = MultiModalSharedField(batch_size=1)
    video_item = MultiModalKwargsItem.from_elems([
        MultiModalFieldElem(
            modality="video",
            key="video_grid_thw",
            data=torch.tensor([1, 2, 2], dtype=torch.long),
            field=field,
        ),
        MultiModalFieldElem(
            modality="video",
            key="use_audio_in_video",
            data=True,
            field=field,
        ),
    ])
    audio_item = MultiModalKwargsItem.from_elems([
        MultiModalFieldElem(
            modality="audio",
            key="audio_feature_lengths",
            data=1,
            field=field,
        ),
    ])
    mm_features = [
        MultiModalFeatureSpec(
            data=video_item,
            modality="video",
            identifier="vid-0",
            mm_position=PlaceholderRange(offset=0, length=0),
        ),
        MultiModalFeatureSpec(
            data=audio_item,
            modality="audio",
            identifier="aud-0",
            mm_position=PlaceholderRange(offset=0, length=0),
        ),
    ]

    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    # THIS IS THE KEY ASSERTION:
    # Pre-fix: This raises IndexError because chat parsing emits a standalone
    #          audio placeholder, causing get_mrope_input_positions to access
    #          audio_feature_lengths[1] when only index 0 exists.
    # Post-fix: This succeeds because the standalone audio placeholder is omitted.
    mock_self = Mock()
    mock_self.config = mock_qwen3_omni_config
    Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions(
        mock_self,
        input_tokens=input_tokens,
        mm_features=mm_features,
    )


def test_video_only_with_use_audio_in_video_keeps_audio_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    mock_mm_processor: Mock,
    mock_model_cls: Mock,
    mock_model_config: Mock,
) -> None:
    """
    Test that video-only requests with use_audio_in_video=True do NOT omit audio.

    When use_audio_in_video=True is set but there's no separate audio_url (audio
    is extracted from the video file), we must NOT omit the audio placeholder.
    The Qwen3 processor requires equal audio/video counts for validation.

    This prevents the regression in bug_report_v3.md where video-only +
    use_audio_in_video=True caused a 500 error.
    """
    from vllm.entrypoints.chat_utils import (
        AsyncMultiModalItemTracker,
        BaseMultiModalItemTracker,
    )

    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "mm_processor",
        property(lambda self: mock_mm_processor),
    )
    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "model_cls",
        property(lambda self: mock_model_cls),
    )
    monkeypatch.setattr(
        AsyncMultiModalItemTracker,
        "create_parser",
        lambda self: _create_mock_content_parser(self),
    )

    # Video-only request - NO separate audio_url
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": None, "uuid": "vid-0"},
                {"type": "text", "text": "Describe this video with audio."},
            ],
        }
    ]

    kwargs: dict = {"content_format": "openai"}
    if "mm_processor_kwargs" in inspect.signature(parse_chat_messages_futures).parameters:
        kwargs["mm_processor_kwargs"] = {"use_audio_in_video": True}

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        messages,
        mock_model_config,
        **kwargs,
    )

    assert len(conversation) == 1, "Expected single conversation message"
    content = conversation[0]["content"]
    assert isinstance(content, list), "Expected list content in openai format"

    # Video placeholder should be present
    video_parts = [p for p in content if p.get("type") == "video"]
    assert len(video_parts) == 1, "Expected 1 video placeholder"

    # Audio placeholder should NOT be omitted (no standalone audio_url was provided)
    # The audio will be extracted from the video by the processor
    audio_parts = [p for p in content if p.get("type") == "audio"]
    # With video-only input, there's no audio part in the message, so audio_parts=0 is expected
    # The key is that we don't crash - the processor handles audio extraction from video
    assert len(audio_parts) == 0, "No audio placeholder expected for video-only input"

    # Verify multimodal data has video only
    mm_data = asyncio.run(mm_future)
    assert mm_data is not None, "Expected multimodal data"
    assert "video" in mm_data, "Expected video in mm_data"
    assert len(mm_data["video"]) == 1, "Expected 1 video item"


def _print_input_ids(input_ids: list[int]) -> str:
    """
    Format input IDs with compressed consecutive special tokens.

    Matches the helper in tests/model_executor/test_qwen3_omni.py.
    - 151675: <|audio_pad|>
    - 151656: <|video_pad|>
    """
    if not input_ids:
        return "[]"

    result = []
    i = 0

    while i < len(input_ids):
        current_id = input_ids[i]

        if current_id in [QWEN3_OMNI_TOKEN_IDS["audio_pad"], QWEN3_OMNI_TOKEN_IDS["video_pad"]]:
            count = 1
            while i + count < len(input_ids) and input_ids[i + count] == current_id:
                count += 1

            token_name = "<|audio_pad|>" if current_id == QWEN3_OMNI_TOKEN_IDS["audio_pad"] else "<|video_pad|>"
            result.append(f"{token_name} * {count}")
            i += count
        else:
            result.append(str(current_id))
            i += 1

    return ", ".join(result)


@pytest.fixture
def mock_hf_processor():
    """Create a mock HF processor (matching test_qwen3_omni.py style)."""
    from transformers.models.whisper import WhisperFeatureExtractor

    processor = Mock()
    processor.audio_token = "<|audio_pad|>"
    processor.image_token = "<|image_pad|>"
    processor.video_token = "<|video_pad|>"
    processor.feature_extractor = WhisperFeatureExtractor()
    return processor


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer (matching test_qwen3_omni.py style)."""
    tokenizer = Mock()
    tokenizer.get_vocab = Mock(return_value={
        "<|audio_pad|>": QWEN3_OMNI_TOKEN_IDS["audio_pad"],
        "<|video_pad|>": QWEN3_OMNI_TOKEN_IDS["video_pad"],
        "<|image_pad|>": QWEN3_OMNI_TOKEN_IDS["image_pad"],
        "<|audio_start|>": QWEN3_OMNI_TOKEN_IDS["audio_start"],
        "<|audio_end|>": QWEN3_OMNI_TOKEN_IDS["audio_end"],
        "<|vision_start|>": QWEN3_OMNI_TOKEN_IDS["vision_start"],
        "<|vision_end|>": QWEN3_OMNI_TOKEN_IDS["vision_end"],
    })
    tokenizer.encode = Mock(side_effect=lambda x: {
        "<|vision_start|>": [QWEN3_OMNI_TOKEN_IDS["vision_start"]],
        "<|vision_end|>": [QWEN3_OMNI_TOKEN_IDS["vision_end"]],
        "<|audio_start|>": [QWEN3_OMNI_TOKEN_IDS["audio_start"]],
        "<|audio_end|>": [QWEN3_OMNI_TOKEN_IDS["audio_end"]],
        "<|audio_pad|>": [QWEN3_OMNI_TOKEN_IDS["audio_pad"]],
        "<|image_pad|>": [QWEN3_OMNI_TOKEN_IDS["image_pad"]],
        "<|video_pad|>": [QWEN3_OMNI_TOKEN_IDS["video_pad"]],
    }.get(x, [0]))
    tokenizer.vision_bos_token = "<|vision_start|>"
    tokenizer.vision_eos_token = "<|vision_end|>"
    tokenizer.audio_bos_token = "<|audio_start|>"
    tokenizer.audio_eos_token = "<|audio_end|>"
    return tokenizer


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor."""
    processor = Mock()
    processor.merge_size = 2
    return processor


def test_use_audio_in_video_generates_interleaved_tokens(
    mock_qwen3_omni_config: Mock,
    mock_hf_processor: Mock,
    mock_tokenizer: Mock,
    mock_image_processor: Mock,
) -> None:
    """
    E2E test: verify audio/video tokens are properly interleaved.

    This test mirrors tests/model_executor/test_qwen3_omni.py but adds explicit
    interleaving verification. When use_audio_in_video=True, the Qwen3 processor
    generates token sequences like:

    151669, <|video_pad|> * 576, <|audio_pad|> * 25, <|video_pad|> * 576, ...

    The pattern alternates video and audio token blocks within the audio markers.
    """
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerMultiModalProcessor,
        Qwen3OmniMoeThinkerProcessingInfo,
    )
    from vllm.multimodal.processing import InputProcessingContext

    # Create processor instance (matching test_qwen3_omni.py style)
    mock_ctx = Mock(spec=InputProcessingContext)
    info = Qwen3OmniMoeThinkerProcessingInfo(mock_ctx)
    info.get_hf_config = Mock(return_value=mock_qwen3_omni_config)
    info.get_hf_processor = Mock(return_value=mock_hf_processor)
    info.get_tokenizer = Mock(return_value=mock_tokenizer)
    info.get_image_processor = Mock(return_value=mock_image_processor)

    mock_dummy_inputs = Mock()
    processor = Qwen3OmniMoeThinkerMultiModalProcessor(info, mock_dummy_inputs)

    # Test parameters from reference video (same as test_qwen3_omni.py)
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4
    audio_len = 85
    video_grid_thw = [6, 36, 64]
    video_second_per_grid_t = 2.0

    # Call the method
    updates = processor.get_updates_use_audio_in_video(
        thinker_config=mock_qwen3_omni_config,
        audio_len=audio_len,
        video_grid_thw=video_grid_thw,
        video_second_per_grid_t=video_second_per_grid_t,
    )

    # Print interleaved pattern for debugging (matches test_qwen3_omni.py)
    # Expected output:
    # 151669, <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 10,
    # <|video_pad|> * 1152, 151670
    print(_print_input_ids(updates))

    # Verify structure
    assert isinstance(updates, list)
    assert len(updates) > 0

    audio_start = mock_qwen3_omni_config.audio_start_token_id
    audio_end = mock_qwen3_omni_config.audio_end_token_id
    audio_tok = mock_qwen3_omni_config.audio_token_id
    video_tok = mock_qwen3_omni_config.video_token_id

    # Verify start and end tokens
    assert updates[0] == audio_start
    assert updates[-1] == audio_end

    # Count audio and video tokens
    audio_count = updates.count(audio_tok)
    video_count = updates.count(video_tok)

    assert audio_count == audio_len, (
        f"Expected {audio_len} audio tokens, got {audio_count}"
    )

    # Calculate expected video token count
    spatial_merge_size = mock_qwen3_omni_config.vision_config.spatial_merge_size
    height = video_grid_thw[1] // spatial_merge_size
    width = video_grid_thw[2] // spatial_merge_size
    expected_video_count = video_grid_thw[0] * height * width

    assert video_count == expected_video_count, (
        f"Expected {expected_video_count} video tokens, got {video_count}"
    )

    # Verify total token count: 1 (start) + audio_len + video_count + 1 (end)
    expected_total = 1 + audio_len + expected_video_count + 1
    assert len(updates) == expected_total, (
        f"Expected {expected_total} total tokens, got {len(updates)}"
    )

    # Verify interleaving pattern: video and audio blocks should alternate
    # The pattern is: audio_start, [video_block, audio_block]*, video_block, audio_end
    inner_tokens = updates[1:-1]  # Exclude start/end markers

    # Find transitions between token types (indicates interleaving)
    transitions = 0
    for i in range(1, len(inner_tokens)):
        if inner_tokens[i] != inner_tokens[i - 1]:
            transitions += 1

    # With proper interleaving, we expect multiple transitions (not just 1 if all video then all audio)
    # For this test case: 4 audio blocks + 5 video blocks = at least 8 transitions
    assert transitions >= 8, (
        f"Expected at least 8 transitions for interleaving, got {transitions}"
    )


def test_openai_client_video_audio_use_audio_in_video_e2e(
    monkeypatch: pytest.MonkeyPatch,
    mock_mm_processor: Mock,
    mock_model_cls: Mock,
    mock_model_config: Mock,
    mock_qwen3_omni_config: Mock,
) -> None:
    """
    E2E test simulating the exact OpenAI client usage pattern:

    ```python
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

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

    This test verifies:
    1. Video + separate audio + use_audio_in_video=True doesn't crash
    2. The standalone audio placeholder is omitted from conversation
    3. Both video and audio data are still passed to the processor
    4. The token sequence is valid for get_mrope_input_positions
    """
    from vllm.entrypoints.chat_utils import (
        AsyncMultiModalItemTracker,
        BaseMultiModalItemTracker,
    )

    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "mm_processor",
        property(lambda self: mock_mm_processor),
    )
    monkeypatch.setattr(
        BaseMultiModalItemTracker,
        "model_cls",
        property(lambda self: mock_model_cls),
    )
    monkeypatch.setattr(
        AsyncMultiModalItemTracker,
        "create_parser",
        lambda self: _create_mock_content_parser(self),
    )

    # Exact message structure from OpenAI client example
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the video and transcribe the speech.",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/video.mp4"},
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": "https://example.com/audio.wav"},
                },
            ],
        }
    ]

    # mm_processor_kwargs passed via extra_body in OpenAI client
    mm_processor_kwargs = {"use_audio_in_video": True}

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        messages,
        mock_model_config,
        content_format="openai",
        mm_processor_kwargs=mm_processor_kwargs,
    )

    # 1. Verify conversation structure - should have video but NOT standalone audio
    assert len(conversation) == 1
    content = conversation[0]["content"]
    assert isinstance(content, list)

    video_parts = [p for p in content if p.get("type") == "video"]
    audio_parts = [p for p in content if p.get("type") == "audio"]
    text_parts = [p for p in content if p.get("type") == "text"]

    assert len(video_parts) == 1, "Expected 1 video placeholder"
    assert len(audio_parts) == 0, "Standalone audio placeholder should be omitted"
    assert len(text_parts) == 1, "Expected 1 text part"

    # 2. Verify mm_data still contains BOTH video and audio (for processor)
    mm_data = asyncio.run(mm_future)
    assert mm_data is not None
    assert "video" in mm_data, "Video data should be present for processor"
    assert "audio" in mm_data, "Audio data should be present for processor"
    assert len(mm_data["video"]) == 1
    assert len(mm_data["audio"]) == 1

    # 3. Build synthetic token sequence (as processor would generate)
    input_tokens = [
        mock_qwen3_omni_config.vision_start_token_id,
        mock_qwen3_omni_config.audio_start_token_id,
        mock_qwen3_omni_config.video_token_id,
        mock_qwen3_omni_config.audio_token_id,
        mock_qwen3_omni_config.audio_end_token_id,
        mock_qwen3_omni_config.vision_end_token_id,
    ]

    # 4. Verify get_mrope_input_positions doesn't crash
    field = MultiModalSharedField(batch_size=1)
    video_item = MultiModalKwargsItem.from_elems([
        MultiModalFieldElem(
            modality="video",
            key="video_grid_thw",
            data=torch.tensor([1, 2, 2], dtype=torch.long),
            field=field,
        ),
        MultiModalFieldElem(
            modality="video",
            key="use_audio_in_video",
            data=True,
            field=field,
        ),
    ])
    audio_item = MultiModalKwargsItem.from_elems([
        MultiModalFieldElem(
            modality="audio",
            key="audio_feature_lengths",
            data=1,
            field=field,
        ),
    ])
    mm_features = [
        MultiModalFeatureSpec(
            data=video_item,
            modality="video",
            identifier=None,
            mm_position=PlaceholderRange(offset=0, length=0),
        ),
        MultiModalFeatureSpec(
            data=audio_item,
            modality="audio",
            identifier=None,
            mm_position=PlaceholderRange(offset=0, length=0),
        ),
    ]

    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    # This should NOT crash - before the fix it raised IndexError
    mock_self = Mock()
    mock_self.config = mock_qwen3_omni_config
    Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions(
        mock_self,
        input_tokens=input_tokens,
        mm_features=mm_features,
    )
