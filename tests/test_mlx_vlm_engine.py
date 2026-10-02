# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""MLX completion metadata without loading MLX or a model."""

from dataclasses import dataclass

import pytest
from PIL import Image

from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.utils.generation_utils import GenerationStopper


@dataclass
class _Chunk:
    """The completion fields of mlx-vlm's GenerationResult."""

    text: str
    generation_tokens: int
    finish_reason: str | None = None


@pytest.fixture
def engine() -> MlxVlmEngine:
    engine = MlxVlmEngine(MlxVlmEngineOptions(), artifacts_path=None)
    engine._initialized = True
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = {}
    engine.apply_chat_template = lambda *args, **kwargs: "Prompt"
    return engine


@pytest.mark.parametrize(
    ("chunks", "expected_reason", "expected_tokens"),
    [
        (
            [_Chunk("A", 1), _Chunk("B", 2), _Chunk("C", 3), _Chunk("", 3, "length")],
            VlmStopReason.LENGTH,
            3,
        ),
        ([_Chunk("A", 1), _Chunk("", 2, "stop")], VlmStopReason.END_OF_SEQUENCE, 2),
        # EOS at the cap is still a normal completion; count alone cannot tell.
        (
            [_Chunk("A", 1), _Chunk("B", 2), _Chunk("C", 3, "stop")],
            VlmStopReason.END_OF_SEQUENCE,
            3,
        ),
        ([_Chunk("", 0, "length")], VlmStopReason.LENGTH, 0),
        ([_Chunk("A", 1)], VlmStopReason.UNSPECIFIED, 1),
    ],
)
def test_mlx_preserves_terminal_status_and_cumulative_count(
    engine, chunks, expected_reason, expected_tokens
):
    engine.stream_generate = lambda *args, **kwargs: iter(chunks)

    (output,) = engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                max_new_tokens=3,
            )
        ]
    )

    assert output.text == "".join(chunk.text for chunk in chunks)
    assert output.stop_reason == expected_reason
    assert output.metadata["num_tokens"] == expected_tokens


class _StopAtB(GenerationStopper):
    def should_stop(self, text: str) -> bool:
        return text.endswith("B")


@pytest.mark.parametrize("use_custom_stopper", [False, True])
def test_mlx_local_stopping_emits_a_recognized_reason(engine, use_custom_stopper):
    consumed = []

    def stream(*args, **kwargs):
        for chunk in [_Chunk("A", 1), _Chunk("B", 2), _Chunk("C", 3)]:
            consumed.append(chunk.text)
            yield chunk

    engine.stream_generate = stream
    request = VlmEngineInput(
        image=Image.new("RGB", (8, 8), "white"),
        prompt="Prompt",
        stop_strings=[] if use_custom_stopper else ["B"],
        extra_generation_config={"custom_stopping_criteria": [_StopAtB()]}
        if use_custom_stopper
        else {},
    )

    (output,) = engine.predict_batch([request])

    assert output.text == "AB"
    assert output.stop_reason == VlmStopReason.STOP_SEQUENCE
    assert output.metadata["num_tokens"] == 2
    assert consumed == ["A", "B"]
