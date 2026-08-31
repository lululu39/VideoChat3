import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_timelens_100k import (
    GROUNDING_PROMPT,
    GroundingRecord,
    build_messages,
    duration_balanced_select,
    excluded_by_filter,
    format_response,
    normalize_query,
    normalize_spans,
    probe_video,
)


def record(duration: float, index: int) -> GroundingRecord:
    return GroundingRecord(
        source="source",
        video_path=f"source/{index}.mp4",
        duration=duration,
        query=f"event {index}",
        spans=((1.0, 2.0),),
        source_row=index,
        event_index=0,
    )


def test_timelens_query_span_and_response_match_official_recipe():
    assert normalize_query("  A   person opens the door... ") == "A person opens the door"
    spans = normalize_spans([[1, 2.25], [4.0, 5]])
    assert spans == ((1.0, 2.25), (4.0, 5.0))
    assert format_response(spans) == (
        "The event happens in 1.0 - 2.2 seconds, 4.0 - 5.0 seconds."
    )


def test_visual_filter_is_stricter_than_official_audio_filter():
    assert excluded_by_filter("When is a sound heard?", "official")
    assert excluded_by_filter("When does the speaker mention the recipe?", "visual")
    assert not excluded_by_filter("When does the person open the door?", "visual")
    assert not excluded_by_filter("When does the speaker mention the recipe?", "official")


def test_duration_balancing_matches_timelens_selection_contract():
    records = []
    for bucket in range(9):
        duration = bucket * 30 + 10 if bucket < 8 else 300
        records.extend(record(duration, bucket * 100 + index) for index in range(10))
    selected, report = duration_balanced_select(records, target_size=45, seed=42)
    assert len(selected) == 45
    assert report["per_bucket_target"] == 5
    assert all(value["selected"] == 5 for value in report["buckets"].values())
    selected_again, _ = duration_balanced_select(records, target_size=45, seed=42)
    assert selected == selected_again


def test_timelens_converter_builds_videochat3_message():
    item = record(12.0, 7)
    metadata = {
        "total_num_frames": 300,
        "duration": 12.0,
        "fps": 25.0,
        "width": 640,
        "height": 360,
        "video_backend": "decord",
    }
    messages = build_messages(item, metadata)
    assert messages[0]["content"][0]["video_url"]["url"] == "source/7.mp4"
    assert messages[0]["content"][0]["video_metadata"] == metadata
    assert messages[0]["content"][1]["text"] == (
        "<VIDEO_CONTEXT>\n" + GROUNDING_PROMPT.format("event 7")
    )
    assert messages[1]["content"] == "The event happens in 1.0 - 2.0 seconds."


def test_timelens_media_probe_uses_decoded_metadata():
    metadata = probe_video(REPO_ROOT / "xtuner-videochat3/tests/resource/tennis.mp4")
    assert metadata["total_num_frames"] > 0
    assert metadata["duration"] > 0
    assert metadata["fps"] > 0
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    assert metadata["video_backend"] == "decord"
