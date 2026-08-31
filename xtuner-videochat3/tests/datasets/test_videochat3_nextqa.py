import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_nextqa import build_messages, find_media, probe_video


def test_nextqa_converter_builds_open_ended_message():
    metadata = {
        "total_num_frames": 120,
        "duration": 4.0,
        "fps": 30.0,
        "width": 640,
        "height": 360,
        "video_backend": "decord",
    }
    messages = build_messages(
        video="0000/123.mp4",
        metadata=metadata,
        question="what happened after the cup fell",
        answer="the child picked it up",
    )

    video_item = messages[0]["content"][0]
    assert video_item["video_url"]["url"] == "0000/123.mp4"
    assert video_item["video_metadata"] == metadata
    assert messages[0]["content"][1]["text"] == (
        "<VIDEO_CONTEXT>\nwhat happened after the cup fell"
    )
    assert messages[1]["content"] == "the child picked it up"


def test_nextqa_media_probe_uses_real_video_metadata():
    path = REPO_ROOT / "xtuner-videochat3/tests/resource/tennis.mp4"
    metadata = probe_video(path)
    assert metadata["total_num_frames"] > 0
    assert metadata["duration"] > 0
    assert metadata["fps"] > 0
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    assert metadata["video_backend"] == "decord"


def test_nextqa_media_index_is_relative_to_nested_layout(tmp_path: Path):
    video = tmp_path / "0001/123.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not-empty")
    assert find_media(tmp_path) == {"123": video}
