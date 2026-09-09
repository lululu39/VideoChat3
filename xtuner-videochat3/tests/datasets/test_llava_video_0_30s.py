import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from scripts.prepare_llava_video_0_30s import (
    build_messages, extract_archive, normalize_conversation, probe_video,
    sample_count, split_for_video, video_key,
)


def test_multiturn_preserves_options_answers_and_only_one_video():
    row = {"conversations": [
        {"from": "human", "value": "<image>\nWhat happens first?\nA. Opens\nB. Closes"},
        {"from": "gpt", "value": "A. Opens"},
        {"from": "human", "value": "What happens next?"},
        {"from": "gpt", "value": "Closes."}]}
    messages = build_messages("a.mp4", {}, normalize_conversation(row))
    assert len(messages) == 4
    assert messages[0]["content"][1]["text"] == "<VIDEO_CONTEXT>\nWhat happens first?\nA. Opens\nB. Closes"
    assert messages[1]["content"] == "A. Opens"
    assert messages[2]["content"] == "What happens next?"


def test_bad_conversation_roles_fail():
    with pytest.raises(ValueError):
        normalize_conversation({"conversations": [{"from": "gpt", "value": "a"},
                                                   {"from": "human", "value": "q"}]})


def test_source_video_split_is_independent_of_question_or_repackaging():
    assert video_key("ytb_abcdefghijk.mp4") == video_key("v_abcdefghijk.mp4")
    assert video_key("sta/RPY8D_0.0_15.2.mp4") == "RPY8D"
    assert video_key("NUsG9BgSes0_210.0_360.0") == "NUsG9BgSes0"
    assert video_key("v_NUsG9BgSes0_210.0_360.0.mp4") == "NUsG9BgSes0"
    assert video_key("v_123456789") == "v_123456789"  # Native 11-character YouTube ID.
    assert video_key("ytb_v_123456789.mp4") == "v_123456789"
    assert video_key(video_key("v_v_123456789.mp4")) == "v_123456789"
    assert video_key("academic_source/youcook2/101/abcdefghijk/split_0.mp4") == "abcdefghijk"
    assert video_key("academic_source/youcook2/101/abcdefghijk/split_1.mp4") == "abcdefghijk"
    assert video_key("academic_source/youcook2/101/ABCDEFGHIJK/split_0.mp4") == "ABCDEFGHIJK"
    for i in range(100):
        assert split_for_video(f"ytb_{i:011d}.mp4") == split_for_video(f"v_{i:011d}.mp4")
    assert {split_for_video(f"{i}.mp4") for i in range(1000)} == {"train", "heldout"}


def test_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../outside.mp4")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "media", tmp_path / "markers")
    assert not (tmp_path / "outside.mp4").exists()


def test_sample_count_never_repeats_missing_frames():
    assert sample_count({"total_num_frames": 1000}) == 64
    assert sample_count({"total_num_frames": 61}) == 60
    assert sample_count({"total_num_frames": 7}) == 4


def test_decode_validates_actual_training_frame_selection():
    metadata = probe_video(REPO_ROOT / "xtuner-videochat3/tests/resource/tennis.mp4")
    assert metadata["duration"] > 0 and metadata["fps"] > 0
    assert metadata["height"] > 0 and metadata["width"] > 0
    assert sample_count(metadata) % 4 == 0


def test_prepare_pipeline_excludes_missing_and_benchmark_media_and_splits_video(tmp_path, monkeypatch):
    import scripts.prepare_llava_video_0_30s as prepare

    train = next(f"{i}.mp4" for i in range(1000) if split_for_video(f"{i}.mp4") == "train")
    heldout = next(f"{i}.mp4" for i in range(1000) if split_for_video(f"{i}.mp4") == "heldout")
    media = tmp_path / "videos"
    media.mkdir()
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i",
                    str(REPO_ROOT / "xtuner-videochat3/tests/resource/tennis.mp4"),
                    "-t", "2", "-c", "copy", str(clip)], check=True)
    for name in (train, heldout, "benchmark.mp4"):
        shutil.copyfile(clip, media / name)
    conv = [{"from": "human", "value": "<image>\nWhat happens first?"},
            {"from": "gpt", "value": "A player serves."},
            {"from": "human", "value": "Then what?"},
            {"from": "gpt", "value": "The ball moves."}]
    rows = [{"id": str(i), "video": v, "conversations": conv}
            for i, v in enumerate([train, train, heldout, "missing.mp4", "benchmark.mp4"])]
    annotation = tmp_path / "source.json"
    annotation.write_text(json.dumps(rows))
    excludes = tmp_path / "exclude.txt"
    excludes.write_text("benchmark\n")
    monkeypatch.setattr(prepare, "source_files", lambda root: [("academic", "oe", annotation)])
    monkeypatch.setattr(prepare, "EXPECTED", {"academic": {"oe": len(rows)}})
    monkeypatch.setattr(prepare, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *a, **k: SimpleNamespace(siblings=[])))
    monkeypatch.setattr(sys, "argv", ["prepare", "--dataset-root", str(tmp_path),
                        "--skip-download", "--skip-extract", "--exclude-video-ids", str(excludes),
                        "--workers", "1"])
    prepare.main()
    summary = json.loads((tmp_path / "llava_video_0_30s_prepare_summary.json").read_text())
    counts = summary["conversion"]["llava_0_30s_academic_oe"]
    assert counts["duplicate_conversations"] == 1
    assert counts["benchmark_excluded_records"] == 1
    assert counts["decode_rejected_records"] == 1
    assert counts["train_records"] == 1 and counts["train_qa_turns"] == 2
    assert counts["heldout_records"] == 2 and counts["heldout_qa_turns"] == 2
    assert "missing.mp4" in summary["missing_or_empty_videos"]
    output = tmp_path / "videochat3_annotations/llava_0_30s_academic_oe_heldout.jsonl"
    assert all(len(json.loads(line)["messages"]) == 2 for line in output.read_text().splitlines())
