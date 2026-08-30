import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_videochat_flash_longvid import (
    convert_conversations,
    inspect_frame_directory,
)
from xtuner.v1.datasets.mllm_tokenize_fn.video_utils import (
    VideoChat3VideoMetadata,
)
from xtuner.v1.datasets.mllm_tokenize_fn.videochat3_tokenize_fn import (
    get_video_frame_sample_indices,
)


class FakeVideoProcessor:
    num_frames = None
    fps = 1.0
    min_frames = 64
    max_frames = 512

    def get_num_sampled_frames(self, metadata, num_frames=None, fps=None):
        assert num_frames is None
        assert fps == 1.0
        count = int(metadata.total_num_frames / metadata.fps * fps)
        return min(min(max(count, self.min_frames), self.max_frames), metadata.total_num_frames)


@pytest.mark.parametrize(
    ("total_frames", "expected_frames"),
    [(63, 60), (143, 140), (512, 512), (1058, 512)],
)
def test_longvid_frame_sampling_matches_official_recipe(total_frames, expected_frames):
    metadata = VideoChat3VideoMetadata(
        total_num_frames=total_frames,
        fps=1.0,
        duration=float(total_frames),
        width=16,
        height=12,
        video_backend="img2",
    )
    indices = get_video_frame_sample_indices(
        metadata,
        FakeVideoProcessor(),
        frame_multiple=4,
    )

    assert len(indices) == expected_frames
    assert indices[0] == 0
    assert indices[-1] == total_frames - 1
    assert all(left <= right for left, right in zip(indices, indices[1:]))


def test_longvid_converter_builds_videochat3_message(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(1, 9):
        Image.new("RGB", (16, 12), color=(index, 0, 0)).save(
            frame_dir / f"{index:05d}.jpg"
        )

    metadata = inspect_frame_directory(frame_dir, fps=1.0, frame_multiple=4)
    messages = convert_conversations(
        [
            {"from": "human", "value": "<image>\nWhat happened first?"},
            {"from": "gpt", "value": "The person opened the door."},
        ],
        video="video-id",
        metadata=metadata,
    )

    video_item = messages[0]["content"][0]
    text_item = messages[0]["content"][1]
    assert video_item["video_url"]["url"] == "video-id"
    assert video_item["video_metadata"]["total_num_frames"] == 8
    assert video_item["video_metadata"]["fps"] == 1.0
    assert video_item["video_metadata"]["video_backend"] == "img2"
    assert text_item["text"] == "<VIDEO_CONTEXT>\nWhat happened first?"
    assert messages[1]["content"] == "The person opened the door."


def test_longvid_converter_rejects_non_contiguous_frames(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for name in ("00001.jpg", "00003.jpg", "00004.jpg", "00005.jpg"):
        Image.new("RGB", (16, 12)).save(frame_dir / name)

    with pytest.raises(ValueError, match="not contiguous"):
        inspect_frame_directory(frame_dir, fps=1.0, frame_multiple=4)
