import json
import sys
from types import SimpleNamespace

from xtuner.v1._writer.wandb_writer import WandbWriter


class FakeRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, metrics, step):
        self.logs.append((metrics, step))

    def finish(self):
        self.finished = True


def test_wandb_writer_initializes_logs_and_keeps_local_jsonl(tmp_path, monkeypatch):
    fake_run = FakeRun()
    init_kwargs = {}

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs),
    )
    writer = WandbWriter(
        log_dir=tmp_path,
        entity="LVSM-Experiment",
        project="videochat3",
        name="vc3-4b-lact-fw4-ve-s3-lite-v1",
        run_id="vc3-4b-lact-fw4-ve-s3-lite-v1",
        group="videochat3-lact-stage3-ve",
        tags=["lact", "vision-encoder-only"],
        resume="allow",
        config={"global_batch_size": 128},
    )
    writer.add_scalar(tag="loss/mean", scalar_value=1.25, global_step=3)
    writer.add_scalars(
        tag_scalar_dict={"lr": 5e-6, "runtime_info/tgs": 1234.0},
        global_step=4,
    )
    writer.close()

    assert init_kwargs["entity"] == "LVSM-Experiment"
    assert init_kwargs["project"] == "videochat3"
    assert init_kwargs["name"] == "vc3-4b-lact-fw4-ve-s3-lite-v1"
    assert init_kwargs["id"] == "vc3-4b-lact-fw4-ve-s3-lite-v1"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["settings"] == {"base_url": "https://api.wandb.ai"}
    assert fake_run.logs == [
        ({"loss/mean": 1.25}, 3),
        ({"lr": 5e-6, "runtime_info/tgs": 1234.0}, 4),
    ]
    assert fake_run.finished

    local_metrics = [json.loads(line) for line in (tmp_path / "tracker.jsonl").read_text().splitlines()]
    assert local_metrics == [
        {"loss/mean": 1.25, "step": 3},
        {"lr": 5e-6, "runtime_info/tgs": 1234.0, "step": 4},
    ]
