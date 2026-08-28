from pathlib import Path
from typing import Any

from .jsonl_writer import JsonlWriter


class WandbWriter:
    """Rank-zero W&B writer with a local JSONL copy of every metric."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        entity: str,
        project: str,
        name: str,
        base_url: str = "https://api.wandb.ai",
        run_id: str | None = None,
        group: str | None = None,
        job_type: str = "train",
        tags: list[str] | None = None,
        resume: str = "allow",
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ):
        if log_dir is None:
            log_dir = Path()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking requires the `wandb` package; reproduce the project environment with `uv sync --frozen`."
            ) from error

        self._local_writer = JsonlWriter(log_dir=log_dir)
        self._run = wandb.init(
            entity=entity,
            project=project,
            name=name,
            id=run_id,
            group=group,
            job_type=job_type,
            tags=tags,
            resume=resume,
            mode=mode,
            dir=str(log_dir),
            config=config,
            settings=wandb.Settings(base_url=base_url),
        )
        if self._run is None:
            self._local_writer.close()
            raise RuntimeError("wandb.init() did not return a run")

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        self._local_writer.add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self._run.log({tag: scalar_value}, step=global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        self._local_writer.add_scalars(
            tag_scalar_dict=tag_scalar_dict,
            global_step=global_step,
        )
        self._run.log(tag_scalar_dict, step=global_step)

    def close(self) -> None:
        self._local_writer.close()
        self._run.finish()
