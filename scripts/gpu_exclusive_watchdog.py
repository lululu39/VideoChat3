#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminate GPU compute processes outside one allowed process tree."
    )
    parser.add_argument("--allow-root-pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--grace", type=float, default=2.0)
    return parser.parse_args()


def parent_pid(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[3])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def is_descendant(pid: int, root_pid: int) -> bool:
    visited = set()
    while pid > 1 and pid not in visited:
        if pid == root_pid:
            return True
        visited.add(pid)
        parent = parent_pid(pid)
        if parent is None:
            return False
        pid = parent
    return False


def gpu_compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }


def log_event(log_path: Path, event: dict) -> None:
    event["time_utc"] = datetime.now(timezone.utc).isoformat()
    with log_path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def terminate_process(pid: int, grace: float, log_path: Path) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        log_event(log_path, {"event": "sigterm", "pid": pid})
    except ProcessLookupError:
        return
    except PermissionError as error:
        log_event(
            log_path,
            {"event": "permission_error", "pid": pid, "error": str(error)},
        )
        return

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(min(0.1, grace))
    try:
        os.kill(pid, signal.SIGKILL)
        log_event(log_path, {"event": "sigkill", "pid": pid})
    except ProcessLookupError:
        pass


def main() -> None:
    args = parse_args()
    if args.interval <= 0 or args.grace < 0:
        raise ValueError("interval must be positive and grace non-negative")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_event(
        args.log,
        {
            "event": "started",
            "watchdog_pid": os.getpid(),
            "allow_root_pid": args.allow_root_pid,
        },
    )
    while Path(f"/proc/{args.allow_root_pid}").exists():
        unauthorized = sorted(
            pid
            for pid in gpu_compute_pids()
            if not is_descendant(pid, args.allow_root_pid)
        )
        for pid in unauthorized:
            terminate_process(pid, args.grace, args.log)
        time.sleep(args.interval)
    log_event(args.log, {"event": "stopped", "reason": "allow_root_exited"})


if __name__ == "__main__":
    main()
