#!/usr/bin/env python3
"""Generate a self-contained HTML map of VideoChat3 code entry points."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "videochat3_code_map.html"

CURRENT_PIPELINE = [
    (
        "Experiment launcher",
        "xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_LINEAR16_DELTA_FWProj_train_timelens_r4_v12.sh",
        "Pins the active v12 environment: Linear16 + Delta, R4 select, 8K pack, no FW ratio clip.",
    ),
    (
        "Dataset wrapper",
        "xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_VE_train_timelens.sh",
        "Resolves the TimeLens manifest and dataset-specific pixel/frame budgets.",
    ),
    (
        "Distributed wrapper",
        "xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_VE_train_stage3.sh",
        "Starts torchrun, W&B, and the exclusive-GPU watchdog.",
    ),
    (
        "Generic SFT entry",
        "xtuner-videochat3/training_scripts/run_sft.sh",
        "Invokes XTuner SFT with the selected Python config.",
    ),
    (
        "Training config",
        "xtuner-videochat3/training_configs/stage3/VideoChat3_4B_LACT_VE_train_stage3.py",
        "Builds model/data/optimizer/FSDP/loss/W&B configs from environment variables.",
    ),
    (
        "Model config",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/videochat3_config.py",
        "Declares Base/LACT vision, projector, and composite 4B model configurations.",
    ),
    (
        "LACT vision",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/modeling_vision_lact.py",
        "Implements SwiGLU/Linear fast weights, Muon/Delta state updates, gates, and recurrent scans.",
    ),
    (
        "Base vision + FSDP",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/modeling_vision.py",
        "Implements packed four-frame ViT attention, patch merge, activation checkpointing, and FSDP.",
    ),
    (
        "Macro temporal selection",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/macro_temporal.py",
        "Selects/means final chunk outputs and keeps video boundaries, timestamps, and placeholders aligned.",
    ),
    (
        "Projector",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/modeling_projector.py",
        "Projects merged vision tokens into the Qwen3 language-model width.",
    ),
    (
        "Composite VLM",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/modeling_videochat3.py",
        "Replaces visual placeholders, runs the frozen LM, and owns HF save/export integration.",
    ),
    (
        "Tokenizer / placeholders",
        "xtuner-videochat3/xtuner/v1/datasets/mllm_tokenize_fn/videochat3_tokenize_fn.py",
        "Samples video frames and constructs synchronized visual placeholders and timestamps.",
    ),
    (
        "Packing",
        "xtuner-videochat3/xtuner/v1/datasets/packing.py",
        "Packs tokenized examples to the configured 8K sequence budget.",
    ),
    (
        "Trainer",
        "xtuner-videochat3/xtuner/v1/engine/vision_compose_train_engine.py",
        "Runs vision-composed SFT, checkpointing, gradient norm/clipping, and HF export.",
    ),
]

DESCRIPTIONS = {
    "AGENTS.md": "Stable project memo: environment, model semantics, training stability, datasets, and evaluation policy.",
    "exp_results.md": "Versioned experiment configurations, checkpoint diagnostics, and native-Accuracy results.",
    "exp_logs.md": "Historical evaluation/run logs retained for audit.",
    "data.md": "Dataset research and selection notes.",
    "pyproject.toml": "Canonical uv/Python dependency and editable-package configuration.",
    "scripts/prepare_timelens_100k.py": "Converts the official TimeLens release into pure-video VideoChat3 manifests.",
    "scripts/sample_timelens_videochat3.py": "Builds the deterministic seed-42 random-half TimeLens subset.",
    "scripts/benchmark_videochat3_lact.py": "Benchmarks Base and LACT vision/full-VLM latency on real checkpoints.",
    "scripts/inspect_videochat3_lact_checkpoint.py": "Audits gates, FW deltas, and frozen Base tensor integrity.",
    "scripts/prepare_videochat3_core_eval_data.py": "Prepares the fixed public core-evaluation datasets.",
    "scripts/eval_videochat3_lact_core.sh": "Eight-GPU resumable native core evaluation launcher.",
    "scripts/gpu_exclusive_watchdog.py": "Terminates GPU processes outside the allowed training process tree.",
    "vlmevalkit-videochat3/vlmeval/vlm/videochat3/model.py": "VLMEvalKit adapter for VideoChat3 checkpoint loading and native generation.",
    "vlmevalkit-videochat3/vlmeval/vlm/videochat3/prompt.py": "VideoChat3-specific VLMEvalKit prompt construction.",
    "vlmevalkit-videochat3/vlmeval/config.py": "VLMEvalKit model registry and runtime configuration.",
    "vlmevalkit-videochat3/vlmeval/dataset/video_dataset_config.py": "Video benchmark registry and sampling settings.",
    "xtuner-videochat3/xtuner/v1/config/fsdp.py": "XTuner FSDP configuration schema.",
    "xtuner-videochat3/xtuner/v1/config/optim.py": "Optimizer parameter grouping, including ViT/FW/projector learning rates.",
    "xtuner-videochat3/xtuner/v1/loss/chunk_loss.py": "Chunked causal-language-model loss used by the Stage 3 config.",
    "xtuner-videochat3/xtuner/v1/utils/grad_norm.py": "Distributed true-global gradient norm computation.",
}

ACTIVE = {
    item[1] for item in CURRENT_PIPELINE
} | {
    "scripts/prepare_timelens_100k.py",
    "scripts/sample_timelens_videochat3.py",
    "AGENTS.md",
    "exp_results.md",
}


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def candidate_paths() -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        text=True,
    )
    selected = []
    curated_paths = set(DESCRIPTIONS) | {item[1] for item in CURRENT_PIPELINE}
    config_suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
    script_suffixes = {".py", ".sh", ".ipynb"}
    for path in output.splitlines():
        pure = Path(path)
        parts = pure.parts
        name = pure.name.lower()
        if "__pycache__" in parts or pure.suffix == ".pyc":
            continue
        if path == "videochat3_code_map.html":
            continue
        is_config = (
            pure.suffix.lower() in config_suffixes
            and (
                "config" in name
                or any(part in {"config", "configs", "training_configs"} for part in parts)
                or pure.name == "pyproject.toml"
            )
        )
        is_script = pure.suffix.lower() in script_suffixes and (
            any(part in {"scripts", "training_scripts", "tools_model_inits"} for part in parts)
            or path in {"vlmevalkit-videochat3/run.py", "setup.py"}
        )
        is_modeling = pure.suffix == ".py" and name.startswith(
            ("modeling_", "configuration_", "processing_")
        )
        is_project_doc = path in {
            "AGENTS.md",
            "data.md",
            "exp_results.md",
            "exp_logs.md",
            "pyproject.toml",
        }
        if path in curated_paths or is_config or is_script or is_modeling or is_project_doc:
            selected.append(path)
    return sorted(set(selected))


def category(path: str) -> str:
    name = Path(path).name.lower()
    if path in {item[1] for item in CURRENT_PIPELINE}:
        return "current-pipeline"
    if path.startswith("xtuner-videochat3/training_configs/"):
        return "training-config"
    if path.startswith("xtuner-videochat3/training_scripts/"):
        return "training-launcher"
    if "/compose/videochat3/" in path:
        if "/hf_" in path:
            return "hf-export"
        return "videochat3-model"
    if path.startswith("scripts/"):
        return "project-script"
    if path.startswith("vlmevalkit-videochat3/configs/"):
        return "eval-config"
    if path.startswith("vlmevalkit-videochat3/") and (
        "videochat3" in name or "/scripts/" in path
    ):
        return "eval-integration"
    if name.startswith(("modeling_", "configuration_", "processing_")):
        return "modeling-other"
    if "config" in path.lower() or Path(path).suffix in {".toml", ".yaml", ".yml"}:
        return "config-other"
    return "script-other"


def scope(path: str) -> str:
    core_prefixes = (
        "scripts/",
        "xtuner-videochat3/training_configs/",
        "xtuner-videochat3/training_scripts/",
        "xtuner-videochat3/xtuner/v1/model/compose/videochat3/",
        "xtuner-videochat3/xtuner/v1/datasets/",
        "xtuner-videochat3/xtuner/v1/engine/",
        "xtuner-videochat3/xtuner/v1/config/",
        "xtuner-videochat3/xtuner/v1/loss/",
        "vlmevalkit-videochat3/configs/",
        "vlmevalkit-videochat3/vlmeval/vlm/videochat3",
    )
    if path.startswith(core_prefixes) or path in DESCRIPTIONS:
        return "project"
    return "upstream"


def status(path: str) -> str:
    if path in ACTIVE:
        return "active"
    lower = path.lower()
    retired_markers = (
        "longvid",
        "nextqa",
        "stage3_lightweight",
        "_v2",
        "_v3",
        "_v4",
        "_v5",
        "_v6",
        "_v7",
        "_v8",
        "_v9",
        "_v10",
        "_v11",
    )
    if scope(path) == "project" and any(marker in lower for marker in retired_markers):
        return "retired"
    return "reference"


def description(path: str, file_category: str) -> str:
    if path in DESCRIPTIONS:
        return DESCRIPTIONS[path]
    name = Path(path).stem.replace("_", " ")
    templates = {
        "training-config": "Executable VideoChat3 training configuration.",
        "training-launcher": "Shell launcher that pins a reproducible training recipe.",
        "videochat3-model": "Core VideoChat3 model/config implementation.",
        "hf-export": "Self-contained Hugging Face configuration/model/processor export.",
        "project-script": "Project utility for data preparation, diagnostics, evaluation, or reporting.",
        "eval-config": "VLMEvalKit benchmark/model configuration.",
        "eval-integration": "VLMEvalKit integration or evaluation utility.",
        "modeling-other": "Vendored or auxiliary model implementation.",
        "config-other": "Framework, example, CI, or upstream configuration.",
        "script-other": "Framework, example, CI, or upstream script.",
        "current-pipeline": "Current v12 execution-path dependency.",
    }
    return f"{templates[file_category]} ({name})"


def line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except (OSError, UnicodeDecodeError):
        return None


def inventory() -> list[dict[str, object]]:
    rows = []
    for path in candidate_paths():
        full_path = ROOT / path
        if not full_path.is_file():
            continue
        file_category = category(path)
        rows.append(
            {
                "path": path,
                "name": full_path.name,
                "href": path,
                "category": file_category,
                "scope": scope(path),
                "status": status(path),
                "description": description(path, file_category),
                "lines": line_count(full_path),
                "bytes": full_path.stat().st_size,
            }
        )
    return rows


def render(output: Path) -> None:
    rows = inventory()
    project_count = sum(row["scope"] == "project" for row in rows)
    active_count = sum(row["status"] == "active" for row in rows)
    revision = run_git("rev-parse", "--short", "HEAD")
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    pipeline_html = "".join(
        f"""
        <li class="flow-step">
          <span class="flow-index">{index:02d}</span>
          <div><strong>{html.escape(label)}</strong>
          <a href="{html.escape(path, quote=True)}">{html.escape(path)}</a>
          <p>{html.escape(note)}</p></div>
        </li>"""
        for index, (label, path, note) in enumerate(CURRENT_PIPELINE, 1)
    )
    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VideoChat3 / LACT Code Map</title>
  <style>
    :root {{ color-scheme: dark; --bg:#090d14; --panel:#101722; --panel2:#151f2d; --text:#e8eef7; --muted:#91a0b5; --line:#273449; --blue:#65a8ff; --cyan:#4ed8c5; --amber:#f7c66b; --red:#ff7c8a; --green:#75d79d; }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:radial-gradient(circle at 80% -20%,#16304b 0,transparent 36rem),var(--bg); color:var(--text); font:14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    code,.mono {{ font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; }}
    .shell {{ max-width:1500px; margin:auto; padding:28px; }}
    header {{ border:1px solid var(--line); background:linear-gradient(135deg,rgba(21,31,45,.96),rgba(10,16,25,.92)); padding:30px; border-radius:18px; box-shadow:0 20px 70px rgba(0,0,0,.3); }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-weight:750; font-size:12px; }}
    h1 {{ margin:.25rem 0 .5rem; font-size:clamp(30px,4vw,54px); letter-spacing:-.045em; }}
    h2 {{ margin:0 0 14px; font-size:22px; }} h3 {{ margin:0; font-size:15px; }}
    .lede {{ color:var(--muted); max-width:980px; font-size:16px; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:12px; margin-top:22px; }}
    .stat {{ padding:14px 16px; background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:12px; }} .stat b {{ display:block; font-size:24px; }} .stat span {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr); gap:18px; margin-top:18px; }}
    section {{ margin-top:18px; background:rgba(16,23,34,.9); border:1px solid var(--line); border-radius:16px; padding:22px; }}
    .callout {{ border-left:3px solid var(--cyan); background:rgba(78,216,197,.07); padding:14px 16px; border-radius:8px; color:#cceee9; }}
    .config-table {{ width:100%; border-collapse:collapse; }} .config-table td {{ padding:8px 9px; border-bottom:1px solid var(--line); }} .config-table td:first-child {{ color:var(--muted); width:42%; }}
    .flow {{ list-style:none; margin:0; padding:0; display:grid; gap:10px; }} .flow-step {{ display:grid; grid-template-columns:38px 1fr; gap:11px; padding:12px; background:var(--panel2); border:1px solid var(--line); border-radius:11px; }} .flow-index {{ color:var(--cyan); font:700 13px/1.8 "SFMono-Regular",monospace; }} .flow-step a {{ display:block; overflow-wrap:anywhere; font-family:"SFMono-Regular",monospace; font-size:12px; }} .flow-step p {{ color:var(--muted); margin:3px 0 0; }}
    .toolbar {{ position:sticky; top:0; z-index:5; display:grid; grid-template-columns:minmax(240px,1fr) repeat(3,minmax(130px,190px)); gap:10px; padding:12px; margin:0 -8px 15px; background:rgba(9,13,20,.94); backdrop-filter:blur(15px); border:1px solid var(--line); border-radius:13px; }}
    input,select {{ width:100%; color:var(--text); background:#0d141e; border:1px solid #34445b; border-radius:9px; padding:10px 11px; outline:none; }} input:focus,select:focus {{ border-color:var(--blue); }}
    .inventory-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; }} .inventory-head span {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:10px; }}
    .card {{ min-width:0; background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:14px; }} .card[data-status="active"] {{ border-color:rgba(117,215,157,.5); box-shadow:inset 3px 0 var(--green); }} .card[data-status="retired"] {{ opacity:.76; }}
    .card-top {{ display:flex; justify-content:space-between; gap:8px; }} .file-link {{ font:700 13px/1.4 "SFMono-Regular",monospace; overflow-wrap:anywhere; }} .card p {{ color:var(--muted); margin:8px 0; min-height:42px; }}
    .badges {{ display:flex; gap:5px; flex-wrap:wrap; }} .badge {{ border:1px solid var(--line); background:#0c131d; color:#aebbd0; border-radius:999px; padding:2px 7px; font-size:10px; text-transform:uppercase; letter-spacing:.05em; }} .badge.active {{ color:var(--green); }} .badge.retired {{ color:var(--red); }} .badge.project {{ color:var(--cyan); }}
    .meta {{ color:#73839a; font:11px/1.4 "SFMono-Regular",monospace; }}
    .empty {{ display:none; color:var(--muted); text-align:center; padding:45px; }} footer {{ color:var(--muted); text-align:center; padding:28px 0 8px; }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns:1fr 1fr; }} .toolbar {{ grid-template-columns:1fr 1fr; }} }}
    @media(max-width:560px) {{ .shell {{ padding:14px; }} header,section {{ padding:17px; }} .stats,.toolbar {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main class="shell">
  <header>
    <div class="eyebrow">Repository navigation · revision {html.escape(revision)}</div>
    <h1>VideoChat3 / LACT Code Map</h1>
    <p class="lede">从当前 v12 的实际调用链出发，链接全部项目级训练配置、脚本、模型实现、HF 导出和评测入口；底部穷举索引同时覆盖 vendored XTuner/VLMEvalKit 中所有匹配的 config、script 与 modeling 文件。</p>
    <div class="stats">
      <div class="stat"><b>{len(rows)}</b><span>索引文件</span></div>
      <div class="stat"><b>{project_count}</b><span>项目相关</span></div>
      <div class="stat"><b>{active_count}</b><span>当前调用链</span></div>
      <div class="stat"><b>{html.escape(revision)}</b><span>Git revision</span></div>
    </div>
  </header>

  <div class="grid">
    <section>
      <h2>当前 v12 调用链</h2>
      <ol class="flow">{pipeline_html}</ol>
    </section>
    <div>
      <section>
        <h2>v12 配置快照</h2>
        <table class="config-table">
          <tr><td>Vision memory</td><td><code>linear</code>, 16 heads, 72×72/head</td></tr>
          <tr><td>Inner update</td><td><code>delta</code>, base write strength 0.01</td></tr>
          <tr><td>Temporal</td><td>4-frame state update; final R4 token selection</td></tr>
          <tr><td>Trainable</td><td>Linear-FW 143,887,104 + projector 33,039,616</td></tr>
          <tr><td>Frozen</td><td>Original ViT + Qwen3 4B LM</td></tr>
          <tr><td>LR</td><td>2e-5 → 1e-6 cosine, 3% warmup</td></tr>
          <tr><td>Stabilization</td><td>No FW ratio clip; global FSDP clip 1.0</td></tr>
          <tr><td>Packing</td><td>8K, global batch 16, 1,485 packs / 93 steps</td></tr>
          <tr><td>Parallelism</td><td>Ordinary FSDP, layer-major FW group 1</td></tr>
        </table>
      </section>
      <section>
        <h2>权威记录</h2>
        <p><a href="AGENTS.md">AGENTS.md</a>：稳定代码结论与操作规则。</p>
        <p><a href="exp_results.md">exp_results.md</a>：版本化实验配置与结果。</p>
        <p><a href="data.md">data.md</a>：数据集调研与选择。</p>
        <div class="callout">当前 native evaluation 只报告 Accuracy。历史 teacher-forced/NLL 脚本保留作审计，不属于默认评测协议。DDP 与 Linear cross-layer/chunk-major 实验代码已回退，生产路径为普通 FSDP + group 1。</div>
      </section>
    </div>
  </div>

  <section id="inventory">
    <div class="inventory-head"><h2>完整文件索引</h2><span id="visible-count"></span></div>
    <div class="toolbar">
      <input id="query" type="search" placeholder="搜索路径、用途或标签…" autocomplete="off">
      <select id="scope"><option value="all">全部 scope</option><option value="project">project</option><option value="upstream">upstream</option></select>
      <select id="status"><option value="all">全部状态</option><option value="active">active</option><option value="reference">reference</option><option value="retired">retired</option></select>
      <select id="category"><option value="all">全部类别</option></select>
    </div>
    <div id="cards" class="cards"></div><div id="empty" class="empty">没有匹配文件。</div>
  </section>
  <footer>Generated {html.escape(generated)} by <a href="scripts/generate_videochat3_code_map.py">scripts/generate_videochat3_code_map.py</a>. Relative links assume this HTML remains at repository root.</footer>
</main>
<script id="inventory-data" type="application/json">{data_json}</script>
<script>
  const rows=JSON.parse(document.getElementById('inventory-data').textContent);
  const cards=document.getElementById('cards'), query=document.getElementById('query'), scope=document.getElementById('scope'), status=document.getElementById('status'), category=document.getElementById('category');
  [...new Set(rows.map(r=>r.category))].sort().forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;category.appendChild(o);}});
  const esc=s=>s.replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  function render(){{
    const q=query.value.trim().toLowerCase();
    const filtered=rows.filter(r=>(scope.value==='all'||r.scope===scope.value)&&(status.value==='all'||r.status===status.value)&&(category.value==='all'||r.category===category.value)&&(!q||`${{r.path}} ${{r.description}} ${{r.category}} ${{r.scope}} ${{r.status}}`.toLowerCase().includes(q)));
    cards.innerHTML=filtered.map(r=>`<article class="card" data-status="${{esc(r.status)}}"><div class="card-top"><a class="file-link" href="${{esc(r.href)}}">${{esc(r.path)}}</a></div><p>${{esc(r.description)}}</p><div class="badges"><span class="badge ${{esc(r.status)}}">${{esc(r.status)}}</span><span class="badge ${{esc(r.scope)}}">${{esc(r.scope)}}</span><span class="badge">${{esc(r.category)}}</span></div><div class="meta">${{r.lines===null?'binary/unknown':r.lines.toLocaleString()+' lines'}} · ${{r.bytes.toLocaleString()}} bytes</div></article>`).join('');
    document.getElementById('visible-count').textContent=`显示 ${{filtered.length}} / ${{rows.length}}`;
    document.getElementById('empty').style.display=filtered.length?'none':'block';
  }}
  [query,scope,status,category].forEach(el=>el.addEventListener(el===query?'input':'change',render)); render();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(
        f"wrote {output.relative_to(ROOT)}: {len(rows)} files "
        f"({project_count} project, {active_count} active)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    render(output)


if __name__ == "__main__":
    main()
