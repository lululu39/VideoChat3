#!/usr/bin/env python3
"""TimeLens TAS training using the exact exported HF graph and ordinary DDP.

One video per microbatch, full state BPTT, chunk checkpointing, frozen BF16 LM,
FP32 trainable slow/master parameters and AdamW state. No exclusive GPU watchdog.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor
from xtuner.v1.model.compose.videochat3.hf_tas_export import prepare_tas_assets, refresh_tas_code


class TimeLensDataset(Dataset):
    def __init__(self, manifest, checkpoint, max_frames=448, limit=None, native_template=False):
        self.spec = next(iter(json.loads(Path(manifest).read_text()).values()))
        self.rows = [json.loads(line) for line in Path(self.spec["anno_path"]).read_text().splitlines()]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.max_frames = max_frames
        self.native_template = native_template
        self.processor.video_processor.video_max_total_pixels = 14680064
        ending = "<|im_end|>" if native_template else "<|im_end|>\n"
        self.answer_lengths = [len(self.processor.tokenizer.encode(
            row["messages"][1]["content"]+ending, add_special_tokens=False)) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from decord import VideoReader, cpu
        row = self.rows[index]
        content = row["messages"][0]["content"]
        item = next(x for x in content if x["type"] == "video_url")
        path = Path(self.spec["media_root"])/item["video_url"]["url"]
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        meta = dict(item["video_metadata"])
        # Same min/max/multiple and uniform sampling as the v26 native loader.
        count = min(min(max(int(meta["total_num_frames"]/meta["fps"]*2), 64), self.max_frames), len(reader))
        count = count//4*4
        if count == 0:
            raise ValueError(f"Video has fewer than four frames: {path}")
        indices = np.linspace(0, len(reader)-1, count).round().astype(int)
        frames = reader.get_batch(indices).asnumpy()
        meta.update(total_num_frames=len(reader), duration=len(reader)/meta["fps"], frames_indices=indices.tolist())
        question = next(x["text"] for x in content if x["type"] == "text").replace("<VIDEO_CONTEXT>", "").strip()
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, videos=[frames], video_metadata=[meta],
                                do_sample_frames=False, return_tensors="pt",
                                size={"shortest_edge": 784, "longest_edge": 50176})
        answer = row["messages"][1]["content"]
        if self.native_template:
            from xtuner.v1.data_proto.messages import ChatMessages
            from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
            n = int((inputs["input_ids"] == self.processor.video_token_id).sum())
            block = self.processor.vision_start_token + self.processor.video_token*n + self.processor.vision_end_token
            original_question = next(x["text"] for x in content if x["type"] == "text")
            messages = ChatMessages(messages=[
                {"role": "user", "content": original_question.replace("<VIDEO_CONTEXT>", block)},
                {"role": "assistant", "content": answer},
            ])
            tokenized = messages.tokenize(self.processor.tokenizer, CHAT_TEMPLATE_MAP["videochat3"])
            inputs["input_ids"] = torch.tensor([tokenized["input_ids"]])
            inputs["labels"] = torch.tensor([tokenized["labels"]])
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            assert int((inputs["labels"] != -100).sum()) == self.answer_lengths[index]
            inputs["sample_id"] = row["id"]
            if inputs["input_ids"].shape[-1] > 4096:
                raise ValueError("Expanded TAS sequence exceeds 4K; never truncate the reference sample")
            return dict(inputs)
        suffix = self.processor.tokenizer.encode(answer+"<|im_end|>\n", add_special_tokens=False)
        suffix = torch.tensor([suffix], dtype=torch.long)
        prefix = inputs["input_ids"]
        inputs["input_ids"] = torch.cat((prefix, suffix), -1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["labels"] = torch.cat((torch.full_like(prefix, -100), suffix), -1)
        if inputs["input_ids"].shape[-1] > 4096:
            raise ValueError("TAS training sequence exceeds 4K; do not silently truncate video inputs")
        inputs["sample_id"] = row["id"]
        return dict(inputs)


def single_collate(items):
    assert len(items) == 1
    return items[0]


def wandb_metrics(stats):
    """Use the existing XTuner dashboard keys; retain raw loss in local JSONL."""
    payload = {k: v for k, v in stats.items() if k != "loss"}
    payload.update({
        "loss/total_loss": stats["loss"],
        "loss/reduced_llm_loss": stats["loss"],
        "time/step_time": stats["seconds"],
        "memory/max_memory_GB": stats["max_allocated_gb"] * 1e9 / 1024**3,
        "memory/reserved_memory_GB": stats["max_reserved_gb"] * 1e9 / 1024**3,
        "lr_groups/vit": stats["lr"],
        "lr_groups/tas": stats["lr"],
        "lr_groups/projector": stats["lr"],
    })
    return payload


class TrainingLoss(nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm

    def forward(self, labels, **inputs):
        hidden = self.vlm.model(**inputs, use_cache=False).last_hidden_state
        # Native next-token SFT CE, restricted to supervised answer positions.
        # Avoid allocating vocabulary logits for the entire visual prefix.
        target = labels[:, 1:]
        keep = target != -100
        logits = self.vlm.lm_head(hidden[:, :-1][keep])
        return nn.functional.cross_entropy(logits.float(), target[keep])


def checkpoint_save(vlm, optimizer, step, args, source, stats):
    dest = args.output/f"hf-{step}"
    dest.mkdir(parents=True, exist_ok=False)
    prepare_tas_assets(source, dest)
    state = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in vlm.state_dict().items()}
    if vlm.lm_head.weight is vlm.model.language_model.embed_tokens.weight:
        state.pop("lm_head.weight", None)
    vlm.save_pretrained(dest, state_dict=state, max_shard_size="4GB")
    refresh_tas_code(dest)
    config_path = dest/"config.json"
    exported = json.loads(config_path.read_text())
    exported["torch_dtype"] = "bfloat16"
    if "dtype" in exported:
        exported["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(exported, indent=2)+"\n")
    # Keep full FP32 trainable state for strict resume/diagnostics; frozen tensors
    # remain in HF and are bitwise identical to initialization.
    trainable = {k: v.detach().cpu() for k, v in vlm.named_parameters() if v.requires_grad}
    torch.save(dict(step=step, optimizer=optimizer.state_dict(), trainable=trainable,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), stats=stats),
               args.output/"resume.tmp")
    (args.output/"resume.tmp").replace(args.output/"resume.pt")
    (args.output/"latest_checkpoint.txt").write_text(str(dest.resolve())+"\n")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/mnt/localssd/VideoChat3/VideoChat3-4B-TAS-init"))
    p.add_argument("--manifest", type=Path, default=Path("/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-ratio", type=float, default=.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=448)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--wandb-name")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reference-plan", type=Path,
                   help="v19/v29 exact pack/step plan; enables beta2=.95, native labels and square loss")
    args = p.parse_args()
    rank, local, world = (int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    assert args.global_batch % world == 0
    accumulation = args.global_batch//world
    plan = json.loads(args.reference_plan.read_text()) if args.reference_plan else None
    if plan:
        assert plan["world"] == world and plan["global_packs"] == args.global_batch
        assert plan["seed"] == args.seed, "Reference sampler seed differs"
        assert args.limit is None, "A subset would invalidate the reference plan"
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True,
              dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(local)
    model.freeze_for_tas_training()
    model.model.vision_tower.tas.collect_metrics = True
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=0., betas=(.9, .95 if plan else .999),
                                 eps=1e-8, fused=False if plan else True, foreach=False if plan else None)
    dataset = TimeLensDataset(args.manifest, args.checkpoint, args.max_frames, args.limit,
                              native_template=bool(plan))
    micro_plan = None
    if plan:
        micro_plan = [(step, index, i == len(rows[rank])-1)
                      for step, rows in enumerate(plan["steps"])
                      for i, index in enumerate(rows[rank])]
        sampler = [index for _, index, _ in micro_plan]
        denominators = [sum(math.sqrt(dataset.answer_lengths[i]) for rows in step for i in rows)
                        for step in plan["steps"]]
    else:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.workers,
                        collate_fn=single_collate, pin_memory=True,
                        persistent_workers=args.workers > 0)
    total_steps = plan["total_steps"] if plan else math.ceil(len(loader)/accumulation)
    warmup = int(total_steps*args.warmup_ratio) if plan else max(1, math.ceil(total_steps*args.warmup_ratio))
    first_step = 0
    if args.resume:
        saved = torch.load(args.output/"resume.pt", map_location="cpu", weights_only=False)
        state = model.state_dict()
        for k, value in saved["trainable"].items():
            state[k].copy_(value)
        optimizer.load_state_dict(saved["optimizer"])
        first_step = saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    wrapped = TrainingLoss(model)
    if world > 1:
        wrapped = DDP(wrapped, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    run = None
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(world_size=world, samples=len(dataset), total_steps=total_steps,
                  optimizer_betas=[.9, .95 if plan else .999], optimizer_eps=1e-8,
                  optimizer_foreach=False if plan else None, optimizer_fused=not bool(plan),
                  loss_reduction="square" if plan else "sample", native_training_template=bool(plan),
                  warmup_steps=warmup,
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  uv_lock_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/"uv.lock").read_bytes()).hexdigest(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if plan:
        config.update(reference_plan_sha256=hashlib.sha256(args.reference_plan.read_bytes()).hexdigest(),
                      reference_pack_recipe={k: v for k, v in plan.items() if k not in ("steps", "rank_pack_order")})
    if rank == 0:
        (args.output/"training_config.json").write_text(json.dumps(config, indent=2)+"\n")
        print(json.dumps(config), flush=True)
        if args.wandb_name:
            import wandb
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
            public = os.environ.get("WANDB_PUBLIC_API_KEY")
            if public:
                os.environ["WANDB_API_KEY"] = public
            else:
                os.environ.pop("WANDB_API_KEY", None)
            run = wandb.init(entity="LVSM-Experiment", project="videochat3", name=args.wandb_name,
                             id=args.wandb_name, resume="allow", config=config, dir=str(args.output),
                             group="videochat3-lact-stage3-ve", job_type="train",
                             tags=["tas", "videochat3-4b", "timelens-100k-visual-random12624", "vit-tas-projector"])
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    losses, terminal_gates = [], []
    completed = first_step
    for batch_idx, batch in enumerate(loader):
        step = micro_plan[batch_idx][0] if plan else batch_idx//accumulation
        if step < first_step:
            continue
        if args.max_steps is not None and step >= args.max_steps:
            break
        sample_id = batch.pop("sample_id")
        batch = {k: v.to(local, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Keep timestamps FP32 even while pixels use BF16 autocast.
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)
        last_micro = micro_plan[batch_idx][2] if plan else (batch_idx+1)%accumulation == 0 or batch_idx+1 == len(loader)
        if plan:
            answer_length = int((batch["labels"][:, 1:] != -100).sum())
            assert answer_length == dataset.answer_lengths[micro_plan[batch_idx][1]]
            loss_weight = world*math.sqrt(answer_length)/denominators[step]
        else:
            group_size = min(accumulation, len(loader)-step*accumulation)
            loss_weight = 1/group_size
        with wrapped.no_sync() if world > 1 and not last_micro else nullcontext():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(**batch)
            # Snapshot before backward checkpoint recomputation overwrites this
            # diagnostic with an earlier chunk. Values refer to terminal writes.
            gate = model.model.vision_tower.tas.last_gate
            terminal_gates.append(torch.stack((gate.mean(), gate.min(), gate.max())))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {sample_id}")
            (loss*loss_weight).backward()
        losses.append(loss.detach()*loss_weight)
        if not last_micro:
            continue
        norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        factor = step/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))
        lr = args.lr*factor if step < warmup else args.min_lr+(args.lr-args.min_lr)*factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        avg = torch.stack(losses).sum()
        if world > 1:
            dist.all_reduce(avg); avg /= world
        peaks = torch.tensor([torch.cuda.max_memory_allocated()/1e9, torch.cuda.max_memory_reserved()/1e9], device=local)
        if world > 1:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        completed = step+1
        stats = dict(step=completed, loss=avg.item(), grad_norm=norm.item(), lr=lr,
                     seconds=time.monotonic()-start, max_allocated_gb=peaks[0].item(), max_reserved_gb=peaks[1].item())
        if plan:
            stats["runtime_info/global_examples_this_step"] = sum(map(len, plan["steps"][step]))
            stats["runtime_info/global_examples_consumed"] = sum(sum(map(len, s)) for s in plan["steps"][:completed])
        gate_stats = torch.stack(terminal_gates).mean(0)
        if world > 1:
            dist.all_reduce(gate_stats); gate_stats /= world
        stats.update(terminal_gate_mean=gate_stats[0].item(),
                     terminal_gate_sample_min_mean=gate_stats[1].item(),
                     terminal_gate_sample_max_mean=gate_stats[2].item())
        if rank == 0:
            print(json.dumps(stats), flush=True)
            with (args.output/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(stats)+"\n")
            if run is not None:
                run.log(wandb_metrics(stats), step=completed)
        save = not args.no_save and (completed%args.save_every == 0 or completed == total_steps or completed == args.max_steps)
        if save:
            if rank == 0:
                checkpoint_save(model, optimizer, completed, args, args.checkpoint, stats)
            if world > 1:
                dist.barrier()
        losses, terminal_gates = [], []; start = time.monotonic()
    if rank == 0:
        (args.output/"training_complete.json").write_text(json.dumps(dict(completed_steps=completed, total_steps=total_steps)) + "\n")
        if run is not None:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
