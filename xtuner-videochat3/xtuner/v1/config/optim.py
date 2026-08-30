from abc import abstractmethod
from typing import Literal, Optional, Tuple

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0
    skip_grad_norm_threshold: Annotated[
        float | None, Parameter(help="Gradient norm threshold for skipping optimizer step.")
    ] = None

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8
    foreach: Annotated[Optional[bool], Parameter(help="Use foreach implementation for AdamW")] = None

    def build(self, params):
        return torch.optim.AdamW(
            params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, foreach=self.foreach
        )


class VisionAdamWConfig(AdamWConfig):
    """AdamW config with separate learning rates for multimodal parameter groups."""
    vit_lr: Annotated[Optional[float], Parameter(help="Learning rate for ViT. If None, uses `lr`.")] = None
    lact_lr: Annotated[
        Optional[float], Parameter(help="Learning rate for LACT fast-weight parameters. If None, uses `vit_lr`.")
    ] = None
    lact_gate_lr: Annotated[
        Optional[float],
        Parameter(help="Learning rate for LACT memory gates. If None, gates stay in the LACT fast-weight group."),
    ] = None
    projector_lr: Annotated[Optional[float], Parameter(help="Learning rate for Projector. If None, uses `lr`.")] = None
    llm_lr: Annotated[Optional[float], Parameter(help="Learning rate for LLM. If None, uses `lr`.")] = None

    def build_with_param_groups(
        self,
        vit_params,
        projector_params,
        llm_params,
        lact_params=None,
        lact_gate_params=None,
    ):
        """Build the optimizer while preserving named learning-rate groups."""
        param_groups = []

        vit_lr = self.vit_lr if self.vit_lr is not None else self.lr
        lact_lr = self.lact_lr if self.lact_lr is not None else vit_lr
        lact_gate_lr = self.lact_gate_lr if self.lact_gate_lr is not None else lact_lr
        projector_lr = self.projector_lr if self.projector_lr is not None else self.lr
        llm_lr = self.llm_lr if self.llm_lr is not None else self.lr

        if vit_params:
            param_groups.append({"params": vit_params, "lr": vit_lr, "name": "vit"})
        if lact_params:
            param_groups.append({"params": lact_params, "lr": lact_lr, "name": "lact_fw"})
        if lact_gate_params:
            param_groups.append({"params": lact_gate_params, "lr": lact_gate_lr, "name": "lact_gate"})
        if projector_params:
            param_groups.append({"params": projector_params, "lr": projector_lr, "name": "projector"})
        if llm_params:
            param_groups.append({"params": llm_params, "lr": llm_lr, "name": "llm"})

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            foreach=self.foreach,
        )


class LRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr_type: Annotated[Literal["cosine", "linear", "constant"], Parameter(help="Type of learning rate schedule")] = (
        "constant"
    )
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6
    lr_min_ratio: Annotated[
        Optional[float],
        Parameter(help="Optional minimum LR as a ratio of each parameter group's initial LR."),
    ] = None
