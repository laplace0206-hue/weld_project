from __future__ import annotations

import contextlib
from typing import Any

import torch

from .ultralytics_loss import CustomV8DetectionLoss
from ..models.ultralytics_modules import LiteFPNRefine, WeldAdaptiveBlock, WeldEMA


_REGISTERED_LOSS_TYPE = "ciou"


def register_custom_ultralytics_modules(loss_type: str = "ciou") -> None:
    import ast
    from ultralytics.nn import tasks

    global _REGISTERED_LOSS_TYPE
    _REGISTERED_LOSS_TYPE = loss_type.lower()

    for name, module in {
        "WeldAdaptiveBlock": WeldAdaptiveBlock,
        "WeldEMA": WeldEMA,
        "LiteFPNRefine": LiteFPNRefine,
    }.items():
        setattr(tasks, name, module)

    tasks.CustomV8DetectionLoss = CustomV8DetectionLoss

    def custom_init_criterion(self):
        if getattr(self, "end2end", False):
            return tasks.E2ELoss(self)
        if _REGISTERED_LOSS_TYPE == "focal_eiou":
            return CustomV8DetectionLoss(self)
        return tasks.v8DetectionLoss(self)

    tasks.DetectionModel.init_criterion = custom_init_criterion

    if getattr(tasks.parse_model, "__name__", "") == "custom_parse_model":
        return

    def custom_parse_model(d: dict[str, Any], ch: int, verbose: bool = True):
        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))
        scale = d.get("scale")

        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                tasks.LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            tasks.Conv.default_act = eval(act)
            if verbose:
                tasks.LOGGER.info(f"{tasks.colorstr('activation:')} {act}")

        if verbose:
            tasks.LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch_list = [ch]
        layers, save = [], []
        base_modules = frozenset(
            {
                tasks.Conv,
                tasks.C2f,
                tasks.SPPF,
                tasks.C3,
                WeldAdaptiveBlock,
                WeldEMA,
                LiteFPNRefine,
            }
        )
        repeat_modules = frozenset({tasks.C2f, tasks.C3})

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else tasks.__dict__[m]
            )
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError, SyntaxError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch_list[f], args[0]
                if c2 != nc:
                    c2 = tasks.make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
            elif m is torch.nn.BatchNorm2d:
                c2 = ch_list[f]
                args = [c2]
            elif m is tasks.Concat:
                c2 = sum(ch_list[x] for x in f)
            elif m in {tasks.Detect, tasks.Segment, tasks.Pose, tasks.OBB}:
                c2 = None
                args.extend([reg_max, end2end, [ch_list[x] for x in f]])
                m.legacy = legacy
            else:
                c2 = ch_list[f]

            module = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            module.np = sum(x.numel() for x in module.parameters())
            module.i, module.f, module.type = i, f, str(m)[8:-2].replace("__main__.", "")
            if verbose:
                tasks.LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{module.np:10.0f}  {module.type:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(module)
            if i == 0:
                ch_list = []
            ch_list.append(c2 if c2 is not None else args[0])
        return torch.nn.Sequential(*layers), sorted(save)

    tasks.parse_model = custom_parse_model
