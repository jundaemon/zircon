from dataclasses import dataclass

import torch


def get_weights(path: str, in_: int, outs: list[int]) -> dict[str, torch.Tensor]:
    dims = [in_] + outs
    weights = {}

    with open(path, "rb") as file:
        for i in range(len(outs)):
            weight_bytes = bytearray(file.read(dims[i + 1] * dims[i] * 4))
            weight = torch.frombuffer(weight_bytes, dtype=torch.float32)
            weight = weight.reshape((dims[i + 1], dims[i]))

            bias_bytes = bytearray(file.read(dims[i + 1] * 4))
            bias = torch.frombuffer(bias_bytes, dtype=torch.float32)

            weights[f"layer_{i}.weight"] = weight
            weights[f"layer_{i}.bias"] = bias

    return weights


def get_losses(path: str) -> torch.Tensor:
    with open(path, "r") as file:
        lines = file.readlines()
        losses = torch.empty(len(lines), dtype=torch.float32)
        for i, line in enumerate(lines):
            losses[i] = float(line.strip())

    return losses
