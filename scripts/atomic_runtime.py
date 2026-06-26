#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import sys
from typing import Any


def passed(atom_id: str, summary: str) -> dict[str, Any]:
    return {"atomId": atom_id, "status": "passed", "summary": summary}


def failed(atom_id: str, error: Exception) -> dict[str, Any]:
    return {"atomId": atom_id, "status": "failed", "error": str(error)}


def tensor_summary(tensor: Any) -> str:
    return f"shape={list(tensor.shape)} dtype={str(tensor.dtype).replace('torch.', '')} finite={bool(tensor.isfinite().all())}"


def custom_module(code: str, nn: Any) -> Any:
    allowed = {
        "Linear", "RMSNorm", "LayerNorm", "Dropout", "Identity", "ReLU",
        "ReLU6", "GELU", "SiLU", "Sigmoid", "Tanh", "Softplus", "ELU",
        "CELU", "SELU", "LeakyReLU", "PReLU", "Mish", "Hardtanh",
    }
    try:
        expression = ast.parse(code, mode="eval").body
    except SyntaxError as error:
        raise RuntimeError(f"invalid custom PyTorch module: {error.msg}") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute):
        raise RuntimeError("custom PyTorch code must be one nn.Module constructor")
    if not isinstance(expression.func.value, ast.Name) or expression.func.value.id != "nn" or expression.func.attr not in allowed:
        raise RuntimeError(f"unsupported custom PyTorch module: {code}")
    try:
        args = [ast.literal_eval(argument) for argument in expression.args]
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords if keyword.arg}
    except (ValueError, TypeError) as error:
        raise RuntimeError("custom PyTorch arguments must be literal values") from error
    return getattr(nn, expression.func.attr)(*args, **kwargs)


def run_model(graph: dict[str, Any], supplied_token_ids: list[int] | None = None) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    config = graph["config"]
    hidden_size = int(config["hiddenSize"])
    query_heads = int(config["queryHeads"])
    key_value_heads = int(config["keyValueHeads"])
    head_dim = int(config["headDim"])
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    by_id = {node["id"]: node for node in nodes}
    values: dict[tuple[str, str], Any] = {}
    results: list[dict[str, Any]] = []
    failed_nodes: set[str] = set()
    first_failure: tuple[str, str] | None = None
    model_output: dict[str, Any] | None = None
    last_tensor = None

    token_tensor = None
    if supplied_token_ids is not None:
        if not supplied_token_ids or not all(isinstance(token_id, int) and token_id >= 0 for token_id in supplied_token_ids):
            raise ValueError("tokenIds must be a non-empty list of non-negative integers")
        token_tensor = torch.tensor([supplied_token_ids], dtype=torch.long)

    # Agent plans can add a missing dependency after its consumer in the node
    # array. Execution follows the graph, never creation order.
    node_index = {node["id"]: index for index, node in enumerate(nodes)}
    indegree = {node["id"]: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        source_id, target_id = edge["source"], edge["target"]
        if source_id in outgoing and target_id in indegree:
            outgoing[source_id].append(target_id)
            indegree[target_id] += 1
    queue = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=node_index.get)
    ordered_ids: list[str] = []
    while queue:
        node_id = queue.pop(0)
        ordered_ids.append(node_id)
        for target_id in sorted(outgoing[node_id], key=node_index.get):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                queue.append(target_id)
                queue.sort(key=node_index.get)
    if len(ordered_ids) != len(nodes):
        raise RuntimeError("model graph contains a cycle")
    ordered = [by_id[node_id] for node_id in ordered_ids]

    def connected_inputs(node_id: str) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for edge in edges:
            if edge["target"] != node_id:
                continue
            source_port = str(edge.get("sourcePort", "output"))
            value = values.get((edge["source"], source_port))
            if value is None:
                value = values.get((edge["source"], "output"))
            if value is not None:
                found[str(edge.get("targetPort", "hidden"))] = value
        return found

    def require(inputs: dict[str, Any], *ports: str) -> None:
        missing = [port for port in ports if port not in inputs]
        if missing:
            raise RuntimeError(f"missing connected {', '.join(missing)} input")

    def apply_rope(tensor: Any, base: float) -> Any:
        sequence, dimension = tensor.shape[-2], tensor.shape[-1]
        inverse = 1.0 / (base ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension))
        angles = torch.outer(torch.arange(sequence, dtype=torch.float32), inverse)
        cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(tensor.dtype)
        sine = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(tensor.dtype)
        rotated = torch.stack((-tensor[..., 1::2], tensor[..., ::2]), dim=-1).flatten(-2)
        return tensor * cosine + rotated * sine

    def swiglu(source: Any, intermediate: int) -> Any:
        gate = nn.Linear(source.shape[-1], intermediate, bias=False)
        up = nn.Linear(source.shape[-1], intermediate, bias=False)
        down = nn.Linear(intermediate, source.shape[-1], bias=False)
        return down(F.silu(gate(source)) * up(source))

    def execute_semantic(node: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        atom = str(node.get("atomId"))
        settings = {**config, **node.get("attributes", {})}
        activation_modules = {
            "relu": lambda: nn.ReLU(inplace=bool(settings.get("inplace", False))),
            "relu6": lambda: nn.ReLU6(inplace=bool(settings.get("inplace", False))),
            "gelu": lambda: nn.GELU(approximate=str(settings.get("approximate", "none"))),
            "silu": lambda: nn.SiLU(inplace=bool(settings.get("inplace", False))),
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "softplus": lambda: nn.Softplus(beta=float(settings.get("beta", 1)), threshold=float(settings.get("threshold", 20))),
            "softsign": nn.Softsign,
            "elu": lambda: nn.ELU(alpha=float(settings.get("alpha", 1)), inplace=bool(settings.get("inplace", False))),
            "celu": lambda: nn.CELU(alpha=float(settings.get("alpha", 1)), inplace=bool(settings.get("inplace", False))),
            "selu": lambda: nn.SELU(inplace=bool(settings.get("inplace", False))),
            "leaky-relu": lambda: nn.LeakyReLU(negative_slope=float(settings.get("negativeSlope", 0.01)), inplace=bool(settings.get("inplace", False))),
            "prelu": lambda: nn.PReLU(num_parameters=int(settings.get("numParameters", 1)), init=float(settings.get("init", 0.25))),
            "rrelu": lambda: nn.RReLU(lower=float(settings.get("lower", 0.125)), upper=float(settings.get("upper", 1 / 3)), inplace=bool(settings.get("inplace", False))),
            "mish": lambda: nn.Mish(inplace=bool(settings.get("inplace", False))),
            "hardsigmoid": lambda: nn.Hardsigmoid(inplace=bool(settings.get("inplace", False))),
            "hardswish": lambda: nn.Hardswish(inplace=bool(settings.get("inplace", False))),
            "hardtanh": lambda: nn.Hardtanh(min_val=float(settings.get("minValue", -1)), max_val=float(settings.get("maxValue", 1)), inplace=bool(settings.get("inplace", False))),
            "hardshrink": lambda: nn.Hardshrink(lambd=float(settings.get("lambd", 0.5))),
            "softshrink": lambda: nn.Softshrink(lambd=float(settings.get("lambd", 0.5))),
            "tanhshrink": nn.Tanhshrink,
            "threshold": lambda: nn.Threshold(threshold=float(settings.get("threshold", 0)), value=float(settings.get("value", 0)), inplace=bool(settings.get("inplace", False))),
            "logsigmoid": nn.LogSigmoid,
            "glu": lambda: nn.GLU(dim=int(settings.get("dim", -1))),
        }
        if atom in activation_modules:
            require(inputs, "hidden")
            return {"output": activation_modules[atom]()(inputs["hidden"])}
        if atom == "token-embedding":
            require(inputs, "tokenIds")
            module = nn.Embedding(int(settings["vocabSize"]), int(settings["hiddenSize"]))
            return {"output": module(inputs["tokenIds"])}
        if atom == "learned-position-embedding":
            require(inputs, "tokenIds")
            token_ids = inputs["tokenIds"]
            positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
            module = nn.Embedding(int(settings["maxPositions"]), int(settings["hiddenSize"]))
            return {"output": module(positions).unsqueeze(0).expand(token_ids.shape[0], -1, -1)}
        if atom in {"sinusoidal-position-encoding", "position-ramp"}:
            require(inputs, "tokenIds")
            token_ids = inputs["tokenIds"]
            sequence = token_ids.shape[-1]
            if atom == "position-ramp":
                ramp = torch.linspace(0, 1, sequence, device=token_ids.device)
                return {"output": ramp.view(1, -1, 1).expand(token_ids.shape[0], -1, hidden_size)}
            position = torch.arange(sequence, device=token_ids.device, dtype=torch.float32).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, hidden_size, 2, device=token_ids.device, dtype=torch.float32) * (-torch.log(torch.tensor(float(settings.get("base", 10000)), device=token_ids.device)) / hidden_size))
            encoding = torch.zeros(sequence, hidden_size, device=token_ids.device)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            return {"output": encoding.unsqueeze(0).expand(token_ids.shape[0], -1, -1)}
        if atom in {"rms-norm", "layer-norm"}:
            require(inputs, "hidden")
            epsilon = float(settings.get("epsilon", 1e-6))
            module = nn.RMSNorm(hidden_size, eps=epsilon) if atom == "rms-norm" else nn.LayerNorm(hidden_size, eps=epsilon)
            return {"output": module(inputs["hidden"])}
        if atom in {"scale-norm", "l2-normalization", "group-norm", "batch-norm-1d", "instance-norm-1d", "mean-centering", "standardization", "unit-rms"}:
            require(inputs, "hidden")
            source = inputs["hidden"]
            epsilon = float(settings.get("epsilon", 1e-6))
            if atom == "scale-norm":
                output = source * (float(settings.get("scale", 1)) / source.norm(dim=-1, keepdim=True).clamp_min(epsilon))
            elif atom == "l2-normalization":
                output = F.normalize(source, p=2, dim=-1, eps=epsilon)
            elif atom == "group-norm":
                output = nn.GroupNorm(int(settings.get("numGroups", 8)), hidden_size, eps=epsilon)(source.transpose(1, 2)).transpose(1, 2)
            elif atom == "batch-norm-1d":
                output = nn.BatchNorm1d(hidden_size, eps=epsilon, momentum=float(settings.get("momentum", 0.1)))(source.transpose(1, 2)).transpose(1, 2)
            elif atom == "instance-norm-1d":
                output = nn.InstanceNorm1d(hidden_size, eps=epsilon, affine=bool(settings.get("affine", True)))(source.transpose(1, 2)).transpose(1, 2)
            elif atom == "mean-centering":
                output = source - source.mean(dim=-1, keepdim=True)
            elif atom == "standardization":
                output = (source - source.mean(dim=-1, keepdim=True)) / source.std(dim=-1, keepdim=True, unbiased=False).clamp_min(epsilon)
            else:
                output = source * torch.rsqrt(source.pow(2).mean(dim=-1, keepdim=True) + epsilon)
            return {"output": output}
        if atom == "qkv-projection":
            require(inputs, "hidden")
            source = inputs["hidden"]
            bias = bool(settings.get("bias", False))
            return {
                "q": nn.Linear(hidden_size, query_heads * head_dim, bias=bias)(source),
                "k": nn.Linear(hidden_size, key_value_heads * head_dim, bias=bias)(source),
                "v": nn.Linear(hidden_size, key_value_heads * head_dim, bias=bias)(source),
            }
        if atom in {"query-projection", "key-projection", "value-projection"}:
            require(inputs, "hidden")
            source = inputs["hidden"]
            bias = bool(settings.get("bias", False))
            if atom == "query-projection":
                return {"q": nn.Linear(hidden_size, query_heads * head_dim, bias=bias)(source)}
            output_size = key_value_heads * head_dim
            port = "k" if atom == "key-projection" else "v"
            return {port: nn.Linear(hidden_size, output_size, bias=bias)(source)}
        if atom == "attention-head-layout":
            require(inputs, "q", "k", "v")
            batch, sequence, _ = inputs["q"].shape
            return {
                "qHeads": inputs["q"].view(batch, sequence, query_heads, head_dim).transpose(1, 2),
                "kHeads": inputs["k"].view(batch, sequence, key_value_heads, head_dim).transpose(1, 2),
                "vHeads": inputs["v"].view(batch, sequence, key_value_heads, head_dim).transpose(1, 2),
            }
        if atom == "qk-normalization":
            require(inputs, "q", "k")
            epsilon = float(settings.get("epsilon", 1e-6))
            return {"q": nn.RMSNorm(head_dim, eps=epsilon)(inputs["q"]), "k": nn.RMSNorm(head_dim, eps=epsilon)(inputs["k"])}
        if atom == "rope":
            require(inputs, "q", "k")
            base = float(settings.get("base", 10000))
            return {"q": apply_rope(inputs["q"], base), "k": apply_rope(inputs["k"], base)}
        if atom in {"query-rope", "key-rope"}:
            port = "q" if atom == "query-rope" else "k"
            require(inputs, port)
            return {port: apply_rope(inputs[port], float(settings.get("base", 10000)))}
        if atom == "gqa-kv-expand":
            require(inputs, "k", "v")
            repeats = query_heads // key_value_heads
            return {"k": inputs["k"].repeat_interleave(repeats, dim=1), "v": inputs["v"].repeat_interleave(repeats, dim=1)}
        if atom == "causal-sdpa":
            require(inputs, "q", "k", "v")
            use_gqa = inputs["q"].shape[1] != inputs["k"].shape[1]
            return {"output": F.scaled_dot_product_attention(inputs["q"], inputs["k"], inputs["v"], is_causal=True, enable_gqa=use_gqa)}
        if atom == "eager-causal-attention":
            require(inputs, "q", "k", "v")
            query, key, value = inputs["q"], inputs["k"], inputs["v"]
            scores = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
            mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
            weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
            weights = F.dropout(weights, p=float(settings.get("dropout", 0)), training=False)
            return {"output": torch.matmul(weights, value)}
        if atom == "noncausal-sdpa":
            require(inputs, "q", "k", "v")
            use_gqa = inputs["q"].shape[1] != inputs["k"].shape[1]
            return {"output": F.scaled_dot_product_attention(inputs["q"], inputs["k"], inputs["v"], dropout_p=float(settings.get("dropout", 0)), is_causal=False, enable_gqa=use_gqa)}
        if atom == "attention-scores":
            require(inputs, "q", "k")
            return {"scores": torch.matmul(inputs["q"], inputs["k"].transpose(-2, -1)) * (inputs["q"].shape[-1] ** -0.5)}
        if atom == "attention-softmax":
            require(inputs, "scores")
            return {"output": torch.softmax(inputs["scores"], dim=int(settings.get("dimension", -1)))}
        if atom == "attention-dropout":
            require(inputs, "attention")
            return {"output": F.dropout(inputs["attention"], p=float(settings.get("probability", 0.1)), training=False)}
        if atom == "attention-value-mix":
            require(inputs, "weights", "v")
            return {"output": torch.matmul(inputs["weights"], inputs["v"])}
        if atom == "merge-attention-heads":
            require(inputs, "attention")
            batch, _, sequence, _ = inputs["attention"].shape
            return {"output": inputs["attention"].transpose(1, 2).contiguous().view(batch, sequence, query_heads * head_dim)}
        if atom == "attention-output-projection":
            require(inputs, "hidden")
            return {"output": nn.Linear(query_heads * head_dim, hidden_size, bias=bool(settings.get("bias", False)))(inputs["hidden"])}
        if atom == "image-channel-normalization":
            require(inputs, "image")
            deviation = max(float(settings.get("standardDeviation", 0.5)), 1e-6)
            return {"image": (inputs["image"] - float(settings.get("mean", 0.5))) / deviation}
        if atom == "image-patch-embedding":
            require(inputs, "image")
            patch_size = int(settings.get("patchSize", 16))
            patches = nn.Conv2d(int(settings.get("inputChannels", 3)), hidden_size, kernel_size=patch_size, stride=patch_size, bias=bool(settings.get("bias", True)))(inputs["image"])
            return {"output": patches.flatten(2).transpose(1, 2)}
        if atom == "video-channel-normalization":
            require(inputs, "video")
            deviation = max(float(settings.get("standardDeviation", 0.5)), 1e-6)
            return {"video": (inputs["video"] - float(settings.get("mean", 0.5))) / deviation}
        if atom == "video-tubelet-embedding":
            require(inputs, "video")
            tubelet_size = int(settings.get("tubeletSize", 2))
            patch_size = int(settings.get("patchSize", 16))
            tubelets = nn.Conv3d(int(settings.get("inputChannels", 3)), hidden_size, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size), bias=bool(settings.get("bias", True)))(inputs["video"])
            return {"output": tubelets.flatten(2).transpose(1, 2)}
        if atom == "audio-waveform-normalization":
            require(inputs, "audio")
            centered = inputs["audio"] - inputs["audio"].mean(dim=-1, keepdim=True)
            return {"audio": centered / centered.abs().amax(dim=-1, keepdim=True).clamp_min(float(settings.get("epsilon", 1e-6)))}
        if atom == "audio-preemphasis":
            require(inputs, "audio")
            audio = inputs["audio"]
            coefficient = float(settings.get("coefficient", 0.97))
            return {"audio": torch.cat((audio[..., :1], audio[..., 1:] - coefficient * audio[..., :-1]), dim=-1)}
        if atom == "audio-resample":
            require(inputs, "audio")
            source_rate = max(1, int(settings.get("sourceRate", 48000)))
            target_rate = max(1, int(settings.get("targetRate", 16000)))
            length = max(1, int(inputs["audio"].shape[-1] * target_rate / source_rate))
            return {"audio": F.interpolate(inputs["audio"], size=length, mode="linear", align_corners=False)}
        if atom == "audio-frame-embedding":
            require(inputs, "audio")
            frames = nn.Conv1d(int(settings.get("inputChannels", 1)), hidden_size, kernel_size=int(settings.get("frameSize", 400)), stride=int(settings.get("hopSize", 160)), bias=bool(settings.get("bias", True)))(inputs["audio"])
            return {"output": frames.transpose(1, 2)}
        if atom == "audio-vq-tokenizer":
            require(inputs, "audio")
            latents = nn.Conv1d(int(settings.get("inputChannels", 1)), hidden_size, kernel_size=int(settings.get("frameSize", 400)), stride=int(settings.get("hopSize", 160)))(inputs["audio"]).transpose(1, 2)
            codebook = torch.randn(int(settings.get("codebookSize", 1024)), hidden_size, device=latents.device) * float(settings.get("initialScale", 0.02))
            distances = latents.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=-1) - 2 * torch.matmul(latents, codebook.t())
            return {"tokenIds": distances.argmin(dim=-1)}
        if atom == "audio-codebook-embedding":
            require(inputs, "tokenIds")
            return {"output": nn.Embedding(int(settings.get("codebookSize", 1024)), hidden_size)(inputs["tokenIds"])}
        if atom == "audio-token-decoder":
            require(inputs, "hidden")
            decoded = nn.ConvTranspose1d(hidden_size, int(settings.get("outputChannels", 1)), kernel_size=int(settings.get("frameSize", 400)), stride=int(settings.get("hopSize", 160)), bias=bool(settings.get("bias", True)))(inputs["hidden"].transpose(1, 2))
            return {"audio": torch.tanh(decoded)}
        if atom in {"vision-patch-projection", "conditioning-projection", "image-latent-decoder", "video-latent-decoder", "audio-feature-projector"}:
            require(inputs, "hidden")
            return {"output": nn.Linear(hidden_size, hidden_size, bias=bool(settings.get("bias", True)))(inputs["hidden"])}
        if atom == "modality-type-embedding":
            require(inputs, "hidden")
            embedding = torch.randn(1, 1, hidden_size, device=inputs["hidden"].device) * float(settings.get("initialScale", 0.02))
            return {"output": inputs["hidden"] + embedding}
        if atom == "adaptive-conditioning":
            require(inputs, "content", "conditioning")
            joined = torch.cat((inputs["content"], inputs["conditioning"]), dim=-1)
            return {"output": nn.Linear(hidden_size * 2, hidden_size, bias=bool(settings.get("bias", True)))(joined)}
        if atom == "temporal-depthwise-convolution":
            require(inputs, "hidden")
            kernel = int(settings.get("kernelSize", 3))
            source = inputs["hidden"].transpose(1, 2)
            output = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel, padding=kernel // 2, groups=hidden_size)(source)
            return {"output": output.transpose(1, 2)}
        if atom == "audio-temporal-convolution":
            require(inputs, "hidden")
            kernel = int(settings.get("kernelSize", 5))
            output = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel, padding=kernel // 2, groups=hidden_size)(inputs["hidden"].transpose(1, 2))
            return {"output": output.transpose(1, 2)}
        if atom == "audio-position-embedding":
            require(inputs, "hidden")
            positions = torch.randn(1, int(settings.get("maxFrames", 2048)), hidden_size, device=inputs["hidden"].device) * float(settings.get("initialScale", 0.02))
            return {"output": inputs["hidden"] + positions[:, :inputs["hidden"].shape[1]]}
        if atom == "audio-mean-pooling":
            require(inputs, "hidden")
            return {"output": inputs["hidden"].mean(dim=1, keepdim=True)}
        if atom == "audio-ctc-head":
            require(inputs, "hidden")
            return {"logits": nn.Linear(hidden_size, int(settings.get("vocabSize", 1024)), bias=bool(settings.get("bias", True)))(inputs["hidden"])}
        if atom == "latent-denoiser":
            require(inputs, "hidden")
            source = inputs["hidden"]
            intermediate = int(settings.get("intermediateSize", hidden_size * 4))
            bias = bool(settings.get("bias", True))
            denoised = nn.Linear(intermediate, hidden_size, bias=bias)(F.gelu(nn.Linear(hidden_size, intermediate, bias=bias)(source)))
            return {"output": source + denoised}
        if atom == "residual-add":
            require(inputs, "residual", "branch")
            return {"output": inputs["residual"] + inputs["branch"]}
        if atom == "linear-projection":
            require(inputs, "hidden")
            return {"output": nn.Linear(hidden_size, hidden_size, bias=bool(settings.get("bias", False)))(inputs["hidden"])}
        if atom == "dropout":
            require(inputs, "hidden")
            return {"output": F.dropout(inputs["hidden"], p=float(settings.get("probability", 0.1)), training=False)}
        if atom == "scale":
            require(inputs, "hidden")
            return {"output": inputs["hidden"] * float(settings.get("factor", 1))}
        if atom == "hadamard-product":
            require(inputs, "left", "right")
            return {"output": inputs["left"] * inputs["right"]}
        if atom == "identity":
            require(inputs, "hidden")
            return {"output": inputs["hidden"]}
        if atom in {"subtract", "average", "maximum", "minimum"}:
            require(inputs, "left", "right")
            left, right = inputs["left"], inputs["right"]
            operations = {
                "subtract": lambda: left - right,
                "average": lambda: (left + right) * 0.5,
                "maximum": lambda: torch.maximum(left, right),
                "minimum": lambda: torch.minimum(left, right),
            }
            return {"output": operations[atom]()}
        if atom == "gated-blend":
            require(inputs, "left", "right")
            weight = torch.sigmoid(torch.tensor(float(settings.get("gateInit", 0))))
            return {"output": weight * inputs["left"] + (1 - weight) * inputs["right"]}
        if atom == "concatenate-projection":
            require(inputs, "left", "right")
            joined = torch.cat((inputs["left"], inputs["right"]), dim=-1)
            return {"output": nn.Linear(hidden_size * 2, hidden_size, bias=bool(settings.get("bias", False)))(joined)}
        if atom in {"clamp", "power", "absolute-value", "negate", "stop-gradient", "stochastic-depth"}:
            require(inputs, "hidden")
            source = inputs["hidden"]
            if atom == "clamp":
                output = torch.clamp(source, min=float(settings.get("minimum", -1)), max=float(settings.get("maximum", 1)))
            elif atom == "power":
                output = source.pow(float(settings.get("exponent", 2)))
            elif atom == "absolute-value":
                output = source.abs()
            elif atom == "negate":
                output = -source
            elif atom == "stop-gradient":
                output = source.detach()
            else:
                probability = float(settings.get("probability", 0.1))
                output = F.dropout(source, p=probability, training=False) * (1 - probability)
            return {"output": output}
        if atom == "swiglu-mlp":
            require(inputs, "hidden")
            return {"output": swiglu(inputs["hidden"], int(settings["intermediateSize"]))}
        if atom == "gelu-mlp":
            require(inputs, "hidden")
            source = inputs["hidden"]
            intermediate = int(settings["intermediateSize"])
            bias = bool(settings.get("bias", True))
            hidden = F.gelu(nn.Linear(source.shape[-1], intermediate, bias=bias)(source))
            return {"output": nn.Linear(intermediate, source.shape[-1], bias=bias)(hidden)}
        if atom in {"geglu-mlp", "reglu-mlp"}:
            require(inputs, "hidden")
            source = inputs["hidden"]
            intermediate = int(settings["intermediateSize"])
            bias = bool(settings.get("bias", False))
            gate = nn.Linear(source.shape[-1], intermediate, bias=bias)(source)
            gate = F.gelu(gate) if atom == "geglu-mlp" else F.relu(gate)
            up = nn.Linear(source.shape[-1], intermediate, bias=bias)(source)
            return {"output": nn.Linear(intermediate, source.shape[-1], bias=bias)(gate * up)}
        if atom == "relu-mlp":
            require(inputs, "hidden")
            source = inputs["hidden"]
            intermediate = int(settings["intermediateSize"])
            bias = bool(settings.get("bias", True))
            activated = F.relu(nn.Linear(source.shape[-1], intermediate, bias=bias)(source))
            return {"output": nn.Linear(intermediate, source.shape[-1], bias=bias)(activated)}
        if atom in {"tanh-mlp", "sigmoid-mlp", "mish-mlp", "squared-relu-mlp", "leaky-relu-mlp", "residual-gelu-mlp", "residual-swiglu-mlp"}:
            require(inputs, "hidden")
            source = inputs["hidden"]
            intermediate = int(settings["intermediateSize"])
            bias = bool(settings.get("bias", True))
            if atom == "residual-swiglu-mlp":
                return {"output": source + swiglu(source, intermediate)}
            projected = nn.Linear(source.shape[-1], intermediate, bias=bias)(source)
            activations = {
                "tanh-mlp": lambda: torch.tanh(projected),
                "sigmoid-mlp": lambda: torch.sigmoid(projected),
                "mish-mlp": lambda: F.mish(projected),
                "squared-relu-mlp": lambda: F.relu(projected).pow(2),
                "leaky-relu-mlp": lambda: F.leaky_relu(projected, negative_slope=0.01),
                "residual-gelu-mlp": lambda: F.gelu(projected),
            }
            output = nn.Linear(intermediate, source.shape[-1], bias=bias)(activations[atom]())
            return {"output": source + output if atom == "residual-gelu-mlp" else output}
        if atom == "moe-router":
            require(inputs, "hidden")
            source = inputs["hidden"]
            n_experts = int(settings["nExperts"])
            if n_experts <= 0:
                raise ValueError("moe-router requires nExperts > 0")
            logits = nn.Linear(source.shape[-1], n_experts, bias=bool(settings.get("routerBias", True)))(source).float()
            scoring = str(settings.get("scoringFunction", "sigmoid"))
            if scoring == "softmax":
                scores = torch.softmax(logits, dim=-1)
            elif scoring == "sigmoid":
                scores = torch.sigmoid(logits)
            else:
                raise ValueError(f"unsupported router scoring function: {scoring}")
            return {"scores": scores}
        if atom == "top-k-routing":
            require(inputs, "scores")
            scores = inputs["scores"]
            top_k = int(settings["topK"])
            if top_k <= 0:
                raise ValueError("top-k-routing requires topK > 0")
            filtered = scores
            method = str(settings.get("selectionMethod", "greedy"))
            n_groups = int(settings.get("nExpertGroups", 1))
            top_groups = int(settings.get("topkGroups", n_groups))
            if method == "group-limited-greedy" and n_groups > 0 and scores.shape[-1] % n_groups == 0:
                grouped = scores.view(*scores.shape[:-1], n_groups, scores.shape[-1] // n_groups)
                group_scores = grouped.max(dim=-1).values
                keep_groups = group_scores.topk(min(max(1, top_groups), n_groups), dim=-1).indices
                group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(-1, keep_groups, True)
                filtered = grouped.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).flatten(-2)
            elif method not in {"greedy", "aux-free", "group-limited-greedy"}:
                raise ValueError(f"unsupported top-k selection method: {method}")
            weights, indices = filtered.topk(min(top_k, filtered.shape[-1]), dim=-1)
            if bool(settings.get("normalizeWeights", True)):
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weights = weights * float(settings.get("routedScalingFactor", 1))
            return {"expertIndices": indices, "expertWeights": weights}
        if atom == "deterministic-token-routing":
            require(inputs, "tokenIds")
            token_ids = inputs["tokenIds"]
            experts = int(settings["nExperts"])
            top_k = int(settings["topK"])
            primary = token_ids.remainder(experts)
            routes = [primary]
            for route in range(1, top_k):
                routes.append((primary + route).remainder(experts))
            indices = torch.stack(routes, dim=-1)
            primary_weight = float(settings.get("primaryWeight", 0.5))
            weights = torch.full_like(indices, (1.0 - primary_weight) / max(1, top_k - 1), dtype=torch.float32)
            weights[..., 0] = primary_weight
            return {"expertIndices": indices, "expertWeights": weights}
        if atom == "shared-expert-bank":
            require(inputs, "hidden")
            outputs = [swiglu(inputs["hidden"], int(settings["intermediateSize"])) for _ in range(int(settings["nSharedExperts"]))]
            return {"output": sum(outputs)}
        if atom == "routed-expert-bank":
            require(inputs, "hidden", "expertIndices", "expertWeights")
            source, indices, weights = inputs["hidden"], inputs["expertIndices"], inputs["expertWeights"]
            output = torch.zeros_like(source)
            for expert in range(int(settings["nExperts"])):
                mask = indices == expert
                token_mask = mask.any(dim=-1)
                if token_mask.any():
                    selected_weight = (weights[token_mask] * mask[token_mask]).sum(dim=-1, keepdim=True)
                    output[token_mask] += swiglu(source[token_mask], int(settings["intermediateSize"])) * selected_weight
            return {"output": output}
        if atom == "branch-gated-merge":
            require(inputs, "shared", "routed")
            return {"output": float(settings["sharedGateInit"]) * inputs["shared"] + float(settings["routedGateInit"]) * inputs["routed"]}
        if atom == "expert-merge":
            require(inputs, "routed", "shared")
            merge = str(settings.get("merge", "sum"))
            if merge == "sum":
                output = inputs["routed"] + inputs["shared"]
            elif merge == "gated-sum":
                gate = torch.sigmoid(torch.zeros((), dtype=inputs["routed"].dtype))
                output = gate * inputs["routed"] + (1.0 - gate) * inputs["shared"]
            else:
                raise ValueError(f"unsupported expert merge: {merge}")
            return {"output": output}
        if atom == "load-balancing-loss":
            require(inputs, "scores")
            scores = inputs["scores"]
            coefficient = float(settings.get("coefficient", 0.001))
            return {"loss": coefficient * (scores.mean(dim=(0, 1)) * scores.shape[-1]).pow(2).mean()}
        if atom == "lm-head":
            require(inputs, "hidden")
            return {"logits": nn.Linear(hidden_size, int(settings["vocabSize"]), bias=bool(settings.get("bias", False)))(inputs["hidden"])}
        if atom in {"greedy-token-decoder", "top-k-token-sampler", "multinomial-token-sampler"}:
            require(inputs, "logits")
            logits = inputs["logits"]
            if atom == "greedy-token-decoder":
                token_ids = torch.argmax(logits, dim=-1)
            elif atom == "top-k-token-sampler":
                top_k = max(1, min(int(settings.get("topK", 50)), logits.shape[-1]))
                token_ids = torch.topk(logits / max(float(settings.get("temperature", 1)), 1e-6), k=top_k, dim=-1).indices[..., 0]
            else:
                probabilities = F.softmax(logits / max(float(settings.get("temperature", 1)), 1e-6), dim=-1)
                token_ids = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1).reshape(probabilities.shape[:-1])
            return {"tokenIds": token_ids}
        if atom == "log-softmax":
            require(inputs, "logits")
            return {"logits": F.log_softmax(inputs["logits"], dim=int(settings.get("dimension", -1)))}
        if atom in {"softmax-output", "temperature-scaling", "logits-clamp"}:
            require(inputs, "logits")
            logits = inputs["logits"]
            if atom == "softmax-output":
                output = torch.softmax(logits, dim=int(settings.get("dimension", -1)))
            elif atom == "temperature-scaling":
                output = logits / max(float(settings.get("temperature", 1)), 1e-6)
            else:
                output = torch.clamp(logits, min=float(settings.get("minimum", -30)), max=float(settings.get("maximum", 30)))
            return {"logits": output}
        if atom == "cross-entropy-loss":
            require(inputs, "logits", "labels")
            return {"loss": F.cross_entropy(inputs["logits"].flatten(0, 1), inputs["labels"].flatten(), ignore_index=int(settings.get("ignoreIndex", -100)))}
        if atom == "label-smoothed-cross-entropy":
            require(inputs, "logits", "labels")
            return {"loss": F.cross_entropy(inputs["logits"].flatten(0, 1), inputs["labels"].flatten(), ignore_index=int(settings.get("ignoreIndex", -100)), label_smoothing=float(settings.get("smoothing", 0.1)))}
        if atom == "router-entropy-loss":
            require(inputs, "scores")
            scores = inputs["scores"]
            probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            coefficient = float(settings.get("coefficient", 0.001))
            return {"loss": -coefficient * (probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean()}
        if atom in {"mean-squared-error", "l1-loss", "binary-cross-entropy"}:
            if atom == "binary-cross-entropy":
                require(inputs, "logits", "targets")
                return {"loss": F.binary_cross_entropy_with_logits(inputs["logits"], inputs["targets"].sigmoid())}
            require(inputs, "prediction", "target")
            loss = F.mse_loss(inputs["prediction"], inputs["target"]) if atom == "mean-squared-error" else F.l1_loss(inputs["prediction"], inputs["target"])
            return {"loss": loss}
        if atom in {"logits-z-loss", "output-entropy-loss"}:
            require(inputs, "logits")
            logits = inputs["logits"]
            coefficient = float(settings.get("coefficient", 0.001))
            if atom == "logits-z-loss":
                return {"loss": coefficient * torch.logsumexp(logits, dim=-1).pow(2).mean()}
            log_probabilities = F.log_softmax(logits, dim=-1)
            return {"loss": -coefficient * (log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()}
        raise RuntimeError(f"unsupported semantic atom: {atom}")

    for node in ordered:
        atom_id = node["id"]
        blocked_by = next((edge["source"] for edge in edges if edge["target"] == atom_id and edge["source"] in failed_nodes), None)
        if blocked_by is not None:
            error = RuntimeError(f"blocked by failed dependency {blocked_by}")
            failed_nodes.add(atom_id)
            results.append(failed(atom_id, error))
            continue
        try:
            kind = node["kind"]
            inputs = connected_inputs(atom_id)
            produced: dict[str, Any]
            if kind == "input":
                outgoing = [edge for edge in edges if edge["source"] == atom_id]
                token_input = any(edge.get("sourcePort") == "tokenIds" or edge.get("targetPort") == "tokenIds" for edge in outgoing)
                labels_input = node.get("role") == "labels" or any(edge.get("sourcePort") == "labels" or edge.get("targetPort") == "labels" for edge in outgoing)
                if node.get("role") == "image":
                    value = torch.randn(2, 3, 32, 64)
                elif node.get("role") == "video":
                    value = torch.randn(2, 3, 4, 32, 32)
                elif node.get("role") == "audio":
                    value = torch.randn(2, 1, 3200)
                elif token_input:
                    value = token_tensor if token_tensor is not None else torch.randint(0, 64, (2, 8))
                elif labels_input:
                    batch, sequence = token_tensor.shape if token_tensor is not None else (2, 8)
                    value = torch.randint(0, min(64, hidden_size), (batch, sequence), dtype=torch.long)
                else:
                    batch, sequence = token_tensor.shape if token_tensor is not None else (2, 8)
                    value = torch.randn(batch, sequence, hidden_size)
                produced = {"output": value}
                for edge in outgoing:
                    produced[str(edge.get("sourcePort", "output"))] = value
            elif kind == "semantic":
                produced = execute_semantic(node, inputs)
            elif kind == "linear":
                if not inputs:
                    raise RuntimeError("linear atom has no connected input")
                attributes = node.get("attributes", {})
                source = next(iter(inputs.values()))
                value = nn.Linear(int(attributes["inFeatures"]), int(attributes["outFeatures"]), bias=bool(attributes.get("bias", False)))(source)
                produced = {"output": value}
            elif kind == "sdpa":
                sources = {by_id[edge["source"]]["role"]: values.get((edge["source"], str(edge.get("sourcePort", "output")))) for edge in edges if edge["target"] == atom_id}
                missing = [role for role in ("query", "key", "value") if sources.get(role) is None]
                if missing:
                    raise RuntimeError(f"missing connected {', '.join(missing)} input")
                batch, sequence, _ = sources["query"].shape
                q = sources["query"].view(batch, sequence, query_heads, head_dim).transpose(1, 2)
                k = sources["key"].view(batch, sequence, key_value_heads, head_dim).transpose(1, 2)
                v = sources["value"].view(batch, sequence, key_value_heads, head_dim).transpose(1, 2)
                repeats = query_heads // key_value_heads
                value = F.scaled_dot_product_attention(q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1), is_causal=True).transpose(1, 2).reshape(batch, sequence, -1)
                produced = {"output": value}
            elif kind == "custom-pytorch":
                if not inputs:
                    raise RuntimeError("custom atom has no executable input")
                source = next(iter(inputs.values()))
                code = str(node.get("code", ""))
                produced = {"output": custom_module(code, nn)(source)}
            else:
                raise RuntimeError(f"unsupported executable atom kind: {kind}")
            for port, value in produced.items():
                values[(atom_id, port)] = value
            if node.get("atomId") == "lm-head" and "logits" in produced:
                logits = produced["logits"]
                probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
                top_probabilities, top_token_ids = torch.topk(probabilities, k=min(5, probabilities.numel()))
                model_output = {
                    "kind": "logits",
                    "tensorShape": list(logits.shape),
                    "logitsShape": list(logits.shape),
                    "predictedTokenId": int(top_token_ids[0]),
                    "topTokenIds": [int(token_id) for token_id in top_token_ids],
                    "topProbabilities": [round(float(probability), 6) for probability in top_probabilities],
                }
            if node.get("atomId") in {"greedy-token-decoder", "top-k-token-sampler", "multinomial-token-sampler"} and "tokenIds" in produced:
                generated = produced["tokenIds"]
                model_output = {
                    "kind": "token-ids",
                    "tensorShape": list(generated.shape),
                    "predictedTokenId": int(generated[0, -1]),
                    "topTokenIds": [int(generated[0, -1])],
                    "topProbabilities": [1.0],
                }
            first_value = next(iter(produced.values()))
            last_tensor = first_value
            results.append(passed(atom_id, tensor_summary(first_value)))
        except Exception as error:
            results.append(failed(atom_id, error))
            failed_nodes.add(atom_id)
            if first_failure is None:
                first_failure = (atom_id, str(error))

    if model_output is None and last_tensor is not None and hasattr(last_tensor, "shape"):
        model_output = {"kind": "tensor", "tensorShape": list(last_tensor.shape)}
    if first_failure is not None:
        return {"engine": "pytorch", "status": "failed", "currentAtomId": first_failure[0], "error": first_failure[1], "modelOutput": model_output, "results": results}
    return {"engine": "pytorch", "status": "completed", "modelOutput": model_output, "results": results}


def build_tokenizer(pipeline: dict[str, Any]):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

    steps = pipeline.get("steps", [])
    model_step = next((step for step in steps if step["atom"] == "bpe-model"), None)
    if model_step is None:
        raise RuntimeError("pipeline requires a BPE model atom")
    tokenizer = Tokenizer(models.BPE(unk_token=model_step["settings"].get("unkToken", "<unk>")))
    normalize_step = next((step for step in steps if step["atom"] == "unicode-normalize"), None)
    if normalize_step:
        tokenizer.normalizer = getattr(normalizers, str(normalize_step["settings"]["form"]))()
    pretokenize_step = next((step for step in steps if step["atom"] == "byte-level-pretokenize"), None)
    if pretokenize_step:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=bool(pretokenize_step["settings"].get("addPrefixSpace", False))
        )
    if any(step["atom"] == "byte-level-decode" for step in steps):
        tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def run_tokenizer(pipeline: dict[str, Any], sample: str) -> dict[str, Any]:
    from tokenizers import normalizers, pre_tokenizers, trainers

    results: list[dict[str, Any]] = []
    tokenizer = None
    token_ids: list[int] | None = None
    for step in pipeline.get("steps", []):
        atom_id = step["id"]
        settings = step.get("settings", {})
        try:
            atom = step["atom"]
            if atom == "unicode-normalize":
                form = str(settings["form"])
                output = getattr(normalizers, form)().normalize_str(sample)
                summary = f"{form}: {output}"
            elif atom == "byte-level-pretokenize":
                pieces = pre_tokenizers.ByteLevel(
                    add_prefix_space=bool(settings.get("addPrefixSpace", False))
                ).pre_tokenize_str(sample)
                summary = f"pieces={len(pieces)} first={pieces[0][0] if pieces else ''}"
            elif atom == "bpe-model":
                tokenizer = build_tokenizer(pipeline)
                summary = f"BPE model unk={settings.get('unkToken', '<unk>')}"
            elif atom == "bpe-trainer":
                tokenizer = tokenizer or build_tokenizer(pipeline)
                trainer = trainers.BpeTrainer(
                    vocab_size=int(settings["vocabSize"]),
                    special_tokens=list(settings.get("specialTokens", [])),
                )
                tokenizer.train_from_iterator(
                    [sample, "LABO AI atomic tokenizer", "typed blocks compile to Python and Rust"],
                    trainer=trainer,
                )
                summary = f"trained vocab={tokenizer.get_vocab_size()}"
            elif atom == "byte-level-decode":
                tokenizer = tokenizer or build_tokenizer(pipeline)
                if tokenizer.get_vocab_size() == 0:
                    tokenizer.train_from_iterator([sample], trainer=trainers.BpeTrainer(vocab_size=256))
                encoded = tokenizer.encode(sample)
                token_ids = encoded.ids
                summary = f"ids={encoded.ids} decoded={tokenizer.decode(encoded.ids)}"
            else:
                raise RuntimeError(f"unsupported tokenizer atom: {atom}")
            results.append(passed(atom_id, summary))
        except Exception as error:
            results.append(failed(atom_id, error))
            return {
                "engine": "tokenizers",
                "status": "failed",
                "currentAtomId": atom_id,
                "error": str(error),
                "results": results,
            }
    if token_ids is None and tokenizer is not None:
        token_ids = tokenizer.encode(sample).ids
    return {"engine": "tokenizers", "status": "completed", "tokenIds": token_ids or [], "results": results}


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("kind") == "model":
        output = run_model(payload["graph"], payload.get("tokenIds"))
    elif payload.get("kind") == "tokenizer":
        output = run_tokenizer(payload["pipeline"], str(payload.get("sample", "LABO AI")))
    else:
        raise ValueError("kind must be model or tokenizer")
    json.dump(output, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
