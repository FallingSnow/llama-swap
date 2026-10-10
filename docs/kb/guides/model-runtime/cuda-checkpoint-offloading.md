---
title: Offloading a model with cuda-checkpoint
summary: How swapOutStrategy suspends a model's GPU state instead of killing it, for fast swap-back without a cold start.
category: guides
tags: [cuda-checkpoint, offload, vram, gpu, sleep, suspend, restore, swap]
config_keys: [models.*.swapOutStrategy, models.*.ttl]
updated: 2026-10-09
---

# Offloading a model with cuda-checkpoint

By default, unloading a model (via `ttl`, the API, or a swap) **kills the
process**, and swapping it back in starts a fresh one. That means every swap-back
is a full cold start: re-loading weights, rebuilding kernels, re-compiling.

`swapOutStrategy: cuda-checkpoint` changes that. When a model is unloaded, llama-swap
runs NVIDIA's `cuda-checkpoint` against the process, which copies its GPU memory
to the host and **releases the GPU while the process keeps running**. Swapping it
back in restores that state, so the model serves again in a fraction of the time
of a cold start.

## The setting

```yaml
models:
  qwen-coder:
    swapOutStrategy: cuda-checkpoint   # suspend/restore instead of kill/restart
    ttl: 300
    cmd: llama-server --port ${PORT} -m /models/qwen.gguf
```

| `swapOutStrategy` | behaviour on unload |
| --- | --- |
| `""` (default) | kill the process; restart it on swap-back |
| `cuda-checkpoint` | suspend the process's CUDA state (GPU freed, process alive); restore it on swap-back |

Only one value is implemented today. The field is a "swap-out strategy" so other
offload methods (for example vLLM's built-in sleep) can be added without changing
how the rest of llama-swap tracks a model's state.

## Requirements

- **Linux only.** On other platforms llama-swap logs a warning and falls back to
  kill/restart.
- **NVIDIA display driver 550 or higher.**
- **The `cuda-checkpoint` binary on `PATH`** of the llama-swap process. Download
  it from the [NVIDIA cuda-checkpoint repo](https://github.com/NVIDIA/cuda-checkpoint).
  In a container, install it into the same image as llama-swap.

## What goes wrong

- **cuda-checkpoint is not installed** → the suspend fails, llama-swap logs it,
  and the model is killed and restarted like normal. The swap still succeeds.
- **The model uses UVM or `cuMemExportToShareableHandle` IPC memory** → cuda-checkpoint
  cannot suspend it, so it falls back to kill/restart.
- **Restore fails** (driver error, or the process died while suspended) →
  llama-swap tears down the process and starts a fresh one. The swap still
  succeeds, just as a cold start.
- **The process exits while suspended** → it is treated as stopped and a fresh
  start happens on the next swap-in.

Every listed failure mode falls back to the normal kill/restart path, so a
failed suspend degrades to the same outcome as if no sleeper were configured.
The trade-off is time: a suspend copies GPU memory to the host and is far
slower than a plain kill, so a swap-out with a sleeper can take longer than an
equivalent kill — under a tight `unloadTimeout`, a very large model may exceed
the stop window.
