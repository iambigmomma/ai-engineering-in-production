# AI Engineering in Production

A hands-on lab series covering the 5-layer production hardening stack for AI inference.

> Think of deploying an AI model like opening a restaurant. Having a great chef (model) is just the beginning — you also need a secure front door, a smart maitre d', backup plans for equipment failures, coordination across kitchens, and a well-stocked pantry.

## The 5-Layer Stack

| Layer | Name | What It Does | Status |
|-------|------|-------------|--------|
| 1 | [**Zero-Trust Edge**](layer-1-edge/) | Secure the front door — Cloudflare Tunnel, Access, WAF, rate limiting | Complete |
| 2 | [**Inference Gateway**](layer-2-inference-gateway/) | The maitre d' — model routing, cost-aware routing, load balancing | In Progress |
| 3 | [**GPU Operations**](layer-3-gpu-operations/) | When the oven breaks — GPU health, OOM recovery, failure management | In Progress |
| 4 | [**Multi-Node GPU Communication**](layer-4-multi-node/) | Coordinating kitchens — tensor/pipeline parallelism, NCCL tuning | Planned |
| 5 | [**Model Storage**](layer-5-storage/) | The pantry — model caching, pull optimization, registry patterns | Planned |

## Who This Is For

Engineers and architects deploying AI inference workloads in production. Each layer is a standalone lab you can run on a single GPU server with Docker.

## How to Use

Start from Layer 1 and work your way up. Each layer builds on the previous one:

1. **Layer 1** gets you a working, secured inference endpoint
2. **Layer 2** adds intelligent routing across multiple models
3. **Layer 3** adds operational resilience when GPUs fail
4. **Layer 4** scales inference across multiple machines
5. **Layer 5** optimizes how models are stored and loaded

## Prerequisites

- NVIDIA GPU with working drivers
- Docker with NVIDIA runtime
- HuggingFace account with model access
