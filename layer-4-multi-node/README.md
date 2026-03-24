# Layer 4: Multi-Node GPU Communication

> **Goal**: Scale inference beyond a single machine by distributing vLLM across multiple GPU nodes with tensor parallelism and pipeline parallelism.

## Background

In Layers 1-3, everything runs on a single machine with one GPU. This works for an 8B model, but larger models (70B, 405B) don't fit in a single GPU's memory. Even for models that do fit, you may want more throughput than one GPU can provide.

There are two strategies for spreading a model across multiple GPUs:

| Strategy | How it works | Best for |
|---|---|---|
| **Tensor Parallelism (TP)** | Splits each layer's weight matrices across GPUs. All GPUs work on the same layer simultaneously. | GPUs on the same machine (needs fast interconnect like NVLink) |
| **Pipeline Parallelism (PP)** | Assigns different layers to different GPUs. Data flows through GPUs sequentially. | GPUs across different machines (tolerates slower network) |

```
Tensor Parallelism (TP=2):
  Layer 1: [GPU 0 handles left half] [GPU 1 handles right half]
  Layer 2: [GPU 0 handles left half] [GPU 1 handles right half]
  ...

Pipeline Parallelism (PP=2):
  GPU 0: Layer 1, Layer 2, ..., Layer 16
  GPU 1: Layer 17, Layer 18, ..., Layer 32
```

## Architecture

```
┌──────────────────────────────────────────────┐
│  Node 1 (Head)                               │
│  ┌──────────┐  ┌──────────┐                  │
│  │ GPU 0    │  │ GPU 1    │                  │
│  │ (TP=0)   │  │ (TP=1)   │                  │
│  └──────────┘  └──────────┘                  │
│  ┌──────────────────────────────────────────┐│
│  │ vLLM (rank 0 - head)                     ││
│  │ Serves API on :8000                      ││
│  └──────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────┐│
│  │ Ray Head Node                             ││
│  └──────────────────────────────────────────┘│
└──────────────────┬───────────────────────────┘
                   │ Network (TCP/NCCL)
┌──────────────────┴───────────────────────────┐
│  Node 2 (Worker)                             │
│  ┌──────────┐  ┌──────────┐                  │
│  │ GPU 0    │  │ GPU 1    │                  │
│  │ (TP=2)   │  │ (TP=3)   │                  │
│  └──────────┘  └──────────┘                  │
│  ┌──────────────────────────────────────────┐│
│  │ vLLM (rank 1 - worker)                   ││
│  └──────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────┐│
│  │ Ray Worker Node                           ││
│  └──────────────────────────────────────────┘│
└──────────────────────────────────────────────┘
```

---

## Prerequisites

- Two or more machines with NVIDIA GPUs
- Network connectivity between machines (ideally 10Gbps+ for TP, 1Gbps minimum for PP)
- Docker with NVIDIA runtime on all machines
- Same vLLM version on all machines

> **Single machine with multiple GPUs?** You can still follow this lab using tensor parallelism on one machine. Skip the multi-node networking steps and just increase `--tensor-parallel-size`.

---

## Part A: Single-Node Multi-GPU (Tensor Parallelism)

Start here if you have multiple GPUs in one machine, or if you want to understand TP before going multi-node.

### Step 1: Check Your GPU Topology

**What you're doing**: Understanding how your GPUs are connected to each other.

**Why**: Tensor parallelism requires constant communication between GPUs. NVLink is 10-20x faster than PCIe for GPU-to-GPU transfers. Your GPU topology determines how fast TP can be.

```bash
# Show GPU topology
nvidia-smi topo -m

# Check if NVLink is available
nvidia-smi nvlink --status
```

### Step 2: Run vLLM with Tensor Parallelism

```bash
docker stop vllm-server && docker rm vllm-server

# TP=2: split model across 2 GPUs
docker run -d \
  --name vllm-server \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

**Key parameter**: `--tensor-parallel-size 2` tells vLLM to split the model across 2 GPUs.

> **`--ipc=host`**: Required for NCCL (NVIDIA's multi-GPU communication library) to use shared memory for fast inter-GPU data transfer.

### Step 3: Verify Multi-GPU Usage

```bash
# Check that both GPUs are being used
nvidia-smi

# You should see vLLM processes on both GPUs with roughly equal memory usage
```

### Step 4: Benchmark Single vs Multi-GPU

```bash
# Time a request with TP=2
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write 500 words about distributed computing."}],"max_tokens":512}' > /dev/null

# Compare with the single-GPU time from Layer 1
# TP=2 should be faster for generation (more compute) but may have
# slightly higher first-token latency (communication overhead)
```

---

## Part B: Multi-Node Setup with Ray

vLLM uses [Ray](https://www.ray.io/) for multi-node distributed inference. Ray handles the networking, process management, and GPU assignment across machines.

### Step 5: Start Ray Head Node (Machine 1)

**What you're doing**: Starting the Ray cluster head on your first machine.

**Why**: Ray uses a head-worker architecture. The head node coordinates all workers and is where you submit jobs (run vLLM).

```bash
# On Machine 1
docker run -d \
  --name ray-head \
  --gpus all \
  --network host \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  bash -c "ray start --head --port=6379 && sleep infinity"
```

Check the Ray dashboard:
```bash
# Ray dashboard is available at http://<machine-1-ip>:8265
curl http://localhost:8265/api/cluster_status
```

### Step 6: Join Ray Worker Node (Machine 2)

**What you're doing**: Adding a second machine's GPUs to the Ray cluster.

```bash
# On Machine 2 — replace <HEAD_NODE_IP> with Machine 1's IP
docker run -d \
  --name ray-worker \
  --gpus all \
  --network host \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  bash -c "ray start --address=<HEAD_NODE_IP>:6379 && sleep infinity"
```

### Step 7: Verify the Cluster

```bash
# On Machine 1
docker exec ray-head ray status

# You should see both nodes and all GPUs listed
# Example output:
# Nodes: 2
# Resources: GPU: 4.0 (2 per node)
```

### Step 8: Run vLLM Across the Cluster

```bash
# On Machine 1 (head node)
docker exec ray-head vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192
```

With `--tensor-parallel-size 4` and 2 GPUs per node, vLLM will use all 4 GPUs across both machines.

### Step 9: Test Multi-Node Inference

```bash
# From any machine that can reach Machine 1
curl http://<HEAD_NODE_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello from the cluster!"}],"max_tokens":50}'
```

---

## Part C: Running Larger Models

The whole point of multi-GPU is running models that don't fit on a single GPU.

### Step 10: Serve a 70B Model

```bash
# Llama 3.1 70B needs ~140GB VRAM in FP16
# With 4x 48GB GPUs = 192GB total, you have enough room

docker exec ray-head vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

> **Note**: You need to accept the Llama 3.1 70B license on HuggingFace separately from the 8B license.

---

## Part D: Network Optimization

### Step 11: Check NCCL Communication

**What you're doing**: Verifying that NCCL (the GPU communication library) is using the fastest available transport.

**Why**: NCCL will use NVLink within a machine and TCP/IP across machines. For multi-node, you want to make sure it's using the right network interface.

```bash
# Check NCCL environment
docker exec ray-head env | grep NCCL

# Set the correct network interface for multi-node NCCL
# Replace eth0 with your high-speed network interface
docker exec ray-head bash -c "export NCCL_SOCKET_IFNAME=eth0"
```

### Step 12: Benchmark Network Throughput

```bash
# Test network bandwidth between nodes
# On Machine 1:
iperf3 -s

# On Machine 2:
iperf3 -c <HEAD_NODE_IP>

# For TP across nodes, you want at least 10 Gbps
# For PP, 1 Gbps is usually sufficient
```

---

## Verification Checklist

```bash
# 1. Multi-GPU: both GPUs are being used
nvidia-smi | grep vllm

# 2. Ray cluster: all nodes connected
docker exec ray-head ray status

# 3. Inference works across GPUs
curl -s -o /dev/null -w "HTTP %{http_code}, Time: %{time_total}s\n" \
  http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'

# 4. GPU memory is balanced across GPUs
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

---

## What You've Built

```
Layer 1-3:
  Single GPU, single machine

Layer 4:
  Multiple GPUs, single or multiple machines
  Tensor parallelism for fast inference
  Pipeline parallelism for large models
  Ray cluster for multi-node coordination
```

## Next: Layer 5

Layer 5 covers **Model Storage & Loading** — efficient model downloading, caching, and fast loading strategies.

---

## Reference

- [vLLM distributed inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [Ray cluster setup](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
