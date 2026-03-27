# Layer 5: Model Storage & Loading

> **Goal**: Optimize how models are stored, downloaded, cached, and loaded to minimize startup time and storage costs.

## Background

When you run `vllm serve meta-llama/Llama-3.1-8B-Instruct` for the first time, it:

1. Downloads ~16GB of model weights from HuggingFace
2. Stores them in a local cache (`~/.cache/huggingface/`)
3. Loads them from disk into CPU RAM
4. Transfers them from CPU RAM to GPU VRAM
5. Only then starts serving inference

Each step takes time:

| Step | Time (8B model) | Time (70B model) |
|---|---|---|
| Download (first time) | 5-15 min | 30-60 min |
| Load from disk to RAM | 10-30 sec | 1-3 min |
| Transfer RAM to VRAM | 5-10 sec | 30-60 sec |
| **Total cold start** | **~15 min** | **~60 min** |
| **Warm start (cached)** | **~30 sec** | **~3 min** |

In production, cold start time directly impacts your ability to:
- Recover from failures (Layer 3 recovery is only as fast as model loading)
- Scale up quickly when traffic spikes
- Switch between models efficiently

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Storage Tiers                               │
│                                              │
│  ┌────────────────┐   Fastest, most expensive│
│  │ GPU VRAM       │   (48GB)                 │
│  │ Active model   │                          │
│  └───────┬────────┘                          │
│          │ load                               │
│  ┌───────▼────────┐                          │
│  │ System RAM     │   Fast, limited          │
│  │ Model buffer   │   (typically 64-256GB)   │
│  └───────┬────────┘                          │
│          │ read                               │
│  ┌───────▼────────┐                          │
│  │ Local SSD      │   Fast reads             │
│  │ HF cache       │   (1-4TB)               │
│  └───────┬────────┘                          │
│          │ download                           │
│  ┌───────▼────────┐                          │
│  │ Remote Storage │   Slow, cheap, unlimited │
│  │ HuggingFace    │                          │
│  │ S3 / GCS       │                          │
│  └────────────────┘                          │
└─────────────────────────────────────────────┘
```

---

## Prerequisites

- Layer 1-3 completed
- Sufficient disk space for model caching (at least 50GB free)
- HuggingFace token with access to gated models

---

## Part A: Understanding the HuggingFace Cache

### Step 1: Explore the Cache Structure

**What you're doing**: Understanding where HuggingFace stores downloaded models and how the cache works.

**Why**: Knowing the cache structure helps you manage storage, pre-download models, and set up shared caches across containers.

```bash
# Find where the cache is
echo $HF_HOME  # Usually ~/.cache/huggingface

# Check cache size
du -sh ~/.cache/huggingface/hub/

# List cached models
ls ~/.cache/huggingface/hub/ | grep models--

# Example output:
# models--meta-llama--Llama-3.1-8B-Instruct
# models--meta-llama--Llama-3.2-3B-Instruct
```

### Step 2: Check What's Using Disk Space

```bash
# Detailed breakdown per model
du -sh ~/.cache/huggingface/hub/models--*

# Check for duplicate snapshots (old versions)
find ~/.cache/huggingface/hub/ -name "snapshots" -type d -exec ls {} \;
```

HuggingFace uses content-addressable storage with symlinks. Each model version gets a snapshot directory. Old snapshots aren't deleted automatically.

### Step 3: Clean Up Old Snapshots

```bash
# Use huggingface-cli to manage the cache
pip install -U huggingface_hub

# Scan for deletable files
huggingface-cli scan-cache

# Delete specific revisions (interactive)
huggingface-cli delete-cache
```

---

## Part B: Pre-Downloading Models

### Step 4: Download Models Before You Need Them

**What you're doing**: Downloading model weights to the local cache without running vLLM.

**Why**: In production, you don't want model download time in your startup path. Pre-download models so that restarts and scaling only need the fast load-from-disk step.

```bash
# Download a model to cache without running it
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Download a specific file type only (e.g., safetensors)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --include "*.safetensors" "*.json" "tokenizer*"

# Verify it's cached
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/
```

### Step 5: Share Cache with Docker Containers

**What you're doing**: Mounting the host's HuggingFace cache into Docker containers so models don't need to be re-downloaded per container.

**Why**: Without this, each `docker run` downloads the full model again (~16GB for 8B). With a shared cache, containers start in seconds.

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

The key flag is `-v ~/.cache/huggingface:/root/.cache/huggingface` — this mounts your host cache into the container.

### Step 6: Measure Warm Start Time

```bash
# Time a container restart (model already cached)
docker stop vllm-server
time (docker start vllm-server && \
  while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 1; done && \
  echo "vLLM is ready")

# This should be ~30-60 seconds for an 8B model (vs 15+ minutes for cold start)
```

---

## Part C: Safetensors and Loading Optimization

### Step 7: Understand Model File Formats

**What you're doing**: Learning the difference between model file formats and why it matters for loading speed.

| Format | Extension | Loading speed | Safety |
|---|---|---|---|
| **Safetensors** | `.safetensors` | Fast (memory-mapped) | Safe (no arbitrary code) |
| **PyTorch pickle** | `.bin` | Slower | Unsafe (can execute code) |
| **GGUF** | `.gguf` | Fast | Safe (used by llama.cpp) |

**Why**: vLLM uses safetensors by default. Safetensors files can be memory-mapped, meaning the OS can load them directly from disk to RAM without copying — this makes loading significantly faster, especially for large models.

```bash
# Check which format your cached model uses
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/

# You should see .safetensors files
# If you see .bin files instead, the model is using the older format
```

### Step 8: Enable Memory-Mapped Loading

vLLM automatically uses memory-mapped loading for safetensors files. You can verify this by watching the loading process:

```bash
docker stop vllm-server && docker start vllm-server
docker logs -f vllm-server 2>&1 | grep -i "loading\|mmap\|safetensor"
```

---

## Part D: Remote Storage (S3/GCS)

For teams with multiple GPU machines, storing models on a shared remote storage avoids downloading the same model multiple times.

### Step 9: Set Up S3-Compatible Storage

**What you're doing**: Configuring a shared model storage that all GPU nodes can access.

**Why**: If you have 10 GPU machines, you don't want to download 16GB x 10 = 160GB from HuggingFace. Download once to S3, then all machines pull from your local S3 at network speed.

> **DigitalOcean Spaces** is S3-compatible and works as a model store.

```bash
# Install s3cmd or aws cli
pip install awscli

# Configure for DigitalOcean Spaces (or AWS S3)
aws configure
# Set your access key, secret, and region

# Upload a model to S3
aws s3 sync ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/ \
  s3://your-bucket/models/Llama-3.1-8B-Instruct/ \
  --endpoint-url https://nyc3.digitaloceanspaces.com

# On another machine, download from S3 (much faster than HuggingFace)
aws s3 sync s3://your-bucket/models/Llama-3.1-8B-Instruct/ \
  ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/ \
  --endpoint-url https://nyc3.digitaloceanspaces.com
```

### Step 10: Use vLLM with a Local Model Path

Instead of downloading from HuggingFace every time, point vLLM to a local directory:

```bash
# Download model to a specific local path
MODEL_DIR=/opt/models/Llama-3.1-8B-Instruct
mkdir -p $MODEL_DIR

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir $MODEL_DIR

# Run vLLM with local path
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -v /opt/models:/models:ro \
  vllm/vllm-openai:latest \
  --model /models/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

---

## Part E: Model Loading Strategy Matrix

### Step 11: Choose the Right Strategy

| Scenario | Strategy | Cold start | Storage cost |
|---|---|---|---|
| **Dev/testing** | Download from HuggingFace on demand | Slow (minutes) | Low |
| **Single prod server** | Pre-download + shared Docker cache | Fast (seconds) | Medium |
| **Multi-node cluster** | S3 + local cache on each node | Medium (first pull) | Medium |
| **Frequent model swaps** | Keep multiple models pre-loaded in cache | Fast | High |
| **Edge deployment** | GGUF quantized + local storage | Very fast | Low |

Create a script to automate model pre-download for your setup:

```bash
# scripts/preload-models.sh
#!/bin/bash
# Pre-download all models needed for this deployment

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
)

for model in "${MODELS[@]}"; do
    echo "Downloading $model..."
    huggingface-cli download "$model" --include "*.safetensors" "*.json" "tokenizer*"
    echo "Done: $model"
done

echo "All models pre-downloaded."
du -sh ~/.cache/huggingface/hub/models--*
```

---

## Verification Checklist

```bash
# 1. Model cache is populated
ls ~/.cache/huggingface/hub/models--*

# 2. Docker container uses shared cache
docker inspect vllm-server --format='{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | grep huggingface

# 3. Warm restart is fast (under 60 seconds for 8B)
docker stop vllm-server
time (docker start vllm-server && \
  while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 1; done && \
  echo "Ready")

# 4. Model uses safetensors format
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/*.safetensors
```

---

## What You've Built

```
The full 5-layer stack:

Layer 1: Zero-Trust Edge     → Secure access to your inference server
Layer 2: Two-Level Proxy     → Multi-model routing and load balancing
Layer 3: GPU Reliability     → Auto-detection and recovery from failures
Layer 4: Multi-Node GPU      → Scale inference across multiple machines
Layer 5: Model Storage       → Fast model loading and efficient storage
```

Together, these 5 layers form a production-grade AI inference infrastructure that is secure, scalable, reliable, and efficient.

---

## Moving to Kubernetes

Model storage and loading patterns translate well to Kubernetes:

- **Shared HuggingFace cache** → Use a PersistentVolumeClaim (PVC) with `ReadWriteMany` access mode. All vLLM pods mount the same volume, so models are downloaded once and shared across pods
- **Pre-download models** → Run a Kubernetes Job or init container that downloads models before the main vLLM container starts. This turns cold starts into warm starts
- **S3/Spaces remote storage** → Same `aws s3 sync` pattern, but run from an init container. Or use [CSI drivers](https://docs.digitalocean.com/products/kubernetes/how-to/add-volumes/) to mount Spaces directly as a volume
- **Model registry** → Use DigitalOcean Container Registry (DOCR) or a dedicated model registry. Package model weights as OCI artifacts for versioned, cacheable distribution
- **Fast loading with safetensors** → Same benefits in Kubernetes. Memory-mapped loading works regardless of orchestration layer

The key Kubernetes advantage for model storage: PVCs persist across pod restarts, so model downloads survive crashes without re-downloading. Combined with node-local caching (via `hostPath` volumes), you get near-instant warm starts even when pods are rescheduled.

---

## Reference

- [HuggingFace Hub documentation](https://huggingface.co/docs/hub/en/index)
- [Safetensors format](https://huggingface.co/docs/safetensors/en/index)
- [vLLM model loading](https://docs.vllm.ai/)
- [DigitalOcean Spaces (S3-compatible)](https://docs.digitalocean.com/products/spaces/)
