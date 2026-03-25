# Layer 3: GPU Reliability & Failure Management

> **Goal**: Build a system that detects GPU failures, recovers automatically, and keeps serving inference with minimal downtime.

## Background

GPUs fail. Not often, but when they do, the failure modes are subtle and varied:

| Failure type | What happens | How it looks to users |
|---|---|---|
| CUDA OOM | GPU runs out of memory mid-inference | HTTP 500 or hung request |
| GPU hang | GPU stops responding to commands | Request times out (524) |
| ECC error | Memory corruption, silent wrong answers | Incorrect model output |
| Driver crash | NVIDIA driver becomes unresponsive | All requests fail |
| Thermal throttle | GPU overheats, slows down | Requests take 5-10x longer |

In Layer 1 we built a health check script. In this layer, we turn that into an **automated monitoring and recovery system**.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Monitoring Layer                                │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ GPU Monitor │  │ vLLM Health Monitor      │  │
│  │ (nvidia-smi)│  │ (inference check)        │  │
│  └──────┬──────┘  └────────────┬─────────────┘  │
│         │                      │                 │
│         ▼                      ▼                 │
│  ┌─────────────────────────────────────────────┐│
│  │ Recovery Controller                         ││
│  │ - Restart vLLM container                    ││
│  │ - Clear GPU memory                          ││
│  │ - Alert & log                               ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  vLLM Server                                     │
│  Auto-restarted when unhealthy                   │
└─────────────────────────────────────────────────┘
```

---

## Prerequisites

- Layer 1 and Layer 2 completed
- `nvidia-smi` available on host
- Docker installed

---

## Part A: GPU Monitoring

### Step 1: Create a GPU Health Monitor Script

**What you're doing**: Building a script that continuously checks GPU health metrics beyond just "is the process running."

**Why**: The Layer 1 health check runs once when you call it. In production, you need continuous monitoring that catches problems as they happen, not when a user complains.

Create `scripts/gpu-monitor.sh`:

```bash
#!/bin/bash
# GPU Health Monitor
# Checks GPU status every INTERVAL seconds and logs warnings/errors

INTERVAL=${1:-10}  # Default: check every 10 seconds
LOG_FILE="logs/gpu-monitor.log"
mkdir -p logs

echo "$(date) - GPU Monitor started (interval: ${INTERVAL}s)" | tee -a "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check 1: Is nvidia-smi responding?
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "$TIMESTAMP [CRITICAL] nvidia-smi not responding - GPU driver may have crashed" | tee -a "$LOG_FILE"
        # Attempt recovery
        ./scripts/recover.sh "driver_crash"
        sleep "$INTERVAL"
        continue
    fi

    # Check 2: GPU temperature
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if [ "$TEMP" -gt 85 ]; then
        echo "$TIMESTAMP [WARNING] GPU temperature: ${TEMP}C (threshold: 85C)" | tee -a "$LOG_FILE"
    fi
    if [ "$TEMP" -gt 95 ]; then
        echo "$TIMESTAMP [CRITICAL] GPU temperature: ${TEMP}C - thermal throttling likely" | tee -a "$LOG_FILE"
    fi

    # Check 3: GPU memory usage
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
    if [ "$MEM_PCT" -gt 95 ]; then
        echo "$TIMESTAMP [WARNING] GPU memory: ${MEM_PCT}% (${MEM_USED}/${MEM_TOTAL} MiB)" | tee -a "$LOG_FILE"
    fi

    # Check 4: GPU utilization (0% for extended time might indicate a hang)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Check 5: ECC errors
    ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    if [ "$ECC_ERRORS" != "N/A" ] && [ "$ECC_ERRORS" != "0" ]; then
        echo "$TIMESTAMP [CRITICAL] ECC uncorrected errors: $ECC_ERRORS" | tee -a "$LOG_FILE"
    fi

    # Check 6: vLLM inference test
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
        http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"OK"}],"max_tokens":1}')

    if [ "$HTTP_CODE" != "200" ]; then
        echo "$TIMESTAMP [ERROR] Inference check failed (HTTP $HTTP_CODE)" | tee -a "$LOG_FILE"
        ./scripts/recover.sh "inference_failed"
    fi

    sleep "$INTERVAL"
done
```

### Step 2: Create the Recovery Script

**What you're doing**: Building an automated recovery script that tries to fix common GPU/vLLM failures.

**Why**: At 3 AM when your GPU hangs, you don't want to wake up. The system should try to fix itself and only alert you if it can't.

Create `scripts/recover.sh`:

```bash
#!/bin/bash
# Recovery script for GPU/vLLM failures
# Usage: ./recover.sh <failure_type>

FAILURE_TYPE=${1:-"unknown"}
LOG_FILE="logs/recovery.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
MAX_RETRIES=3

mkdir -p logs

echo "$TIMESTAMP [RECOVERY] Starting recovery for: $FAILURE_TYPE" | tee -a "$LOG_FILE"

recover_vllm() {
    echo "$TIMESTAMP [RECOVERY] Attempting vLLM restart (attempt $1/$MAX_RETRIES)" | tee -a "$LOG_FILE"

    # Step 1: Stop the container gracefully
    docker stop vllm-server --time 30 2>/dev/null
    docker rm vllm-server 2>/dev/null

    # Step 2: Clear any stuck GPU processes
    # (fuser kills processes using the GPU device files)
    sudo fuser -k /dev/nvidia* 2>/dev/null
    sleep 5

    # Step 3: Restart vLLM
    docker run -d \
        --name vllm-server \
        --gpus all \
        -p 8000:8000 \
        -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
        vllm/vllm-openai:latest \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9

    # Step 4: Wait for model to load
    echo "$TIMESTAMP [RECOVERY] Waiting for model to load..." | tee -a "$LOG_FILE"
    for i in $(seq 1 60); do
        if curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/health | grep -q "200"; then
            echo "$TIMESTAMP [RECOVERY] vLLM is back online after ${i}s" | tee -a "$LOG_FILE"
            return 0
        fi
        sleep 5
    done

    echo "$TIMESTAMP [RECOVERY] vLLM failed to come back after 5 minutes" | tee -a "$LOG_FILE"
    return 1
}

case "$FAILURE_TYPE" in
    "inference_failed")
        for attempt in $(seq 1 $MAX_RETRIES); do
            if recover_vllm "$attempt"; then
                echo "$TIMESTAMP [RECOVERY] Success on attempt $attempt" | tee -a "$LOG_FILE"
                exit 0
            fi
        done
        echo "$TIMESTAMP [RECOVERY] FAILED after $MAX_RETRIES attempts - manual intervention required" | tee -a "$LOG_FILE"
        exit 1
        ;;

    "driver_crash")
        echo "$TIMESTAMP [RECOVERY] GPU driver crash detected - attempting nvidia module reload" | tee -a "$LOG_FILE"
        # This is a last resort — usually requires the machine to be rebooted
        docker stop vllm-server 2>/dev/null
        docker rm vllm-server 2>/dev/null
        sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
        sudo modprobe nvidia
        sleep 10
        if nvidia-smi > /dev/null 2>&1; then
            echo "$TIMESTAMP [RECOVERY] GPU driver reloaded successfully" | tee -a "$LOG_FILE"
            recover_vllm "1"
        else
            echo "$TIMESTAMP [RECOVERY] GPU driver reload FAILED - reboot required" | tee -a "$LOG_FILE"
            exit 1
        fi
        ;;

    *)
        echo "$TIMESTAMP [RECOVERY] Unknown failure type: $FAILURE_TYPE" | tee -a "$LOG_FILE"
        recover_vllm "1"
        ;;
esac
```

---

## Part B: Docker-Level Auto-Restart

### Step 3: Use Docker Restart Policies

**What you're doing**: Configuring Docker to automatically restart vLLM if it crashes.

**Why**: This is the simplest form of self-healing. Docker monitors the container process — if it exits, Docker restarts it. No custom scripts needed for basic crash recovery.

```bash
# Stop and remove the existing container
docker stop vllm-server && docker rm vllm-server

# Re-run with restart policy
docker run -d \
  --name vllm-server \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

**Restart policies explained**:

| Policy | Behavior |
|---|---|
| `no` | Never restart (default) |
| `on-failure` | Restart only if the process exits with non-zero code |
| `unless-stopped` | Always restart, unless you explicitly `docker stop` it |
| `always` | Always restart, even after `docker stop` + daemon restart |

We use `unless-stopped` so the container comes back after crashes and reboots, but stays stopped when you intentionally stop it.

### Step 4: Test Auto-Restart

```bash
# Verify the container is running
docker ps | grep vllm-server

# Kill the process inside the container (simulates a crash)
docker exec vllm-server kill 1

# Wait a few seconds, then check — Docker should have restarted it
sleep 5
docker ps | grep vllm-server
# STATUS should show "Up X seconds" (recently restarted)

# Check restart count
docker inspect vllm-server --format='{{.RestartCount}}'
```

---

## Part C: Docker Compose for the Full Stack

### Step 5: Create a Docker Compose File

**What you're doing**: Defining the entire Layer 1-3 stack (vLLM + nginx + monitor) in a single `docker-compose.yml` so you can start/stop everything together.

**Why**: Managing 3+ containers with individual `docker run` commands is error-prone. Docker Compose gives you one command to start everything, with restart policies, volume mounts, and networking handled declaratively.

Create `docker-compose.yml`:

```yaml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --max-model-len 8192
      --gpu-memory-utilization 0.9
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  api-gateway:
    image: nginx:alpine
    container_name: api-gateway
    network_mode: host
    volumes:
      - ./configs/api-gateway.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      vllm-server:
        condition: service_healthy
    restart: unless-stopped

  gpu-monitor:
    image: nvidia/cuda:12.4.0-base-ubuntu22.04
    container_name: gpu-monitor
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./scripts:/scripts:ro
      - ./logs:/logs
    command: bash /scripts/gpu-monitor.sh 30
    restart: unless-stopped
```

### Step 6: Run the Full Stack

```bash
# Start everything
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f gpu-monitor

# Stop everything
docker compose down
```

---

## Verification Checklist

```bash
# 1. GPU monitor is running and logging
tail -5 logs/gpu-monitor.log

# 2. Auto-restart works
docker exec vllm-server kill 1
sleep 10
docker ps | grep vllm-server  # should be running again

# 3. Health check in docker compose works
docker inspect vllm-server --format='{{.State.Health.Status}}'
# Expected: healthy

# 4. Recovery script works
./scripts/recover.sh "inference_failed"
```

---

## What You've Built

```
Layer 1-2:
  Manual monitoring, manual restart when things break

Layer 3:
  GPU monitoring → auto-detection → auto-recovery → logging
  Docker restart policies for basic crash recovery
  Docker Compose for declarative stack management
```

## Next: Layer 4

Layer 4 covers **Multi-Node GPU Communication** — scaling inference across multiple machines with distributed vLLM.

---

## Reference

- [NVIDIA SMI documentation](https://developer.nvidia.com/system-management-interface)
- [Docker restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/)
- [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
