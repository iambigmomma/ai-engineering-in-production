# Layer 2: Inference Gateway

> The Maitre d' — decides which kitchen station handles your order

**Goal**: Build an intelligent inference gateway that routes requests to the right model, on the right GPU, at the right cost.

## Scope (v2)

Four routing patterns for production inference:

1. **Edge-to-pod routing** — Cloudflare tunnel → cloudflared → NGINX → vLLM (covered below)
2. **Model-based routing** — Route by model name/size in request body (covered below)
3. **Cost-aware routing** — Small requests → small GPUs, complex → large GPUs (planned)
4. **Disaggregation-aware routing** — Separate prefill and decode phases to different server pools (planned)

---

## Background

In Layer 1, we added nginx as a simple reverse proxy between cloudflared and vLLM. It works, but it's a single-purpose passthrough — one model, one backend, no routing logic.

In production, you need to:
- Serve **multiple models** (e.g., a fast small model for simple tasks + a large model for complex ones)
- **Route requests** to the right model based on the request content
- **Load balance** across multiple vLLM instances for the same model
- **Manage API access** with quotas, usage tracking, and key management beyond what Cloudflare Access provides

This is where a two-level proxy architecture comes in.

## Architecture

```
cloudflared (from Layer 1)
  │
  ▼
┌─────────────────────────────────────────────┐
│  Level 1: API Gateway (port 8080)           │
│  ┌─────────────────────────────────────────┐│
│  │ - Request validation                    ││
│  │ - Model routing (/v1/chat/completions)  ││
│  │ - API key management (application-level)││
│  │ - Usage tracking & logging              ││
│  │ - Request/response transformation       ││
│  └────────────────┬────────────────────────┘│
│                   │                          │
│  Level 2: Load Balancer                      │
│  ┌────────────────┼────────────────────────┐│
│  │         ┌──────┴──────┐                 ││
│  │         ▼             ▼                 ││
│  │  ┌────────────┐ ┌────────────┐          ││
│  │  │ vLLM :8000 │ │ vLLM :8001 │  ...     ││
│  │  │ (Model A)  │ │ (Model B)  │          ││
│  │  └────────────┘ └────────────┘          ││
│  └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

### Why two levels?

| Level | Responsibility | Concern |
|---|---|---|
| **Level 1: API Gateway** | Who is calling, what do they want, are they allowed? | Application logic |
| **Level 2: Load Balancer** | Which backend instance should handle this request? | Infrastructure logic |

Separating these concerns means you can change routing rules without touching load balancing, and scale backends without changing API logic.

---

## Prerequisites

- Layer 1 completed (vLLM running, cloudflared tunnel working)
- Docker with NVIDIA runtime
- At least 48GB VRAM (for running 2 models simultaneously, or 1 model with room to spare)

---

## Part A: Multi-Model Setup

### Step 1: Run a Second vLLM Instance with a Smaller Model

**What you're doing**: Running a second, smaller model alongside Llama 3.1 8B. This gives you a fast model for simple tasks and a larger model for complex ones.

**Why**: In production, not every request needs your biggest model. A smaller model responds faster and uses less GPU memory. Routing simple requests to a small model saves GPU capacity for the requests that actually need it.

```bash
# Check current GPU memory usage
nvidia-smi

# Start a smaller model on a different port
docker run -d \
  --name vllm-small \
  --gpus all \
  -p 8001:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.3

# Wait for it to load
docker logs -f vllm-small
```

> **Note on VRAM**: Llama 3.1 8B uses ~16GB and Llama 3.2 3B uses ~6GB. With `gpu-memory-utilization` set to 0.9 and 0.3 respectively, you should have enough room on a 48GB GPU. If not, reduce `--gpu-memory-utilization` on the 8B model to 0.6.

### Step 2: Verify Both Models Are Running

```bash
# Test the 8B model (port 8000)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Test the 3B model (port 8001)
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

## Part B: Level 1 — API Gateway

The API gateway sits in front of all vLLM instances. Every request goes through it. It handles routing, validation, and logging.

We'll use **nginx** with Lua scripting (OpenResty) for this, since we already have nginx experience from Layer 1. For more complex setups, you could use Kong, Envoy, or a custom Python/Go service.

### Step 3: Create the API Gateway Config

**What you're doing**: Building an nginx config that routes requests to different vLLM backends based on the `model` field in the request body.

**Why**: Clients send requests to a single endpoint. The gateway reads the `model` field and forwards to the correct backend. The client doesn't need to know which port or server each model runs on.

```bash
mkdir -p ~/ai-engineering-in-production/layer-2-proxy/configs
```

Create the gateway config at `configs/api-gateway.conf`:

```nginx
# Model routing map
upstream model_8b {
    server 127.0.0.1:8000;
}

upstream model_3b {
    server 127.0.0.1:8001;
}

log_format gateway '$remote_addr - [$time_local] "$request" '
                   '$status $body_bytes_sent '
                   'upstream=$upstream_addr '
                   'response_time=$upstream_response_time '
                   'req_id=$req_id '
                   'model=$http_x_model_name';

server {
    listen 8080;

    location /v1/chat/completions {
        set $req_id $request_id;
        if ($http_x_request_id) {
            set $req_id $http_x_request_id;
        }

        # Default to 8B model
        set $backend "model_8b";

        # Route based on X-Model-Route header
        # (clients set this, or a middleware sets it based on request content)
        if ($http_x_model_route = "small") {
            set $backend "model_3b";
        }

        proxy_pass http://$backend;
        proxy_set_header X-Request-ID $req_id;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;

        # Timeout
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 30s;

        access_log /var/log/nginx/gateway_access.log gateway;
    }

    # Health check endpoint — checks all backends
    location /health {
        default_type application/json;
        return 200 '{"status":"ok"}';
    }

    # List available models
    location /v1/models {
        proxy_pass http://model_8b;
    }
}
```

### Step 4: Run the API Gateway

```bash
docker run -d \
  --name api-gateway \
  --network host \
  -v ~/ai-engineering-in-production/layer-2-proxy/configs/api-gateway.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

### Step 5: Test Model Routing

```bash
# Default route → 8B model
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"What model are you?"}],"max_tokens":20}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Model: {r[\"model\"]}, Response: {r[\"choices\"][0][\"message\"][\"content\"]}')"

# Explicit route to small model
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Model-Route: small" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"What model are you?"}],"max_tokens":20}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Model: {r[\"model\"]}, Response: {r[\"choices\"][0][\"message\"][\"content\"]}')"
```

### Step 6: Check Gateway Logs

```bash
docker exec api-gateway cat /var/log/nginx/gateway_access.log
```

You should see entries with `upstream=127.0.0.1:8000` or `upstream=127.0.0.1:8001` depending on the route, along with `response_time` and `req_id`.

---

## Part C: Level 2 — Load Balancing

### Step 7: Scale Up a Model with Multiple Instances

**What you're doing**: Running multiple instances of the same model for redundancy and throughput.

**Why**: A single vLLM instance has a concurrency limit (`--max-num-seqs`). When all slots are full, new requests queue. Multiple instances let you handle more concurrent requests.

> **Note**: This requires enough VRAM. On a single 48GB GPU, you may not have room for multiple instances of an 8B model. You can either: (a) run multiple instances of the 3B model, or (b) use this as a reference for multi-GPU setups covered in Layer 4.

If you have the VRAM, start a second instance of the 3B model:

```bash
docker run -d \
  --name vllm-small-2 \
  --gpus all \
  -p 8002:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.2
```

### Step 8: Update the Gateway Config for Load Balancing

Update the `model_3b` upstream in `configs/api-gateway.conf`:

```nginx
upstream model_3b {
    least_conn;              # Send to the instance with fewest active connections
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
```

Then restart the gateway:

```bash
docker restart api-gateway
```

### Step 9: Test Load Balancing

```bash
# Send 10 requests to the small model and check which backend handled each
for i in $(seq 1 10); do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-Model-Route: small" \
    -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null &
done
wait

# Check which upstream was used
docker exec api-gateway cat /var/log/nginx/gateway_access.log | tail -10
```

You should see requests distributed between `127.0.0.1:8001` and `127.0.0.1:8002`.

---

## Part D: Health-Aware Routing

### Step 10: Add Backend Health Checks

**What you're doing**: Configuring the gateway to detect when a vLLM instance is unhealthy and stop sending traffic to it.

**Why**: Without health checks, the gateway sends requests to crashed or overloaded backends, causing user-facing errors. With health checks, unhealthy backends are automatically removed from the pool and re-added when they recover.

Update `configs/api-gateway.conf`:

```nginx
upstream model_3b {
    least_conn;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
}
```

- `max_fails=3` — after 3 failed requests, mark the backend as down
- `fail_timeout=30s` — wait 30 seconds before trying the backend again

### Step 11: Test Failover

```bash
# Stop one instance
docker stop vllm-small-2

# Send requests — they should all go to the remaining instance
for i in $(seq 1 5); do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -H "X-Model-Route: small" \
    -H "Content-Type: application/json" \
    http://localhost:8080/v1/chat/completions \
    -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
done
# All should return 200

# Restart the instance
docker start vllm-small-2

# After ~30s, it will be added back to the pool
```

---

## Verification Checklist

```bash
# 1. Both models respond through the gateway
curl -s -o /dev/null -w "8B model: %{http_code}\n" \
  http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

curl -s -o /dev/null -w "3B model: %{http_code}\n" \
  http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Model-Route: small" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# 2. Gateway logs show correct routing
docker exec api-gateway cat /var/log/nginx/gateway_access.log | tail -5

# 3. Load balancing distributes requests
# (check upstream addresses in logs)

# 4. Failover works
# (stop a backend, verify requests still succeed)
```

---

## What You've Built

```
Layer 1:
  cloudflared → nginx (simple proxy) → vLLM

Layer 2:
  cloudflared → API Gateway (routing + logging) → Load Balancer → multiple vLLM instances
```

You now have a gateway that can serve multiple models through a single endpoint, balance load across instances, and automatically route around failures.

## Next: Layer 3

Layer 3 focuses on **GPU Reliability & Failure Management** — what happens when a GPU throws an error, how to detect it, and how to recover automatically.

---

## Reference

- [nginx upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)
- [nginx load balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)
- [vLLM multi-model serving](https://docs.vllm.ai/)
