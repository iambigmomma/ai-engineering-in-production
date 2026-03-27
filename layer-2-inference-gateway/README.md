# Layer 2: Inference Gateway

> The Maitre d' — decides which kitchen station handles your order

**Goal**: Build an intelligent inference gateway that routes requests to the right model, on the right GPU, at the right cost.

## Scope

Four routing patterns for production inference:

1. **Edge-to-pod routing** — Cloudflare tunnel → cloudflared → NGINX → vLLM (Part A-B)
2. **Model-based routing** — Route by model name/size in request body (Part B)
3. **Cost-aware routing** — Small requests → small GPUs, complex → large GPUs (Part E)
4. **Disaggregation-aware routing** — Separate prefill and decode phases to different server pools (Part F)

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

### Step 1: Create the API Gateway Config

**What you're doing**: Building an nginx config that routes requests to different vLLM backends based on the `model` field in the request body.

**Why**: Clients send requests to a single endpoint. The gateway reads the `model` field and forwards to the correct backend. The client doesn't need to know which port or server each model runs on.

```bash
mkdir -p ~/ai-engineering-in-production/layer-2-inference-gateway/configs
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

### Step 2: Run the API Gateway

```bash
docker run -d \
  --name api-gateway \
  --network host \
  -v ~/ai-engineering-in-production/layer-2-inference-gateway/configs/api-gateway.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

### Step 3: Test Model Routing

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

### Step 4: Check Gateway Logs

```bash
docker exec api-gateway cat /var/log/nginx/gateway_access.log
```

You should see entries with `upstream=127.0.0.1:8000` or `upstream=127.0.0.1:8001` depending on the route, along with `response_time` and `req_id`.

---

## Part C: Level 2 — Load Balancing

### Step 1: Scale Up a Model with Multiple Instances

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

### Step 2: Update the Gateway Config for Load Balancing

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

### Step 3: Test Load Balancing

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

### Step 1: Add Backend Health Checks

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

Restart the gateway to apply the changes:

```bash
docker restart api-gateway
```

### Step 2: Test Failover

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

## Part E: Cost-Aware Routing

### Step 1: Understand the Cost-Aware Routing Concept

**What you're doing**: Configuring the gateway to route requests based on estimated complexity — simple requests go to small, cheap GPUs; complex requests go to large, powerful GPUs.

**Why**: Not every request needs your biggest GPU. A simple "translate this sentence" request doesn't need an 8B model on an RTX 6000. Routing it to a 3B model on a smaller GPU saves capacity for the requests that actually need it. This is **GPU Arbitrage** — matching request cost to GPU cost.

The key insight: you already have model-based routing from Part B (using `X-Model-Route` header). Cost-aware routing builds on this by **automatically deciding** which model to use based on the request itself, instead of requiring the client to choose.

### Step 2: Add Request Inspection to the Gateway

**What you're doing**: Adding Lua scripting to nginx that inspects the request body and routes based on estimated complexity.

**Why**: The client sends a request to a single endpoint. The gateway reads the message content, estimates complexity, and routes to the appropriate model. The client doesn't need to know which model or GPU handles its request.

Create a new config file at `configs/cost-aware-gateway.conf`:

```nginx
# Cost-Aware Routing Gateway
# Routes requests based on estimated complexity

upstream model_large {
    server 127.0.0.1:8000;  # 8B model on powerful GPU
}

upstream model_small {
    least_conn;
    server 127.0.0.1:8001;  # 3B model on smaller GPU
    server 127.0.0.1:8002;  # 3B model replica
}

log_format cost_log '$remote_addr - [$time_local] "$request" '
                    '$status $body_bytes_sent '
                    'upstream=$upstream_addr '
                    'response_time=$upstream_response_time '
                    'route=$sent_http_x_route_decision '
                    'req_id=$req_id';

server {
    listen 8090;

    location /v1/chat/completions {
        set $req_id $request_id;
        if ($http_x_request_id) {
            set $req_id $http_x_request_id;
        }

        # Default to small model (cost-efficient)
        set $backend "model_small";
        set $route_decision "small-default";

        # Route to large model based on hints
        # Option 1: Client provides explicit hint
        if ($http_x_model_route = "large") {
            set $backend "model_large";
            set $route_decision "large-client-hint";
        }

        # Option 2: Large max_tokens suggests complex request
        # (nginx can't parse JSON natively without Lua, so we use header-based hints)
        if ($http_x_max_tokens ~ "^[5-9][0-9]{2,}$") {
            set $backend "model_large";
            set $route_decision "large-high-tokens";
        }
        if ($http_x_max_tokens ~ "^[0-9]{4,}$") {
            set $backend "model_large";
            set $route_decision "large-high-tokens";
        }

        proxy_pass http://$backend;
        proxy_set_header X-Request-ID $req_id;
        proxy_set_header Host $host;

        # Pass back the routing decision for observability
        add_header X-Route-Decision $route_decision;

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

        access_log /var/log/nginx/cost_routing_access.log cost_log;
    }

    location /health {
        default_type application/json;
        return 200 '{"status":"ok","routing":"cost-aware"}';
    }

    location /v1/models {
        proxy_pass http://model_large;
    }
}
```

> **Note on complexity detection**: Pure nginx can't parse JSON request bodies. We use two approaches here: (1) the client sends an `X-Model-Route` header as an explicit hint, or (2) the client sends an `X-Max-Tokens` header that nginx can inspect. For full JSON body inspection, you'd use OpenResty (nginx + Lua) or a dedicated routing service — see the "Going Further" section at the end.

### Step 3: Run the Cost-Aware Gateway

```bash
# Stop the existing gateway if running
docker stop api-gateway 2>/dev/null && docker rm api-gateway 2>/dev/null

# Start with cost-aware config on port 8090
docker run -d \
  --name cost-gateway \
  --network host \
  -v ~/ai-engineering-in-production/layer-2-inference-gateway/configs/cost-aware-gateway.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

### Step 4: Test Cost-Aware Routing

```bash
# Test 1: Default route → small model (cost-efficient)
echo "--- Test 1: Default (should route to small) ---"
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Model: {r[\"model\"]}')"

# Test 2: Explicit large hint → large model
echo "--- Test 2: Client hint large ---"
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Model-Route: large" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Explain quantum computing"}],"max_tokens":200}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Model: {r[\"model\"]}')"

# Test 3: High token count hint → large model
echo "--- Test 3: High token count (should route to large) ---"
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Max-Tokens: 1024" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write a long essay"}],"max_tokens":1024}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Model: {r[\"model\"]}')"
```

> **Latency observation**: You'll notice a clear difference in response time across these three tests. Test 1 (simple prompt, small model) returns almost instantly. Test 2 and Test 3 take noticeably longer. This is NOT because nginx is slow at routing decisions — nginx adds less than 1ms of overhead. The latency difference comes entirely from the inference itself: more tokens to generate = more GPU compute time. This is exactly why cost-aware routing matters: if you send a "Say hi" request to the 8B model, you're paying 8B-model latency for a 3B-model task.

### Step 5: Verify Routing Decisions in Logs

```bash
docker exec cost-gateway cat /var/log/nginx/cost_routing_access.log
```

You should see entries with `route=small-default`, `route=large-client-hint`, or `route=large-high-tokens` showing which routing decision was made for each request.

### The Cost-Aware Routing Decision Matrix

| Signal | Route to Small (3B) | Route to Large (8B) |
|--------|---------------------|---------------------|
| No hint provided | Default | — |
| `X-Model-Route: small` | Explicit | — |
| `X-Model-Route: large` | — | Explicit |
| `X-Max-Tokens` < 500 | Default | — |
| `X-Max-Tokens` >= 500 | — | High complexity |
| Short simple prompt | Default | — |
| Long complex prompt | — | Needs full JSON parsing (Lua/custom service) |

---

## Part F: Disaggregation-Aware Routing (Concepts)

> This section is conceptual — it explains the architecture pattern and how it applies to your gateway, without requiring additional infrastructure to run.

### Step 1: Understand Prefill vs Decode Disaggregation

**What you're doing**: Learning the architectural pattern that separates the two phases of LLM inference — prefill and decode — onto different server pools.

**Why**: In standard LLM inference, one GPU handles both phases:

1. **Prefill** (prompt processing) — reads your entire input prompt, computes attention for all tokens at once. This is **compute-bound** — it benefits from raw GPU compute power.
2. **Decode** (token generation) — generates output tokens one at a time, reading from KV cache. This is **memory-bandwidth-bound** — it benefits from fast memory access.

These two phases have completely different hardware requirements:

| Phase | Bottleneck | Ideal GPU | Analogy |
|-------|-----------|-----------|---------|
| **Prefill** | Compute (FLOPS) | High compute, less memory | Prep kitchen — chopping, measuring, mixing (lots of parallel work) |
| **Decode** | Memory bandwidth | High bandwidth, less compute | Line cook — plating one dish at a time (sequential, fast access to ingredients) |

When both phases share one GPU, they fight for resources. Separating them lets each phase run on hardware optimized for its workload.

### Step 2: How Disaggregated Routing Would Work

In a disaggregated architecture, the gateway makes a two-stage routing decision:

```
Client Request
  │
  ▼
┌─────────────────────────────────────────────┐
│  Inference Gateway (Maitre d')              │
│                                             │
│  Stage 1: Route to Prefill Pool             │
│  ┌─────────────┐  ┌─────────────┐          │
│  │ Prefill GPU  │  │ Prefill GPU  │  ...    │
│  │ (compute)    │  │ (compute)    │         │
│  └──────┬──────┘  └──────┬──────┘          │
│         │                │                  │
│  Stage 2: Route to Decode Pool              │
│  ┌─────────────┐  ┌─────────────┐          │
│  │ Decode GPU   │  │ Decode GPU   │  ...    │
│  │ (bandwidth)  │  │ (bandwidth)  │         │
│  └──────┬──────┘  └──────┬──────┘          │
│         │                │                  │
│         ▼                ▼                  │
│  KV Cache Transfer: Prefill → Decode        │
└─────────────────────────────────────────────┘
  │
  ▼
Response back to client
```

### Step 3: Map to Your Current Architecture

Even without a full disaggregated setup, you can start thinking in these terms:

| What you have now | Disaggregated equivalent |
|-------------------|-------------------------|
| 8B model (port 8000) | Could serve as "prefill-optimized" (more compute) |
| 3B model (port 8001) | Could serve as "decode-optimized" (lighter, faster per token) |
| Cost-aware routing (Part E) | Foundation for disaggregation-aware routing |
| NGINX gateway | Would become the disaggregation router |

### Step 4: When to Actually Implement Disaggregation

Disaggregation makes sense when:

- You have **multiple GPUs** with different characteristics (e.g., high-compute vs high-bandwidth)
- Your workload has a **bimodal pattern** — some requests have long prompts (prefill-heavy), others have long outputs (decode-heavy)
- You're running at **scale** where the efficiency gain justifies the architectural complexity
- Your inference framework supports it — **vLLM's disaggregated prefill** feature, **NVIDIA Dynamo**, or **llm-d** on Kubernetes

For a single-GPU setup, standard model-based routing (Part B) and cost-aware routing (Part E) give you most of the benefit with much less complexity.

### Reference Architecture for Future Implementation

When you're ready to implement disaggregation, the nginx config would look something like:

```nginx
# Conceptual — requires vLLM disaggregated prefill support or custom routing service

upstream prefill_pool {
    # GPUs optimized for compute (high FLOPS)
    server prefill-gpu-1:8000;
    server prefill-gpu-2:8000;
}

upstream decode_pool {
    # GPUs optimized for memory bandwidth
    server decode-gpu-1:8000;
    server decode-gpu-2:8000;
}

# In practice, disaggregation routing requires a stateful service
# (not just nginx) because:
# 1. The KV cache from prefill must be transferred to the decode GPU
# 2. The router needs to track which decode GPU has the KV cache
# 3. Subsequent requests in a conversation should go to the same decode GPU
#
# Tools that handle this:
# - vLLM disaggregated prefill (experimental)
# - NVIDIA Dynamo
# - llm-d on Kubernetes
```

---

## Verification Checklist

```bash
# 1. Both models respond through the model-based gateway
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

# 5. Cost-aware routing works (if running Part E)
curl -s -o /dev/null -w "Default (small): %{http_code}\n" \
  http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

curl -s -o /dev/null -w "Hint large: %{http_code}\n" \
  http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Model-Route: large" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# 6. Cost-aware routing logs show decisions
docker exec cost-gateway cat /var/log/nginx/cost_routing_access.log | tail -5
```

---

## What You've Built

```
Layer 1:
  cloudflared → nginx (simple proxy) → vLLM

Layer 2 (model-based):
  cloudflared → API Gateway (routing + logging) → Load Balancer → multiple vLLM instances

Layer 2 (cost-aware):
  cloudflared → Cost-Aware Gateway → small model pool (cheap requests)
                                   → large model pool (complex requests)

Layer 2 (disaggregation — conceptual):
  Gateway → Prefill Pool (compute-heavy GPUs) → KV Cache transfer → Decode Pool (bandwidth-heavy GPUs)
```

You now have a gateway that can:
- Serve multiple models through a single endpoint
- Balance load across instances
- Route around failures automatically
- Route by cost — simple requests to small GPUs, complex to large GPUs
- Understand the disaggregation pattern for future implementation

## Going Further

For full JSON body inspection (routing based on prompt content, not just headers), consider:

- **OpenResty** — nginx + Lua scripting, can parse JSON request bodies
- **Custom routing service** — a lightweight Python/Go service that inspects requests and proxies to the right backend
- **Envoy proxy** — supports Lua filters and WASM for request inspection

## Next: Layer 3

Layer 3 focuses on **GPU Operations** — what happens when a GPU throws an error at 3am, how to detect it before users notice, and how to recover automatically.

---

## Moving to Kubernetes

The routing concepts in this lab map directly to Kubernetes:

- **nginx gateway config** → Becomes a ConfigMap mounted into an NGINX Ingress Controller, or use Kubernetes [Gateway API](https://gateway-api.sigs.k8s.io/) with HTTPRoute resources for native routing rules
- **Multiple vLLM instances** → Each model becomes its own Deployment + Service. Load balancing is handled natively by Kubernetes Services (`type: ClusterIP`)
- **Model-based routing** → Ingress rules or HTTPRoute `matches` based on headers — same `X-Model-Route` header pattern
- **Cost-aware routing** → Same nginx config in a ConfigMap, or implement as a custom controller that watches request metrics
- **Health-aware failover** → Kubernetes readiness probes automatically remove unhealthy pods from Service endpoints — you get this for free
- **Disaggregation** → Frameworks like [llm-d](https://llm-d.ai/) and [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) are built specifically for disaggregated inference on Kubernetes

The nginx configs you wrote in this lab can be used almost unchanged in Kubernetes — they just get mounted as ConfigMaps instead of Docker volumes.

---

## Reference

- [nginx upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)
- [nginx load balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)
- [vLLM multi-model serving](https://docs.vllm.ai/)
- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/serving/disagg_prefill.html)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — disaggregated inference framework
- [llm-d](https://llm-d.ai/) — Kubernetes-native disaggregated inference
