# Layer 1: Zero-Trust Edge for AI Inference

> **Goal**: Deploy a local vLLM inference server, expose it securely through a Cloudflare Tunnel, then harden it with 6 production-grade fixes.

## Background & Inspiration

This lab is inspired by [tunnels-for-ai-inference](https://github.com/rxsalad/tunnels-for-ai-inference), which demonstrates how Cloudflare Tunnel can expose vLLM inference servers to the internet without opening inbound ports.

This lab adapts the same idea so you can run it on **a single GPU server with Docker** — no Kubernetes cluster needed. On top of the basic setup, we layer in 6 production hardening practices:

| What we start with | What we add |
|---|---|
| vLLM serving Llama 3.1 8B | - |
| Cloudflare Tunnel (outbound-only) | - |
| HTTPS enforcement | - |
| - | Fix 1: Authentication (Service Tokens) |
| - | Fix 2: Rate Limiting |
| - | Fix 3: SSE Streaming Support |
| - | Fix 4: Timeout Tuning |
| - | Fix 5: 3-Layer Health Checks |
| - | Fix 6: Observability (Request ID) |

---

## Prerequisites

- NVIDIA GPU with working drivers (`nvidia-smi` should show your GPU)
- Docker with NVIDIA runtime installed
- HuggingFace account with:
  - An access token ([create here](https://huggingface.co/settings/tokens))
  - Llama 3.1 license accepted ([accept here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

---

## Part A: Basic Setup (Get Inference Working Through a Tunnel)

### Step 1: Verify Your GPU

**What you're doing**: Confirming Docker can see and use your NVIDIA GPU.

**Why**: vLLM needs GPU access. If this doesn't work, nothing else will.

```bash
nvidia-smi
```

You should see your RTX 6000 Ada with 48GB VRAM. Then verify Docker can access it:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this prints the same GPU info, you're good. If it fails, your NVIDIA Container Toolkit isn't set up correctly.

### Step 2: Start vLLM Server

**What you're doing**: Running vLLM as a Docker container that serves Llama 3.1 8B with an OpenAI-compatible API.

**Why**: vLLM is the industry standard for high-throughput LLM serving. It uses PagedAttention to maximize GPU memory efficiency. The OpenAI-compatible API means any client that works with OpenAI will work with your server.

```bash
export HF_TOKEN=<your_token_here>

docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

**Key parameters explained**:
- `--gpus all` — gives the container access to all GPUs
- `--max-model-len 8192` — max sequence length (input + output tokens). Higher = more VRAM
- `--gpu-memory-utilization 0.9` — use 90% of VRAM. Leaves 10% headroom to avoid OOM

**Wait for the model to load** (takes a few minutes the first time as it downloads ~16GB):

```bash
docker logs -f vllm-server
```

Look for a line like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to exit the log stream once you see it.

### Step 3: Test Local Inference

**What you're doing**: Sending a chat completion request directly to vLLM to confirm it works.

**Why**: Always verify locally before adding network layers. If this fails, the problem is vLLM, not the tunnel.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in 10 words or less"}],
    "max_tokens": 50
  }'
```

You should get a JSON response with the model's reply. Also test the health endpoint:

```bash
curl http://localhost:8000/health
```

This should return HTTP 200 with an empty body.

### Step 4: Install cloudflared

**What you're doing**: Installing the Cloudflare Tunnel client on your machine.

**Why**: `cloudflared` creates an outbound-only encrypted connection to Cloudflare's edge network. Traffic flows: Internet -> Cloudflare Edge -> Tunnel -> Your Machine. No inbound ports needed.

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### Step 5: Create a Quick Tunnel

**What you're doing**: Creating a temporary public URL that points to your local vLLM server.

**Why**: Quick tunnels require no Cloudflare account, no domain, no configuration. Perfect for testing. In production you'd use a persistent tunnel with a custom domain.

```bash
cloudflared tunnel --url http://localhost:8000
```

This will output something like:
```
Your quick Tunnel has been created! Visit it at:
https://random-words.trycloudflare.com
```

**Keep this terminal open** — the tunnel dies when you close it.

> **Important**: Quick tunnels (`trycloudflare.com`) are for testing only. They do NOT support Cloudflare Access (authentication, rate limiting, etc.). You will need to upgrade to a persistent tunnel before starting Part B. Don't skip ahead trying to set up Service Tokens on a quick tunnel — it won't work.

### Step 6: Test Through the Tunnel

**What you're doing**: Sending the same inference request, but now through the public internet via Cloudflare.

**Why**: This proves the full path works: your request goes to Cloudflare's edge, through the encrypted tunnel, to your local vLLM server, and back.

Open a **new terminal** and run:

```bash
curl https://<YOUR_TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello from the tunnel!"}],
    "max_tokens": 50
  }'
```

If you get a response, congratulations — you now have a GPU inference server accessible from anywhere in the world.

**The problem**: So does everyone else. Anyone with your URL can use your GPU for free. That's what Part B fixes.

---

## Upgrade: Quick Tunnel to Persistent Tunnel

Part A uses a quick tunnel (`trycloudflare.com`) — it's temporary and doesn't support Cloudflare Access (authentication, rate limiting, etc.). Before starting Part B, you need to upgrade to a **persistent tunnel** with your own domain.

### Prerequisites for this step

- A **Cloudflare account** (free tier works)
- A **domain** you own, added to Cloudflare (e.g., `isfusion.cloud`)
- The `cloudflared` CLI installed (done in Step 4)

### Step 7: Authenticate cloudflared

**What you're doing**: Linking your local `cloudflared` CLI to your Cloudflare account.

**Why**: Quick tunnels are anonymous. Persistent tunnels need to know which Cloudflare account and domain to use.

```bash
cloudflared tunnel login
```

This opens a browser window. Select your domain (e.g., `isfusion.cloud`) and authorize. After success, a certificate is saved to `~/.cloudflared/cert.pem`.

### Step 8: Create a Named Tunnel

**What you're doing**: Creating a persistent tunnel that won't disappear when you close the terminal.

**Why**: Quick tunnels give you a random URL that changes every time. A named tunnel gets a fixed Tunnel ID that you can attach to your own domain.

```bash
cloudflared tunnel create ai-inference
```

Output will look like:
```
Created tunnel ai-inference with id abcd1234-5678-90ab-cdef-1234567890ab
```

**Save this Tunnel ID** — you'll need it for the config file. A credentials file is also created at `~/.cloudflared/<TUNNEL_ID>.json`.

### Step 9: Route Your Domain to the Tunnel

**What you're doing**: Creating a DNS record that points your subdomain to the tunnel.

**Why**: This is what connects `inference.isfusion.cloud` (public) to your tunnel (private). Cloudflare creates a CNAME record automatically.

```bash
cloudflared tunnel route dns ai-inference inference.isfusion.cloud
```

You can verify in the Cloudflare Dashboard → DNS → Records. You should see a CNAME pointing to `<TUNNEL_ID>.cfargotunnel.com`.

### Step 10: Create the Tunnel Config File

**What you're doing**: Telling cloudflared which traffic to route where.

**Why**: The config file defines the mapping: requests to `inference.isfusion.cloud` get forwarded to `http://localhost:8000` (your vLLM server).

```bash
cat > ~/.cloudflared/config.yml << EOF
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /home/$USER/.cloudflared/<YOUR_TUNNEL_ID>.json

ingress:
  - hostname: inference.isfusion.cloud
    service: http://localhost:8000
  - service: http_status:404
EOF
```

### Step 11: Run the Persistent Tunnel

```bash
cloudflared tunnel run ai-inference
```

**Keep this terminal open** (or run it as a systemd service — see below).

Test it:
```bash
curl https://inference.isfusion.cloud/health
```

Should return HTTP 200.

#### (Optional) Run as a systemd service

So you don't need to keep a terminal open:

```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared  # auto-start on boot
```

### Step 12: Stop the Quick Tunnel

You can now close the quick tunnel terminal from Step 5. You no longer need it.

---

## Part B: Production Hardening (6 Fixes)

### Architecture After Hardening

```
Internet
  │
  ▼
┌─────────────────────────────────────────────┐
│  Cloudflare Edge                            │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Auth (Fix 1)│→ │ Rate Limit (Fix 2)   │  │
│  └─────────────┘  └──────────────────────┘  │
│         │                                   │
│  ┌──────▼──────────────────────────────────┐│
│  │ Timeout Tuning (Fix 4)                  ││
│  │ CF-Ray Header (Fix 6)                   ││
│  └─────────────────────────────────────────┘│
└────────────────────┬────────────────────────┘
                     │ Tunnel (outbound-only)
                     ▼
┌─────────────────────────────────────────────┐
│  Your Machine                               │
│  ┌─────────────────────────────────────────┐│
│  │ cloudflared                             ││
│  │  SSE passthrough (Fix 3)                ││
│  │  Timeout tuning (Fix 4)                 ││
│  └────────────────┬────────────────────────┘│
│                   ▼                          │
│  ┌─────────────────────────────────────────┐│
│  │ nginx :8080 (Fix 6)                     ││
│  │  X-Request-ID generation & logging      ││
│  │  SSE support / Timeout tuning           ││
│  └────────────────┬────────────────────────┘│
│                   ▼                          │
│  ┌─────────────────────────────────────────┐│
│  │ vLLM :8000                              ││
│  │  Health Checks (Fix 5)                  ││
│  └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

### Fix 1: Authentication (Service Tokens)

**Problem**: Your persistent tunnel endpoint is open to anyone who knows the URL.

**Solution**: Use Cloudflare Access with Service Tokens. Requests are authenticated at Cloudflare's edge — unauthorized traffic never reaches your server.

**Steps**:

#### 1a. Create a Service Token

1. Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/)
2. Navigate to **Access** → **Service Auth** → **Service Tokens**
3. Click **Create Service Token**, give it a name (e.g., `ai-inference-token`)
4. **Immediately copy and save** the `CF-Access-Client-Id` and `CF-Access-Client-Secret`. The secret is only shown once.

#### 1b. Create an Access Application

1. In Zero Trust Dashboard, go to **Access** → **Applications**
2. Click **Add an application** → Select **Self-hosted**
3. Configure the application:
   - **Application name**: `AI Inference`
   - **Session Duration**: `24 hours` (or your preference)
   - **Application domain**: `inference.isfusion.cloud` (your tunnel hostname)
4. Click **Next** to configure the policy

#### 1c. Create the Access Policy (this is the critical part)

1. **Policy name**: `Service Token Auth`
2. **Action**: Select **Service Auth** (NOT "Allow" — this is the most common mistake)
3. **Include rule**:
   - Selector: **Service Token**
   - Value: Select the token you created in Step 1a
4. Click **Save**

> **Why "Service Auth" and not "Allow"?** The `Allow` action requires an identity (email login, SSO, etc.). `Service Auth` is specifically designed for machine-to-machine access using `CF-Access-Client-Id` / `CF-Access-Client-Secret` headers. If you use `Allow`, the Service Token headers will be recognized but the request will still be redirected to a login page.

4. **Test without token** (should get blocked):
```bash
curl https://<YOUR_TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "test"}]}'
# Should return 403 or redirect to login page
```

5. **Test with token** (should work):
```bash
curl https://<YOUR_TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <your_client_id>" \
  -H "CF-Access-Client-Secret: <your_client_secret>" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "test"}]}'
```

**Why Service Tokens, not API keys?**: Service Tokens are Cloudflare-native and checked at the edge before traffic even reaches your tunnel. API keys would require vLLM or a middleware to validate them, which means unauthorized traffic still hits your server.

Save your token config for later:

```bash
# Save to layer-1-edge/configs/access-token.env (DO NOT commit this file)
cat > configs/access-token.env << 'EOF'
CF_ACCESS_CLIENT_ID=<your_client_id>
CF_ACCESS_CLIENT_SECRET=<your_client_secret>
EOF
```

#### Troubleshooting Fix 1

**Got a 302 redirect even with correct token headers?**

Check your Access Policy action. This is the #1 mistake:

| Action | Behavior with Service Token |
|---|---|
| `Service Auth` | Validates the token headers, grants access |
| `Allow` | Ignores the token headers, redirects to login page |

If you see `HTTP 302` with `location: https://...cloudflareaccess.com/cdn-cgi/access/login/...`, your policy action is set to `Allow`. Change it to `Service Auth`.

**curl command not working?**

Three common mistakes when writing multi-line curl commands:

```bash
# WRONG: spaces after backslash (invisible but breaks the command)
curl https://example.com \
  -H "Content-Type: application/json"

# WRONG: missing closing quote
curl https://example.com \
  -H "CF-Access-Client-Secret: abc123 \
  -d '...'

# WRONG: JSON content split across lines
curl https://example.com \
  -d '{"content": "Hello from
  the world"}'
```

Safest approach: put the JSON body on a single line:

```bash
curl https://inference.isfusion.cloud/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <your_id>" \
  -H "CF-Access-Client-Secret: <your_secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Fix 2: Rate Limiting

**Problem**: Even authenticated users can send unlimited requests, exhausting your GPU.

**Solution**: Configure rate limiting in Cloudflare WAF.

> **Free plan note**: Cloudflare's free plan only allows **1 rate limiting rule**. The most important one is Burst Protection — it prevents someone (or a buggy script) from hammering your GPU with rapid-fire requests. If you upgrade to Pro or higher, you can add Sustained and Daily rules as well.

**Steps**:

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → select your domain (`isfusion.cloud`)
2. Left menu: **Security** → **WAF**
3. Top tab: **Rate limiting rules**
4. Click **Create rule**
5. Fill in:

| Field | Value |
|---|---|
| **Rule name** | `Burst Protection` |
| **If incoming requests match...** | Field: `URI Path`, Operator: `contains`, Value: `/v1/chat/completions` |
| **Rate** | `10` requests per `10 seconds` |
| **Then take action** | `Block` |
| **Duration** | `60 seconds` |

6. **Double-check the Expression Preview** before deploying. It should show exactly:
   ```
   (http.request.uri.path contains "/v1/chat/completions")
   ```
   > **Common pitfall**: If you copy-paste the path value, you might accidentally include a trailing space. The expression would become `"/v1/chat/completions "` (with a space at the end) and **never match any request**, because the actual URI path doesn't end with a space. If your rule shows 0 events, check for this first.

7. Click **Deploy**

**Test the burst limit**:

A single batch of 15 requests may not trigger the block due to how Cloudflare counts — there is a slight delay between when the threshold is exceeded and when blocking kicks in. To reliably test, send two rounds back-to-back:

```bash
# Round 1: fill up the quota
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "R1-$i: %{http_code}\n" \
    -H "CF-Access-Client-Id: <your_id>" \
    -H "CF-Access-Client-Secret: <your_secret>" \
    -H "Content-Type: application/json" \
    https://inference.isfusion.cloud/v1/chat/completions \
    -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' &
done
wait

# Round 2: should be blocked
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "R2-$i: %{http_code}\n" \
    -H "CF-Access-Client-Id: <your_id>" \
    -H "CF-Access-Client-Secret: <your_secret>" \
    -H "Content-Type: application/json" \
    https://inference.isfusion.cloud/v1/chat/completions \
    -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' &
done
wait
# Round 1 should return 200, Round 2 should return 429
```

> **Note**: Cloudflare's rate limit counter is not instant. It won't block exactly at request #11. Instead, it detects the threshold was exceeded and blocks the **next batch** of requests. This is normal — the protection still works, just with a slight delay.

**Why burst protection is the priority**: Your endpoint already requires a Service Token (Fix 1), so random internet traffic can't reach it. The main risk is an authenticated client sending too many requests too fast — a runaway loop, a misconfigured retry, or testing gone wrong. Burst protection catches all of these.

#### (Pro plan and above) Additional rules

If you have Cloudflare Pro or higher, add these two rules for deeper protection:

| Rule name | Path Match | Rate | Action |
|---|---|---|---|
| `Sustained Protection` | URI Path contains `/v1/chat/completions` | 100 requests per 1 minute | Managed Challenge |
| `Daily Limit` | URI Path starts with `/v1/` | 1000 requests per 1 hour | Block (1 hour) |

### Fix 3: Streaming Support (SSE)

**Problem**: LLM responses can take seconds to generate. Without streaming, users stare at a blank screen until the full response is ready.

**Solution**: Enable Server-Sent Events (SSE) streaming. vLLM already supports this natively — you just need to make sure cloudflared and any proxy in front of it don't buffer the response.

**Steps**:

1. **Test streaming locally** (should already work):
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write a haiku about GPUs"}],"max_tokens":50,"stream":true}'
```

You should see tokens arriving one by one as `data: {...}` lines. The `-N` flag disables curl's output buffering so you see them in real-time.

2. **Test streaming through the tunnel**:
```bash
curl -N https://<YOUR_DOMAIN>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <your_id>" \
  -H "CF-Access-Client-Secret: <your_secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write a haiku about GPUs"}],"max_tokens":50,"stream":true}'
```

3. **If streaming is choppy through the tunnel**, make sure your cloudflared config (see `configs/cloudflared-config.yml`) has `disableChunkedEncoding: false` in the `originRequest` section. This is already included in the config we created in the Upgrade section.

**Why this matters for AI**: ChatGPT, Claude, and every production LLM product uses streaming. Users perceive streaming as 10x faster even though total generation time is the same. First-token latency becomes the key metric.

### Fix 4: Timeout Tuning

**Problem**: Default timeouts (30-60s) will kill long-context requests. A prompt with 4K+ tokens on Llama 3.1 8B can easily take 60-120 seconds.

**Solution**: Increase timeouts at every layer of the stack.

**Where timeouts live**:

| Layer | Default | What to set | Where to configure |
|---|---|---|---|
| Cloudflare proxy | 100s | Cannot change on free plan | Cloudflare Dashboard (Enterprise only) |
| cloudflared | 30s connect, 90s keepalive | `connectTimeout: 30s`, `keepAliveTimeout: 120s` | `~/.cloudflared/config.yml` (see `configs/cloudflared-config.yml`) |
| nginx (if using Fix 6) | 60s | `proxy_read_timeout: 300s` | `configs/nginx-proxy.conf` |
| vLLM | No timeout | Control via `--max-model-len` and `--max-num-seqs` | Docker run command |

The cloudflared and nginx timeouts are already configured in the config files we created earlier (`configs/cloudflared-config.yml` and `configs/nginx-proxy.conf`).

**Test with a long request**:

```bash
time curl https://<YOUR_DOMAIN>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <your_id>" \
  -H "CF-Access-Client-Secret: <your_secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write a detailed 2000-word essay about the history of GPU computing."}],"max_tokens":2048}'
```

If this completes without a `524` (timeout) error, your tuning is working. On an RTX 6000 Ada, this takes around 25-30 seconds — well within Cloudflare's 100-second free plan limit. Slower GPUs or higher concurrency may need Enterprise-level timeout adjustments.

**Why this matters**: The #1 production issue with AI inference behind reverse proxies is timeout errors on long requests. A 2048-token generation can take 30-90 seconds depending on GPU and load. If any layer in the chain has a shorter timeout, the user gets a cryptic `524 A timeout occurred` error from Cloudflare.

### Fix 5: Health Checks (3-Layer)

**Problem**: You have no way to know if the system is actually healthy. "The server is running" doesn't mean "the model is loaded and inference is working."

**Solution**: Implement 3 layers of health checks, each catching a different type of failure.

#### What the 3 layers check

| Layer | What it checks | How | What failure it catches |
|---|---|---|---|
| **Layer 1: Process** | Is the vLLM container running? | `docker ps` | Container crashed, OOM killed, or never started |
| **Layer 2: API** | Is the vLLM HTTP server responding? | `curl /health` | Model still loading, process hung, port not bound |
| **Layer 3: Inference** | Can the model actually generate text? | Send a real chat completion | GPU error (CUDA OOM), model corrupted, weights not loaded |

Each layer builds on the previous one. Layer 1 passing but Layer 2 failing means the container is up but the server inside isn't ready. Layer 2 passing but Layer 3 failing means the API is alive but the GPU/model has a problem — this is the sneakiest failure because everything *looks* fine until you try inference.

#### Create the health check script

```bash
# Script: scripts/health-check.sh
cat > scripts/health-check.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "=== Layer 1: Process Check ==="
if docker ps --format '{{.Names}}' | grep -q vllm-server; then
  echo "PASS: vllm-server container is running"
else
  echo "FAIL: vllm-server container is not running"
  exit 1
fi

echo ""
echo "=== Layer 2: API Check ==="
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)
if [ "$HTTP_CODE" = "200" ]; then
  echo "PASS: /health returned 200"
else
  echo "FAIL: /health returned $HTTP_CODE"
  exit 1
fi

echo ""
echo "=== Layer 3: Inference Check ==="
RESPONSE=$(curl -s -w '\n%{http_code}' http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 5
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ] && echo "$BODY" | grep -q '"choices"'; then
  echo "PASS: Inference working"
  echo "Response: $(echo $BODY | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$BODY")"
else
  echo "FAIL: Inference check failed (HTTP $HTTP_CODE)"
  exit 1
fi

echo ""
echo "=== All health checks passed ==="
SCRIPT

chmod +x scripts/health-check.sh
```

2. **Run it**:
```bash
./scripts/health-check.sh
```

3. **Cloudflare-side health check**: If using a persistent tunnel, configure health checks in the Cloudflare dashboard to periodically hit `/health` and alert you if it goes down.

**Why 3 layers?**: Process check catches crashes. API check catches model loading failures. Inference check catches GPU errors (e.g., CUDA OOM) that don't crash the process but silently break inference.

### Fix 6: Observability (Request ID Propagation)

**Problem**: When a request fails, you can't trace it from the client through Cloudflare to vLLM. Debugging is guesswork.

**Solution**: Add an nginx reverse proxy between cloudflared and vLLM that generates and logs request IDs. Combined with Cloudflare's built-in `CF-Ray` header, you get end-to-end traceability.

**How request tracing works after this fix**:

```
Client sends request
  │
  ▼
Cloudflare Edge ── adds CF-Ray header (e.g., cf-ray: 9e0ff0baad41-YYZ)
  │
  ▼
cloudflared tunnel
  │
  ▼
nginx (port 8080) ── generates X-Request-ID (or uses client-provided one)
  │                   logs: req_id=test-123 in access log
  ▼
vLLM (port 8000) ── processes inference
```

**Steps**:

#### 6a. Verify Cloudflare's built-in tracing

Every request through Cloudflare already gets a `CF-Ray` header:

```bash
curl -s -i https://<YOUR_DOMAIN>/health \
  -H "CF-Access-Client-Id: <your_id>" \
  -H "CF-Access-Client-Secret: <your_secret>" 2>&1 | grep cf-ray

# Output: cf-ray: 9e0ff0baad4139c3-YYZ
```

This gives you Cloudflare-side tracing. But you also need server-side tracing.

#### 6b. Create the nginx proxy config

The nginx config does three things: generates a `X-Request-ID` for every request (or uses one the client provides), logs it, and also handles SSE streaming and timeout tuning (from Fix 3 and Fix 4).

The config file is at `configs/nginx-proxy.conf`:

```nginx
log_format req_id_log '$remote_addr - [$time_local] "$request" '
                       '$status $body_bytes_sent '
                       'req_id=$req_id';

server {
    listen 8080;

    location / {
        # Generate a unique request ID if not present
        set $req_id $request_id;
        if ($http_x_request_id) {
            set $req_id $http_x_request_id;
        }

        proxy_pass http://localhost:8000;
        proxy_set_header X-Request-ID $req_id;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE support (Fix 3)
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;

        # Timeout tuning (Fix 4)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 30s;

        # Logging with request ID
        access_log /var/log/nginx/vllm_access.log req_id_log;
    }
}
```

> **Important**: The `log_format` directive must be outside the `server` block (nginx requires it at the `http` level). The config above works because Cloudflare loads it into `/etc/nginx/conf.d/`, which is inside the `http` block of the main nginx config.

#### 6c. Run nginx as a sidecar

```bash
docker run -d \
  --name vllm-proxy \
  --network host \
  -v ~/ai-engineering-in-production/layer-1-edge/configs/nginx-proxy.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

> Use the full path to avoid issues with `$(pwd)` pointing to the wrong directory.

#### 6d. Update cloudflared to point to nginx

In your `~/.cloudflared/config.yml`, change the service from port 8000 to 8080:

```yaml
ingress:
  - hostname: <YOUR_DOMAIN>
    service: http://localhost:8080   # was 8000, now points to nginx
```

Then restart cloudflared for the change to take effect.

#### 6e. Test Request ID propagation

```bash
# Send a request with a custom Request ID
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -H "X-Request-ID: test-123" \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <your_id>" \
  -H "CF-Access-Client-Secret: <your_secret>" \
  https://<YOUR_DOMAIN>/v1/chat/completions \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'

# Check nginx log — you should see req_id=test-123
docker exec vllm-proxy cat /var/log/nginx/vllm_access.log | grep "test-123"
```

Expected log output:
```
127.0.0.1 - [24/Mar/2026:15:47:14 +0000] "POST /v1/chat/completions HTTP/1.1" 200 600 req_id=test-123
```

**Why this matters**: In production, when a user reports "my request failed," you need to trace that specific request through Cloudflare (`CF-Ray`), through nginx (`X-Request-ID`), to vLLM logs. Without this, you're grep-ing through thousands of log lines by timestamp and guessing which one is the right one.

---

## Verification Checklist

After completing all fixes, run through this checklist:

```bash
# 1. Health check passes
./scripts/health-check.sh

# 2. Unauthenticated request is blocked
curl -s -o /dev/null -w '%{http_code}' https://<TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"test"}]}'
# Expected: 403

# 3. Authenticated request works
curl -s -o /dev/null -w '%{http_code}' https://<TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <id>" \
  -H "CF-Access-Client-Secret: <secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":5}'
# Expected: 200

# 4. Streaming works through tunnel
curl -N https://<TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <id>" \
  -H "CF-Access-Client-Secret: <secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"count to 10"}],"max_tokens":50,"stream":true}'
# Expected: data: chunks arriving incrementally

# 5. Long request doesn't timeout
time curl https://<TUNNEL_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "CF-Access-Client-Id: <id>" \
  -H "CF-Access-Client-Secret: <secret>" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write 500 words about Linux."}],"max_tokens":1024}'
# Expected: completes without 524 error

# 6. Request ID propagates
curl -i https://<TUNNEL_URL>/health \
  -H "X-Request-ID: verify-123" \
  -H "CF-Access-Client-Id: <id>" \
  -H "CF-Access-Client-Secret: <secret>"
# Expected: see cf-ray in response, verify-123 in nginx logs
```

---

## What You've Built

```
Basic setup:
  Internet → Cloudflare → Tunnel → vLLM

After hardening:
  Internet → Auth → Rate Limit → Tunnel → nginx (Request ID + SSE + Timeout) → vLLM
  (authenticated, rate-limited, observable, streaming, resilient)
```

## Next: Layer 2

With the edge layer secured, Layer 2 introduces a **Two-Level Proxy Architecture** — adding an API gateway between cloudflared and vLLM for request routing, load balancing, and model management.

---

## Moving to Kubernetes

The concepts in this lab apply directly to Kubernetes deployments:

- **Cloudflare Tunnel** → Run `cloudflared` as a Kubernetes Deployment, or use the [Cloudflare Tunnel Operator](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/deploy-tunnels/deployment-guides/kubernetes/) with a Helm chart
- **nginx reverse proxy** → Use an NGINX Ingress Controller or a standalone nginx Deployment with your config in a ConfigMap
- **Service Tokens & Rate Limiting** → These are configured in Cloudflare's dashboard, not on your server — they work identically regardless of whether the backend is Docker or Kubernetes
- **Health checks** → Map directly to Kubernetes liveness and readiness probes on your vLLM pods
- **vLLM container** → Becomes a Deployment with GPU resource requests (`nvidia.com/gpu: 1`)

The key insight: everything you configured here (nginx config, cloudflared tunnel, Cloudflare Access policies) stays the same. Kubernetes only changes how you deploy and manage the containers, not what they do.

---

## Reference

- [rxsalad/tunnels-for-ai-inference](https://github.com/rxsalad/tunnels-for-ai-inference) — Inspiration for this lab
- [Cloudflare Tunnel Docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Cloudflare Access Service Tokens](https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/)
