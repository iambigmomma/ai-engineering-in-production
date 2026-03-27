# Competitive Positioning: AI Production Engineering

> **INTERNAL - DO NOT PUBLISH**
> This document is for Jeff's strategic reference only. Do not share externally or include in public content.

## The Three-Way Complementary Model

| Dimension | Baseten Book (Philip Kiely) | DO Platform | Jeff's Expedition |
|-----------|---------------------------|-------------|-------------------|
| **Core question** | How to make inference fast & cheap | Where to run inference safely | When to use, when NOT to, what if it breaks |
| **Perspective** | Inference platform vendor | Cloud infrastructure provider | Production operations architect |
| **Security/Edge** | Not covered (managed platform) | Security Compliance Governance | Layer 1: Zero-Trust Edge |
| **Routing** | Basic load balancing concepts | Inference Endpoints + Functions | Layer 2: Inference Gateway (4 patterns) |
| **GPU optimization** | Deep theory (quantization, KV cache, batching) | GPU Droplets + DOKS | Layer 3: GPU Ops (operational focus) |
| **Multi-node** | Parallelism theory | Multi-GPU infrastructure | Layer 4: Multi-Node Communication |
| **Disaggregation** | Prefill/Decode theory | Implicit in Inference Functions | Layer 2: Disaggregation-aware routing |

## Key Gaps to Exploit

Baseten's book has ZERO coverage on:
- Security (zero-trust, WAF, auth) — they assume managed platform handles it
- Operational recovery (OOM, GPU crashes, driver issues) — they cover optimization, not operations
- Decision frameworks (when to use vs. when NOT to use) — they cover how, not when

## How to Reference in Content (without naming competitors)

### In articles:
> "Most resources on AI inference focus on optimization — how to make it fast and cheap.
> This series focuses on the complementary challenge: how to run inference workloads
> in production safely, reliably, and at scale."

### In workshops:
> "Today we're covering the production operations layer. There are excellent resources
> for inference optimization theory. What we're covering is what happens AFTER you've
> optimized — how do you expose it securely, route intelligently, and recover when
> things break?"

### In conference talks:
> "The industry has excellent resources for inference optimization. What's missing is
> the production engineering layer — the security, reliability, and operational patterns
> that turn a fast model into a production service."
