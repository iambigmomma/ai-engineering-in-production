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
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}')

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
