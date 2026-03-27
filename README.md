# The AI Inference Kitchen
## 從米其林餐廳學 AI Production Engineering

---

## 核心類比

**你在經營一家米其林餐廳。**

每天有成千上萬的客人想來用餐。他們期待：完美的餐點、快速的上菜、24/7 不打烊。

| 餐廳 | AI Inference |
|-----|-------------|
| 客人 | Request |
| 餐點 | Response |
| 主廚 | vLLM |
| 爐灶/烤箱 | GPU |
| 週六晚上 | Traffic spike |
| VIP 客人 | Priority request |
| 菜單 | Model |
| 食譜 | Model weights |

---

## The Story: 從私廚到米其林

### Chapter 0: 在家辦私廚（PoC）

你是一個超棒的廚師。朋友來家裡吃飯，都說你做的菜比餐廳還好吃。

「你應該開一家餐廳！」

**這就是 PoC 階段**：你有一個很厲害的模型（主廚），在本地跑得很好（家裡的廚房）。但是，**在家做菜**和**開餐廳**是完全不同的事。

開餐廳需要五件事：

1. 一道安全的大門
2. 一位聰明的領班
3. 一套設備維運制度
4. 多廚房協作能力
5. 一個管理良好的倉庫

這就是我們的五層架構。

---

### Chapter 1: 餐廳入口（[Layer 1: Zero-Trust Edge](layer-1-edge/)）

**故事**：你的餐廳太火爆了。每天有人排隊到街尾。

| 問題 | 解決方案 | 技術對應 |
|-----|---------|---------|
| 不想讓所有人知道地址 | 私人車道 | **Cloudflare Tunnel** |
| 確認客人身份 | 訂位確認碼 | **Service Auth Token** |
| 控制入場人數 | 每時段限制 | **Rate Limiting** |
| 防止暴民衝入 | 保全團隊 | **DDoS Protection** |
| 一道一道上菜 | 上菜節奏 | **Streaming (SSE)** |

**Tutorial**：「開餐廳第一課：誰能進來？」

---

### Chapter 2: 領班出場（[Layer 2: Inference Gateway](layer-2-inference-gateway/)）

**故事**：客人進門了，但你現在有三個廚房 — 快炒區、主菜區、甜點區。誰來決定每張單送去哪裡？

這就是**領班（Maitre d'）**的工作。他站在入口，看著點單，做四個決策：

| 決策 | 領班怎麼做 | 技術對應 |
|-----|-----------|---------|
| 客人從哪裡進來的？ | 確認是從正門進來、經過保全檢查的 | **Edge-to-pod routing** |
| 這張單該去哪個廚房？ | 看點的是什麼菜，送去對應廚房 | **Model-based routing** |
| 簡單的菜不要浪費大廚 | 炒飯送快炒區，牛排送主菜區 | **Cost-aware routing** |
| 備料和烹飪可以分開 | 備料間先切好，再送去對應爐台 | **Disaggregation-aware routing** |

**Tutorial**：「開餐廳第二課：每張單送去哪個廚房？」

---

### Chapter 3: 設備維運（[Layer 3: GPU Operations](layer-3-gpu-operations/)）

**故事**：週六晚上，餐廳爆滿。突然，主力烤箱壞了。

這不是「會不會壞」的問題，是「壞了怎麼辦」的問題。

| 策略 | 說明 | 技術對應 |
|-----|------|---------|
| 設備健檢 | 每小時巡檢溫度、瓦斯壓力，不等到壞了才發現 | **GPU health monitoring** |
| 備用烤箱 | 壞了馬上切換，客人甚至不會發現 | **OOM recovery** |
| 簡化菜單 | 暫停複雜料理，保住基本出餐 | **Graceful degradation** |
| 換品牌的爐子 | 德國爐換成義大利爐，火候要重新調 | **NVIDIA / AMD portability** |
| 升級設備不停業 | 凌晨換新爐，早上照常開門 | **Driver lifecycle management** |

**Tutorial**：「開餐廳第三課：爐子壞了怎麼辦？」

---

### Chapter 4: 多廚房協作（[Layer 4: Multi-Node GPU Communication](layer-4-multi-node/)）

**故事**：接了一場 200 人的婚宴。一個廚房做不完，三個廚房要同時出菜。

| 挑戰 | 解決方案 | 技術對應 |
|-----|---------|---------|
| 一道菜分多站同做 | 前菜台、主菜台同時處理同一道菜的不同部分 | **Tensor parallelism** |
| 流水線出菜 | 第一站做完傳給第二站 | **Pipeline parallelism** |
| 快速溝通 | 專用對講機頻道，不靠吼的 | **NCCL/RCCL** |
| 走道動線設計 | 廚房之間的傳菜通道要夠寬 | **Inter-node networking** |

**Tutorial**：「開餐廳第四課：多個廚房怎麼協作？」

---

### Chapter 5: 食材倉儲（[Layer 5: Model Storage](layer-5-storage/)）

**故事**：每天開店前，要從倉庫拿食材、切好備料、熱好爐子。第一位客人能多快吃到菜，取決於你的備料做得多好。

| 工作 | 說明 | 技術對應 |
|-----|------|---------|
| 中央倉庫 | 所有分店共用的大冰庫 | **Remote storage (S3/Spaces)** |
| 店內冰箱 | 今天要用的食材先搬過來 | **Local cache (HF cache)** |
| 備料台 | 切好、醃好、隨時能下鍋 | **Pre-download + warm start** |
| 食材標準化 | 所有食材用統一規格包裝，拆開就能用 | **Safetensors format** |
| 多套菜單備料 | 倉庫裡預存不同菜系的食材 | **Multi-model caching** |

**Tutorial**：「開餐廳第五課：開店前要準備什麼？」

---

## 延伸場景

### DDoS Attack
> 「有人雇了 1000 人來門口排隊，但都不點餐。」
> 解法：保全在門口篩選，只讓有訂位的人進來。

### GPU OOM
> 「客人點了超大份量，盤子裝不下。」
> 解法：設定最大份量限制（max_model_len）。

### Cold Start
> 「凌晨 3 點有客人來，但廚師還在睡覺，食材還在倉庫。」
> 解法：要嘛 24hr 待命（keep-warm），要嘛接受第一位客人要等。

### Model Switching
> 「今天突然要換菜單，但備料都是昨天的菜。」
> 解法：倉庫裡預存多套食材（pre-download multiple models）。

---

## Prerequisites

- NVIDIA GPU with working drivers
- Docker with NVIDIA runtime
- HuggingFace account with model access
