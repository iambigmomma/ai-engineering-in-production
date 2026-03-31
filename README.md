# The AI Inference Kitchen
## Learn AI Production Engineering from a Michelin Restaurant

---

## Core Analogy

**You're running a Michelin-starred restaurant.**

Thousands of guests want to dine every day. They expect: perfect dishes, fast service, and 24/7 availability.

| Restaurant | AI Inference |
|-----------|-------------|
| Guest | Request |
| Dish | Response |
| Head Chef | vLLM |
| Stove / Oven | GPU |
| Saturday night | Traffic spike |
| VIP guest | Priority request |
| Menu | Model |
| Recipe | Model weights |

---

## The Story: From Home Cooking to Michelin Star

### Chapter 0: Home Kitchen (PoC)

You're an amazing cook. Every friend who eats at your place says your food is better than any restaurant.

"You should open a restaurant!"

**This is the PoC stage**: you have a great model (head chef) that runs well locally (your home kitchen). But **cooking at home** and **running a restaurant** are completely different things.

Running a restaurant requires five things:

1. A secure front door
2. A smart maitre d'
3. An equipment maintenance system
4. Multi-kitchen coordination
5. A well-managed warehouse

This is our five-layer architecture.

---

### Chapter 1: The Restaurant Entrance ([Layer 1: Zero-Trust Edge](layer-1-edge/))

**Story**: Your restaurant is wildly popular. People queue down the street every day.

| Problem | Solution | Technical Mapping |
|---------|----------|-------------------|
| Don't want everyone to know the address | Private driveway | **Cloudflare Tunnel** |
| Verify guest identity | Reservation confirmation code | **Service Auth Token** |
| Control admission rate | Per-timeslot limits | **Rate Limiting** |
| Prevent mob rushes | Security team | **DDoS Protection** |
| Serve course by course | Pacing the service | **Streaming (SSE)** |

**Tutorial**: "Restaurant Lesson 1: Who gets in?"

---

### Chapter 2: The Maitre d' ([Layer 2: Inference Gateway](layer-2-inference-gateway/))

**Story**: Guests are in the door, but now you have three kitchens — stir-fry station, main course station, and dessert station. Who decides where each order goes?

That's the **Maitre d'**'s job. Standing at the entrance, reading the order ticket, making four decisions:

| Decision | What the Maitre d' Does | Technical Mapping |
|----------|------------------------|-------------------|
| Where did the guest come from? | Confirm they entered through the front door and passed security | **Edge-to-pod routing** |
| Which kitchen gets this order? | Read what was ordered, send to the right kitchen | **Model-based routing** |
| Don't waste the head chef on simple dishes | Fried rice goes to stir-fry, steak goes to main course | **Cost-aware routing** |
| Separate prep from cooking | Prep kitchen cuts ingredients first, then sends to the right station | **Disaggregation-aware routing** |

**Tutorial**: "Restaurant Lesson 2: Which kitchen gets each order?"

---

### Chapter 3: Equipment Operations ([Layer 3: GPU Operations](layer-3-gpu-operations/))

**Story**: Saturday night, the restaurant is packed. Suddenly, the main oven breaks down.

This isn't a question of "will it break" — it's "what do we do when it breaks."

| Strategy | Description | Technical Mapping |
|----------|-------------|-------------------|
| Equipment health checks | Hourly inspections of temperature and gas pressure — don't wait until it breaks | **GPU health monitoring** |
| Backup oven | Switch immediately — guests won't even notice | **OOM recovery** |
| Simplified menu | Pause complex dishes, keep basic service running | **Graceful degradation** |
| Swap oven brands | Switch from German to Italian ovens — heat calibration needs adjustment | **NVIDIA / AMD portability** |
| Upgrade without closing | Replace ovens at midnight, open as usual in the morning | **Driver lifecycle management** |

**Tutorial**: "Restaurant Lesson 3: What happens when the oven breaks?"

---

### Chapter 4: Multi-Kitchen Coordination ([Layer 4: Multi-Node GPU Communication](layer-4-multi-node/))

**Story**: You've booked a 200-person wedding banquet. One kitchen can't handle it — three kitchens need to serve simultaneously.

| Challenge | Solution | Technical Mapping |
|-----------|----------|-------------------|
| One dish prepared across multiple stations | Appetizer and main course stations work on different parts of the same dish simultaneously | **Tensor parallelism** |
| Assembly-line service | Station 1 finishes and passes to Station 2 | **Pipeline parallelism** |
| Fast communication | Dedicated radio channels, no shouting | **NCCL/RCCL** |
| Traffic flow design | Passageways between kitchens must be wide enough | **Inter-node networking** |

**Tutorial**: "Restaurant Lesson 4: How do multiple kitchens coordinate?"

---

### Chapter 5: Ingredient Storage ([Layer 5: Model Storage](layer-5-storage/))

**Story**: Every day before opening, you retrieve ingredients from the warehouse, prep them, and preheat the ovens. How fast the first guest gets served depends on how well you've prepped.

| Task | Description | Technical Mapping |
|------|-------------|-------------------|
| Central warehouse | A large shared cold storage for all branches | **Remote storage (S3/Spaces)** |
| In-store fridge | Move today's ingredients in advance | **Local cache (HF cache)** |
| Prep station | Cut, marinate, ready to cook at any moment | **Pre-download + warm start** |
| Standardized ingredients | All ingredients in uniform packaging — open and use | **Safetensors format** |
| Multi-menu prep | Pre-stock ingredients for different cuisines in the warehouse | **Multi-model caching** |

**Tutorial**: "Restaurant Lesson 5: What to prepare before opening?"

---

## Extended Scenarios

### DDoS Attack
> "Someone hired 1,000 people to queue at the door, but none of them order."
> Solution: Security screens at the door — only those with reservations get in.

### GPU OOM
> "A guest orders a mega-sized portion — the plate can't hold it."
> Solution: Set a maximum portion size (`max_model_len`).

### Cold Start
> "A guest shows up at 3 AM, but the chef is asleep and ingredients are still in the warehouse."
> Solution: Either keep staff on standby 24/7 (keep-warm), or accept that the first guest has to wait.

### Model Switching
> "Today's menu changes suddenly, but all the prep is for yesterday's dishes."
> Solution: Pre-stock multiple sets of ingredients in the warehouse (pre-download multiple models).

---

## Prerequisites

- NVIDIA GPU with working drivers
- Docker with NVIDIA runtime
- HuggingFace account with model access
