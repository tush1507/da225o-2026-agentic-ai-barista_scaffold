# Block 4 — Agent Design Patterns

Block 3 gave you the foundation: typed state, specialist agents, and explicit routing. Block 4 applies that foundation to three patterns that appear repeatedly in real production systems. Each option is self-contained — you can study them independently or in order.

---

## Option A — Memory Pattern (`option_a_memory.py`)

### The problem
Block 3's pipeline is stateless across sessions. Every invocation starts fresh — the agent has no idea whether this is the customer's first visit or their fiftieth. Personalisation is impossible without memory.

### The pattern
An external store (here a JSON file, in production a database or vector store) holds a user profile keyed by `user_id`. A `PreferenceAgent` runs **before** the order pipeline and:
1. Loads the profile from the store
2. Generates a personalised greeting based on past visits
3. Writes `preferences` and `personalized_greeting` to state

The downstream `OrderAgent` reads those fields and uses them to fill in defaults (e.g. preferred milk) before the user even specifies them.

```
[START]
   ↓
[PreferenceAgent]  ← reads from external store, writes preferences to state
   ↓
[OrderAgent+Memory]  ← reads preferences, places order, saves updated profile
   ↓
[END]
```

### Key insight
The memory store is accessed as a **tool**, not a direct function call. This means swapping the backing store (JSON → SQLite → Redis → a vector DB for semantic memory) requires changing only `load_preferences()` and `save_preferences()` — the agent code is untouched.

### What to observe
Run the script three times with the same `user_id`. Watch how the greeting changes from "Welcome!" to "Welcome back!" and how the agent starts suggesting the customer's previous order.

---

## Option B — Dynamic Tool + Shared State (`option_b_surge.py`)

### The problem
Pricing in the real world isn't static. Demand, time of day, queue length — all of these should influence what a customer pays. But BillingAgent in Block 3 hardcodes a 10% discount. How do you inject live data into a pipeline without coupling agents together?

### The pattern
`SurgeAgent` runs **before** BillingAgent. It:
1. Calls a live tool (`get_queue_length`) to read current demand
2. Applies business rules (queue 0–10: normal, 11–20: +15%, 20+: +30%)
3. Writes `surge_multiplier` and `surge_reason` to state

`BillingAgent` reads `surge_multiplier` from state and multiplies the base price — it has **no knowledge** that a SurgeAgent exists.

```
[START]
   ↓
[SurgeAgent]  ← reads live queue length, writes surge_multiplier to state
   ↓
[BillingAgent]  ← reads surge_multiplier from state, computes final price
   ↓
[END]
```

### Key insight
Agents communicate via **state, not direct calls**. SurgeAgent never calls BillingAgent. BillingAgent never calls SurgeAgent. You could replace SurgeAgent with a WeatherAgent, a TimeOfDayAgent, or a LoyaltyAgent — BillingAgent's code stays the same.

### What to observe
Run the script several times. Because `get_current_queue_length()` is randomised, you'll see different surge levels (and prices) on each run. This simulates a real-world dynamic pricing system.

---

## Option C — Parallel Fan-out (`option_c_parallel.py`)

### The problem
Block 3's pipeline is strictly sequential — OrderAgent finishes, then InventoryAgent starts, then BillingAgent. But `InventoryAgent` and `NutritionAgent` are completely independent: neither needs the other's output. Running them sequentially wastes time.

### The pattern
A `fan_out` node forks execution into two simultaneous branches. LangGraph runs both in parallel and automatically waits for both to complete before allowing `merge_and_bill` to run.

```
[START]
   ↓
[fan_out]
   ↙         ↘
[inventory]  [nutrition]   ← run simultaneously
   ↘         ↙
  [merge_and_bill]         ← LangGraph waits for both
   ↓
[END]
```

### Key insight
The fork is just two `add_edge` calls from the same source node — LangGraph handles the synchronisation automatically. Adding a third parallel branch (e.g. `AllergenAgent`) requires zero changes to existing agents:

```python
graph.add_edge("fan_out", "allergen")
graph.add_edge("allergen", "merge_and_bill")
```

### What to observe
The elapsed time printed at the end should be roughly the time of one agent call, not two — because both run concurrently. Compare this to a sequential version where you'd pay the latency cost twice.

---

## Pattern Comparison

| | Option A | Option B | Option C |
|---|---|---|---|
| **Core idea** | Persist knowledge across sessions | Inject live data into pricing | Run independent agents simultaneously |
| **State role** | Carries memory into downstream agents | Carries live signal between agents | Carries partial results until merge |
| **New concept** | External memory store as tool | Dynamic tool + state handoff | Parallel edges + automatic join |
| **Real-world analogy** | CRM / customer history | Ride-share surge pricing | Microservices fan-out |

---

## Combining the Patterns

These patterns compose. A production barista system might use all three:

1. **Memory** (Option A) — personalise before ordering
2. **Surge** (Option B) — adjust price based on demand
3. **Parallel** (Option C) — check inventory and nutrition simultaneously

The LangGraph state and node model scales to all of this without changing the agent code — only `graph.py` and `state.py` need to grow.
