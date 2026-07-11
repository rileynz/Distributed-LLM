# Distributed LLM

Split AI inference across multiple machines. Each machine runs a slice of the model, passes its result to the next, and together they generate text that none could run alone.

Built from scratch — custom transformer, TCP wire protocol, live registry, and streaming output. The same core idea as [Petals](https://github.com/bigscience-workshop/petals) and [SPAN's XFRA](https://span.io), at a learnable scale.

---

## Quick start

```bash
pip install -r requirements.txt
python create_checkpoint.py   # run once
python run.py                 # interactive launcher
```

The launcher shows a live dashboard — registry status, connected nodes, chain health — and guides you through everything else.

**Startup order:** Registry → Nodes → Client. Each in its own terminal.

---

## How it works

1. The **registry** is the network phonebook. Nodes announce their layer range on startup; the coordinator asks for the current live chain before each request.
2. Each **node** loads the full model weights but only computes its assigned layer range. It receives a tensor from the previous node, runs its layers, and forwards the result to the next.
3. The **coordinator** tokenises a prompt, sends it to the first node, and waits for logits to come back through the whole chain. Tokens are streamed to the terminal as they're generated.
4. **Hot reload** — run `python train.py` or `python download_model.py` at any time. Nodes detect the change and reload weights within a few seconds, no restart needed.

---

## Two machines on the same wifi

1. Copy the whole folder (including `model.pt`) to both machines.
2. Start the registry on machine A — note its local IP (`ipconfig` on Windows, `ip addr` on Linux/Mac).
3. On each machine, run `python run.py` → **3 NODE** → enter the registry IP when prompted. The launcher auto-assigns the right layer range.
4. Run the client from either machine.

**Firewall:** Windows will prompt to allow the port — click Allow. On the same local network, no router config is needed. For machines on different networks, [Tailscale](https://tailscale.com) (free) makes them act like they're on the same LAN.

---

## Downloading real AI models

```bash
python run.py   # → 5 MODELS
```

The model manager benchmarks your network, recommends models that will fit your hardware, and downloads them across all nodes simultaneously. Switching to a new model hot-reloads into running nodes automatically.

**Supported model families:** GPT-2 (all sizes), DistilGPT-2, GPT-Neo, OPT, TinyLlama.

**Recommended starting point:** TinyLlama 1.1B — 2.2 GB download, runs on a single laptop with 4GB+ RAM, produces genuinely coherent output. Far better than any GPT-2 variant at the same size.

---

## Training the built-in model

The default `model.pt` has random weights (output is noise). To teach it real language:

```bash
python train.py --data data/sample.txt --steps 1500
```

Use a bigger text file for much better results. Any plain-text file works — public domain books from [gutenberg.org](https://www.gutenberg.org) are ideal. 500KB+ of text with 3000+ steps will produce recognisable word patterns.

Running nodes pick up the new weights automatically.

---

## File guide

| File | Purpose |
|---|---|
| `run.py` | Live dashboard + interactive launcher for all options |
| `registry/server.py` | HTTP phonebook — nodes announce here, coordinator queries here |
| `node/server.py` | Node agent — loads model slice, serves inference, hot-reloads |
| `coordinator/client.py` | Sends prompts, streams output, prints per-node timing |
| `download_model.py` | Benchmark → recommend → distributed download → switch model |
| `setup_wizard.py` | First-time guided setup with firewall configuration |
| `train.py` | Train the built-in model on a text file |
| `create_checkpoint.py` | Create a fresh random-weight checkpoint |
| `models/catalogue.py` | List of downloadable models with hardware requirements |
| `models/hf_wrapper.py` | Adapter that makes HuggingFace models speak the node protocol |
| `shared/protocol.py` | Length-prefixed TCP wire format for sending tensors between nodes |
| `config.py` | Model shape and defaults |
| `ui.py` | Terminal colour helpers, box drawing, spinner |

---

## Registry API

| Endpoint | Method | Description |
|---|---|---|
| `/announce` | POST | Node registers: `{host, port, start_layer, end_layer, label, hw_specs}` |
| `/heartbeat` | POST | Node keepalive every 5s |
| `/chain` | GET | Ordered list of live nodes — 503 if layers have gaps |
| `/status` | GET | All nodes, alive/dead, requests served, hardware specs |
| `/network_specs` | GET | Hardware specs across all alive nodes (used by recommender) |
| `/health` | GET | Liveness check |

---

## Honest limitations

**Central registry** — if the registry goes down, no new requests can be routed. In-flight requests complete fine. A full peer-to-peer DHT (like Petals uses) would remove this single point of failure but is significantly more complex.

**No load balancing** — requests always go through the same linear chain. Two parallel requests queue up rather than being split across spare capacity.

**No KV cache** — each new token reprocesses the full sequence from scratch. This is the biggest reason generation is slower than it could be.

**Pickle for tensor transport** — fine on a trusted local network; never expose node ports to the open internet.

---

## What's next

- KV cache (biggest speed win)
- Live web dashboard (map of nodes, request flow animation)
- Node failure recovery and automatic rerouting
- More model families (Mistral, Phi-3)
- Streaming HTTP API so other apps can use the network
