# Contributing

Thanks for taking a look at this project. It's a from-scratch learning/portfolio
project first, so contributions that keep it readable and well-tested matter
more here than in most repos.

## Getting set up

```bash
git clone <this repo>
cd distributed-llm
pip install -r requirements.txt
python create_checkpoint.py
python run.py
```

`run.py` → **1 SETUP** walks you through starting a registry and a couple of
local nodes on one machine, so you can try the whole pipeline without any
other hardware.

## Before opening a PR

- **Run the thing.** There's no CI yet (see below), so the best signal right
  now is: does `python run.py` still work end to end — registry, a couple of
  nodes, and a generation through the client?
- **Keep the interface contracts intact.** `forward_embed` / `forward_blocks`
  / `forward_head` is the shared contract every model wrapper (`TinyGPT`,
  `GPT2NodeModel`, `OPTNodeModel`, `LlamaNodeModel`) implements identically —
  if you change one, the others (and `node/server.py`) need to keep working
  unmodified.
- **If you touch model loading or caching,** the strongest way to prove
  correctness is a numerical comparison: run the same input through your
  change and through the old path, and diff the output logits. That's how
  KV caching, quantization, and lazy loading were each verified in this repo
  — "it doesn't crash" isn't the same as "it's correct."
- **Small, focused PRs.** Easier to review, easier to bisect if something
  breaks later.

## Good first issues

These are scoped to be doable without understanding the whole codebase:

- **Add a model family** — pattern-match `models/catalogue.py`'s existing
  entries (pick a HuggingFace model id, fill in size/layer count/RAM
  estimates) and confirm it loads via `python download_model.py`. Llama-family
  models (Mistral, Phi-3, Qwen2, Gemma) usually work with zero changes to
  `models/hf_wrapper.py` since they share the same block structure.
- **A test for `shared/protocol.py`** — send a tensor through
  `send_msg`/`recv_msg` over a real socket pair and assert it round-trips
  unchanged. There's no test suite yet; this would be a good first one.
- **A test for the registry's `/chain` gap detection** — announce nodes with
  a gap in their layer ranges and assert `/chain` returns a 503 with a clear
  error instead of a broken chain.

## Bigger projects, if you want to go deeper

- Per-node shard downloads (each node fetches only its own layers, the
  download-side counterpart to lazy loading)
- Continuous batching across concurrent requests
- Node failure recovery / automatic rerouting mid-generation

See the README's "What's next" section for the full, current list.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened instead,
and the relevant terminal output. If it's a correctness bug (wrong output
rather than a crash), a minimal repro prompt + model helps a lot.
