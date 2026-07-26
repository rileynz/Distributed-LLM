Distributed LLM Universal 0.3.0 — Linux
===============================================

Debian or Ubuntu:

  sudo apt install ./distributed-llm_0.3.0_all.deb

Other modern 64-bit Linux systems:

  chmod +x Distributed-LLM-Universal-0.3.0-Linux.run
  sudo ./Distributed-LLM-Universal-0.3.0-Linux.run

Desktop systems can then open Distributed LLM from the applications menu.

Create a headless coordinator:

  sudo -u distributed-llm distributed-llm setup --configure-only
  sudo systemctl enable --now distributed-llm

Join a headless worker:

  sudo -u distributed-llm distributed-llm join \
    --coordinator http://COORDINATOR-IP:7000 \
    --code SIX-DIGIT-CODE \
    --name worker-1 \
    --configure-only

  sudo systemctl enable --now distributed-llm

Use llama.cpp RPC only on a trusted private LAN or VPN. Never expose TCP port
50052 to the internet.
