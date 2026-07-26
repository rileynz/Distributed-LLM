# Admin-Free Portable Worker

The portable worker joins an existing Distributed LLM Universal coordinator
without installing an application, system service, Python, firewall rule, or
device driver. Extract the package into a user-writable folder and run it.

It is intended for computers you own or are authorized to use. It does not
bypass operating-system, school, workplace, endpoint-security, or application
control policies.

## Windows

1. Extract the complete ZIP. Do not run the executable from inside the ZIP.
2. Double-click `DistributedLLM-Worker.exe` or `Start-Portable-Worker.cmd`.
3. Enter the coordinator URL, six-digit join code, and a worker name.
4. Approve the request from the coordinator dashboard.
5. Keep the worker window open while the device contributes to inference.

No administrator prompt is used. Configuration and caches are stored under the
current user's local application-data directory.

To keep the worker's data beside the executable instead, run:

```powershell
DistributedLLM-Worker.exe --portable-data
```

## Linux

```bash
./DistributedLLM-Worker portable-worker \
  --coordinator http://192.168.1.50:7000 \
  --code 123456 \
  --name linux-worker
```

No `sudo` is required. If the archive lost its executable bit, the owner of the
extracted file can restore it with `chmod +x DistributedLLM-Worker`.

## How admin-free networking works

- `llama.cpp` RPC binds only to `127.0.0.1` on the worker.
- The worker opens an authenticated TLS connection to coordinator port `7443`.
- The coordinator exposes a loopback-only proxy endpoint to its own
  `llama-server`.
- Multiple binary RPC streams are multiplexed through the outbound connection.
- The coordinator certificate fingerprint learned during approved enrollment
  is pinned by the worker.
- Node tokens authenticate the connection and can be revoked by removing the
  worker from the dashboard.

The coordinator must be reachable on its control and tunnel ports. The worker
does not accept inbound LAN connections and does not request a firewall change.

## Source-only mode

Install the Python dependency into your own account and use the same code:

```bash
python -m pip install --user -r requirements.txt
python run.py portable-worker \
  --coordinator http://192.168.1.50:7000 \
  --code 123456 \
  --name source-worker
```

## Limitations

- Existing CPU or GPU drivers are used; portable mode cannot install drivers.
- A managed device can still block unknown executables or network connections.
- The worker runs only while the user is signed in and the process is open.
- The initial join flow is intended for a trusted LAN. The persistent RPC
  tunnel is encrypted, but the current management dashboard still uses HTTP.
- Unsigned Windows builds can trigger a Microsoft SmartScreen warning. Code
  signing is a separate release step and does not require changes to worker
  privileges.
