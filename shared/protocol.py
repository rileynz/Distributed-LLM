"""
A tiny length-prefixed message protocol over TCP sockets.

Every message is: [8 bytes = length of payload][pickled payload bytes]

Pickle is used so we can send Python dicts containing PyTorch tensors
without writing our own tensor serialization format. This is fine for
a local demo between machines you trust on your own network — do not
unpickle data from the open internet, since pickle can execute
arbitrary code (see README "Security note" for why and what a real
version would use instead, e.g. JSON + numpy/safetensors).
"""

import pickle
import struct

LENGTH_HEADER_SIZE = 8  # bytes, unsigned 64-bit length prefix


def send_msg(sock, obj):
    payload = pickle.dumps(obj)
    header = struct.pack(">Q", len(payload))
    sock.sendall(header + payload)


def recv_exact(sock, n):
    """Reads exactly n bytes from the socket, or raises ConnectionError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed before expected data arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    header = recv_exact(sock, LENGTH_HEADER_SIZE)
    (length,) = struct.unpack(">Q", header)
    payload = recv_exact(sock, length)
    return pickle.loads(payload)
