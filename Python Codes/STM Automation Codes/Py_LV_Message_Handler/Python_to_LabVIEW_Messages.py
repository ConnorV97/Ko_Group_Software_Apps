
"""
test_qmh_connection.py
-----------------------
Quick smoke-test for the LabVIEW QMH TCP connection.
Run this AFTER your LabVIEW TCP listener loop is running and waiting.

Usage:
    python test_qmh_connection.py
    python test_qmh_connection.py --host 192.168.1.10 --port 5005
"""

import argparse
import json
import socket
import struct


# --- bare-minimum framing helpers (no class needed for a quick test) --------

def send_msg(sock, message, data=None):
    name = message.encode("utf-8")
    name_frame = struct.pack(">I", len(name)) + name
    payload = json.dumps(data or {}).encode("utf-8")
    data_frame = struct.pack(">I", len(payload)) + payload
    sock.sendall(name_frame + data_frame)


def recv_msg(sock):
    def recv_exactly(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by LabVIEW")
            buf += chunk
        return buf

    length = struct.unpack(">I", recv_exactly(4))[0]
    body   = recv_exactly(length)
    parsed = json.loads(body.decode("utf-8"))
    print(f"  <<< RECEIVED {parsed}")
    return parsed


# --- test cases -------------------------------------------------------------

def run_tests(host, port, timeout):
    print(f"\nConnecting to LabVIEW at {host}:{port} ...")
    with socket.create_connection((host, port), timeout) as sock:
        sock.settimeout(timeout)
        print("Connected.\n")

        # --- Test 3: Fire-and-forget (no reply expected) ----------------------
        # Send a config-style command. LabVIEW just acts on it, no reply.
        # We give it a short window and catch the timeout — that IS the pass.
        print("Test 3 — Scan (Command)")
        send_msg(sock, "Scan")
        sock.settimeout(2.0)   # short timeout — we don't expect a reply
        try:
            unexpected = recv_msg(sock)
            print(f"  NOTE: LabVIEW replied (optional): {unexpected}")
        except (socket.timeout, TimeoutError):
            print("  No reply received — expected for fire-and-forget.")
            print("  PASS\n")
        finally:
            sock.settimeout(timeout)  # restore normal timeout

        print("All tests complete.")


# --- entry point ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LabVIEW QMH TCP connection")
    parser.add_argument("--host",    default="127.0.0.1", help="LabVIEW machine IP")
    parser.add_argument("--port",    default=5005, type=int)
    parser.add_argument("--timeout", default=10.0, type=float, help="Socket timeout (s)")
    args = parser.parse_args()

    run_tests(args.host, args.port, args.timeout)


# class LabVIEWQMHClient:
#     def __init__(self, host= "127.0.0.1", port= 5005, timeout = 10.0):
#         self.host = host
#         self.port = port
#         self.timeout = timeout
#         self.sock = None
#
#     def connect(self):
#         self.sock = socket.create_connection((self.host, self.port), self.timeout)
#         self.sock.settimeout(self.timeout)
#         return self
#
#     def disconnect(self):
#         if self.sock:
#             self.sock.close()
#             self.sock = None
#
#     def __enter__(self):
#         return self.connect()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.disconnect()
#
#
#     def _frame(self,data_bytes):
#
#         """Established 4-byte big-endian length header to byte string"""
#
#         return struct.pack(">I", len(data_bytes)) + data_bytes
#
#     def send_msg(self, message, data=None):
#
#         """Sends two frames to LabVIEW
#         - Frame 1 = message frame (plain UTF-8 string)
#         - Frame 2 = data frame (UTF-8 JSON string)
#         No reply is expected from LabVIEW.
#         """
#         name_frame = self._frame(message.encode("utf-8"))
#         data_frame = salef._frame(json.dumps(data or {}).encode("utf-8"))
#         self.sock.sendall(name_frame + data_frame)
#
#     def run_experiment(self, temperature, field, scan):
#
#         params = {
#           "temperature": temperature,
#           "field": field,
#           "scan": scan
#         }
#
#         self.send_msg("RunExperiment", params)
#
#     def stop(self):
#         self.send_msg("Stop")
#
#     def set_temperature(self, temperature):
#         self.send_msg("SetTemperature", {"temperature": temperature})
#
#     def set_field(self, field):
#         self.send_msg("SetField", {"field": field})



