"""
AI Sentinel — Live Safe Threat Generator
Generates realistic network reconnaissance (PortScan) and rapid TCP connection bursts
so the live network sniffer can capture real packets from your network interface in real time.
"""

import argparse
import socket
import time


def scan_ports(target_ip="192.168.43.1", start_port=70, end_port=150):
    """Generate a rapid TCP PortScan reconnaissance burst."""
    print(f"\n[!] AI Sentinel Threat Generator: Reconnaissance / PortScan")
    print(f"[*] Target IP:   {target_ip}")
    print(f"[*] Port Range:  {start_port} - {end_port}")
    print(f"[*] Sending probe packets...")

    count = 0
    start = time.time()
    for port in range(start_port, end_port + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.04)
            s.connect_ex((target_ip, port))
            s.close()
            count += 1
        except Exception:
            pass

    elapsed = time.time() - start
    print(f"[+] Completed! Sent {count} probe packets in {elapsed:.2f} seconds.")
    print(f"[*] Check live sniffer output to observe captured reconnaissance flows.\n")


def connection_flood(target_ip="192.168.43.1", port=80, connections=100):
    """Generate a high-rate TCP connection burst."""
    print(f"\n[!] AI Sentinel Threat Generator: Connection Flood")
    print(f"[*] Target IP:   {target_ip}:{port}")
    print(f"[*] Burst size:  {connections} rapid connections")
    print(f"[*] Transmitting packet burst...")

    start = time.time()
    for _ in range(connections):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.03)
            s.connect_ex((target_ip, port))
            s.close()
        except Exception:
            pass

    elapsed = time.time() - start
    print(f"[+] Completed! Sent {connections} connections in {elapsed:.2f} seconds.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate live network traffic for AI Sentinel demo.")
    parser.add_argument("--type", choices=["portscan", "flood"], default="portscan", help="Type of threat traffic to generate")
    parser.add_argument("--target", default="192.168.43.1", help="Target IP for traffic generation")
    args = parser.parse_args()

    if args.type == "portscan":
        scan_ports(target_ip=args.target)
    else:
        connection_flood(target_ip=args.target)

