
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json

from scapy.all import IP, TCP, UDP, conf, sniff


WIFI_INTERFACE_INDEX = 16
CAPTURE_DURATION_SECONDS = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "reports" / "live_flows.json"


def get_wifi_interface():
    """Return the selected Wi-Fi interface."""
    interface = conf.ifaces.dev_from_index(WIFI_INTERFACE_INDEX)

    if interface is None:
        raise RuntimeError(
            f"Network interface index {WIFI_INTERFACE_INDEX} was not found."
        )

    if not interface.ip:
        raise RuntimeError(
            f"Interface {interface.name} has no IPv4 address."
        )

    return interface


def capture_flows(duration=CAPTURE_DURATION_SECONDS):
    """Capture IPv4 TCP/UDP packets and group bidirectional flows."""
    interface = get_wifi_interface()
    local_ip = interface.ip
    flows = {}

    def process_packet(packet):
        if IP not in packet:
            return

        ip = packet[IP]

        if TCP in packet:
            transport = packet[TCP]
            protocol = "TCP"
        elif UDP in packet:
            transport = packet[UDP]
            protocol = "UDP"
        else:
            return

        src_ip = ip.src
        dst_ip = ip.dst
        src_port = int(transport.sport)
        dst_port = int(transport.dport)

        # Create a consistent key for both directions of a flow.
        endpoints = tuple(sorted((
            (src_ip, src_port),
            (dst_ip, dst_port),
        )))
        key = (protocol, endpoints)

        timestamp = float(packet.time)
        packet_bytes = int(ip.len)

        # Orient local traffic as local -> remote.
        if src_ip == local_ip:
            forward_packet = True
        elif dst_ip == local_ip:
            forward_packet = False
        else:
            forward_packet = (src_ip, src_port) == endpoints[0]

        if key not in flows:
            if src_ip == local_ip:
                source_ip, source_port = src_ip, src_port
                destination_ip, destination_port = dst_ip, dst_port
            elif dst_ip == local_ip:
                source_ip, source_port = dst_ip, dst_port
                destination_ip, destination_port = src_ip, src_port
            else:
                (source_ip, source_port), (
                    destination_ip, destination_port
                ) = endpoints

            flows[key] = {
                "protocol": protocol,
                "source_ip": source_ip,
                "source_port": source_port,
                "destination_ip": destination_ip,
                "destination_port": destination_port,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "total_packets": 0,
                "total_ip_bytes": 0,
                "forward_packets": 0,
                "backward_packets": 0,
                "forward_ip_bytes": 0,
                "backward_ip_bytes": 0,
                "tcp_flags": defaultdict(int),
                "packet_records": [],
            }

        flow = flows[key]

        flow["first_seen"] = min(flow["first_seen"], timestamp)
        flow["last_seen"] = max(flow["last_seen"], timestamp)
        flow["total_packets"] += 1
        flow["total_ip_bytes"] += packet_bytes

        if forward_packet:
            flow["forward_packets"] += 1
            flow["forward_ip_bytes"] += packet_bytes
            direction = "forward"
        else:
            flow["backward_packets"] += 1
            flow["backward_ip_bytes"] += packet_bytes
            direction = "backward"

        # Record packet-level data for later feature engineering.
        ip_header_length = (
            int(ip.ihl) * 4 if ip.ihl is not None else 20
        )

        if TCP in packet:
            tcp_header_length = (
                int(transport.dataofs) * 4
                if transport.dataofs is not None
                else 20
            )
            tcp_flags = int(transport.flags)
            tcp_window = int(transport.window)

            for flag_value, flag_name in (
                (0x01, "FIN"),
                (0x02, "SYN"),
                (0x04, "RST"),
                (0x08, "PSH"),
                (0x10, "ACK"),
                (0x20, "URG"),
                (0x40, "ECE"),
                (0x80, "CWR"),
            ):
                if tcp_flags & flag_value:
                    flow["tcp_flags"][flag_name] += 1
        else:
            tcp_header_length = 8
            tcp_flags = 0
            tcp_window = None

        flow["packet_records"].append({
            "timestamp": timestamp,
            "direction": direction,
            "ip_length": packet_bytes,
            "ip_header_length": ip_header_length,
            "transport_header_length": tcp_header_length,
            "payload_length": len(transport.payload),
            "tcp_flags": tcp_flags,
            "tcp_window": tcp_window,
        })

    print(f"Interface: {interface.name}")
    print(f"Local IPv4: {local_ip}")
    print(f"Capture duration: {duration} seconds")
    print("Capturing IPv4 TCP/UDP packet metadata...")
    print("Generate ordinary network activity during capture.")

    sniff(
        iface=interface,
        filter="ip",
        timeout=duration,
        store=False,
        prn=process_packet,
    )

    results = []

    for flow in flows.values():
        duration_seconds = max(
            0.0, flow["last_seen"] - flow["first_seen"]
        )

        results.append({
            **flow,
            "first_seen": datetime.fromtimestamp(
                flow["first_seen"], timezone.utc
            ).isoformat(),
            "last_seen": datetime.fromtimestamp(
                flow["last_seen"], timezone.utc
            ).isoformat(),
            "duration_seconds": duration_seconds,
            "tcp_flags": dict(flow["tcp_flags"]),
        })

    results.sort(
        key=lambda item: item["total_packets"],
        reverse=True,
    )

    print(f"\nUnique bidirectional flows: {len(results)}")

    for index, flow in enumerate(results[:20], start=1):
        print(
            f"\nFlow {index}: "
            f"{flow['source_ip']}:{flow['source_port']} -> "
            f"{flow['destination_ip']}:{flow['destination_port']} "
            f"({flow['protocol']})"
        )
        print(f"  Packets: {flow['total_packets']}")
        print(f"  IP bytes: {flow['total_ip_bytes']}")
        print(
            "  Forward/backward packets: "
            f"{flow['forward_packets']}/{flow['backward_packets']}"
        )
        print(f"  Duration: {flow['duration_seconds']:.6f} seconds")
        print(f"  TCP flags: {flow['tcp_flags']}")
        print(f"  Packet records: {len(flow['packet_records'])}")

    return results


if __name__ == "__main__":
    captured_flows = capture_flows()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(captured_flows, file, indent=2)

    print(f"\nSaved flow metadata to: {OUTPUT_PATH}")
