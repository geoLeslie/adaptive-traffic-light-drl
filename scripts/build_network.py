"""Build sumo/network/intersection.net.xml from the plain XML source files.

Usage (from the project root, venv active):
    python scripts/build_network.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import sumolib

ROOT = Path(__file__).resolve().parents[1]
NET_DIR = ROOT / "sumo" / "network"
NAME = "intersection"


def build() -> Path:
    out = NET_DIR / f"{NAME}.net.xml"
    cmd = [
        sumolib.checkBinary("netconvert"),
        "--node-files", str(NET_DIR / f"{NAME}.nod.xml"),
        "--edge-files", str(NET_DIR / f"{NAME}.edg.xml"),
        "--connection-files", str(NET_DIR / f"{NAME}.con.xml"),
        "--tllogic-files", str(NET_DIR / f"{NAME}.tll.xml"),
        "--no-turnarounds", "true",
        "--lefthand", "true",  # Indonesia drives on the left
        "--output-file", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def verify(net_path: Path) -> None:
    """Fail loudly if the built network drifts from the project scope."""
    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    # Every approach: right, straight, left (no U-turns) -> 12 movements.
    expected = {
        (f"{a}2C", f"C2{b}")
        for a in "NSEW" for b in "NSEW" if a != b
    }
    # edge.getOutgoing() -> {to_edge: [connections]}
    found = {
        (e.getID(), to.getID())
        for e in net.getEdges() if e.getID().endswith("2C")
        for to in e.getOutgoing()
    }
    if found != expected:
        sys.exit(f"Unexpected connections: {sorted(found)}")

    # tls.getLinks() -> {link_index: [(in_lane, out_lane, via_lane_id)]}
    tls = net.getTLS("C")
    links = {
        idx: (conns[0][0].getEdge().getID(), conns[0][1].getEdge().getID())
        for idx, conns in tls.getLinks().items()
    }
    print("Connections OK:", sorted(found))
    print("Signal link indices:", dict(sorted(links.items())))
    for prog in tls.getPrograms().values():
        for i, p in enumerate(prog.getPhases()):
            print(f"  phase {i}: {p.state} {p.duration:>4.0f}s {p.name}")


if __name__ == "__main__":
    verify(build())
