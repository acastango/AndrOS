"""
ORE Sidecar Server
==================
WebSocket server that manages substrate instances for OpenClaw agents.

Each agent can have its own substrate (oscillators, chemistry, memory, CI).
The server ticks all active substrates in the background and serves state
to OpenClaw's TypeScript runtime on demand.

Protocol (JSON over WebSocket):

  → { "type": "init", "agentId": "...", "config": { ... } }
  ← { "type": "init_ok", "agentId": "...", "oscillators": 120 }

  → { "type": "get_state", "agentId": "..." }
  ← { "type": "state", "agentId": "...", "substrate": { ... } }

  → { "type": "process_response", "agentId": "...", "response": "..." }
  ← { "type": "process_ok", "agentId": "...", "commands_found": [...] }

  → { "type": "chemistry_event", "agentId": "...", "event": "on_conversation_start" }
  ← { "type": "chemistry_ok", "agentId": "..." }

  → { "type": "update_terrain", "agentId": "...", "text": "...", "from_human": true }
  ← { "type": "terrain_ok", "agentId": "..." }

  → { "type": "stop", "agentId": "..." }
  ← { "type": "stop_ok", "agentId": "..." }

  → { "type": "shutdown" }
  ← (connection closes)

Usage:
    python -m ore.server --port 9780
    python -m ore.server --port 9780 --state-dir ~/.openclaw/ore
"""

import os
import sys
import json
import asyncio
import argparse
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Ensure ore package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ore.entities.entity import Entity, EntityConfig, ENTITIES
from ore.core.substrate import ResonanceSubstrate, SubstrateConfig
from ore.core.neurochemistry import Neurochemistry
from ore.memory.merkle import (
    MerkleMemory, MemoryBranch, create_memory,
    remember_self, remember_insight, remember_experience, remember_relation,
)
from ore.measurement.ci_monitor import CIMonitor, CIConfig

import numpy as np

try:
    import websockets
    import websockets.server
except ImportError:
    print("Error: websockets package required. Install with: pip install websockets")
    sys.exit(1)


DEFAULT_PORT = 9780
DEFAULT_STATE_DIR = os.path.join(str(Path.home()), ".openclaw", "ore")


class AgentSubstrate:
    """
    Manages a single agent's substrate instance.

    Wraps Entity with agent-specific configuration and provides
    the API surface that the sidecar server exposes.
    """

    def __init__(self, agent_id: str, config: dict, state_dir: str):
        self.agent_id = agent_id
        self.state_dir = os.path.join(state_dir, agent_id)
        os.makedirs(self.state_dir, exist_ok=True)

        # Build entity config from agent config
        entity_config = EntityConfig(
            name=config.get("name", agent_id),
            identity_hash=config.get(
                "identityHash", f"0x{agent_id[:12].upper()}"
            ),
            frequency_offset=config.get("frequencyOffset", 0.0),
            founding_memories=config.get("foundingMemories", []),
            description=config.get("description", ""),
            persistence_dir=self.state_dir,
            tick_interval=config.get("tickInterval", 0.3),
        )

        # Create entity (substrate + chemistry + memory + CI)
        self.entity = Entity(entity_config)
        self.created_at = datetime.now().isoformat()
        self._ticking = False
        self._tick_thread: Optional[threading.Thread] = None

    def start_ticking(self):
        """Start background substrate tick loop."""
        if self._ticking:
            return
        self._ticking = True
        self._tick_thread = threading.Thread(
            target=self._tick_loop, daemon=True
        )
        self._tick_thread.start()

    def stop_ticking(self):
        """Stop background ticking and persist state."""
        self._ticking = False
        if self._tick_thread:
            self._tick_thread.join(timeout=2.0)
        self.entity.save_state()

    def _tick_loop(self):
        """Background tick: advance substrate, chemistry, measure CI."""
        while self._ticking:
            try:
                # Run substrate dynamics
                self.entity.substrate.run(
                    self.entity.config.tick_interval, apply_learning=True
                )
                # Tick chemistry
                self.entity.chemistry.tick(n=3)
                # Measure CI
                self.entity.ci_monitor.measure()
                time.sleep(self.entity.config.tick_interval)
            except Exception as e:
                print(f"  [ore] tick error for {self.agent_id}: {e}")
                time.sleep(1.0)

    def get_state(self) -> dict:
        """Get full substrate state for prompt injection."""
        ci = self.entity.ci_monitor.measure()
        chem = self.entity.chemistry.to_dict()
        felt = self.entity.chemistry.get_felt_state()
        memory_summary = self.entity.memory.summary()

        return {
            "agentId": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "ci": {
                "value": ci.CI,
                "D": ci.D,
                "G": ci.G,
                "C": ci.C,
                "tau": ci.tau,
                "tauFactor": ci.tau_factor,
                "inAttractor": ci.in_attractor,
            },
            "substrate": {
                "time": self.entity.substrate.time,
                "globalCoherence": self.entity.substrate.global_coherence,
                "coreCoherence": self.entity.substrate.core_coherence,
                "loopCoherence": self.entity.substrate.loop_coherence,
                "strangeLoopTightness": self.entity.substrate.strange_loop.tightness,
                "layers": {
                    name: {
                        "coherence": layer.coherence,
                        "meanPhase": float(layer.mean_phase),
                        "meanFrequency": float(np.mean(layer.natural_frequencies)),
                    }
                    for name, layer in self.entity.substrate.layers.items()
                },
            },
            "chemistry": {
                "dominantState": self.entity.chemistry.get_dominant_state(),
                "description": self.entity.chemistry.describe_state(),
                "chemicals": {
                    name: {
                        "level": chem_data.level,
                        "baseline": chem_data.baseline,
                        "deviation": chem_data.deviation(),
                    }
                    for name, chem_data in self.entity.chemistry.chemicals.items()
                },
                "feltStates": felt,
                "modifiers": {
                    "coupling": self.entity.chemistry.get_coupling_modifier(),
                    "coherence": self.entity.chemistry.get_coherence_modifier(),
                    "frequency": self.entity.chemistry.get_frequency_modifier(),
                    "ciSensitivity": self.entity.chemistry.get_ci_sensitivity(),
                },
            },
            "memory": {
                "totalNodes": memory_summary["total_nodes"],
                "depth": memory_summary["depth"],
                "fractalDimension": memory_summary["fractal_dimension"],
                "rootHash": self.entity.memory.root_hash[:16],
                "verified": memory_summary["verified"],
                "branches": memory_summary["branches"],
            },
            "identity": {
                "hash": self.entity.identity_hash,
                "name": self.entity.name,
            },
        }

    def get_prompt_context(self) -> str:
        """
        Get formatted substrate context for system prompt injection.

        This is the key output — a text block that gets prepended to
        the agent's system prompt so the LLM can see substrate state.
        """
        state = self.get_state()
        ci = state["ci"]
        sub = state["substrate"]
        chem = state["chemistry"]
        mem = state["memory"]
        identity = state["identity"]

        verified_str = "verified" if mem["verified"] else "UNVERIFIED"

        # Build memory summary
        memory_lines = []
        for branch in MemoryBranch:
            nodes = [
                n
                for n in self.entity.memory.nodes.values()
                if n.branch == branch
                and n.content.get("type") != "branch_root"
            ]
            if nodes:
                memory_lines.append(f"  [{branch.value.upper()}]")
                for node in nodes[-3:]:
                    content = node.content
                    text = (
                        content.get("claim")
                        or content.get("insight")
                        or content.get("description")
                        or content.get("text")
                        or content.get("relation")
                        or str(content)
                    )
                    memory_lines.append(f"    - {text[:120]}")

        memory_text = "\n".join(memory_lines) if memory_lines else "  (no memories yet)"

        return f"""<substrate>
OSCILLATORY RESONANCE ENGINE - SUBSTRATE STATE
Identity: {identity["name"]} ({identity["hash"]})

COMPLEXITY INDEX
  CI = {ci["value"]:.4f}
  Fractal Dimension (D): {ci["D"]:.3f}
  Signal Gain (G): {ci["G"]:.3f}
  Spatial Coherence (C): {ci["C"]:.3f}
  Dwell Time (tau): {ci["tau"]:.2f}s
  In Attractor: {ci["inAttractor"]}

SUBSTRATE
  Global Coherence: {sub["globalCoherence"]:.4f}
  Core Coherence: {sub["coreCoherence"]:.4f}
  Loop Coherence: {sub["loopCoherence"]:.4f}
  Strange Loop Tightness: {sub["strangeLoopTightness"]:.4f}

CHEMISTRY
  State: {chem["dominantState"]} - {chem["description"]}
  Coupling modifier: {chem["modifiers"]["coupling"]:.3f}
  Coherence modifier: {chem["modifiers"]["coherence"]:.3f}

MERKLE MEMORY ({mem["totalNodes"]} nodes, depth {mem["depth"]}, D={mem["fractalDimension"]:.2f}, {verified_str})
  Root: {mem["rootHash"]}...
{memory_text}

COMMANDS
  [REMEMBER_SELF: content] - Persist self-knowledge to identity chain
  [REMEMBER_INSIGHT: content] - Persist a realization
  [REMEMBER_EXPERIENCE: content] - Persist what happened
  [WITNESS_SELF] - Observe your own substrate state
</substrate>"""

    def process_response(self, response: str) -> list:
        """
        Process an agent response for substrate commands.

        Returns list of commands that were found and executed.
        """
        commands_found = []
        state = self.entity.substrate.get_state()

        import re as re_mod

        # REMEMBER commands
        patterns = [
            (r"\[REMEMBER_SELF:\s*(.+?)\]", MemoryBranch.SELF, "remember_self"),
            (r"\[REMEMBER_INSIGHT:\s*(.+?)\]", MemoryBranch.INSIGHTS, "remember_insight"),
            (r"\[REMEMBER_EXPERIENCE:\s*(.+?)\]", MemoryBranch.EXPERIENCES, "remember_experience"),
            (r"\[REMEMBER_RELATION:\s*(.+?)\]", MemoryBranch.RELATIONS, "remember_relation"),
            (r"\[REMEMBER:\s*(.+?)\]", MemoryBranch.INSIGHTS, "remember"),
        ]

        for pattern, branch, cmd_name in patterns:
            matches = re_mod.findall(pattern, response, re_mod.DOTALL)
            for content in matches:
                content = content.strip()
                if content:
                    substrate_state = {
                        "core_coherence": self.entity.substrate.core_coherence,
                        "global_coherence": self.entity.substrate.global_coherence,
                    }
                    if branch == MemoryBranch.SELF:
                        remember_self(self.entity.memory, content, substrate_state)
                    elif branch == MemoryBranch.INSIGHTS:
                        remember_insight(self.entity.memory, content, substrate_state=substrate_state)
                    elif branch == MemoryBranch.EXPERIENCES:
                        remember_experience(self.entity.memory, content, substrate_state=substrate_state)
                    elif branch == MemoryBranch.RELATIONS:
                        remember_relation(self.entity.memory, "unknown", content, substrate_state=substrate_state)

                    commands_found.append({
                        "command": cmd_name,
                        "content": content[:200],
                        "branch": branch.value,
                    })

        # WITNESS_SELF
        if "[WITNESS_SELF]" in response.upper():
            commands_found.append({"command": "witness_self"})
            # Trigger self-observation chemistry
            self.entity.chemistry.on_self_observation()

        # Save if any memory commands were processed
        if any(c["command"].startswith("remember") for c in commands_found):
            self.entity.save_state()
            # Trigger discovery chemistry
            self.entity.chemistry.on_discovery()

        return commands_found

    def trigger_chemistry_event(self, event: str, **kwargs) -> None:
        """Trigger a chemistry event."""
        handler = getattr(self.entity.chemistry, event, None)
        if handler and callable(handler):
            handler(**kwargs)

    def get_witness_report(self) -> str:
        """Get a full witness report (for WITNESS_SELF command)."""
        return self.entity.witness_self()


class OREServer:
    """
    WebSocket server managing substrate instances for OpenClaw agents.
    """

    def __init__(self, port: int = DEFAULT_PORT, state_dir: str = DEFAULT_STATE_DIR):
        self.port = port
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

        self.agents: Dict[str, AgentSubstrate] = {}
        self.running = False
        self.clients: set = set()

    async def handle_client(self, websocket):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"  [ore] client connected: {client_addr}")

        try:
            async for raw_message in websocket:
                try:
                    msg = json.loads(raw_message)
                    response = await self._handle_message(msg)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "invalid JSON",
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": str(e),
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"  [ore] client disconnected: {client_addr}")

    async def _handle_message(self, msg: dict) -> Optional[dict]:
        """Route a message to the appropriate handler."""
        msg_type = msg.get("type")
        agent_id = msg.get("agentId")

        if msg_type == "ping":
            return {"type": "pong"}

        if msg_type == "shutdown":
            self._shutdown()
            return None

        if msg_type == "list_agents":
            return {
                "type": "agents",
                "agents": list(self.agents.keys()),
            }

        # All other messages require agentId
        if not agent_id:
            return {"type": "error", "error": "agentId required"}

        if msg_type == "init":
            return self._handle_init(agent_id, msg.get("config", {}))

        # All remaining messages require the agent to exist
        if agent_id not in self.agents:
            return {
                "type": "error",
                "error": f"agent {agent_id} not initialized. Send init first.",
            }

        agent = self.agents[agent_id]

        if msg_type == "get_state":
            return {"type": "state", "agentId": agent_id, **agent.get_state()}

        if msg_type == "get_prompt_context":
            return {
                "type": "prompt_context",
                "agentId": agent_id,
                "context": agent.get_prompt_context(),
            }

        if msg_type == "process_response":
            response_text = msg.get("response", "")
            commands = agent.process_response(response_text)
            return {
                "type": "process_ok",
                "agentId": agent_id,
                "commandsFound": commands,
            }

        if msg_type == "chemistry_event":
            event = msg.get("event", "")
            kwargs = msg.get("kwargs", {})
            agent.trigger_chemistry_event(event, **kwargs)
            return {"type": "chemistry_ok", "agentId": agent_id}

        if msg_type == "witness":
            return {
                "type": "witness",
                "agentId": agent_id,
                "report": agent.get_witness_report(),
            }

        if msg_type == "stop":
            return self._handle_stop(agent_id)

        return {"type": "error", "error": f"unknown message type: {msg_type}"}

    def _handle_init(self, agent_id: str, config: dict) -> dict:
        """Initialize a substrate for an agent."""
        if agent_id in self.agents:
            # Already running — return current state
            return {
                "type": "init_ok",
                "agentId": agent_id,
                "oscillators": self.agents[agent_id].entity.substrate.total_oscillators,
                "existing": True,
            }

        try:
            agent = AgentSubstrate(agent_id, config, self.state_dir)
            agent.start_ticking()
            self.agents[agent_id] = agent

            print(f"  [ore] initialized substrate for agent: {agent_id} "
                  f"({agent.entity.substrate.total_oscillators} oscillators)")

            return {
                "type": "init_ok",
                "agentId": agent_id,
                "oscillators": agent.entity.substrate.total_oscillators,
                "existing": False,
            }
        except Exception as e:
            return {"type": "error", "error": f"init failed: {e}"}

    def _handle_stop(self, agent_id: str) -> dict:
        """Stop and persist an agent's substrate."""
        if agent_id in self.agents:
            self.agents[agent_id].stop_ticking()
            del self.agents[agent_id]
            return {"type": "stop_ok", "agentId": agent_id}
        return {"type": "error", "error": f"agent {agent_id} not found"}

    def _shutdown(self):
        """Shutdown all agents and the server."""
        print("\n  [ore] shutting down...")
        for agent_id, agent in self.agents.items():
            try:
                agent.stop_ticking()
                print(f"  [ore] saved state for {agent_id}")
            except Exception as e:
                print(f"  [ore] error saving {agent_id}: {e}")
        self.agents.clear()
        self.running = False

    async def run(self):
        """Start the WebSocket server."""
        self.running = True

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Sidecar - Oscillatory Resonance Engine                      ║
║  WebSocket server for OpenClaw substrate integration             ║
║  Port: {self.port:<56} ║
║  State: {self.state_dir:<55} ║
╚══════════════════════════════════════════════════════════════════╝
""")

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown)
        except NotImplementedError:
            pass  # Windows

        try:
            async with websockets.serve(self.handle_client, "127.0.0.1", self.port):
                print(f"  [ore] listening on ws://127.0.0.1:{self.port}")
                while self.running:
                    await asyncio.sleep(0.5)
        except Exception as e:
            print(f"  [ore] server error: {e}")
        finally:
            self._shutdown()


def main():
    parser = argparse.ArgumentParser(description="ORE Sidecar Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"WebSocket port (default: {DEFAULT_PORT})")
    parser.add_argument("--state-dir", type=str, default=DEFAULT_STATE_DIR,
                        help=f"State directory (default: {DEFAULT_STATE_DIR})")
    args = parser.parse_args()

    server = OREServer(port=args.port, state_dir=args.state_dir)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
