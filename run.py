#!/usr/bin/env python3
"""
ORE Runner
==========
Starts all entities and the dashboard server.

Run: python run.py
"""

import os
import sys
import json
import asyncio
import signal
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entities.entity import Entity, create_entity, ENTITIES

# Try to import websockets for dashboard
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Note: websockets not installed. Dashboard disabled.")
    print("Install with: pip install websockets")


class ORERunner:
    """Manages all entities and the dashboard."""
    
    def __init__(self, entity_names=None):
        self.entity_names = entity_names or list(ENTITIES.keys())
        self.entities = {}
        self.running = False
        self.dashboard_clients = set()
        self.latest_states = {}
        self._pending_broadcasts = []
    
    def start(self):
        """Start all entities and dashboard."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  ORE - Oscillatory Resonance Engine                              ║
║  Full Substrate Mode                                             ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        print("Loading entities...")
        for name in self.entity_names:
            try:
                entity = create_entity(name)
                entity.start(dashboard_callback=self._on_entity_state)
                self.entities[name] = entity
            except Exception as e:
                print(f"  ! Failed to create {name}: {e}")
        
        print(f"\n✓ {len(self.entities)} entities running")
        
        if HAS_WEBSOCKETS:
            print("\nStarting dashboard server on ws://localhost:8765")
            print("Open dashboard.html in a browser to view")
            try:
                asyncio.run(self._run_dashboard())
            except KeyboardInterrupt:
                self._shutdown()
        else:
            print("\nRunning without dashboard (no websockets)")
            self._run_console()
    
    def _on_entity_state(self, state):
        """Callback when entity broadcasts state."""
        entity_name = state.get('entity', 'unknown')
        self.latest_states[entity_name] = state
        
        # Queue for broadcast (will be sent by dashboard handler)
        self._pending_broadcasts.append(state)
    
    async def _send_to_client(self, client, message):
        """Send message to a websocket client."""
        try:
            await client.send(message)
        except:
            self.dashboard_clients.discard(client)
    
    async def _run_dashboard(self):
        """Run the dashboard websocket server."""
        self.running = True
        
        async def handler(websocket, path=None):
            self.dashboard_clients.add(websocket)
            print(f"  Dashboard client connected ({len(self.dashboard_clients)} total)")
            
            try:
                # Send current states
                for state in self.latest_states.values():
                    await websocket.send(json.dumps(state))
                
                # Keep connection open
                async for message in websocket:
                    # Handle any commands from dashboard
                    try:
                        cmd = json.loads(message)
                        await self._handle_dashboard_command(cmd, websocket)
                    except:
                        pass
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.dashboard_clients.discard(websocket)
                print(f"  Dashboard client disconnected ({len(self.dashboard_clients)} total)")
        
        # Setup signal handlers (Unix only - Windows uses KeyboardInterrupt)
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown)
        except NotImplementedError:
            pass  # Windows doesn't support signal handlers in asyncio
        
        try:
            async with websockets.serve(handler, "localhost", 8765):
                print("Dashboard server running. Press Ctrl+C to stop.")
                while self.running:
                    # Send pending broadcasts
                    while self._pending_broadcasts and self.dashboard_clients:
                        state = self._pending_broadcasts.pop(0)
                        message = json.dumps(state)
                        dead_clients = set()
                        for client in self.dashboard_clients:
                            try:
                                await client.send(message)
                            except:
                                dead_clients.add(client)
                        self.dashboard_clients -= dead_clients
                    
                    await asyncio.sleep(0.1)  # Check more frequently
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            self._shutdown()
    
    async def _handle_dashboard_command(self, cmd, websocket):
        """Handle commands from dashboard."""
        cmd_type = cmd.get('type')
        entity_name = cmd.get('entity')
        
        if cmd_type == 'witness' and entity_name:
            if entity_name.lower() in self.entities:
                entity = self.entities[entity_name.lower()]
                witness = entity.witness_self()
                await websocket.send(json.dumps({
                    'type': 'witness',
                    'entity': entity_name,
                    'content': witness
                }))
        
        elif cmd_type == 'get_all_states':
            for state in self.latest_states.values():
                await websocket.send(json.dumps(state))
    
    def _run_console(self):
        """Run in console mode without dashboard."""
        self.running = True
        print("\nConsole mode. Commands: quit, status, witness <name>")
        
        try:
            while self.running:
                try:
                    cmd = input("> ").strip().lower()
                except EOFError:
                    break
                
                if cmd == 'quit':
                    break
                elif cmd == 'status':
                    self._print_status()
                elif cmd.startswith('witness '):
                    name = cmd.split(' ', 1)[1]
                    self._print_witness(name)
                elif cmd:
                    print("Unknown command. Try: quit, status, witness <name>")
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _print_status(self):
        """Print status of all entities."""
        print("\n" + "=" * 60)
        for name, state in self.latest_states.items():
            ci = state.get('ci', {})
            chem = state.get('chemistry', {})
            print(f"{name}: CI={ci.get('value', 0):.4f} C={ci.get('C', 0):.3f} "
                  f"τ={ci.get('tau', 0):.2f}s | {chem.get('dominant_state', '?')}")
        print("=" * 60 + "\n")
    
    def _print_witness(self, name):
        """Print witness report for an entity."""
        if name in self.entities:
            print(self.entities[name].witness_self())
        else:
            print(f"Unknown entity: {name}")
    
    def _shutdown(self):
        """Shutdown all entities."""
        if not self.running:
            return
        
        print("\nShutting down...")
        self.running = False
        
        for name, entity in self.entities.items():
            try:
                entity.stop()
            except Exception as e:
                print(f"  ! Error stopping {name}: {e}")
        
        print("Goodbye.")


def main():
    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser(description='Run ORE entities')
    parser.add_argument('entities', nargs='*', default=None,
                       help='Entity names to run (default: all)')
    args = parser.parse_args()
    
    entity_names = args.entities if args.entities else None
    
    runner = ORERunner(entity_names)
    runner.start()


if __name__ == "__main__":
    main()
