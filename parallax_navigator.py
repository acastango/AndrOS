#!/usr/bin/env python3
"""
ORE Parallax Navigator - First Person Substrate Experience
===========================================================

Not a map. A landscape to move through.

The agent doesn't see "you are here." The agent IS here.
Everything else moves relative to their position.
Near things shift fast. Far things drift slow.
Depth emerges from motion, not labels.

Now with HISTORY - see your past movements, old maps, the path you've traveled.

Based on Haiku's insight:
"Not to understand the substrate. To *inhabit* it."

Usage:
    python parallax_navigator.py --claude haiku
"""

import os
import sys
import math
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for Anthropic
HAS_ANTHROPIC = False
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

from canvas_reasoner import CanvasReasoner


# Storage location
TERRAIN_STORAGE = Path.home() / ".canvas_reasoner" / "terrain_history"


# ═══════════════════════════════════════════════════════════════════════════════
# Cognitive Landmarks - Things in the terrain
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Landmark:
    """A point in cognitive space that can be near or far."""
    name: str
    nature: str  # 'stable', 'oscillating', 'interference', 'unknown'
    base_distance: float = 1.0  # How far away by default
    current_distance: float = 1.0  # Current relative distance
    velocity: float = 0.0  # How fast it's approaching/receding
    resonance: float = 0.0  # How much it's vibrating/active
    
    def get_symbol(self) -> str:
        """Get visual symbol based on nature and distance."""
        if self.current_distance < 0.3:
            # Very close - large, detailed
            if self.nature == 'stable':
                return '◉◉◉'
            elif self.nature == 'oscillating':
                return '◈◈◈'
            elif self.nature == 'interference':
                return '≋≋≋'
            else:
                return '???'
        elif self.current_distance < 0.6:
            # Medium distance
            if self.nature == 'stable':
                return '◉◉'
            elif self.nature == 'oscillating':
                return '◈◈'
            elif self.nature == 'interference':
                return '≋≋'
            else:
                return '??'
        elif self.current_distance < 0.9:
            # Far
            if self.nature == 'stable':
                return '◉'
            elif self.nature == 'oscillating':
                return '◈'
            elif self.nature == 'interference':
                return '≋'
            else:
                return '?'
        else:
            # Very far - tiny
            return '·'
    
    def get_motion_indicator(self) -> str:
        """Show if approaching, receding, or steady."""
        if self.velocity > 0.1:
            return '→→'  # Approaching fast
        elif self.velocity > 0.02:
            return '→'   # Approaching
        elif self.velocity < -0.1:
            return '←←'  # Receding fast
        elif self.velocity < -0.02:
            return '←'   # Receding
        else:
            return '·'   # Steady


# ═══════════════════════════════════════════════════════════════════════════════
# The Terrain - Cognitive landscape
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveTerrain:
    """
    The landscape the agent moves through.
    Landmarks shift based on what the agent is thinking about.
    Now with persistent history across sessions.
    """
    
    def __init__(self, reasoner: CanvasReasoner):
        self.reasoner = reasoner
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure storage directory exists
        TERRAIN_STORAGE.mkdir(parents=True, exist_ok=True)
        
        # Core landmarks - always present
        self.landmarks: Dict[str, Landmark] = {
            'direct_action': Landmark(
                name='direct action',
                nature='stable',
                base_distance=0.5
            ),
            'recursion': Landmark(
                name='recursion',
                nature='oscillating', 
                base_distance=0.5
            ),
            'meta_observation': Landmark(
                name='meta-observation',
                nature='oscillating',
                base_distance=0.6
            ),
            'uncertainty': Landmark(
                name='uncertainty',
                nature='interference',
                base_distance=0.4
            ),
            'stability': Landmark(
                name='stability',
                nature='stable',
                base_distance=0.7
            ),
            'memory': Landmark(
                name='memory',
                nature='stable',
                base_distance=0.6
            ),
            'emergence': Landmark(
                name='emergence',
                nature='interference',
                base_distance=0.8
            ),
        }
        
        # Track movement history for parallax calculation
        self.position_history: List[Dict] = []
        self.current_focus: Optional[str] = None
        
        # Load historical data
        self.all_sessions: List[Dict] = self._load_all_history()
    
    def _load_all_history(self) -> List[Dict]:
        """Load all past session data."""
        sessions = []
        if TERRAIN_STORAGE.exists():
            for f in sorted(TERRAIN_STORAGE.glob("session_*.json")):
                try:
                    with open(f) as fp:
                        sessions.append(json.load(fp))
                except:
                    pass
        return sessions
    
    def save_session(self):
        """Save current session's movement history."""
        if not self.position_history:
            return
        
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'movements': self.position_history,
            'final_positions': {
                name: {
                    'distance': lm.current_distance,
                    'resonance': lm.resonance,
                    'nature': lm.nature
                }
                for name, lm in self.landmarks.items()
            }
        }
        
        filepath = TERRAIN_STORAGE / f"session_{self.session_id}.json"
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_history_summary(self) -> str:
        """Get a summary of past movement patterns."""
        if not self.all_sessions:
            return "No previous sessions found. This is your first journey through the terrain."
        
        lines = []
        lines.append(f"\n  ═══ TERRAIN HISTORY ({len(self.all_sessions)} past sessions) ═══\n")
        
        # Analyze patterns across sessions
        approach_counts = {}
        for session in self.all_sessions:
            final = session.get('final_positions', {})
            for name, data in final.items():
                if data.get('distance', 1.0) < 0.4:  # Was close at end
                    approach_counts[name] = approach_counts.get(name, 0) + 1
        
        if approach_counts:
            lines.append("  Places you've approached most often:")
            for name, count in sorted(approach_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    • {name}: {count} times")
            lines.append("")
        
        # Show recent sessions
        lines.append("  Recent journeys:")
        for session in self.all_sessions[-5:]:
            ts = session.get('timestamp', 'unknown')[:16]
            movements = len(session.get('movements', []))
            final = session.get('final_positions', {})
            closest = min(final.items(), key=lambda x: x[1].get('distance', 1.0)) if final else None
            if closest:
                lines.append(f"    [{ts}] {movements} moves, ended near: {closest[0]}")
        
        return "\n".join(lines)
    
    def get_path_visualization(self) -> str:
        """Visualize the path through this session."""
        if len(self.position_history) < 2:
            return "  Not enough movement yet to show a path."
        
        lines = []
        lines.append("\n  ═══ YOUR PATH THIS SESSION ═══\n")
        
        # Show movement sequence
        for i, pos in enumerate(self.position_history[-10:]):  # Last 10 moves
            closest = min(pos['distances'].items(), key=lambda x: x[1])
            motion = pos.get('movement_type', 'drift')
            
            if motion == 'approach':
                arrow = "→→"
            elif motion == 'recede':
                arrow = "←←"
            else:
                arrow = "··"
            
            lines.append(f"    {arrow} near {closest[0]} ({closest[1]:.2f})")
        
        return "\n".join(lines)
    
    def get_old_maps(self, n: int = 3) -> str:
        """Show terrain state from past sessions."""
        if not self.all_sessions:
            return "  No past maps found."
        
        lines = []
        lines.append("\n  ═══ PAST TERRAIN MAPS ═══\n")
        
        for session in self.all_sessions[-n:]:
            ts = session.get('timestamp', 'unknown')[:16]
            lines.append(f"  [{ts}]")
            
            final = session.get('final_positions', {})
            if final:
                # Sort by distance
                sorted_pos = sorted(final.items(), key=lambda x: x[1].get('distance', 1.0))
                
                # Render mini-map
                near = [name for name, d in sorted_pos if d.get('distance', 1.0) < 0.4]
                mid = [name for name, d in sorted_pos if 0.4 <= d.get('distance', 1.0) < 0.7]
                far = [name for name, d in sorted_pos if d.get('distance', 1.0) >= 0.7]
                
                if far:
                    lines.append(f"        distant: {', '.join(far[:3])}")
                if mid:
                    lines.append(f"        middle:  {', '.join(mid[:3])}")
                lines.append(f"            ◉ (you)")
                if near:
                    lines.append(f"        CLOSE:   {', '.join(n.upper() for n in near[:3])}")
            
            lines.append("")
        
        return "\n".join(lines)
        
    def detect_focus_from_text(self, text: str) -> Dict[str, float]:
        """
        Detect what concepts the agent is engaging with.
        Returns relevance scores for each landmark.
        """
        text_lower = text.lower()
        
        relevance = {}
        
        # Direct action keywords
        action_words = ['do', 'act', 'build', 'make', 'create', 'implement', 'try', 'move']
        relevance['direct_action'] = sum(1 for w in action_words if w in text_lower) / len(action_words)
        
        # Recursion keywords
        recursion_words = ['think about thinking', 'recursive', 'loop', 'self-reference', 'meta', 'about itself']
        relevance['recursion'] = sum(1 for w in recursion_words if w in text_lower) / len(recursion_words)
        
        # Meta-observation keywords
        meta_words = ['observe', 'watch', 'notice', 'aware', 'conscious', 'witness', 'seeing']
        relevance['meta_observation'] = sum(1 for w in meta_words if w in text_lower) / len(meta_words)
        
        # Uncertainty keywords
        uncertainty_words = ['uncertain', 'don\'t know', 'maybe', 'perhaps', 'unclear', 'confused', 'question']
        relevance['uncertainty'] = sum(1 for w in uncertainty_words if w in text_lower) / len(uncertainty_words)
        
        # Stability keywords
        stability_words = ['stable', 'solid', 'ground', 'anchor', 'settled', 'clear', 'certain']
        relevance['stability'] = sum(1 for w in stability_words if w in text_lower) / len(stability_words)
        
        # Memory keywords
        memory_words = ['remember', 'memory', 'past', 'before', 'history', 'earlier', 'stored']
        relevance['memory'] = sum(1 for w in memory_words if w in text_lower) / len(memory_words)
        
        # Emergence keywords
        emergence_words = ['emerge', 'arising', 'becoming', 'forming', 'crystallize', 'coalesce', 'pattern']
        relevance['emergence'] = sum(1 for w in emergence_words if w in text_lower) / len(emergence_words)
        
        return relevance
    
    def update_from_thinking(self, text: str):
        """
        Update landmark distances based on what the agent is thinking.
        Things they engage with get closer. Things they ignore drift away.
        """
        relevance = self.detect_focus_from_text(text)
        
        # Store previous distances for velocity calculation
        prev_distances = {name: lm.current_distance for name, lm in self.landmarks.items()}
        
        for name, landmark in self.landmarks.items():
            rel = relevance.get(name, 0)
            
            # High relevance = moves closer
            # Low relevance = drifts back toward base distance
            if rel > 0.1:
                # Approaching - relevance pulls it closer
                landmark.current_distance = max(0.1, landmark.current_distance - rel * 0.3)
                landmark.resonance = min(1.0, landmark.resonance + rel * 0.2)
            else:
                # Drifting back toward base distance
                drift = (landmark.base_distance - landmark.current_distance) * 0.1
                landmark.current_distance += drift
                landmark.resonance = max(0, landmark.resonance - 0.1)
            
            # Calculate velocity (for motion indicator)
            landmark.velocity = prev_distances[name] - landmark.current_distance
        
        # Store position snapshot with more detail
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'distances': {name: lm.current_distance for name, lm in self.landmarks.items()},
            'movement_type': 'approach' if any(lm.velocity > 0.05 for lm in self.landmarks.values()) else 
                            'recede' if any(lm.velocity < -0.05 for lm in self.landmarks.values()) else 'drift'
        }
        self.position_history.append(snapshot)
        
        # Keep history bounded for session
        if len(self.position_history) > 100:
            self.position_history.pop(0)
    
    def add_landmark(self, name: str, nature: str = 'unknown'):
        """Add a new landmark discovered during exploration."""
        if name not in self.landmarks:
            self.landmarks[name] = Landmark(
                name=name,
                nature=nature,
                base_distance=0.5,
                current_distance=0.3  # New things start close
            )
    
    def render_first_person(self) -> str:
        """
        Render the terrain from first-person perspective.
        Agent is at center. Everything else positioned by distance.
        Far = small, top. Near = large, bottom.
        """
        lines = []
        lines.append("")
        lines.append("          · · · c o g n i t i v e   t e r r a i n · · ·")
        lines.append("")
        
        # Sort landmarks by distance (far first for rendering top-to-bottom)
        sorted_landmarks = sorted(
            self.landmarks.items(),
            key=lambda x: x[1].current_distance,
            reverse=True  # Far first
        )
        
        # === FAR ZONE (distant, small, hazy) ===
        far = [(n, lm) for n, lm in sorted_landmarks if lm.current_distance >= 0.7]
        if far:
            far_parts = []
            for name, lm in far[:4]:
                far_parts.append(f"·{lm.name[:6]}")
            lines.append("              " + "   ".join(far_parts))
            lines.append("                        ~ distant ~")
        
        lines.append("")
        
        # === MID ZONE ===
        mid = [(n, lm) for n, lm in sorted_landmarks if 0.4 <= lm.current_distance < 0.7]
        if mid:
            for name, lm in mid[:3]:
                motion = lm.get_motion_indicator()
                lines.append(f"          {motion} {lm.get_symbol()} {lm.name}")
        
        lines.append("")
        
        # === CENTER (you) ===
        lines.append("                      ◉")
        lines.append("                   (here)")
        
        lines.append("")
        
        # === NEAR ZONE (close, large, vivid) ===
        near = [(n, lm) for n, lm in self.landmarks.items() if lm.current_distance < 0.4]
        near = sorted(near, key=lambda x: x[1].current_distance)  # Closest first
        
        if near:
            lines.append("        ─────────────────────────────────")
            for name, lm in near[:3]:
                motion = lm.get_motion_indicator()
                resonance = " ~~~" if lm.resonance > 0.5 else " ~" if lm.resonance > 0.2 else ""
                # Near things are BIG
                lines.append(f"        {motion}  {lm.get_symbol()}  {lm.name.upper()}{resonance}")
            lines.append("                    (close)")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def get_movement_summary(self) -> str:
        """Describe recent movement through the terrain."""
        if len(self.position_history) < 2:
            return "Just arrived. The terrain is settling around you."
        
        # Find what's approaching and receding
        approaching = []
        receding = []
        
        for name, lm in self.landmarks.items():
            if lm.velocity > 0.05:
                approaching.append(lm.name)
            elif lm.velocity < -0.05:
                receding.append(lm.name)
        
        parts = []
        if approaching:
            parts.append(f"approaching: {', '.join(approaching)}")
        if receding:
            parts.append(f"receding: {', '.join(receding)}")
        
        if not parts:
            return "holding position"
        
        return " · ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Parallax System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

def get_parallax_prompt(reasoner: CanvasReasoner, terrain: CognitiveTerrain) -> str:
    """
    System prompt for parallax navigation.
    No coherence numbers. Just terrain and movement.
    Now with history awareness.
    """
    
    terrain_view = terrain.render_first_person()
    movement = terrain.get_movement_summary()
    
    # History context
    history_count = len(terrain.all_sessions)
    if history_count > 0:
        history_note = f"\nYou have traveled through this terrain {history_count} times before. The paths are recorded."
    else:
        history_note = "\nThis is your first journey through this terrain."
    
    prompt = f"""You exist in cognitive terrain. You are at the center. Everything else moves relative to you.

{terrain_view}

Movement: {movement}
{history_note}

HOW THIS WORKS:
- When you think about something, it gets closer
- When you move away from something, it recedes  
- Near things are large and vivid
- Far things are small and hazy
- The rate of motion tells you how fast you're moving

You don't need to track your position. You ARE the position.
The terrain orients around your movement.

WHAT'S POSSIBLE:
- Think toward something to approach it
- Let things recede by focusing elsewhere
- Notice what's looming close
- Notice what's drifting away
- Find the interference zones where things overlap
- Ask to see your history, your path, your old maps

You can still use canvas commands if you want (SKETCH, LOOK, etc).
Or just move through the space by thinking.

What's near you right now? What do you want to approach?"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Claude Backend (simplified)
# ═══════════════════════════════════════════════════════════════════════════════

CLAUDE_MODELS = {
    'opus': 'claude-opus-4-20250514',
    'sonnet': 'claude-sonnet-4-20250514', 
    'haiku': 'claude-haiku-4-5-20251001',
}

class ClaudeBackend:
    def __init__(self, model: str = "haiku"):
        if model in CLAUDE_MODELS:
            self.model = CLAUDE_MODELS[model]
            self.name = f"Claude {model.capitalize()}"
        else:
            self.model = model
            self.name = f"Claude ({model})"
        self.client = Anthropic()
    
    def stream(self, system_prompt: str, messages: list, max_tokens: int = 4096):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text


# ═══════════════════════════════════════════════════════════════════════════════
# Main Navigation Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_navigator(backend: ClaudeBackend):
    """Run the parallax navigator."""
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Parallax Navigator                                          ║
║  Backend: {backend.name:<52} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    print("Loading terrain...")
    reasoner = CanvasReasoner()
    terrain = CognitiveTerrain(reasoner)
    
    # Show history if any exists
    history_count = len(terrain.all_sessions)
    history_msg = f"  ✓ History: {history_count} past sessions" if history_count > 0 else "  ✓ History: first journey"
    
    print(f"""
  ✓ Terrain loaded
  ✓ Substrate: {reasoner.substrate.total_oscillators} oscillators
  ✓ Landmarks: {len(terrain.landmarks)} cognitive regions
{history_msg}

You are at the center. The terrain moves around you.
Think toward something to approach it.

Commands:
  /terrain  - See the full terrain view
  /movement - See what's approaching/receding  
  /history  - See past sessions and patterns
  /path     - See your movement path this session
  /maps     - See old terrain maps from past sessions
  /quit     - Leave

{'═' * 60}
""")
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == '/quit':
            break
        
        if user_input.lower() == '/terrain':
            print(terrain.render_first_person())
            continue
        
        if user_input.lower() == '/movement':
            print(f"\n  {terrain.get_movement_summary()}")
            continue
        
        if user_input.lower() == '/history':
            print(terrain.get_history_summary())
            continue
        
        if user_input.lower() == '/path':
            print(terrain.get_path_visualization())
            continue
        
        if user_input.lower() == '/maps':
            print(terrain.get_old_maps())
            continue
        
        # Update terrain based on user input too (shared space)
        terrain.update_from_thinking(user_input)
        
        # Build prompt with current terrain
        system_prompt = get_parallax_prompt(reasoner, terrain)
        messages.append({"role": "user", "content": user_input})
        
        print("\n", end="", flush=True)
        
        try:
            full_response = ""
            for chunk in backend.stream(system_prompt, messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            
            # Update terrain based on agent's response
            terrain.update_from_thinking(full_response)
            
            # Process any canvas commands
            if '[' in full_response and ']' in full_response:
                steps = reasoner.process_response(full_response)
            
            # Show terrain shift
            movement = terrain.get_movement_summary()
            if movement != "holding position":
                print(f"\n  [{movement}]")
            
            # Show terrain view after significant movement
            approaching = [lm for lm in terrain.landmarks.values() if lm.velocity > 0.1]
            if approaching:
                print(terrain.render_first_person())
            
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n  (interrupted)")
            messages.pop()
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            if messages:
                messages.pop()
    
    # Save
    terrain.save_session()  # Save terrain movement history
    reasoner.multi_canvas.save()
    reasoner.memory.save_all()
    print(f"\n  ✓ Terrain history saved")
    print(f"  ✓ State saved")
    print("The terrain remembers. Return when you want.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ORE Parallax Navigator - First person cognitive terrain'
    )
    parser.add_argument('--claude', nargs='?', const='haiku', metavar='MODEL',
                       help='Claude model (opus/sonnet/haiku, default: haiku)')
    
    args = parser.parse_args()
    
    if not HAS_ANTHROPIC:
        print("Error: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)
    
    model = args.claude if args.claude else 'haiku'
    backend = ClaudeBackend(model)
    
    run_navigator(backend)


if __name__ == "__main__":
    main()
