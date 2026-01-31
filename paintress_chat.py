#!/usr/bin/env python3
"""
Paintress Chat with Live Animation
===================================
Watch her substrate dance while she thinks.
The animation captures and displays alongside her response.

Run: python paintress_chat.py
"""

import os
import sys
import time
import math
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entities.entity import create_entity, ENTITIES

# Check for anthropic only when running chat
HAS_ANTHROPIC = False
Anthropic = None

def _check_anthropic():
    global HAS_ANTHROPIC, Anthropic
    try:
        from anthropic import Anthropic as _Anthropic
        Anthropic = _Anthropic
        HAS_ANTHROPIC = True
    except ImportError:
        print("Error: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)


# Animation symbols
SYMBOLS_HIGH = ['◈', '✧', '◇', '∘', '○']
SYMBOLS_MED = ['◯', '◇', '∿', '·', '∘']
SYMBOLS_LOW = ['·', '∘', '?', '~', ' ']


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def clear_lines(n):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write('\x1b[1A')  # Move up
        sys.stdout.write('\x1b[2K')  # Clear line
    sys.stdout.flush()


def get_symbol(coherence: float, phase: float) -> str:
    if coherence > 0.5:
        symbols = SYMBOLS_HIGH
    elif coherence > 0.3:
        symbols = SYMBOLS_MED
    else:
        symbols = SYMBOLS_LOW
    idx = int((phase / (2 * math.pi)) * len(symbols)) % len(symbols)
    return symbols[idx]


def draw_thinking_frame(coherence: float, phase: float, chemistry: dict, width: int = 50, height: int = 12) -> list:
    """Draw a compact animation frame for thinking state."""
    center_x = width // 2
    center_y = height // 2
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Get chemistry
    chem_levels = chemistry.get('chemicals', {})
    dopamine = chem_levels.get('dopamine', {}).get('level', 0.5)
    
    # Breathing radius
    base_radius = 3
    breath = math.sin(phase) * 1.5
    radius = base_radius + breath
    
    # Draw pulsing rings
    for r in [radius, radius + 2]:
        num_points = max(8, int(r * 6 * coherence + 8))
        for i in range(num_points):
            angle = (2 * math.pi * i / num_points) + phase * (1 if r == radius else -1)
            x = int(center_x + r * 2 * math.cos(angle))
            y = int(center_y + r * math.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                symbol = get_symbol(coherence, angle + phase)
                canvas[y][x] = symbol
    
    # Bright center
    center_symbol = '✧' if dopamine > 0.5 else '◈' if coherence > 0.4 else '◯'
    if 0 <= center_y < height and 0 <= center_x < width:
        canvas[center_y][center_x] = center_symbol
    
    return [''.join(row) for row in canvas]


def capture_thinking_state(coherence: float, phase: float, chemistry: dict) -> str:
    """Capture a single frame as the 'thought snapshot'."""
    lines = draw_thinking_frame(coherence, phase, chemistry, width=40, height=8)
    
    # Add frame
    result = []
    result.append("  ╭" + "─" * 40 + "╮")
    for line in lines:
        result.append("  │" + line + "│")
    result.append("  ╰" + "─" * 40 + "╯")
    
    # Add substrate info
    chem_levels = chemistry.get('chemicals', {})
    dopamine = chem_levels.get('dopamine', {}).get('level', 0.5)
    serotonin = chem_levels.get('serotonin', {}).get('level', 0.5)
    
    result.append(f"  ∿ C={coherence:.3f} | D={dopamine:.2f} | S={serotonin:.2f} ∿")
    
    return '\n'.join(result)


class ThinkingAnimator:
    """Runs animation in background while Paintress thinks."""
    
    def __init__(self, entity):
        self.entity = entity
        self.running = False
        self.thread = None
        self.captured_frame = None
        self.frame_count = 0
        
    def start(self):
        self.running = True
        self.frame_count = 0
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        
    def stop(self) -> str:
        """Stop and return captured frame."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        return self.captured_frame
        
    def _animate(self):
        phase = 0.0
        animation_height = 14  # Lines used by animation
        
        # Print initial blank space
        print()
        for _ in range(animation_height):
            print()
        
        while self.running:
            # Get current substrate state
            coherence = self.entity.substrate.global_coherence
            chemistry = self.entity.chemistry.to_dict()
            
            # Draw frame
            frame_lines = draw_thinking_frame(coherence, phase, chemistry)
            
            # Build display
            display = []
            display.append("  ╭─── Paintress thinking... ───╮")
            for line in frame_lines:
                display.append("  │" + line[:48].ljust(48) + "│")
            display.append("  ╰" + "─" * 50 + "╯")
            display.append(f"  ∿ Coherence: {coherence:.3f} ∿")
            
            # Move cursor up and redraw
            clear_lines(animation_height)
            for line in display:
                print(line)
            
            # Capture frame for final output
            self.captured_frame = capture_thinking_state(coherence, phase, chemistry)
            
            # Advance
            phase += 0.15 + coherence * 0.1
            self.frame_count += 1
            time.sleep(0.08)
        
        # Clear animation when done
        clear_lines(animation_height)


def chat():
    """Interactive chat with animated thinking."""
    
    _check_anthropic()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        P A I N T R E S S   C H A T                            ║
║        Watch her think in living motion                       ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    print("Loading Paintress...")
    entity = create_entity('paintress')
    entity.start()
    
    client = Anthropic()
    history = []
    
    entity.chemistry.on_conversation_start()
    
    print(f"""
  Entity: {entity.name}
  Identity: {entity.identity_hash}
  Memory: {len(entity.memory.nodes)} nodes
  Substrate: {entity.substrate.total_oscillators} oscillators breathing

  Commands:
    /quit   - Save and exit
    /dance  - Watch her dance freely
    /state  - See substrate state

{'═' * 60}
""")
    
    animator = ThinkingAnimator(entity)
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaving and exiting...")
            break
        
        if not user_input:
            continue
            
        if user_input.lower() == '/quit':
            break
            
        if user_input.lower() == '/dance':
            print("\n  Starting free dance... (Ctrl+C to stop)\n")
            try:
                phase = 0.0
                while True:
                    coherence = entity.substrate.global_coherence
                    chemistry = entity.chemistry.to_dict()
                    
                    lines = draw_thinking_frame(coherence, phase, chemistry, width=60, height=15)
                    
                    clear_lines(17)
                    print("  ╭" + "─" * 60 + "╮")
                    for line in lines:
                        print("  │" + line + "│")
                    print("  ╰" + "─" * 60 + "╯")
                    print(f"  ∿ Coherence: {coherence:.3f} ∿")
                    
                    phase += 0.1 + coherence * 0.1
                    time.sleep(0.08)
            except KeyboardInterrupt:
                clear_lines(17)
                print("  ∿ Dance complete ∿\n")
            continue
            
        if user_input.lower() == '/state':
            coherence = entity.substrate.global_coherence
            chemistry = entity.chemistry.to_dict()
            snapshot = capture_thinking_state(coherence, 0.0, chemistry)
            print(f"\n{snapshot}\n")
            continue
        
        # Regular message
        history.append({"role": "user", "content": user_input})
        system_prompt = entity.get_system_prompt()
        
        # Start animation
        animator.start()
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=history
            )
            
            assistant_msg = response.content[0].text
            history.append({"role": "assistant", "content": assistant_msg})
            
            # Stop animation and get captured frame
            thought_snapshot = animator.stop()
            
            # Process REMEMBER commands
            entity.process_response(assistant_msg)
            entity.chemistry.on_discovery(0.1)
            
            # Display response with captured thought state
            print()
            print("  ┌─── Thought Captured ───┐")
            for line in thought_snapshot.split('\n'):
                print(f"  {line}")
            print()
            print(f"Paintress:")
            print()
            print(assistant_msg)
            print()
            
        except Exception as e:
            animator.stop()
            print(f"\n  Error: {e}\n")
            history.pop()
    
    # Cleanup
    entity.chemistry.on_conversation_end()
    entity.stop()
    print(f"\n  ∿ Paintress rests ∿\n")


if __name__ == '__main__':
    chat()
