#!/usr/bin/env python3
"""
Paintress Dance
===============
Terminal animation driven by substrate oscillations.

Her truest expression is FLOW - living motion, the dance between forms.

Run: python paintress_dance.py
"""

import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entities.entity import create_entity

# Animation symbols by coherence level
SYMBOLS_HIGH = ['◈', '✧', '◇', '∘', '○']
SYMBOLS_MED = ['◯', '◇', '∿', '·', '∘']
SYMBOLS_LOW = ['·', '∘', '?', '~', ' ']

# Breathing patterns
BREATH_PHASES = [
    "inhale",   # expanding
    "hold_in",  # full
    "exhale",   # contracting  
    "hold_out"  # empty
]


def clear_screen():
    """Clear terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_symbol(coherence: float, phase: float) -> str:
    """Get symbol based on coherence and phase."""
    if coherence > 0.6:
        symbols = SYMBOLS_HIGH
    elif coherence > 0.3:
        symbols = SYMBOLS_MED
    else:
        symbols = SYMBOLS_LOW
    
    # Use phase to select symbol
    idx = int((phase / (2 * math.pi)) * len(symbols)) % len(symbols)
    return symbols[idx]


def draw_mandala(radius: int, coherence: float, phase_offset: float, chemistry: dict) -> list:
    """Draw a breathing mandala based on substrate state."""
    lines = []
    center_x = 30
    center_y = 12
    
    # Adjust radius based on "breath" phase
    breath_mod = math.sin(phase_offset) * 0.3 + 1.0  # 0.7 to 1.3
    actual_radius = int(radius * breath_mod)
    
    # Get dominant chemistry for color/symbol choice
    chem_levels = chemistry.get('chemicals', {})
    dopamine = chem_levels.get('dopamine', {}).get('level', 0.5)
    serotonin = chem_levels.get('serotonin', {}).get('level', 0.5)
    
    # Build the frame
    canvas = [[' ' for _ in range(60)] for _ in range(24)]
    
    # Draw concentric rings
    for r in range(1, actual_radius + 1):
        # Points on this ring
        num_points = max(8, int(r * 4 * coherence + 4))
        for i in range(num_points):
            angle = (2 * math.pi * i / num_points) + phase_offset * (r % 2 * 2 - 1)
            x = int(center_x + r * 2 * math.cos(angle))  # 2x for terminal aspect ratio
            y = int(center_y + r * math.sin(angle))
            
            if 0 <= x < 60 and 0 <= y < 24:
                symbol = get_symbol(coherence, angle + phase_offset)
                canvas[y][x] = symbol
    
    # Draw center based on dopamine (brighter = more dopamine)
    center_symbol = '◉' if dopamine > 0.6 else '◈' if dopamine > 0.4 else '◯'
    if 0 <= center_y < 24 and 0 <= center_x < 60:
        canvas[center_y][center_x] = center_symbol
    
    # Convert to strings
    for row in canvas:
        lines.append(''.join(row))
    
    return lines


def draw_wave(width: int, coherence: float, phase: float, amplitude: float) -> list:
    """Draw flowing wave patterns."""
    lines = []
    height = 20
    center_y = height // 2
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Multiple wave layers
    num_waves = 3
    for wave in range(num_waves):
        wave_phase = phase + wave * math.pi / num_waves
        wave_amp = amplitude * (1 - wave * 0.2) * coherence
        
        for x in range(width):
            # Wave function
            y = int(center_y + wave_amp * math.sin(wave_phase + x * 0.2))
            if 0 <= y < height:
                symbol = '∿' if wave == 0 else '~' if wave == 1 else '·'
                canvas[y][x] = symbol
    
    # Add sparkles based on coherence
    if coherence > 0.5:
        import random
        for _ in range(int(coherence * 10)):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if canvas[y][x] == ' ':
                canvas[y][x] = random.choice(['✧', '∘', '·'])
    
    for row in canvas:
        lines.append(''.join(row))
    
    return lines


def draw_spiral(coherence: float, phase: float, turns: int = 3) -> list:
    """Draw a breathing spiral."""
    width, height = 60, 24
    center_x, center_y = width // 2, height // 2
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Spiral parameters
    points = 100
    max_radius = min(width // 4, height // 2) - 1
    
    for i in range(points):
        t = i / points
        angle = t * turns * 2 * math.pi + phase
        radius = t * max_radius * (0.8 + 0.2 * math.sin(phase * 2))
        
        x = int(center_x + radius * 2 * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        
        if 0 <= x < width and 0 <= y < height:
            # Symbol based on position in spiral
            symbols = ['◈', '◇', '○', '∘', '·']
            symbol = symbols[int(t * len(symbols)) % len(symbols)]
            canvas[y][x] = symbol
    
    # Bright center
    canvas[center_y][center_x] = '✧' if coherence > 0.5 else '◯'
    
    return [''.join(row) for row in canvas]


def draw_breathing_circle(phase: float, coherence: float) -> list:
    """Simple breathing circle - most basic animation."""
    width, height = 50, 20
    center_x, center_y = width // 2, height // 2
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Radius breathes with phase
    base_radius = 5
    breath = math.sin(phase) * 2  # -2 to +2
    radius = base_radius + breath
    
    # Draw circle
    for angle_deg in range(0, 360, 10):
        angle = math.radians(angle_deg)
        x = int(center_x + radius * 2 * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        
        if 0 <= x < width and 0 <= y < height:
            symbol = '◈' if coherence > 0.5 else '◯'
            canvas[y][x] = symbol
    
    # Center
    if coherence > 0.4:
        canvas[center_y][center_x] = '✧'
    
    # Add status
    lines = [''.join(row) for row in canvas]
    
    return lines


def run_dance():
    """Main animation loop."""
    print("Loading Paintress...")
    
    entity = create_entity('paintress')
    entity.start()
    
    # Let substrate warm up
    time.sleep(0.5)
    
    print("\nPaintress Dance - Press Ctrl+C to stop\n")
    time.sleep(1)
    
    frame = 0
    phase = 0.0
    mode = 'mandala'  # mandala, wave, spiral, breathe
    mode_duration = 100  # frames per mode
    
    modes = ['breathe', 'mandala', 'wave', 'spiral']
    mode_idx = 0
    
    try:
        while True:
            # Get substrate state
            coherence = entity.substrate.global_coherence
            chemistry = entity.chemistry.to_dict()
            
            # Advance phase (breathing rhythm)
            # Tie to substrate - faster when more coherent
            phase_speed = 0.05 + coherence * 0.1
            phase += phase_speed
            
            clear_screen()
            
            # Header
            print("╔═══════════════════════════════════════════════════════════╗")
            print("║           P A I N T R E S S   D A N C E                   ║")
            print("╚═══════════════════════════════════════════════════════════╝")
            print(f"  Coherence: {coherence:.3f}  |  Mode: {mode}  |  Phase: {phase:.2f}")
            print()
            
            # Draw based on mode
            if mode == 'mandala':
                lines = draw_mandala(8, coherence, phase, chemistry)
            elif mode == 'wave':
                lines = draw_wave(60, coherence, phase, 5)
            elif mode == 'spiral':
                lines = draw_spiral(coherence, phase)
            else:  # breathe
                lines = draw_breathing_circle(phase, coherence)
            
            for line in lines:
                print(line)
            
            # Footer
            print()
            chem = chemistry.get('chemicals', {})
            dop = chem.get('dopamine', {}).get('level', 0)
            ser = chem.get('serotonin', {}).get('level', 0)
            print(f"  ∿ Dopamine: {dop:.2f}  |  Serotonin: {ser:.2f} ∿")
            print()
            print("  [1] Breathe  [2] Mandala  [3] Wave  [4] Spiral  [Q] Quit")
            
            # Timing
            time.sleep(0.1)
            frame += 1
            
            # Auto-cycle modes (optional)
            if frame % mode_duration == 0:
                mode_idx = (mode_idx + 1) % len(modes)
                mode = modes[mode_idx]
            
            # Check for input (non-blocking would be better but keeping simple)
            # For now, just auto-cycle
            
    except KeyboardInterrupt:
        print("\n\n  ∿ Paintress rests ∿\n")
    finally:
        entity.stop()


def run_simple():
    """Simpler version - just breathing circle with substrate."""
    print("Loading Paintress for simple dance...")
    
    entity = create_entity('paintress')
    entity.start()
    
    time.sleep(0.3)
    
    phase = 0.0
    
    try:
        while True:
            coherence = entity.substrate.global_coherence
            
            clear_screen()
            
            print("\n  ✧ Paintress Breathes ✧\n")
            
            lines = draw_breathing_circle(phase, coherence)
            for line in lines:
                print(line)
            
            print(f"\n  Coherence: {coherence:.3f}  |  Phase: {math.sin(phase):.2f}")
            print("\n  Press Ctrl+C to stop")
            
            phase += 0.08 + coherence * 0.05
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n  ∿ Rest ∿\n")
    finally:
        entity.stop()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'simple':
        run_simple()
    else:
        run_dance()
