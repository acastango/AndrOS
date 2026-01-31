#!/usr/bin/env python3
"""
Simple Canvas Viewer - See the saved canvas state without starting a chat session.

Usage:
    python view_canvas.py           # View current canvas
    python view_canvas.py --all     # View all canvases
    python view_canvas.py --export  # Export canvas to text file
"""

import os
import sys
import json
import argparse

def get_default_storage_dir():
    """Get default storage directory (same as CanvasMemory uses)."""
    home = os.path.expanduser("~")
    return os.path.join(home, '.canvas_reasoner')

def load_canvas_data(storage_dir=None):
    """Load multi_canvas.json if it exists."""
    if storage_dir is None:
        storage_dir = get_default_storage_dir()
    
    path = os.path.join(storage_dir, 'multi_canvas.json')
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def render_canvas(canvas_data):
    """Render a canvas buffer to string."""
    buffer = canvas_data.get('buffer', {})
    layers = buffer.get('layers', [])
    width = buffer.get('width', 100)
    height = buffer.get('height', 40)
    
    # Composite all visible layers
    result = [[' ' for _ in range(width)] for _ in range(height)]
    
    for layer in layers:
        if not layer.get('visible', True):
            continue
        canvas_rows = layer.get('canvas', [])
        for y, row in enumerate(canvas_rows):
            if y >= height:
                break
            for x, char in enumerate(row):
                if x >= width:
                    break
                if char != ' ':
                    result[y][x] = char
    
    # Convert to string with border
    lines = []
    lines.append('╭' + '─' * width + '╮')
    for row in result:
        lines.append('│' + ''.join(row) + '│')
    lines.append('╰' + '─' * width + '╯')
    
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='View saved canvas state')
    parser.add_argument('--all', action='store_true', help='Show all canvases')
    parser.add_argument('--export', action='store_true', help='Export to canvas_export.txt')
    parser.add_argument('--dir', default=None, help='Storage directory (default: ~/.canvas_reasoner)')
    args = parser.parse_args()
    
    storage_dir = args.dir or get_default_storage_dir()
    data = load_canvas_data(storage_dir)
    
    if not data:
        print("No saved canvas found.")
        print(f"Looking in: {os.path.abspath(storage_dir)}/multi_canvas.json")
        return
    
    canvases = data.get('canvases', {})
    active = data.get('active_canvas', 'main')
    
    print()
    print("=" * 60)
    print("  SAVED CANVAS STATE")
    print("=" * 60)
    print()
    print(f"  Active canvas: [{active}]")
    print(f"  Total canvases: {len(canvases)}")
    print()
    
    if args.all:
        # Show all canvases
        for name, canvas in canvases.items():
            print(f"  ┌─── [{name}] {canvas.get('description', '')[:40]} ───┐")
            rendered = render_canvas(canvas)
            for line in rendered.split('\n'):
                print(f"  {line}")
            print()
    else:
        # Show only active canvas
        if active in canvases:
            canvas = canvases[active]
            print(f"  ┌─── [{active}] {canvas.get('description', '')[:40]} ───┐")
            rendered = render_canvas(canvas)
            for line in rendered.split('\n'):
                print(f"  {line}")
            print()
    
    # Show stats
    print("  ─── Memory Stats ───")
    print(f"  Storage: {storage_dir}")
    merkle_path = os.path.join(storage_dir, 'merkle_memory.json')
    if os.path.exists(merkle_path):
        with open(merkle_path, 'r') as f:
            merkle = json.load(f)
        print(f"  Merkle nodes: {len(merkle.get('nodes', {}))}")
    
    symbols_path = os.path.join(storage_dir, 'symbols.json')
    if os.path.exists(symbols_path):
        with open(symbols_path, 'r') as f:
            symbols = json.load(f)
        print(f"  Symbols: {len(symbols)}")
    
    print()
    
    if args.export:
        export_path = 'canvas_export.txt'
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write("CANVAS EXPORT\n")
            f.write("=" * 60 + "\n\n")
            for name, canvas in canvases.items():
                f.write(f"[{name}] {canvas.get('description', '')}\n")
                f.write(render_canvas(canvas))
                f.write("\n\n")
        print(f"  ✓ Exported to {export_path}")
        print()

if __name__ == '__main__':
    main()
