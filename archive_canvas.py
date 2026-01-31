#!/usr/bin/env python3
"""
Archive Canvas Memories
=======================
Moves existing canvas memory to an archive folder and starts fresh.

Usage:
    python archive_canvas.py                    # Archive with timestamp
    python archive_canvas.py --name "sonnet_v1" # Archive with custom name
"""

import os
import shutil
import argparse
from datetime import datetime

def get_storage_dir():
    """Get default storage directory."""
    home = os.path.expanduser("~")
    return os.path.join(home, '.canvas_reasoner')

def archive_canvas(name=None):
    """Archive current canvas memory to a subfolder."""
    storage_dir = get_storage_dir()
    
    if not os.path.exists(storage_dir):
        print(f"No canvas memory found at {storage_dir}")
        return False
    
    # Check if there's anything to archive
    files = os.listdir(storage_dir)
    data_files = [f for f in files if f.endswith('.json') or f.endswith('.md')]
    
    if not data_files:
        print("No canvas data to archive.")
        return False
    
    # Create archive name
    if name:
        archive_name = f"archive_{name}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"archive_{timestamp}"
    
    archive_dir = os.path.join(storage_dir, archive_name)
    
    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)
    
    # Move all data files to archive
    moved = []
    for f in data_files:
        src = os.path.join(storage_dir, f)
        dst = os.path.join(archive_dir, f)
        shutil.move(src, dst)
        moved.append(f)
    
    # Also move any subdirectories (like journals)
    for item in os.listdir(storage_dir):
        item_path = os.path.join(storage_dir, item)
        if os.path.isdir(item_path) and not item.startswith('archive'):
            dst = os.path.join(archive_dir, item)
            shutil.move(item_path, dst)
            moved.append(f"{item}/")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Canvas Memory Archived                                          ║
╚══════════════════════════════════════════════════════════════════╝

  Archived to: {archive_dir}
  
  Files moved:
""")
    for f in moved:
        print(f"    • {f}")
    
    print(f"""
  Canvas is now fresh. Next session will start clean.
  
  To restore, move files back from:
    {archive_dir}
""")
    return True


def list_archives():
    """List existing archives."""
    storage_dir = get_storage_dir()
    
    if not os.path.exists(storage_dir):
        print("No canvas memory directory found.")
        return
    
    archives = [d for d in os.listdir(storage_dir) 
                if d.startswith('archive') and os.path.isdir(os.path.join(storage_dir, d))]
    
    if not archives:
        print("No archives found.")
        return
    
    print("\nExisting archives:")
    for a in sorted(archives):
        archive_path = os.path.join(storage_dir, a)
        files = os.listdir(archive_path)
        print(f"  • {a} ({len(files)} items)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Archive canvas memories')
    parser.add_argument('--name', '-n', type=str, default=None,
                       help='Custom name for archive (default: timestamp)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List existing archives')
    args = parser.parse_args()
    
    if args.list:
        list_archives()
    else:
        archive_canvas(args.name)


if __name__ == "__main__":
    main()
