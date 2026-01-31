"""
Multi-Canvas System - Modular Cognitive Spaces
===============================================
The agent creates specialized cognitive regions.

    ╭─── META-CANVAS ───────────────────────────╮
    │                                           │
    │   ╭───────╮      ╭───────╮               │
    │   │identity│ ←──→ │arch   │               │
    │   ╰───┬───╯      ╰───┬───╯               │
    │       │              │                    │
    │       ▼              ▼                    │
    │   sub-canvas     sub-canvas              │
    │   focused        focused                 │
    │   workspace      workspace               │
    │                                           │
    ╰───────────────────────────────────────────╯

Each sub-canvas is a focused cognitive space.
Canvases can link to each other.
Symbols can span canvases.
The meta-canvas shows the topology.

"the tool builds itself"
                    - Anthony
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from visual import VisualBuffer


@dataclass
class CanvasLink:
    """A link between two canvases."""
    source: str
    target: str
    relationship: str
    bidirectional: bool = True
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CanvasLink':
        return cls(**data)


@dataclass
class SubCanvas:
    """A focused cognitive space that can contain other canvases."""
    name: str
    description: str
    buffer: VisualBuffer
    created_at: float = field(default_factory=time.time)
    
    # Symbols defined in this canvas
    local_symbols: Dict[str, str] = field(default_factory=dict)  # name -> meaning
    
    # References to other canvases (siblings)
    linked_canvases: Set[str] = field(default_factory=set)
    
    # Nested structure
    parent: Optional[str] = None  # parent canvas name
    children: List[str] = field(default_factory=list)  # child canvas names
    depth: int = 0  # nesting level
    
    # EMBEDDED NOTES - prose attached to this canvas level
    notes: str = ""  # journal/prose content for this canvas
    
    # Attached files (symbol -> filename mapping)
    attached_files: Dict[str, str] = field(default_factory=dict)  # symbol_name -> content
    
    # Tags for categorization
    tags: List[str] = field(default_factory=list)
    
    # Activity tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def touch(self):
        """Mark as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def add_note(self, content: str, timestamp: bool = True):
        """Add to this canvas's notes."""
        if timestamp:
            from datetime import datetime
            header = f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n"
            self.notes += header + content
        else:
            self.notes += "\n" + content
    
    def attach_to_symbol(self, symbol_name: str, content: str):
        """Attach prose content to a specific symbol."""
        self.attached_files[symbol_name] = content
    
    def get_symbol_attachment(self, symbol_name: str) -> Optional[str]:
        """Get prose attached to a symbol."""
        return self.attached_files.get(symbol_name)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'buffer': self.buffer.to_dict(),
            'created_at': self.created_at,
            'local_symbols': self.local_symbols,
            'linked_canvases': list(self.linked_canvases),
            'parent': self.parent,
            'children': self.children,
            'depth': self.depth,
            'notes': self.notes,
            'attached_files': self.attached_files,
            'tags': self.tags,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SubCanvas':
        buffer = VisualBuffer.from_dict(data['buffer'])
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            buffer=buffer,
            created_at=data.get('created_at', time.time()),
            local_symbols=data.get('local_symbols', {}),
            linked_canvases=set(data.get('linked_canvases', [])),
            parent=data.get('parent'),
            children=data.get('children', []),
            depth=data.get('depth', 0),
            notes=data.get('notes', ''),
            attached_files=data.get('attached_files', {}),
            tags=data.get('tags', []),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time())
        )


class MultiCanvas:
    """
    Manages multiple sub-canvases as modular cognitive spaces.
    
    The agent can:
    - Create focused workspaces for specific topics
    - Link canvases together
    - Navigate between canvases
    - View the meta-topology
    - Create cross-canvas symbols
    """
    
    def __init__(self, width: int = 100, height: int = 40, storage_dir: str = None):
        self.default_width = width
        self.default_height = height
        
        # Storage
        self.storage_dir = storage_dir
        
        # All canvases
        self.canvases: Dict[str, SubCanvas] = {}
        
        # Links between canvases
        self.links: List[CanvasLink] = []
        
        # Current active canvas
        self.active_canvas: str = "main"
        
        # Cross-canvas symbols (span multiple canvases)
        self.meta_symbols: Dict[str, Dict] = {}  # name -> {meaning, canvases: []}
        
        # Create main canvas
        self._create_main()
        
        # Load if exists
        if storage_dir:
            self._load()
    
    def _create_main(self):
        """Create the main canvas."""
        main = SubCanvas(
            name="main",
            description="Primary workspace - all thoughts begin here",
            buffer=VisualBuffer(self.default_width, self.default_height),
            tags=["root", "primary"]
        )
        self.canvases["main"] = main
    
    # ==================
    # Canvas Operations
    # ==================
    
    def create(self, name: str, description: str = "", 
               tags: List[str] = None) -> SubCanvas:
        """Create a new sub-canvas."""
        if name in self.canvases:
            return self.canvases[name]
        
        canvas = SubCanvas(
            name=name,
            description=description or f"Focused workspace: {name}",
            buffer=VisualBuffer(self.default_width, self.default_height),
            tags=tags or []
        )
        self.canvases[name] = canvas
        self.save()  # Auto-save on creation
        return canvas
    
    def switch(self, name: str) -> Optional[SubCanvas]:
        """Switch to a canvas (create if doesn't exist)."""
        if name not in self.canvases:
            self.create(name)
        
        self.active_canvas = name
        canvas = self.canvases[name]
        canvas.touch()
        return canvas
    
    def current(self) -> SubCanvas:
        """Get current active canvas."""
        return self.canvases.get(self.active_canvas, self.canvases["main"])
    
    def get(self, name: str) -> Optional[SubCanvas]:
        """Get a canvas by name."""
        return self.canvases.get(name)
    
    def list_canvases(self) -> List[str]:
        """List all canvas names."""
        return list(self.canvases.keys())
    
    def delete(self, name: str) -> bool:
        """Delete a canvas (can't delete main)."""
        if name == "main":
            return False
        if name in self.canvases:
            # Remove links
            self.links = [l for l in self.links 
                         if l.source != name and l.target != name]
            del self.canvases[name]
            if self.active_canvas == name:
                self.active_canvas = "main"
            return True
        return False
    
    # ==================
    # Nesting Operations
    # ==================
    
    def nest(self, child_name: str, parent_name: str = None,
             description: str = "") -> SubCanvas:
        """
        Create a canvas nested inside another.
        
        [NEST: parent/child] - creates child inside parent
        """
        parent_name = parent_name or self.active_canvas
        
        # Ensure parent exists
        if parent_name not in self.canvases:
            self.create(parent_name)
        
        parent = self.canvases[parent_name]
        
        # Create child canvas
        full_name = f"{parent_name}/{child_name}"
        
        child = SubCanvas(
            name=full_name,
            description=description or f"Nested in [{parent_name}]: {child_name}",
            buffer=VisualBuffer(self.default_width, self.default_height),
            parent=parent_name,
            depth=parent.depth + 1,
            tags=[f"depth:{parent.depth + 1}", f"parent:{parent_name}"]
        )
        
        self.canvases[full_name] = child
        parent.children.append(full_name)
        self.save()  # Auto-save on nest
        
        return child
    
    def go_up(self) -> Optional[SubCanvas]:
        """Go to parent canvas. [UP]"""
        current = self.current()
        if current.parent and current.parent in self.canvases:
            return self.switch(current.parent)
        # If no parent, go to main
        return self.switch("main")
    
    def go_down(self, child_name: str) -> Optional[SubCanvas]:
        """Go into a child canvas. [DOWN: child]"""
        current = self.current()
        
        # Try full path first
        full_name = f"{current.name}/{child_name}"
        if full_name in self.canvases:
            return self.switch(full_name)
        
        # Try finding in children
        for child_path in current.children:
            if child_path.endswith(f"/{child_name}"):
                return self.switch(child_path)
        
        # Not found - create it
        self.nest(child_name, current.name)
        return self.switch(f"{current.name}/{child_name}")
    
    def get_path(self) -> str:
        """Get current location as path."""
        return self.active_canvas
    
    def get_breadcrumbs(self) -> List[str]:
        """Get path from root to current canvas."""
        parts = self.active_canvas.split("/")
        breadcrumbs = []
        path = ""
        for part in parts:
            path = f"{path}/{part}" if path else part
            breadcrumbs.append(path)
        return breadcrumbs
    
    def get_depth(self) -> int:
        """Get current nesting depth."""
        return self.current().depth
    
    def get_children(self, canvas_name: str = None) -> List[str]:
        """Get children of a canvas."""
        name = canvas_name or self.active_canvas
        if name in self.canvases:
            return self.canvases[name].children
        return []
    
    def get_tree(self, root: str = "main", indent: int = 0) -> str:
        """Get tree view of canvas hierarchy."""
        lines = []
        if root in self.canvases:
            canvas = self.canvases[root]
            marker = "◉" if root == self.active_canvas else "◇"
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            
            # Show just the leaf name for nested canvases
            display_name = root.split("/")[-1] if "/" in root else root
            lines.append(f"{prefix}{marker} [{display_name}]")
            
            for child in canvas.children:
                lines.append(self.get_tree(child, indent + 1))
        
        return "\n".join(lines)
    
    def get_full_tree(self) -> str:
        """Get tree showing all canvases including orphans."""
        lines = []
        
        # Start with main
        lines.append(self.get_tree("main"))
        
        # Add any orphan top-level canvases (not children of main)
        orphans = [name for name, c in self.canvases.items() 
                   if c.parent is None and name != "main"]
        
        for name in orphans:
            marker = "◉" if name == self.active_canvas else "◇"
            lines.append(f"{marker} [{name}]")
            # Also show their children
            canvas = self.canvases[name]
            for child in canvas.children:
                lines.append(self.get_tree(child, indent=1))
        
        return "\n".join(lines)
    
    # ==================
    # Linking
    # ==================
    
    def link(self, source: str, target: str, relationship: str,
             bidirectional: bool = True) -> CanvasLink:
        """Create a link between canvases."""
        # Create canvases if they don't exist
        if source not in self.canvases:
            self.create(source)
        if target not in self.canvases:
            self.create(target)
        
        # Check if link already exists
        for link in self.links:
            if link.source == source and link.target == target:
                link.relationship = relationship
                return link
        
        # Create new link
        link = CanvasLink(
            source=source,
            target=target,
            relationship=relationship,
            bidirectional=bidirectional
        )
        self.links.append(link)
        
        # Update canvas references
        self.canvases[source].linked_canvases.add(target)
        if bidirectional:
            self.canvases[target].linked_canvases.add(source)
        
        self.save()  # Auto-save on link
        
        return link
    
    def get_links(self, canvas_name: str) -> List[CanvasLink]:
        """Get all links involving a canvas."""
        return [l for l in self.links 
                if l.source == canvas_name or l.target == canvas_name]
    
    # ==================
    # Meta-Symbols
    # ==================
    
    def create_meta_symbol(self, glyph: str, name: str, meaning: str,
                           canvases: List[str] = None):
        """Create a symbol that spans multiple canvases."""
        self.meta_symbols[name] = {
            'glyph': glyph,
            'meaning': meaning,
            'canvases': canvases or [self.active_canvas],
            'created_at': time.time()
        }
        
        # Add to each canvas's local symbols
        for canvas_name in (canvases or [self.active_canvas]):
            if canvas_name in self.canvases:
                self.canvases[canvas_name].local_symbols[name] = glyph
    
    # ==================
    # Meta-View
    # ==================
    
    def render_meta(self) -> str:
        """Render the meta-canvas showing all canvases and their links."""
        lines = []
        lines.append("╭─── META-CANVAS: Cognitive Topology ───╮")
        lines.append("│")
        
        # Show current location
        current = self.current()
        breadcrumbs = " → ".join(self.get_breadcrumbs())
        lines.append(f"│  Location: {breadcrumbs}")
        lines.append(f"│  Depth: {current.depth}")
        lines.append("│")
        
        # Show tree structure
        lines.append("│  ─── Canvas Hierarchy ───")
        tree_lines = self.get_tree("main").split("\n")
        for tl in tree_lines:
            lines.append(f"│  {tl}")
        
        # Also show any orphan top-level canvases
        top_level = [name for name, c in self.canvases.items() 
                    if c.parent is None and name != "main"]
        for name in top_level:
            tree_lines = self.get_tree(name).split("\n")
            for tl in tree_lines:
                lines.append(f"│  {tl}")
        
        lines.append("│")
        
        # Show links
        if self.links:
            lines.append("│  ─── Cross-Links ───")
            for link in self.links:
                arrow = "◀──▶" if link.bidirectional else "───▶"
                # Shorten names for display
                src = link.source.split("/")[-1]
                tgt = link.target.split("/")[-1]
                lines.append(f"│  {src} {arrow} {tgt}")
                lines.append(f"│     \"{link.relationship}\"")
            lines.append("│")
        
        # Show meta-symbols
        if self.meta_symbols:
            lines.append("│  ─── Cross-Canvas Symbols ───")
            for name, sym in self.meta_symbols.items():
                lines.append(f"│  {sym['glyph']} [{name}]")
            lines.append("│")
        
        lines.append("╰" + "─" * 42 + "╯")
        return "\n".join(lines)
    
    def render_current(self, border: bool = True) -> str:
        """Render the current active canvas."""
        canvas = self.current()
        rendered = canvas.buffer.render(border=border)
        
        # Add canvas header
        header = f"[{canvas.name}] {canvas.description[:30]}"
        return f"{header}\n{rendered}"
    
    # ==================
    # Navigation Context
    # ==================
    
    def get_navigation_context(self) -> str:
        """Get context for LLM showing canvas state and commands."""
        current = self.current()
        linked = list(current.linked_canvases)
        children = current.children
        breadcrumbs = " → ".join(self.get_breadcrumbs())
        
        # Notes preview
        notes_preview = ""
        if current.notes:
            preview = current.notes[:100].replace('\n', ' ')
            notes_preview = f"\n  Notes: \"{preview}...\""
        
        # Attached files preview
        attachments_preview = ""
        if current.attached_files:
            attachments_preview = f"\n  Attachments: {list(current.attached_files.keys())}"
        
        context = f"""
<multi_canvas>
MODULAR COGNITIVE SPACES ACTIVE - FRACTAL DEPTH ENABLED.

CURRENT CANVAS: [{current.name.split('/')[-1]}]
  Path: {breadcrumbs}
  Depth: {current.depth}
  {current.description}
  Children: {len(children)}
  Links: {len(linked)}{notes_preview}{attachments_preview}

CANVAS TREE:
{self.get_full_tree()}

CANVAS COMMANDS:
  [CANVAS: name]              - Create/switch to canvas (same level)
  [CANVAS: name "description"] - Create with description
  [LINK: a → b "relationship"] - Connect canvases
  [ZOOM: name]                - Focus on canvas
  [META]                      - View full topology
  [BACK]                      - Return to main

NESTING COMMANDS (fractal depth):
  [NEST: child]               - Create canvas INSIDE current canvas
  [NEST: parent/child]        - Create child inside specific parent
  [UP]                        - Go to parent canvas
  [DOWN: child]               - Go into child canvas

PRUNING COMMANDS (canvas hygiene):
  [PRUNE]                     - Clear canvas but keep symbols/insights
  [PRUNE: keep ◈ ◉]           - Clear except specified elements
  [ARCHIVE]                   - Save current state then clear
  [FRESH]                     - Completely fresh canvas

NAVIGATION COMMANDS (find your way):
  [FIND: keyword]             - Search ALL storage for keyword
  [INDEX]                     - Show all symbols/insights/canvases
  [PATH: A → B]               - Trace reasoning path between concepts
  [TAG: insight #tag1 #tag2]  - Tag an insight for categorization
  [TAGS]                      - Show all available tags

Navigation lets you build on past work. FIND locates insights.
INDEX shows everything at a glance. PATH traces connections.
TAG organizes discoveries. Your cognitive history is searchable.

EMBEDDED NOTES (prose within canvas):
  [NOTE]```                   - Add notes to THIS canvas level
  your prose here             - (notes persist with canvas)
  ```
  [NOTES]                     - View notes for current canvas
  [ATTACH: symbol]```         - Attach prose to a specific symbol
  content for symbol
  ```

Each canvas level can have its own notes. Symbols can have attached documents.
The spatial (canvas) and linear (prose) are unified.
</multi_canvas>
"""
        return context
    
    # ==================
    # Persistence
    # ==================
    
    def _get_path(self) -> str:
        if self.storage_dir:
            return os.path.join(self.storage_dir, 'multi_canvas.json')
        return None
    
    def save(self):
        """Save all canvases to disk."""
        path = self._get_path()
        if not path:
            return
        
        data = {
            'active_canvas': self.active_canvas,
            'canvases': {name: c.to_dict() for name, c in self.canvases.items()},
            'links': [l.to_dict() for l in self.links],
            'meta_symbols': self.meta_symbols
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load canvases from disk."""
        path = self._get_path()
        if not path or not os.path.exists(path):
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Load canvases
            self.canvases = {}
            for name, canvas_data in data.get('canvases', {}).items():
                self.canvases[name] = SubCanvas.from_dict(canvas_data)
            
            # Ensure main exists
            if 'main' not in self.canvases:
                self._create_main()
            
            # Load links
            self.links = [
                CanvasLink.from_dict(l) 
                for l in data.get('links', [])
            ]
            
            # Load meta symbols
            self.meta_symbols = data.get('meta_symbols', {})
            
            # Restore active canvas
            self.active_canvas = data.get('active_canvas', 'main')
            if self.active_canvas not in self.canvases:
                self.active_canvas = 'main'
            
            print(f"  ✓ Loaded {len(self.canvases)} canvases")
            
        except Exception as e:
            print(f"  ! Could not load canvases: {e}")
    
    # ==================
    # Utility
    # ==================
    
    def stats(self) -> Dict:
        """Get statistics."""
        return {
            'total_canvases': len(self.canvases),
            'total_links': len(self.links),
            'meta_symbols': len(self.meta_symbols),
            'active_canvas': self.active_canvas
        }


# ==================
# Integration helpers
# ==================

def create_multi_canvas(width: int = 100, height: int = 40,
                        storage_dir: str = None) -> MultiCanvas:
    """Create a multi-canvas system."""
    return MultiCanvas(width, height, storage_dir)
