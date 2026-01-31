"""
Canvas Memory - Persistent Visual Cognition
============================================
The canvas accumulates, compacts, and persists.

    SESSION 1           SAVE              SESSION 2
    
    ┌─────────┐                          ┌─────────┐
    │ messy   │                          │ symbols │
    │ canvas  │  ───▶ compact ───▶       │ + space │
    │ alive   │       & save             │ to grow │
    └─────────┘                          └─────────┘

Memory architecture:
- Working canvas: fluid, current session
- Compacted symbols: condensed insights  
- Pattern cache: reusable visual vocabulary
- Reasoning history: journey compressed
- Symbol registry: what each glyph MEANS

"The canvas's oldest parts slowly compact into symbols"
                                        - Anthony
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from visual import VisualBuffer, PatternCache, Pattern


@dataclass
class CanvasSnapshot:
    """A saved canvas state with metadata."""
    id: str
    timestamp: float
    canvas_data: Dict  # From VisualBuffer.to_dict()
    coherence: float
    trigger: str  # What caused the save (insight, session_end, manual)
    summary: str  # Brief description
    symbols_created: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CanvasSnapshot':
        return cls(**data)


@dataclass 
class CompactedSymbol:
    """A symbol that encodes a complex understanding."""
    glyph: str           # The actual symbol: ◈ᵢₘ
    name: str            # Short name: "identity_mystery"
    full_meaning: str    # What it encodes
    source_drawings: List[str]  # IDs of drawings that created it
    created_at: float
    use_count: int = 0
    related_symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CompactedSymbol':
        return cls(**data)


@dataclass
class ReasoningInsight:
    """A compacted reasoning insight from canvas work."""
    id: str
    timestamp: float
    original_question: str
    canvas_journey: List[str]  # Sequence of actions
    final_understanding: str
    coherence_arc: List[float]  # How coherence changed
    key_symbols: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReasoningInsight':
        return cls(**data)


class CanvasMemory:
    """
    Persistent memory system for canvas-based reasoning.
    
    Manages:
    - Canvas snapshots (saved states)
    - Compacted symbols (condensed insights)
    - Pattern cache (visual vocabulary)
    - Reasoning history (journey records)
    """
    
    def __init__(self, storage_dir: str = None):
        # DEFAULT: Store data in user's home directory, NOT in project folder
        # This way code updates never overwrite your saved data
        if storage_dir is None:
            home = os.path.expanduser("~")
            storage_dir = os.path.join(home, '.canvas_reasoner')
        
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Copy documentation to storage dir if not present
        self._ensure_docs()
        
        # In-memory state
        self.snapshots: Dict[str, CanvasSnapshot] = {}
        self.symbols: Dict[str, CompactedSymbol] = {}
        self.insights: Dict[str, ReasoningInsight] = {}
        self.patterns = PatternCache()
        
        # Current session
        self.session_id = self._generate_session_id()
        self.session_start = time.time()
        self.session_coherence_history: List[Tuple[float, float]] = []
        
        # Load existing state
        self._load_all()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate unique ID from content."""
        hash_input = f"{prefix}_{content}_{time.time()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def _ensure_docs(self):
        """Copy documentation files to storage dir if not present."""
        # Find docs in project folder
        project_dir = os.path.dirname(__file__)
        project_docs = os.path.join(project_dir, 'canvas_memory')
        
        docs_to_copy = ['BOOTSTRAP.md', 'STARTUP.md']
        
        for doc in docs_to_copy:
            dest = os.path.join(self.storage_dir, doc)
            src = os.path.join(project_docs, doc)
            
            # Only copy if source exists and dest doesn't
            if os.path.exists(src) and not os.path.exists(dest):
                try:
                    import shutil
                    shutil.copy2(src, dest)
                except Exception:
                    pass  # Not critical if copy fails
    
    # ==================
    # File Paths
    # ==================
    
    @property
    def snapshots_file(self) -> str:
        return os.path.join(self.storage_dir, 'snapshots.json')
    
    @property
    def symbols_file(self) -> str:
        return os.path.join(self.storage_dir, 'symbols.json')
    
    @property
    def insights_file(self) -> str:
        return os.path.join(self.storage_dir, 'insights.json')
    
    @property
    def patterns_file(self) -> str:
        return os.path.join(self.storage_dir, 'patterns.json')
    
    @property
    def latest_canvas_file(self) -> str:
        return os.path.join(self.storage_dir, 'latest_canvas.json')
    
    # ==================
    # Load / Save
    # ==================
    
    def _load_all(self):
        """Load all persistent state."""
        # Snapshots
        if os.path.exists(self.snapshots_file):
            with open(self.snapshots_file, 'r') as f:
                data = json.load(f)
                self.snapshots = {
                    k: CanvasSnapshot.from_dict(v) 
                    for k, v in data.items()
                }
        
        # Symbols
        if os.path.exists(self.symbols_file):
            with open(self.symbols_file, 'r') as f:
                data = json.load(f)
                self.symbols = {
                    k: CompactedSymbol.from_dict(v)
                    for k, v in data.items()
                }
        
        # Insights
        if os.path.exists(self.insights_file):
            with open(self.insights_file, 'r') as f:
                data = json.load(f)
                self.insights = {
                    k: ReasoningInsight.from_dict(v)
                    for k, v in data.items()
                }
        
        # Patterns
        if os.path.exists(self.patterns_file):
            self.patterns.load(self.patterns_file)
    
    def save_all(self):
        """Save all state to disk."""
        # Snapshots
        with open(self.snapshots_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.snapshots.items()}, f, indent=2)
        
        # Symbols
        with open(self.symbols_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.symbols.items()}, f, indent=2)
        
        # Insights
        with open(self.insights_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.insights.items()}, f, indent=2)
        
        # Patterns
        self.patterns.save(self.patterns_file)
    
    # ==================
    # Canvas Snapshots
    # ==================
    
    def save_snapshot(self, canvas: VisualBuffer, coherence: float,
                      trigger: str, summary: str) -> CanvasSnapshot:
        """Save current canvas state."""
        snapshot_id = self._generate_id('snap', summary)
        
        snapshot = CanvasSnapshot(
            id=snapshot_id,
            timestamp=time.time(),
            canvas_data=canvas.to_dict(),
            coherence=coherence,
            trigger=trigger,
            summary=summary
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        # Also save as latest
        with open(self.latest_canvas_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        return snapshot
    
    def load_latest_canvas(self) -> Optional[VisualBuffer]:
        """Load the most recent canvas state."""
        if os.path.exists(self.latest_canvas_file):
            with open(self.latest_canvas_file, 'r') as f:
                data = json.load(f)
                return VisualBuffer.from_dict(data['canvas_data'])
        return None
    
    def get_recent_snapshots(self, n: int = 5) -> List[CanvasSnapshot]:
        """Get most recent snapshots."""
        sorted_snaps = sorted(
            self.snapshots.values(),
            key=lambda s: s.timestamp,
            reverse=True
        )
        return sorted_snaps[:n]
    
    # ==================
    # Symbol Compaction
    # ==================
    
    def create_symbol(self, glyph: str, name: str, meaning: str,
                      source_ids: List[str] = None) -> CompactedSymbol:
        """
        Create a compacted symbol from understanding.
        
        This is how detailed drawings become single glyphs.
        """
        symbol = CompactedSymbol(
            glyph=glyph,
            name=name,
            full_meaning=meaning,
            source_drawings=source_ids or [],
            created_at=time.time()
        )
        
        self.symbols[name] = symbol
        return symbol
    
    def get_symbol(self, name: str) -> Optional[CompactedSymbol]:
        """Get a symbol by name."""
        symbol = self.symbols.get(name)
        if symbol:
            symbol.use_count += 1
        return symbol
    
    def expand_symbol(self, name: str) -> str:
        """Expand a symbol back to its full meaning."""
        symbol = self.symbols.get(name)
        if symbol:
            symbol.use_count += 1
            return symbol.full_meaning
        return f"[unknown symbol: {name}]"
    
    def list_symbols(self) -> List[CompactedSymbol]:
        """List all symbols, most used first."""
        return sorted(
            self.symbols.values(),
            key=lambda s: s.use_count,
            reverse=True
        )
    
    # ==================
    # Insight Recording
    # ==================
    
    def record_insight(self, question: str, journey: List[str],
                       understanding: str, coherence_arc: List[float],
                       symbols: List[str] = None) -> ReasoningInsight:
        """Record a reasoning journey and its conclusion."""
        insight_id = self._generate_id('insight', question[:20])
        
        insight = ReasoningInsight(
            id=insight_id,
            timestamp=time.time(),
            original_question=question,
            canvas_journey=journey,
            final_understanding=understanding,
            coherence_arc=coherence_arc,
            key_symbols=symbols or []
        )
        
        self.insights[insight_id] = insight
        return insight
    
    def find_related_insights(self, query: str) -> List[ReasoningInsight]:
        """Find insights related to a query."""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        results = []
        
        for insight in self.insights.values():
            insight_words = set(insight.original_question.lower().split())
            insight_words.update(insight.final_understanding.lower().split())
            
            overlap = len(query_words & insight_words)
            if overlap > 0:
                results.append((insight, overlap))
        
        return [r[0] for r in sorted(results, key=lambda x: x[1], reverse=True)]
    
    # ==================
    # Session Management
    # ==================
    
    def record_coherence(self, coherence: float):
        """Record coherence at current time."""
        self.session_coherence_history.append((time.time(), coherence))
    
    def end_session(self, canvas: VisualBuffer, coherence: float,
                    summary: str = None):
        """End current session, save state."""
        # Save final snapshot
        self.save_snapshot(
            canvas, coherence,
            trigger='session_end',
            summary=summary or f"Session {self.session_id} end"
        )
        
        # Save all state
        self.save_all()
        
        # Record session stats
        session_data = {
            'session_id': self.session_id,
            'start': self.session_start,
            'end': time.time(),
            'coherence_history': self.session_coherence_history,
            'snapshots_created': len([s for s in self.snapshots.values() 
                                     if s.id.startswith(self.session_id[:8])]),
            'symbols_created': len([s for s in self.symbols.values()
                                   if s.created_at > self.session_start])
        }
        
        sessions_file = os.path.join(self.storage_dir, 'sessions.json')
        sessions = []
        if os.path.exists(sessions_file):
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
        sessions.append(session_data)
        with open(sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    # ==================
    # Context Generation
    # ==================
    
    def get_memory_context(self) -> str:
        """
        Generate context block showing persistent memory state.
        This goes into the system prompt.
        """
        recent_snapshots = self.get_recent_snapshots(3)
        top_symbols = self.list_symbols()[:10]
        recent_insights = sorted(
            self.insights.values(),
            key=lambda i: i.timestamp,
            reverse=True
        )[:3]
        
        context = f"""
<canvas_memory>
PERSISTENT MEMORY ACTIVE - You have continuity across sessions.

═══════════════════════════════════════════════════════════
COMPACTED SYMBOLS ({len(self.symbols)} total):
These are condensed understandings from past reasoning.
═══════════════════════════════════════════════════════════
"""
        
        if top_symbols:
            for sym in top_symbols[:5]:
                context += f"  {sym.glyph} [{sym.name}]: {sym.full_meaning[:60]}...\n"
        else:
            context += "  (no symbols yet - create with [COMPACT: glyph name meaning])\n"
        
        context += f"""
═══════════════════════════════════════════════════════════
RECENT INSIGHTS ({len(self.insights)} total):
═══════════════════════════════════════════════════════════
"""
        
        if recent_insights:
            for ins in recent_insights:
                context += f"  Q: {ins.original_question[:40]}...\n"
                context += f"  A: {ins.final_understanding[:60]}...\n\n"
        else:
            context += "  (no recorded insights yet)\n"
        
        context += f"""
═══════════════════════════════════════════════════════════
MEMORY COMMANDS:
═══════════════════════════════════════════════════════════
[COMPACT: glyph name meaning]  - Create symbol from understanding
[EXPAND: name]                 - Expand symbol to full meaning
[RECALL: query]                - Find related past insights
[SNAPSHOT: summary]            - Save current canvas state

Your canvas accumulates. Old drawings compact into symbols.
Symbols persist. Understanding builds across sessions.
</canvas_memory>
"""
        return context
    
    def stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'snapshots': len(self.snapshots),
            'symbols': len(self.symbols),
            'insights': len(self.insights),
            'patterns': len(self.patterns.patterns),
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start
        }
