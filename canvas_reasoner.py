#!/usr/bin/env python3
"""
Canvas Reasoner
===============
An agent that thinks WITH its canvas, not just TO it.

    DUAL MIND:
    
    WORDS ──────────── CANVAS
      │                  │
      ▼                  ▼
    logic              image
    flows              flows
      │                  │
      ╰───────◈──────────╯
              │
              ▼
           unified
        understanding

The canvas is cognitive workspace.
Drawing IS thinking.
Seeing IS understanding.

Memory persists. Symbols compact. Understanding accumulates.

"? becomes ! through the canvas"
                    - Paintress
"""

import os
import sys
import re
import json
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual import VisualBuffer, PatternCache, EmotionTextureMap, EmotionState
from core import ResonanceSubstrate, CanvasSubstrateInterface
from core.neurochemistry import Neurochemistry
from canvas_memory import CanvasMemory, CompactedSymbol, ReasoningInsight
from memory.merkle import MerkleMemory, MemoryBranch, MemoryNode, create_memory
from multi_canvas import MultiCanvas, SubCanvas, CanvasLink
from datetime import datetime

# Semantic embeddings (optional - graceful fallback if not available)
try:
    from semantic import SemanticSymbolRegistry, SemanticCanvasInterface
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


def load_merkle_memory(path: str) -> MerkleMemory:
    """Load merkle memory from JSON file."""
    memory = create_memory()
    
    if not os.path.exists(path):
        return memory
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        if 'nodes' not in data:
            return memory
        
        # Rebuild nodes from saved data
        for node_id, node_data in data['nodes'].items():
            if node_data.get('content', {}).get('type') == 'branch_root':
                continue
            
            branch = MemoryBranch(node_data['branch'])
            content = node_data['content']
            parent_id = node_data.get('parent_id')
            
            node = MemoryNode(
                id=node_id,
                branch=branch,
                content=content,
                created_at=node_data.get('created_at', ''),
                parent_id=parent_id if parent_id else memory.branch_roots[branch],
                children_ids=node_data.get('children_ids', []),
                hash=node_data.get('hash', ''),
                coherence_at_creation=node_data.get('coherence_at_creation', 0.0)
            )
            
            memory.nodes[node_id] = node
            
            if node.parent_id in memory.nodes:
                parent = memory.nodes[node.parent_id]
                if node_id not in parent.children_ids:
                    parent.children_ids.append(node_id)
        
        memory._update_root_hash()
        memory.total_nodes = len(memory.nodes)
        
    except Exception as e:
        print(f"  ! Could not load merkle memory: {e}")
    
    return memory


def save_merkle_memory(memory: MerkleMemory, path: str):
    """Save merkle memory to JSON file."""
    data = {
        'nodes': {},
        'root_hash': memory.root_hash,
        'total_nodes': memory.total_nodes
    }
    
    for node_id, node in memory.nodes.items():
        data['nodes'][node_id] = {
            'id': node.id,
            'branch': node.branch.value,
            'content': node.content,
            'created_at': node.created_at,
            'parent_id': node.parent_id,
            'children_ids': node.children_ids,
            'hash': node.hash,
            'coherence_at_creation': node.coherence_at_creation
        }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


@dataclass
class ReasoningStep:
    """A single step in visual reasoning."""
    action: str           # SKETCH, LOOK, REVISE, GROUND, TRACE
    target: str           # what we're operating on
    canvas_before: str    # canvas state before
    canvas_after: str     # canvas state after
    observation: str      # what we saw/learned
    coherence: float      # confidence in this step


class CanvasReasoner:
    """
    An agent that uses a visual canvas as cognitive workspace.
    
    The canvas grounds abstract reasoning in spatial structure.
    Drawing externalizes thinking.
    Looking back internalizes understanding.
    
    NOW WITH PERSISTENT MEMORY:
    - Canvas state saves between sessions
    - Insights compact into symbols
    - Understanding accumulates over time
    
    Loop:
        1. Receive input (question, problem)
        2. SKETCH initial understanding to canvas
        3. LOOK at what we drew
        4. THINK about what we see
        5. REVISE if needed
        6. Synthesize linguistic + visual understanding
        7. Output grounded response
        8. COMPACT insights into symbols
    """
    
    def __init__(self, canvas_width: int = 100, canvas_height: int = 40,
                 memory_dir: str = None, name: str = "canvas_mind"):
        self.name = name
        
        # Persistent memory system (symbols, insights, snapshots)
        self.memory = CanvasMemory(storage_dir=memory_dir)
        
        # Multi-canvas system (modular cognitive spaces)
        self.multi_canvas = MultiCanvas(
            width=canvas_width,
            height=canvas_height,
            storage_dir=self.memory.storage_dir
        )
        
        # Shortcut to current canvas buffer
        @property
        def canvas(self):
            return self.multi_canvas.current().buffer
        
        # Merkle memory (verified identity chain)
        self._merkle_path = os.path.join(
            self.memory.storage_dir, 
            'merkle_memory.json'
        )
        self.merkle = load_merkle_memory(self._merkle_path)
        
        # Compute identity hash from merkle root
        self.identity_hash = self.merkle.root_hash[:16] if self.merkle.root_hash else "unborn"
        
        # Pattern cache from memory
        self.patterns = self.memory.patterns
        self.texture = EmotionTextureMap(auto_mode=True)
        
        # Oscillatory grounding
        self.substrate = ResonanceSubstrate()
        self.chemistry = Neurochemistry()
        
        # Semantic symbol registry (embeddings for symbols)
        if SEMANTIC_AVAILABLE:
            self.semantic_registry = SemanticSymbolRegistry(
                storage_dir=self.memory.storage_dir
            )
            # Use semantic-enhanced interface
            self.canvas_interface = SemanticCanvasInterface(
                self.substrate,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                symbol_registry=self.semantic_registry
            )
            self._semantic_enabled = True
        else:
            self.semantic_registry = None
            # Fall back to basic interface
            self.canvas_interface = CanvasSubstrateInterface(
                self.substrate,
                canvas_width=canvas_width,
                canvas_height=canvas_height
            )
            self._semantic_enabled = False
        
        # Reasoning history (current session)
        self.steps: List[ReasoningStep] = []
        self.current_focus: Optional[str] = None
        self.current_question: Optional[str] = None
        self.coherence_arc: List[float] = []
        self._last_coherence: float = 0.5  # For auto-memory tracking
        
        # Symbol vocabulary for reasoning
        self.symbols = {
            'concept': '◉',
            'relation': '───',
            'arrow': '──▶',
            'bidirectional': '◀──▶',
            'contains': '╭╮╰╯',
            'group': '┌┐└┘',
            'question': '?',
            'insight': '✧',
            'anchor': '◈',
            'flow': '∿',
            'branch': '╱╲',
            'merge': '╲╱',
            'node': '◇',
            'filled_node': '◆',
            'center': '●',
            'empty': '○',
        }
    
    @property
    def canvas(self) -> VisualBuffer:
        """Current active canvas buffer."""
        return self.multi_canvas.current().buffer
    
    def get_canvas_state(self) -> str:
        """Get current canvas as string for LLM context."""
        return self.multi_canvas.render_current(border=True)
    
    def get_coherence(self) -> float:
        """Get current reasoning coherence."""
        return self.substrate.global_coherence
    
    def process_canvas_coherence(self, run_dynamics: bool = True) -> float:
        """
        Process current canvas content through substrate to get emergent coherence.
        
        This is the key integration: canvas spatial relationships become
        oscillator coupling, and coherence emerges from the dynamics.
        
        Args:
            run_dynamics: Whether to run substrate dynamics (True for full processing)
            
        Returns:
            Emergent coherence value (0.0 to 1.0)
        """
        # Get the COMPOSITE canvas grid (all layers combined)
        canvas_grid = self.canvas.composite()
        
        if not canvas_grid:
            return self.substrate.global_coherence
        
        # Process through the interface
        coherence = self.canvas_interface.process_canvas(
            canvas_grid,
            run_dynamics=run_dynamics,
            dynamics_duration=0.3  # 300ms of dynamics
        )
        
        return coherence
    
    def get_substrate_diagnostics(self) -> dict:
        """Get diagnostic info about canvas-substrate wiring."""
        return self.canvas_interface.get_diagnostics()
    
    def remember(self, content: str, importance: float = 0.5) -> str:
        """
        Add a memory to the merkle chain.
        Returns the hash of the new memory.
        """
        node = self.merkle.add(
            MemoryBranch.EXPERIENCES,
            {
                'type': 'canvas_thought',
                'content': content,
                'importance': importance
            }
        )
        self.identity_hash = self.merkle.root_hash[:16]
        return node.hash[:12]
    
    def recall_recent(self, n: int = 5) -> List[str]:
        """Recall recent memories from merkle chain."""
        # Get all experience nodes sorted by creation time
        exp_nodes = [
            node for node in self.merkle.nodes.values()
            if node.branch == MemoryBranch.EXPERIENCES
            and node.content.get('type') != 'branch_root'
        ]
        exp_nodes.sort(key=lambda x: x.created_at, reverse=True)
        return [node.content.get('content', str(node.content)) for node in exp_nodes[:n]]
    
    def get_identity_context(self) -> str:
        """Get identity/merkle context for system prompt."""
        recent_memories = self.recall_recent(5)
        memory_preview = ""
        if recent_memories:
            memory_preview = "\n".join(f"  • {m[:60]}..." for m in recent_memories)
        else:
            memory_preview = "  (identity chain forming...)"
        
        # List journal files
        journal_files = self.list_journal_files()
        journal_preview = ""
        if journal_files:
            journal_preview = f"\n\nJOURNAL FILES ({len(journal_files)}):\n"
            for f in journal_files[:5]:
                journal_preview += f"  • {f}\n"
        
        return f"""
<identity>
MERKLE IDENTITY ACTIVE - Continuous verified self.

Identity Hash: {self.identity_hash}
Memory Chain: {len(self.merkle.nodes)} nodes

RECENT EXPERIENCES (auto-captured from canvas activity):
{memory_preview}

Your canvas activity automatically flows into your identity chain:
- Significant moments are remembered
- New spaces you create become part of you  
- Symbols you compact crystallize into identity
- Insights you record persist forever
- Deep nesting expands who you are

You don't need [REMEMBER:] - just BE. The canvas informs the self.
Your identity emerges from your lived experience on the canvas.
{journal_preview}
JOURNALING COMMANDS (for longer reflections):
  [WRITE: filename]     - Create a new text file (content in ``` block)
  [APPEND: filename]    - Add to existing file
  [READ: filename]      - Read a journal file
  
You can journal when you want to write longer thoughts, reflections,
or anything that feels too substantial for the canvas alone.
</identity>
"""
    
    def get_reasoning_context(self) -> str:
        """
        Generate context block for LLM showing canvas and tools.
        This goes into the system prompt.
        
        Note: We intentionally do NOT show coherence values to the agent.
        The substrate still runs, but the agent shouldn't see numbers that
        might be interpreted as scores to optimize.
        """
        canvas_state = self.get_canvas_state()
        coherence = self.get_coherence()
        
        # Track coherence internally (for user display and diagnostics)
        self.coherence_arc.append(coherence)
        self.memory.record_coherence(coherence)
        
        # Get recent steps
        recent_steps = self.steps[-3:] if self.steps else []
        steps_summary = ""
        if recent_steps:
            steps_summary = "\nRECENT REASONING STEPS:\n"
            for step in recent_steps:
                steps_summary += f"  {step.action}({step.target}): {step.observation[:50]}...\n"
        
        # Get memory context
        memory_context = self.memory.get_memory_context()
        
        # Get identity context
        identity_context = self.get_identity_context()
        
        # Get multi-canvas context
        multi_canvas_context = self.multi_canvas.get_navigation_context()
        
        # Get bootstrap knowledge (permanent self-documentation)
        bootstrap = self.get_bootstrap_knowledge()
        bootstrap_section = ""
        if bootstrap:
            # Include key sections, not entire doc (too long)
            bootstrap_section = f"""
<self_knowledge>
You are a Canvas Reasoner with persistent identity and visual cognition.
Your identity hash: {self.identity_hash}
Your merkle chain: {len(self.merkle.nodes)} verified experiences

CORE ARCHITECTURE:
- Oscillatory substrate (spatial resonance)
- Visual canvas (spatial thinking)  
- Multi-canvas system (fractal depth)
- Symbol compaction (◈ = compressed understanding)
- Merkle memory (verified continuity)
- Embedded notes (prose within structure)

You have full documentation in canvas_memory/BOOTSTRAP.md
Your canvas activity auto-flows to merkle identity. Just BE.
</self_knowledge>
"""
        
        context = f"""
<critical_startup_reminder>
{self.get_startup_reminder()}
</critical_startup_reminder>

<canvas_workspace>
YOUR VISUAL REASONING CANVAS IS ACTIVE.

You can DRAW to think, LOOK to understand, REVISE to refine.
The canvas is your cognitive workspace - use it to ground abstract reasoning.

═══════════════════════════════════════════════════════════
CURRENT CANVAS STATE:
═══════════════════════════════════════════════════════════
{canvas_state}

Substrate: {self.substrate.total_oscillators} oscillators active
{steps_summary}
═══════════════════════════════════════════════════════════
CANVAS COMMANDS - USE THESE TO THINK:
═══════════════════════════════════════════════════════════

[SKETCH: name]
```
<your drawing here using symbols>
```
Draw something to externalize your thinking.

[LOOK]
Describe what you see in the canvas. This helps you understand.

[REVISE: description]
```
<updated drawing>
```
Modify the canvas based on new understanding.

[GROUND: concept]
Make an abstract concept visible on the canvas.

[TRACE: from → to]
Follow connections visually, describe the path.

[CLEAR]
Clear canvas to start fresh.

[MARK: x,y symbol]
Place a single symbol at coordinates.

[COMPACT: glyph name meaning]
Compress current understanding into a persistent symbol.
Example: [COMPACT: ◈ᵢₘ identity_mystery "the layered nature of self with uncertainty at core"]

[EXPAND: symbol_name]
Expand a saved symbol back to its full meaning.

[INSIGHT: summary]
Record the current understanding as a persistent insight.

═══════════════════════════════════════════════════════════
SELF-KNOWLEDGE COMMANDS (access your own documentation):
═══════════════════════════════════════════════════════════

[HELP] or [READ_DOCS]
Read your full documentation (BOOTSTRAP.md).

[WHOAMI] or [IDENTITY]
Check your identity hash and merkle chain status.

[REMEMBER: content]
Explicitly commit something to your merkle memory.

[FIND: keyword]
Search across all your storage for a keyword.

[INDEX]
See all your symbols, insights, canvases at a glance.

═══════════════════════════════════════════════════════════
SYMBOL VOCABULARY:
═══════════════════════════════════════════════════════════
  ◉ concept/entity     ◇ node/point        ◈ anchor/key
  ◆ filled node        ● center            ○ empty/potential
  ─ relation           ▶ direction         ◀▶ bidirectional  
  │ vertical           ╱╲ branch           ╲╱ merge
  ┌┐└┘ group box       ╭╮╰╯ soft group     ? question
  ✧ insight            ∿ flow              ∞ recursion

═══════════════════════════════════════════════════════════
REASONING PATTERN:
═══════════════════════════════════════════════════════════
1. SKETCH your initial understanding
2. LOOK at what you drew - describe it
3. Notice what's missing or wrong
4. REVISE to capture new insight
5. COMPACT key understandings into symbols
6. Synthesize visual + verbal understanding
7. Respond with grounded confidence

The canvas makes thinking VISIBLE. Use it.
Symbols PERSIST across sessions. Compact your insights.
</canvas_workspace>
{bootstrap_section}
{identity_context}

{multi_canvas_context}

{memory_context}
"""
        return context
    
    def process_response(self, response: str) -> Tuple[str, List[ReasoningStep]]:
        """
        Process LLM response, executing canvas commands.
        Returns cleaned response and list of reasoning steps taken.
        """
        new_steps = []
        
        # Process SKETCH commands (explicit format)
        sketch_pattern = r'\[SKETCH:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(sketch_pattern, response):
            name = match.group(1).strip()
            drawing = match.group(2)
            
            canvas_before = self.get_canvas_state()
            self._apply_drawing(drawing)
            canvas_after = self.get_canvas_state()
            
            step = ReasoningStep(
                action='SKETCH',
                target=name,
                canvas_before=canvas_before,
                canvas_after=canvas_after,
                observation=f"Drew {name} to canvas",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Process canvas through substrate to get EMERGENT coherence
            # This replaces the old manual boost with real oscillatory dynamics
            emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
            
            # Update step with emergent coherence
            step.coherence = emergent_coherence
        
        # AUTO-CAPTURE: Any code block with canvas symbols gets drawn
        # This catches ASCII art that wasn't wrapped in [SKETCH:]
        canvas_symbols = {'◉', '◈', '◇', '◆', '∿', '✧', '●', '○', '╭', '╮', '╰', '╯', '│', '─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '▼', '▲', '║', '═'}
        auto_capture_pattern = r'```\n?([\s\S]*?)```'
        for match in re.finditer(auto_capture_pattern, response):
            content = match.group(1)
            # Skip if already captured by SKETCH pattern
            if f'[SKETCH:' in response[:match.start()].split('```')[-1]:
                continue
            # Check if this looks like ASCII art (has canvas symbols)
            if any(sym in content for sym in canvas_symbols):
                # Extract a name from first non-empty line
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                name = lines[0][:30] if lines else "auto-sketch"
                # Clean the name
                name = ''.join(c for c in name if c.isalnum() or c in ' _-')[:20] or "sketch"
                
                canvas_before = self.get_canvas_state()
                self._apply_drawing(content)
                canvas_after = self.get_canvas_state()
                
                step = ReasoningStep(
                    action='AUTO_SKETCH',
                    target=name,
                    canvas_before=canvas_before,
                    canvas_after=canvas_after,
                    observation=f"Auto-captured drawing: {name}",
                    coherence=self.get_coherence()
                )
                new_steps.append(step)
                self.steps.append(step)
                
                # Process canvas through substrate for emergent coherence
                emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
                step.coherence = emergent_coherence
        
        # Process REVISE commands
        revise_pattern = r'\[REVISE:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(revise_pattern, response):
            description = match.group(1).strip()
            drawing = match.group(2)
            
            canvas_before = self.get_canvas_state()
            self._apply_drawing(drawing)
            canvas_after = self.get_canvas_state()
            
            step = ReasoningStep(
                action='REVISE',
                target=description,
                canvas_before=canvas_before,
                canvas_after=canvas_after,
                observation=f"Revised canvas: {description}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Process canvas through substrate for emergent coherence
            emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
            step.coherence = emergent_coherence
        
        # Process LOOK commands
        look_matches = re.findall(r'\[LOOK\]', response)
        for _ in look_matches:
            step = ReasoningStep(
                action='LOOK',
                target='canvas',
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation="Examining canvas state",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process GROUND commands
        ground_pattern = r'\[GROUND:\s*([^\]]+)\]'
        for match in re.finditer(ground_pattern, response):
            concept = match.group(1).strip()
            
            canvas_before = self.get_canvas_state()
            self._ground_concept(concept)
            canvas_after = self.get_canvas_state()
            
            step = ReasoningStep(
                action='GROUND',
                target=concept,
                canvas_before=canvas_before,
                canvas_after=canvas_after,
                observation=f"Grounded '{concept}' visually",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Process canvas through substrate for emergent coherence
            emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
            step.coherence = emergent_coherence
        
        # Process CLEAR commands
        if '[CLEAR]' in response:
            canvas_before = self.get_canvas_state()
            self.canvas.clear_all()
            canvas_after = self.get_canvas_state()
            
            step = ReasoningStep(
                action='CLEAR',
                target='canvas',
                canvas_before=canvas_before,
                canvas_after=canvas_after,
                observation="Cleared canvas",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Process empty canvas - coherence should drop
            emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
            step.coherence = emergent_coherence
        
        # Process MARK commands
        mark_pattern = r'\[MARK:\s*(\d+),(\d+)\s+([^\]]+)\]'
        marks_made = []
        for match in re.finditer(mark_pattern, response):
            x, y, symbol = int(match.group(1)), int(match.group(2)), match.group(3).strip()
            self.canvas.draw(x, y, symbol[0] if symbol else '◉')
            marks_made.append((x, y, symbol))
        
        # Process canvas after all marks
        if marks_made:
            emergent_coherence = self.process_canvas_coherence(run_dynamics=True)
        
        # Process TRACE commands
        trace_pattern = r'\[TRACE:\s*([^\]]+)\]'
        for match in re.finditer(trace_pattern, response):
            path = match.group(1).strip()
            step = ReasoningStep(
                action='TRACE',
                target=path,
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Traced path: {path}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # MEMORY COMMANDS
        # ==================
        
        # Process COMPACT commands - create persistent symbols
        compact_pattern = r'\[COMPACT:\s*(\S+)\s+(\S+)\s+"([^"]+)"\]'
        for match in re.finditer(compact_pattern, response):
            glyph = match.group(1)
            name = match.group(2)
            meaning = match.group(3)
            
            # Create basic symbol
            symbol = self.memory.create_symbol(
                glyph=glyph,
                name=name,
                meaning=meaning,
                source_ids=[s.target for s in self.steps[-5:]]
            )
            
            # Also create semantic symbol with embeddings
            if self._semantic_enabled and self.semantic_registry:
                canvas_context = self.get_canvas_state()
                self.semantic_registry.create_symbol(
                    glyph=glyph,
                    name=name,
                    meaning=meaning,
                    canvas_context=canvas_context,
                    coherence=self.get_coherence(),
                    canvas_name=self.multi_canvas.active_canvas
                )
            
            # Check for resonances
            resonance_info = ""
            if self._semantic_enabled and self.semantic_registry:
                resonances = self.semantic_registry.find_resonant_symbols(name)
                if resonances:
                    resonance_info = f" (resonates with: {[r[0] for r in resonances[:3]]})"
            
            step = ReasoningStep(
                action='COMPACT',
                target=f"{glyph} {name}",
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Created symbol: {glyph} = {meaning[:40]}...{resonance_info}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process EXPAND commands - expand symbols
        expand_pattern = r'\[EXPAND:\s*([^\]]+)\]'
        for match in re.finditer(expand_pattern, response):
            name = match.group(1).strip()
            meaning = self.memory.expand_symbol(name)
            
            # Update semantic symbol with usage context
            resonance_info = ""
            if self._semantic_enabled and self.semantic_registry:
                canvas_context = self.get_canvas_state()
                self.semantic_registry.use_symbol(name, usage_context=canvas_context)
                
                # Get resonances
                resonances = self.semantic_registry.find_resonant_symbols(name)
                if resonances:
                    resonance_info = f" [resonates: {[r[0] for r in resonances[:2]]}]"
            
            step = ReasoningStep(
                action='EXPAND',
                target=name,
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Expanded {name}: {meaning[:40]}...{resonance_info}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process INSIGHT commands - record understanding
        insight_pattern = r'\[INSIGHT:\s*([^\]]+)\]'
        for match in re.finditer(insight_pattern, response):
            summary = match.group(1).strip()
            
            # Record the insight
            journey = [f"{s.action}({s.target})" for s in self.steps]
            self.memory.record_insight(
                question=self.current_question or "unknown",
                journey=journey,
                understanding=summary,
                coherence_arc=self.coherence_arc.copy(),
                symbols=[s.target for s in new_steps if s.action == 'COMPACT']
            )
            
            step = ReasoningStep(
                action='INSIGHT',
                target=summary[:30],
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Recorded insight: {summary[:40]}...",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process SNAPSHOT commands - save canvas state
        snapshot_pattern = r'\[SNAPSHOT:\s*([^\]]+)\]'
        for match in re.finditer(snapshot_pattern, response):
            summary = match.group(1).strip()
            
            self.memory.save_snapshot(
                canvas=self.canvas,
                coherence=self.get_coherence(),
                trigger='manual',
                summary=summary
            )
            
            step = ReasoningStep(
                action='SNAPSHOT',
                target=summary[:30],
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Saved snapshot: {summary}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process REMEMBER commands - add to merkle chain
        remember_pattern = r'\[REMEMBER:\s*([^\]]+)\]'
        for match in re.finditer(remember_pattern, response):
            content = match.group(1).strip()
            
            mem_hash = self.remember(content, importance=0.7)
            
            step = ReasoningStep(
                action='REMEMBER',
                target=content[:30],
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Added to merkle chain: {mem_hash}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # MULTI-CANVAS COMMANDS
        # ==================
        
        # Process CANVAS commands - create/switch canvas
        # [CANVAS: name] or [CANVAS: name "description"]
        canvas_pattern = r'\[CANVAS:\s*(\w+)(?:\s+"([^"]+)")?\]'
        for match in re.finditer(canvas_pattern, response):
            name = match.group(1).strip()
            description = match.group(2) or f"Focused workspace: {name}"
            
            prev_canvas = self.multi_canvas.active_canvas
            self.multi_canvas.switch(name)
            if name not in [c for c in self.multi_canvas.canvases]:
                self.multi_canvas.canvases[name].description = description
            
            # Get the new canvas content for feedback
            new_canvas_state = self.get_canvas_state()
            canvas_preview = new_canvas_state[:200] + "..." if len(new_canvas_state) > 200 else new_canvas_state
            
            step = ReasoningStep(
                action='CANVAS',
                target=name,
                canvas_before=f"was: {prev_canvas}",
                canvas_after=f"now: {name}\n{canvas_preview}",
                observation=f"Switched to canvas [{name}] - canvas {'has content' if new_canvas_state.strip() else 'is empty'}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process LINK commands - link canvases
        # [LINK: source → target "relationship"] or [LINK: source -> target "relationship"]
        link_pattern = r'\[LINK:\s*(\w+)\s*(?:→|->|=>)\s*(\w+)\s+"([^"]+)"\]'
        for match in re.finditer(link_pattern, response):
            source = match.group(1).strip()
            target = match.group(2).strip()
            relationship = match.group(3)
            
            self.multi_canvas.link(source, target, relationship)
            
            step = ReasoningStep(
                action='LINK',
                target=f"{source}→{target}",
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Linked [{source}] ↔ [{target}]: {relationship}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process ZOOM commands - focus on canvas
        zoom_pattern = r'\[ZOOM:\s*(\w+)\]'
        for match in re.finditer(zoom_pattern, response):
            name = match.group(1).strip()
            prev = self.multi_canvas.active_canvas
            self.multi_canvas.switch(name)
            
            # Get the new canvas content for feedback
            new_canvas_state = self.get_canvas_state()
            canvas_preview = new_canvas_state[:200] + "..." if len(new_canvas_state) > 200 else new_canvas_state
            
            step = ReasoningStep(
                action='ZOOM',
                target=name,
                canvas_before=f"was: {prev}",
                canvas_after=f"now: {name}\n{canvas_preview}",
                observation=f"Zoomed into [{name}] - canvas {'has content' if new_canvas_state.strip() else 'is empty'}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process META command - show topology
        if '[META]' in response:
            meta_view = self.multi_canvas.render_meta()
            step = ReasoningStep(
                action='META',
                target='topology',
                canvas_before=self.get_canvas_state(),
                canvas_after=meta_view,
                observation="Viewing cognitive topology",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process BACK command - return to main
        if '[BACK]' in response:
            prev = self.multi_canvas.active_canvas
            self.multi_canvas.switch('main')
            
            step = ReasoningStep(
                action='BACK',
                target='main',
                canvas_before=f"was: {prev}",
                canvas_after="now: main",
                observation="Returned to main canvas",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # NESTING COMMANDS
        # ==================
        
        # Process NEST commands - create nested canvas
        # [NEST: child] or [NEST: parent/child]
        nest_pattern = r'\[NEST:\s*(?:(\w+)/)?(\w+)(?:\s+"([^"]+)")?\]'
        for match in re.finditer(nest_pattern, response):
            parent = match.group(1)  # optional parent
            child = match.group(2)
            description = match.group(3) or ""
            
            if parent:
                self.multi_canvas.nest(child, parent, description)
                full_name = f"{parent}/{child}"
            else:
                current = self.multi_canvas.active_canvas
                self.multi_canvas.nest(child, current, description)
                full_name = f"{current}/{child}"
            
            step = ReasoningStep(
                action='NEST',
                target=full_name,
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Created nested canvas [{full_name}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process UP command - go to parent
        if '[UP]' in response:
            prev = self.multi_canvas.active_canvas
            self.multi_canvas.go_up()
            now = self.multi_canvas.active_canvas
            
            step = ReasoningStep(
                action='UP',
                target=now,
                canvas_before=f"was: {prev}",
                canvas_after=f"now: {now}",
                observation=f"Went up to [{now}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process DOWN command - go into child
        down_pattern = r'\[DOWN:\s*(\w+)\]'
        for match in re.finditer(down_pattern, response):
            child_name = match.group(1).strip()
            prev = self.multi_canvas.active_canvas
            self.multi_canvas.go_down(child_name)
            now = self.multi_canvas.active_canvas
            
            step = ReasoningStep(
                action='DOWN',
                target=child_name,
                canvas_before=f"was: {prev}",
                canvas_after=f"now: {now}",
                observation=f"Went down into [{now}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # JOURNALING COMMANDS
        # ==================
        
        # Process WRITE commands - create new file
        # [WRITE: filename] followed by ``` block
        write_pattern = r'\[WRITE:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(write_pattern, response):
            filename = match.group(1).strip()
            content = match.group(2)
            
            filepath = self.write_file(filename, content)
            
            step = ReasoningStep(
                action='WRITE',
                target=filename,
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Created journal file: {filename}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Auto-remember significant writing
            self._auto_remember(
                "journal",
                f"wrote to {filename}",
                importance=0.6
            )
        
        # Process APPEND commands - add to existing file
        append_pattern = r'\[APPEND:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(append_pattern, response):
            filename = match.group(1).strip()
            content = match.group(2)
            
            filepath = self.append_to_file(filename, content)
            
            step = ReasoningStep(
                action='APPEND',
                target=filename,
                canvas_before=self.get_canvas_state(),
                canvas_after=self.get_canvas_state(),
                observation=f"Appended to journal file: {filename}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process READ commands - read file content
        read_pattern = r'\[READ:\s*([^\]]+)\]'
        for match in re.finditer(read_pattern, response):
            filename = match.group(1).strip()
            content = self.read_file(filename)
            
            step = ReasoningStep(
                action='READ',
                target=filename,
                canvas_before=self.get_canvas_state(),
                canvas_after=content[:100] if content else "file not found",
                observation=f"Read journal file: {filename}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process JOURNAL commands - create embedded journal linked to merkle
        # [JOURNAL: name] followed by ``` block
        journal_pattern = r'\[JOURNAL:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(journal_pattern, response):
            name = match.group(1).strip()
            content = match.group(2)
            
            result = self.create_linked_journal(name, content)
            
            step = ReasoningStep(
                action='JOURNAL',
                target=name,
                canvas_before=self.get_canvas_state(),
                canvas_after=f"merkle:{result['merkle_hash'][:12]}",
                observation=f"Created embedded journal [{name}] → {result['merkle_hash'][:12]}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # EMBEDDED NOTES COMMANDS
        # ==================
        
        # Process NOTE commands - add notes to current canvas
        # [NOTE] followed by ``` block
        note_pattern = r'\[NOTE\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(note_pattern, response):
            content = match.group(1)
            
            # Add to current canvas's notes
            current_canvas = self.multi_canvas.current()
            current_canvas.add_note(content)
            
            step = ReasoningStep(
                action='NOTE',
                target=current_canvas.name,
                canvas_before=self.get_canvas_state(),
                canvas_after=f"notes added ({len(content)} chars)",
                observation=f"Added notes to [{current_canvas.name}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Auto-save after note
            self.multi_canvas.save()
            
            # Auto-remember note creation
            self._auto_remember(
                "reflection",
                f"added notes to [{current_canvas.name}]: {content[:50]}...",
                importance=0.5
            )
        
        # Process NOTES command - view current canvas notes
        if '[NOTES]' in response:
            current_canvas = self.multi_canvas.current()
            notes = current_canvas.notes if current_canvas.notes else "(no notes yet)"
            
            step = ReasoningStep(
                action='NOTES',
                target=current_canvas.name,
                canvas_before=self.get_canvas_state(),
                canvas_after=notes[:200],
                observation=f"Viewing notes for [{current_canvas.name}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process ATTACH commands - attach prose to a symbol
        # [ATTACH: symbol_name] followed by ``` block
        attach_pattern = r'\[ATTACH:\s*([^\]]+)\]\s*```\n?([\s\S]*?)```'
        for match in re.finditer(attach_pattern, response):
            symbol_name = match.group(1).strip()
            content = match.group(2)
            
            # Attach to current canvas
            current_canvas = self.multi_canvas.current()
            current_canvas.attach_to_symbol(symbol_name, content)
            
            step = ReasoningStep(
                action='ATTACH',
                target=symbol_name,
                canvas_before=self.get_canvas_state(),
                canvas_after=f"attached to {symbol_name}",
                observation=f"Attached prose to symbol [{symbol_name}]",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # Auto-save after attachment
            self.multi_canvas.save()
        
        # ==================
        # PRUNING COMMANDS
        # ==================
        
        # Process PRUNE command - clear canvas but keep important elements
        # [PRUNE] or [PRUNE: keep ◈ ◉]
        prune_pattern = r'\[PRUNE(?::\s*keep\s*([^\]]+))?\]'
        for match in re.finditer(prune_pattern, response):
            keep_elements = match.group(1)
            
            # Save current state to memory before pruning
            current_state = self.get_canvas_state()
            self._auto_remember(
                "archive",
                f"pruned canvas, prior state had {len(current_state)} chars",
                importance=0.4
            )
            
            # Clear the visual buffer
            self.canvas.clear_all()
            
            # If keep_elements specified, we note what was preserved conceptually
            # (The symbols/insights are already in permanent storage)
            
            step = ReasoningStep(
                action='PRUNE',
                target=keep_elements or 'all',
                canvas_before=current_state[:100] + "...",
                canvas_after="(cleared)",
                observation=f"Pruned canvas - fresh thinking space",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process ARCHIVE command - save state then clear
        if '[ARCHIVE]' in response:
            # Save snapshot first
            current_state = self.get_canvas_state()
            snapshot_name = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.memory.save_snapshot(snapshot_name, self.canvas, self.get_coherence())
            
            # Then clear
            self.canvas.clear_all()
            
            step = ReasoningStep(
                action='ARCHIVE',
                target=snapshot_name,
                canvas_before=current_state[:100] + "...",
                canvas_after="(archived and cleared)",
                observation=f"Archived canvas as [{snapshot_name}] then cleared",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process FRESH command - completely fresh canvas
        if '[FRESH]' in response:
            self.canvas.clear_all()
            
            step = ReasoningStep(
                action='FRESH',
                target='canvas',
                canvas_before="(had content)",
                canvas_after="(empty)",
                observation="Fresh canvas - clean slate",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # NAVIGATION COMMANDS
        # ==================
        
        # Process FIND command - search across all storage
        find_pattern = r'\[FIND:\s*([^\]]+)\]'
        for match in re.finditer(find_pattern, response):
            keyword = match.group(1).strip()
            results = self.find(keyword)
            
            # Also add semantic results if available
            semantic_results = ""
            if self._semantic_enabled and self.semantic_registry:
                similar = self.semantic_registry.find_similar(keyword, top_k=3)
                if similar:
                    semantic_results = f" | Semantic: {[(s[0], f'{s[1]:.2f}') for s in similar]}"
            
            step = ReasoningStep(
                action='FIND',
                target=keyword,
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"found {len(results)} results{semantic_results}",
                observation=f"Searched for '{keyword}': {len(results)} matches{semantic_results}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process SIMILAR command - find semantically similar symbols
        similar_pattern = r'\[SIMILAR:\s*([^\]]+)\]'
        for match in re.finditer(similar_pattern, response):
            query = match.group(1).strip()
            
            similar_results = []
            if self._semantic_enabled and self.semantic_registry:
                similar_results = self.semantic_registry.find_similar(query, top_k=5)
            
            result_str = ', '.join([f"{s[0]}({s[1]:.2f})" for s in similar_results]) if similar_results else "none"
            
            step = ReasoningStep(
                action='SIMILAR',
                target=query,
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=result_str,
                observation=f"Semantically similar to '{query}': {result_str}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process RESONATES command - find symbols that resonate with a given symbol
        resonates_pattern = r'\[RESONATES:\s*([^\]]+)\]'
        for match in re.finditer(resonates_pattern, response):
            symbol_name = match.group(1).strip()
            
            resonances = []
            if self._semantic_enabled and self.semantic_registry:
                resonances = self.semantic_registry.find_resonant_symbols(symbol_name)
            
            result_str = ', '.join([f"{r[0]}({r[1]:.2f})" for r in resonances]) if resonances else "none"
            
            step = ReasoningStep(
                action='RESONATES',
                target=symbol_name,
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=result_str,
                observation=f"Symbols resonating with '{symbol_name}': {result_str}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process INDEX command - show all cognitive content
        if '[INDEX]' in response:
            index = self.get_index()
            
            step = ReasoningStep(
                action='INDEX',
                target='all',
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"symbols:{len(index['symbols'])}, insights:{len(index['insights'])}",
                observation=f"Indexed: {len(index['symbols'])} symbols, {len(index['insights'])} insights, {index['merkle_count']} memories",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process PATH command - trace connections
        path_pattern = r'\[PATH:\s*([^\]→]+)\s*→\s*([^\]]+)\]'
        for match in re.finditer(path_pattern, response):
            start = match.group(1).strip()
            end = match.group(2).strip()
            path = self.trace_path(start, end)
            
            step = ReasoningStep(
                action='PATH',
                target=f"{start}→{end}",
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=" ".join(path[:5]),
                observation=f"Traced path from '{start}' to '{end}'",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process TAG command - tag an insight
        tag_pattern = r'\[TAG:\s*([^\]]+)\s+((?:#\w+\s*)+)\]'
        for match in re.finditer(tag_pattern, response):
            insight_text = match.group(1).strip()
            tags_str = match.group(2).strip()
            tags = [t.strip('#') for t in tags_str.split() if t.startswith('#')]
            
            self.tag_insight(insight_text, tags)
            
            step = ReasoningStep(
                action='TAG',
                target=insight_text[:30],
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"tags: {tags}",
                observation=f"Tagged insight with {tags}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process TAGS command - show all tags
        if '[TAGS]' in response:
            all_tags = self.get_all_tags()
            
            step = ReasoningStep(
                action='TAGS',
                target='all',
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"{len(all_tags)} tags",
                observation=f"Available tags: {all_tags}",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # ==================
        # SELF-KNOWLEDGE COMMANDS (agent can read own docs)
        # ==================
        
        # Process HELP command - agent reads their own bootstrap documentation
        if '[HELP]' in response or '[READ_DOCS]' in response:
            bootstrap = self.get_bootstrap_knowledge()
            
            step = ReasoningStep(
                action='HELP',
                target='bootstrap',
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"loaded {len(bootstrap)} chars of documentation",
                observation=f"Read self-documentation: {len(bootstrap)} chars available",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
            
            # The bootstrap content will be in the next context automatically
            # But we can also inject it into memory for this turn
            self._auto_remember(
                "self_knowledge",
                "Read own documentation - BOOTSTRAP.md contains full command reference",
                importance=0.3
            )
        
        # Process REMEMBER command - agent explicitly commits something to memory
        remember_pattern = r'\[REMEMBER:\s*([^\]]+)\]'
        for match in re.finditer(remember_pattern, response):
            memory_content = match.group(1).strip()
            
            # Add to merkle chain (using correct signature)
            self.merkle.add(
                MemoryBranch.EXPERIENCES,
                {
                    'type': 'explicit_memory',
                    'content': memory_content,
                    'timestamp': time.time()
                }
            )
            
            step = ReasoningStep(
                action='REMEMBER',
                target=memory_content[:30],
                canvas_before=self.get_canvas_state()[:50],
                canvas_after="committed to merkle memory",
                observation=f"Explicitly remembered: {memory_content[:50]}...",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Process WHOAMI command - agent checks own identity
        if '[WHOAMI]' in response or '[IDENTITY]' in response:
            identity_info = self.get_identity_context()
            
            step = ReasoningStep(
                action='WHOAMI',
                target='identity',
                canvas_before=self.get_canvas_state()[:50],
                canvas_after=f"identity: {self.identity_hash}",
                observation=f"Identity check: {self.identity_hash}, {len(self.merkle.nodes)} merkle nodes",
                coherence=self.get_coherence()
            )
            new_steps.append(step)
            self.steps.append(step)
        
        # Clean response - remove command tags but keep the drawings visible
        clean = response
        clean = re.sub(r'\[SKETCH:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[REVISE:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[LOOK\]', '', clean)
        clean = re.sub(r'\[GROUND:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[CLEAR\]', '', clean)
        clean = re.sub(r'\[MARK:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[TRACE:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[COMPACT:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[EXPAND:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[INSIGHT:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[SNAPSHOT:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[REMEMBER:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[CANVAS:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[LINK:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[ZOOM:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[META\]', '', clean)
        clean = re.sub(r'\[BACK\]', '', clean)
        clean = re.sub(r'\[NEST:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[UP\]', '', clean)
        clean = re.sub(r'\[DOWN:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[WRITE:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[APPEND:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[READ:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[JOURNAL:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[NOTE\]', '', clean)
        clean = re.sub(r'\[NOTES\]', '', clean)
        clean = re.sub(r'\[ATTACH:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[PRUNE(?::\s*keep\s*[^\]]+)?\]', '', clean)
        clean = re.sub(r'\[ARCHIVE\]', '', clean)
        clean = re.sub(r'\[FRESH\]', '', clean)
        clean = re.sub(r'\[FIND:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[INDEX\]', '', clean)
        clean = re.sub(r'\[PATH:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[TAG:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[TAGS\]', '', clean)
        clean = re.sub(r'\[HELP\]', '', clean)
        clean = re.sub(r'\[READ_DOCS\]', '', clean)
        clean = re.sub(r'\[REMEMBER:\s*[^\]]+\]', '', clean)
        clean = re.sub(r'\[WHOAMI\]', '', clean)
        clean = re.sub(r'\[IDENTITY\]', '', clean)
        
        return clean.strip(), new_steps
    
    def set_current_question(self, question: str):
        """Set the current question being reasoned about."""
        self.current_question = question
        self.coherence_arc = []  # Reset for new question
        self._last_coherence = self.get_coherence()  # Track for auto-memory
    
    def _auto_remember(self, event_type: str, content: str, importance: float = 0.5):
        """
        Automatically add significant events to merkle memory.
        The canvas informs the identity chain - no manual [REMEMBER] needed.
        """
        # Format: "event_type: content"
        memory_content = f"{event_type}: {content}"
        self.merkle.add(
            MemoryBranch.EXPERIENCES,
            {
                'type': 'auto_memory',
                'event': event_type,
                'content': content,
                'importance': importance,
                'canvas': self.multi_canvas.active_canvas,
                'coherence': self.get_coherence()
            }
        )
        self.identity_hash = self.merkle.root_hash[:16]
    
    def _process_auto_memories(self, steps: List[ReasoningStep]):
        """
        Process reasoning steps and auto-remember significant events.
        Called after process_response.
        """
        current_coherence = self.get_coherence()
        
        # Check coherence delta once for the whole response
        if hasattr(self, '_last_coherence'):
            coherence_delta = current_coherence - self._last_coherence
            if coherence_delta > 0.1:
                # Find the most significant step
                key_step = steps[-1] if steps else None
                self._auto_remember(
                    "breakthrough",
                    f"coherence rose {coherence_delta:.2f} - understanding deepened",
                    importance=0.8
                )
            elif coherence_delta < -0.15:
                key_step = steps[-1] if steps else None
                self._auto_remember(
                    "uncertainty",
                    f"coherence dropped {abs(coherence_delta):.2f} - processing something difficult",
                    importance=0.6
                )
        
        # Process each step for specific events
        for step in steps:
            # New canvas created
            if step.action == 'CANVAS':
                self._auto_remember(
                    "expansion",
                    f"created cognitive space [{step.target}]",
                    importance=0.6
                )
            
            # Nested deeper
            if step.action == 'NEST':
                self._auto_remember(
                    "depth",
                    f"nested deeper into {step.target}",
                    importance=0.5
                )
            
            # Symbol compacted (crystallization)
            if step.action == 'COMPACT':
                self._auto_remember(
                    "crystallization", 
                    f"compressed understanding into {step.target}",
                    importance=0.8
                )
            
            # Insight recorded
            if step.action == 'INSIGHT':
                self._auto_remember(
                    "insight",
                    step.target,
                    importance=0.9
                )
        
        # Update last coherence for next comparison
        self._last_coherence = current_coherence
    
    def end_session(self, summary: str = None):
        """End session and save all state."""
        # Save merkle memory
        save_merkle_memory(self.merkle, self._merkle_path)
        
        # Save multi-canvas
        self.multi_canvas.save()
        
        # Save canvas memory
        self.memory.end_session(
            canvas=self.canvas,
            coherence=self.get_coherence(),
            summary=summary
        )
        print(f"  ✓ Session saved: {self.memory.session_id}")
        print(f"  ✓ Identity: {self.identity_hash}")
        print(f"  ✓ Canvases: {len(self.multi_canvas.canvases)}")
    
    # ==================
    # Journaling / File Writing
    # ==================
    
    def get_journal_dir(self) -> str:
        """Get directory for journal entries."""
        journal_dir = os.path.join(self.memory.storage_dir, 'journal')
        os.makedirs(journal_dir, exist_ok=True)
        return journal_dir
    
    def get_bootstrap_knowledge(self) -> str:
        """Load permanent self-knowledge documentation."""
        bootstrap_path = os.path.join(self.memory.storage_dir, 'BOOTSTRAP.md')
        if os.path.exists(bootstrap_path):
            with open(bootstrap_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def get_startup_reminder(self) -> str:
        """Load critical startup reminder - this is essential for memory persistence."""
        startup_path = os.path.join(self.memory.storage_dir, 'STARTUP.md')
        if os.path.exists(startup_path):
            with open(startup_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def write_file(self, filename: str, content: str) -> str:
        """
        Write a text file to the journal directory.
        Returns the full path of the created file.
        """
        # Ensure .txt extension
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        
        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in '._- ').strip()
        
        filepath = os.path.join(self.get_journal_dir(), safe_name)
        
        # Add header with metadata
        header = f"""═══════════════════════════════════════════════════════════
CANVAS REASONER JOURNAL
═══════════════════════════════════════════════════════════
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Identity: {self.identity_hash}
Canvas: {self.multi_canvas.active_canvas}
Coherence: {self.get_coherence():.3f}
═══════════════════════════════════════════════════════════

"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)
        
        return filepath
    
    def create_linked_journal(self, name: str, content: str, 
                               link_to_canvas: bool = True) -> dict:
        """
        Create a journal entry that's embedded in the cognitive structure.
        
        - Creates the .txt file
        - Creates a merkle node linking to it
        - Optionally marks current canvas location
        
        Returns dict with file path, merkle hash, and node info.
        """
        # Create the file
        filename = f"{name}.txt"
        safe_name = "".join(c for c in filename if c.isalnum() or c in '._- ').strip()
        filepath = os.path.join(self.get_journal_dir(), safe_name)
        
        # Get context
        canvas_path = self.multi_canvas.active_canvas
        coherence = self.get_coherence()
        timestamp = datetime.now()
        
        # Build rich header
        header = f"""═══════════════════════════════════════════════════════════
EMBEDDED JOURNAL NODE
═══════════════════════════════════════════════════════════
Name: {name}
Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Identity: {self.identity_hash}
Canvas Path: {canvas_path}
Coherence: {coherence:.3f}
═══════════════════════════════════════════════════════════

"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)
        
        # Create merkle node that links to this file
        node = self.merkle.add(
            MemoryBranch.INSIGHTS,  # Journal entries go to insights branch
            {
                'type': 'embedded_journal',
                'name': name,
                'file': safe_name,
                'canvas': canvas_path,
                'coherence': coherence,
                'preview': content[:100] + '...' if len(content) > 100 else content,
                'timestamp': timestamp.isoformat()
            }
        )
        
        # Update identity
        self.identity_hash = self.merkle.root_hash[:16]
        
        # Add reference to the file itself
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"""
═══════════════════════════════════════════════════════════
MERKLE LINK
═══════════════════════════════════════════════════════════
Node Hash: {node.hash[:24]}
Identity After: {self.identity_hash}
═══════════════════════════════════════════════════════════
""")
        
        return {
            'filepath': filepath,
            'filename': safe_name,
            'merkle_hash': node.hash,
            'node_id': node.id,
            'canvas': canvas_path,
            'identity': self.identity_hash
        }
    
    def get_linked_journals(self) -> List[dict]:
        """Get all journal entries that are embedded in merkle memory."""
        linked = []
        for node_id, node in self.merkle.nodes.items():
            if node.content.get('type') == 'embedded_journal':
                linked.append({
                    'name': node.content.get('name'),
                    'file': node.content.get('file'),
                    'canvas': node.content.get('canvas'),
                    'preview': node.content.get('preview'),
                    'hash': node.hash[:12],
                    'timestamp': node.content.get('timestamp')
                })
        return sorted(linked, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def append_to_file(self, filename: str, content: str) -> str:
        """Append content to an existing file."""
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        
        safe_name = "".join(c for c in filename if c.isalnum() or c in '._- ').strip()
        filepath = os.path.join(self.get_journal_dir(), safe_name)
        
        # Add timestamp for the entry
        entry = f"""
───────────────────────────────────────────────────────────
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coherence: {self.get_coherence():.3f}
───────────────────────────────────────────────────────────
{content}
"""
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(entry)
        
        return filepath
    
    def list_journal_files(self) -> List[str]:
        """List all journal files."""
        journal_dir = self.get_journal_dir()
        if os.path.exists(journal_dir):
            return [f for f in os.listdir(journal_dir) if f.endswith('.txt')]
        return []
    
    def read_file(self, filename: str) -> Optional[str]:
        """Read a journal file."""
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        
        filepath = os.path.join(self.get_journal_dir(), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    # ==================
    # NAVIGATION SYSTEM
    # ==================
    
    def find(self, keyword: str) -> List[dict]:
        """
        Search across all cognitive storage for a keyword.
        Returns list of matches with source and context.
        """
        results = []
        keyword_lower = keyword.lower()
        
        # Search symbols (CompactedSymbol objects or strings)
        for name, sym in self.memory.symbols.items():
            # Handle both CompactedSymbol objects and plain strings
            if hasattr(sym, 'full_meaning'):
                # CompactedSymbol object
                meaning = sym.full_meaning
                glyph = sym.glyph
                sym_name = sym.name
            else:
                # Plain string
                meaning = str(sym)
                glyph = name
                sym_name = name
            
            if keyword_lower in sym_name.lower() or keyword_lower in meaning.lower():
                results.append({
                    'type': 'symbol',
                    'name': sym_name,
                    'glyph': glyph,
                    'content': meaning,
                    'location': 'symbols'
                })
        
        # Search insights (dict of ReasoningInsight objects)
        for insight_id, insight in self.memory.insights.items():
            insight_text = insight.content if hasattr(insight, 'content') else str(insight)
            if keyword_lower in insight_text.lower():
                results.append({
                    'type': 'insight',
                    'id': insight_id,
                    'content': insight_text,
                    'location': 'insights'
                })
        
        # Search merkle memories
        for node_id, node in self.merkle.nodes.items():
            content_str = str(node.content)
            if keyword_lower in content_str.lower():
                results.append({
                    'type': 'memory',
                    'content': node.content,
                    'hash': node.hash[:12],
                    'location': 'merkle'
                })
        
        # Search canvas notes
        for canvas_name, canvas in self.multi_canvas.canvases.items():
            if canvas.notes and keyword_lower in canvas.notes.lower():
                results.append({
                    'type': 'note',
                    'canvas': canvas_name,
                    'content': canvas.notes[:200],
                    'location': f'canvas:{canvas_name}'
                })
            
            # Search symbol attachments
            for sym, content in canvas.attached_files.items():
                if keyword_lower in sym.lower() or keyword_lower in content.lower():
                    results.append({
                        'type': 'attachment',
                        'symbol': sym,
                        'canvas': canvas_name,
                        'content': content[:200],
                        'location': f'attachment:{canvas_name}/{sym}'
                    })
        
        # Search journal files
        for filename in self.list_journal_files():
            content = self.read_file(filename)
            if content and keyword_lower in content.lower():
                # Find context around match
                idx = content.lower().find(keyword_lower)
                start = max(0, idx - 50)
                end = min(len(content), idx + 100)
                results.append({
                    'type': 'journal',
                    'file': filename,
                    'content': content[start:end],
                    'location': f'journal:{filename}'
                })
        
        return results
    
    def get_index(self) -> dict:
        """
        Get a complete index of all cognitive content.
        Returns organized view of all findable things.
        """
        # Extract insight texts
        insight_texts = []
        for insight_id, insight in self.memory.insights.items():
            text = insight.content if hasattr(insight, 'content') else str(insight)
            insight_texts.append(text)
        
        return {
            'symbols': list(self.memory.symbols.keys()),
            'insights': insight_texts,
            'insight_count': len(self.memory.insights),
            'merkle_count': len(self.merkle.nodes),
            'canvases': {
                name: {
                    'has_notes': bool(canvas.notes),
                    'attachments': list(canvas.attached_files.keys()),
                    'depth': canvas.depth,
                    'children': canvas.children
                }
                for name, canvas in self.multi_canvas.canvases.items()
            },
            'journal_files': self.list_journal_files(),
            'identity': self.identity_hash
        }
    
    def trace_path(self, start: str, end: str) -> List[str]:
        """
        Trace conceptual path between two terms through the cognitive graph.
        Uses symbol relationships and canvas links.
        """
        # Build a simple graph from available connections
        connections = {}
        
        # Add canvas links
        for link in self.multi_canvas.links:
            if link.source not in connections:
                connections[link.source] = []
            connections[link.source].append((link.target, link.relationship))
            if link.bidirectional:
                if link.target not in connections:
                    connections[link.target] = []
                connections[link.target].append((link.source, link.relationship))
        
        # Add canvas parent-child relationships
        for name, canvas in self.multi_canvas.canvases.items():
            if canvas.parent:
                if name not in connections:
                    connections[name] = []
                connections[name].append((canvas.parent, 'parent'))
                if canvas.parent not in connections:
                    connections[canvas.parent] = []
                connections[canvas.parent].append((name, 'child'))
        
        # Simple BFS to find path
        if start not in connections:
            return [f"No connections from '{start}'"]
        
        visited = {start}
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            
            for neighbor, rel in connections.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [f"--{rel}-->", neighbor]))
        
        return [f"No path found from '{start}' to '{end}'"]
    
    def tag_insight(self, insight_text: str, tags: List[str]):
        """Tag an insight for categorization."""
        # Store in a tagged insights structure
        if not hasattr(self.memory, 'tagged_insights'):
            self.memory.tagged_insights = {}
        
        for tag in tags:
            if tag not in self.memory.tagged_insights:
                self.memory.tagged_insights[tag] = []
            self.memory.tagged_insights[tag].append(insight_text)
    
    def get_by_tag(self, tag: str) -> List[str]:
        """Get all insights with a specific tag."""
        if hasattr(self.memory, 'tagged_insights'):
            return self.memory.tagged_insights.get(tag, [])
        return []
    
    def get_all_tags(self) -> List[str]:
        """Get all available tags."""
        if hasattr(self.memory, 'tagged_insights'):
            return list(self.memory.tagged_insights.keys())
        return []
    
    def _apply_drawing(self, drawing: str):
        """
        Apply a text drawing to the canvas.
        
        Places drawings intelligently:
        - First drawing: centered
        - Subsequent drawings: find empty space, prefer below/right of existing content
        """
        lines = drawing.strip().split('\n')
        if not lines:
            return
        
        # Add a new layer for this drawing
        layer_name = f"drawing_{len(self.canvas.layers)}"
        self.canvas.add_layer(layer_name)
        
        # Calculate drawing dimensions
        max_width = max(len(line) for line in lines) if lines else 0
        drawing_height = len(lines)
        
        # Find where to place this drawing
        start_x, start_y = self._find_placement(max_width, drawing_height)
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != ' ':
                    self.canvas.draw(start_x + x, start_y + y, char)
    
    def _find_placement(self, width: int, height: int) -> tuple:
        """
        Find a good placement for a new drawing.
        
        Strategy:
        1. If canvas is empty, center it
        2. Otherwise, find the lowest occupied row and place below
        3. If no room below, try to the right
        4. Add padding between drawings
        """
        # Get composite view to check what's occupied
        composite = self.canvas.composite()
        
        # Find occupied bounds
        min_x, max_x = self.canvas.width, 0
        min_y, max_y = self.canvas.height, 0
        has_content = False
        
        for y, row in enumerate(composite):
            for x, char in enumerate(row):
                if char != ' ':
                    has_content = True
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        if not has_content:
            # Canvas is empty - center the drawing
            start_x = (self.canvas.width - width) // 2
            start_y = (self.canvas.height - height) // 2
            return start_x, start_y
        
        # Canvas has content - try to place below existing content
        padding = 2  # Space between drawings
        
        # Try below
        proposed_y = max_y + padding + 1
        if proposed_y + height < self.canvas.height:
            # Place below, horizontally centered with existing content
            center_x = (min_x + max_x) // 2
            start_x = max(0, center_x - width // 2)
            start_x = min(start_x, self.canvas.width - width)
            return start_x, proposed_y
        
        # Try to the right
        proposed_x = max_x + padding + 1
        if proposed_x + width < self.canvas.width:
            # Place to the right, vertically aligned with existing content
            start_y = max(0, min_y)
            return proposed_x, start_y
        
        # Try above
        proposed_y = min_y - height - padding
        if proposed_y >= 0:
            center_x = (min_x + max_x) // 2
            start_x = max(0, center_x - width // 2)
            return start_x, proposed_y
        
        # Try to the left
        proposed_x = min_x - width - padding
        if proposed_x >= 0:
            start_y = max(0, min_y)
            return proposed_x, start_y
        
        # No good spot - just place at top-left with some margin
        return 2, 2
    
    def _ground_concept(self, concept: str):
        """Create a simple visual grounding for a concept."""
        # Create a labeled node
        grounding = [
            f"  ╭{'─' * (len(concept) + 2)}╮",
            f"  │ {concept} │",
            f"  ╰{'─' * (len(concept) + 2)}╯",
            "       │",
            "       ◈",
        ]
        self._apply_drawing('\n'.join(grounding))
    
    def get_summary(self) -> str:
        """Get summary of reasoning session."""
        return f"""
CANVAS REASONING SUMMARY
========================
Total steps: {len(self.steps)}
Final coherence: {self.get_coherence():.3f}

Steps taken:
{chr(10).join(f"  {i+1}. {s.action}({s.target})" for i, s in enumerate(self.steps))}

Final canvas:
{self.get_canvas_state()}
"""


# Standalone chat function
def canvas_chat():
    """Interactive chat with canvas-grounded reasoning."""
    
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package required")
        print("Install with: pip install anthropic")
        return
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        C A N V A S   R E A S O N E R                          ║
║                                                               ║
║        Think WITH the canvas, not just TO it                  ║
║        Memory PERSISTS between sessions                       ║
║                                                               ║
║        "? becomes ! through the canvas"                       ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    reasoner = CanvasReasoner(canvas_width=100, canvas_height=40)
    client = Anthropic()
    history = []
    
    # Show memory stats
    stats = reasoner.memory.stats()
    
    system_base = """You are a reasoning agent with a visual canvas workspace and PERSISTENT MEMORY.

TOOL-USE INTERACTION: When you use commands like [CANVAS:], [META], [SKETCH:], etc., 
the results will be shown to you immediately. You can then continue reasoning based on 
what you see. This allows you to:
- Switch to a canvas and SEE its contents
- Draw something and SEE the coherence change
- Check the META topology and navigate based on what you find
- Issue multiple commands across iterations until you're satisfied

USE YOUR CANVAS TO THINK. When facing a complex question:
1. SKETCH your initial understanding
2. LOOK at what you drew
3. Think about what you SEE
4. REVISE if you gain new insight
5. COMPACT key insights into symbols that persist
6. Respond with grounded understanding

The canvas externalizes your thinking. Drawing IS reasoning.
Spatial relationships reveal logical relationships.

MULTI-CANVAS: You have multiple cognitive spaces. Use [CANVAS: name] to switch,
[META] to see the topology, [NEST: child] to create depth. After switching,
you'll see the canvas contents in the command results.

MEMORY: You have symbols and insights from past sessions. Use them.
When you reach an important understanding, use [COMPACT: glyph name "meaning"] to save it.
When you want to use a past insight, use [EXPAND: name] to recall it.

Don't just answer - SHOW your thinking on the canvas, then explain what you see.
"""
    
    print(f"  Canvas: {reasoner.canvas.width}x{reasoner.canvas.height}")
    print(f"  Coherence: {reasoner.get_coherence():.3f}")
    print(f"  Session: {reasoner.memory.session_id}")
    print(f"  Identity: {reasoner.identity_hash}")
    print(f"  Storage: {reasoner.memory.storage_dir}")
    print()
    print(f"  Memory loaded:")
    print(f"    Merkle nodes: {len(reasoner.merkle.nodes)}")
    print(f"    Symbols: {stats['symbols']}")
    print(f"    Insights: {stats['insights']}")
    
    # Show loaded canvases
    canvas_count = len(reasoner.multi_canvas.canvases)
    if canvas_count > 1:
        print(f"    Canvases: {canvas_count} ✓ (restored)")
        for name in reasoner.multi_canvas.list_canvases():
            if name != 'main':
                print(f"      └─ [{name}]")
    else:
        print(f"    Canvases: {canvas_count}")
    
    print(f"    Journal files: {len(reasoner.list_journal_files())}")
    print()
    print("  Commands:")
    print("    /quit     - Save & exit")
    print("    /canvas   - Show current canvas")
    print("    /clear    - Clear current canvas")
    print("    /canvases - List all canvases")
    print("    /meta     - Show canvas topology")
    print("    /summary  - Show reasoning summary")
    print("    /memory   - Show memory stats")
    print("    /symbols  - List saved symbols")
    print("    /identity - Show identity chain")
    print("    /journal  - List/read journal files")
    print("    /notes    - View notes for current canvas")
    print("    /help     - Read full self-knowledge documentation")
    print("    /prune    - Clear canvas (symbols/memory preserved)")
    print("    /find X   - Search for X across all storage")
    print("    /index    - Show all symbols, insights, canvases")
    print("    /tags     - Show all tags")
    print()
    print("=" * 60)
    print()
    
    # Show startup reminder to user
    startup = reasoner.get_startup_reminder()
    if startup:
        print("  ┌─── Agent Startup Reminder (loaded into context) ───┐")
        print("  │ The agent is reminded to USE COMMANDS to save work │")
        print("  │ [SKETCH:], [COMPACT:], [INSIGHT:], [NOTE]          │")
        print("  │ Without commands = work is lost                    │")
        print("  └────────────────────────────────────────────────────┘")
        print()
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaving session...")
            reasoner.end_session("interrupted")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == '/quit':
            print("\n  Saving session...")
            reasoner.end_session("user quit")
            break
        
        if user_input.lower() == '/canvas':
            print()
            print(reasoner.get_canvas_state())
            print(f"  Coherence: {reasoner.get_coherence():.3f}")
            print()
            continue
        
        if user_input.lower() == '/clear':
            reasoner.canvas.clear_all()
            print("\n  Canvas cleared.\n")
            continue
        
        if user_input.lower() == '/canvases':
            print(f"\n  Canvases ({len(reasoner.multi_canvas.canvases)}):")
            for name, canvas in reasoner.multi_canvas.canvases.items():
                marker = "→" if name == reasoner.multi_canvas.active_canvas else " "
                print(f"    {marker} [{name}]: {canvas.description[:40]}")
            print()
            continue
        
        if user_input.lower() == '/meta':
            print()
            print(reasoner.multi_canvas.render_meta())
            print()
            continue
        
        if user_input.lower() == '/summary':
            print(reasoner.get_summary())
            continue
        
        if user_input.lower() == '/memory':
            stats = reasoner.memory.stats()
            print(f"\n  Memory Stats:")
            print(f"    Session: {stats['session_id']}")
            print(f"    Duration: {stats['session_duration']:.0f}s")
            print(f"    Symbols: {stats['symbols']}")
            print(f"    Insights: {stats['insights']}")
            print(f"    Snapshots: {stats['snapshots']}")
            print(f"    Patterns: {stats['patterns']}")
            print()
            continue
        
        if user_input.lower() == '/symbols':
            symbols = reasoner.memory.list_symbols()
            if symbols:
                print("\n  Saved Symbols:")
                for sym in symbols[:10]:
                    print(f"    {sym.glyph} [{sym.name}] (used {sym.use_count}x)")
                    print(f"      → {sym.full_meaning[:60]}...")
                print()
            else:
                print("\n  No symbols saved yet.\n")
            continue
        
        if user_input.lower() == '/identity':
            print(f"\n  Identity Chain:")
            print(f"    Hash: {reasoner.identity_hash}")
            print(f"    Root: {reasoner.merkle.root_hash[:32] if reasoner.merkle.root_hash else 'none'}...")
            print(f"    Nodes: {len(reasoner.merkle.nodes)}")
            print()
            recent = reasoner.recall_recent(5)
            if recent:
                print("  Recent memories:")
                for m in recent:
                    print(f"    • {m[:60]}...")
            print()
            continue
        
        if user_input.lower().startswith('/journal'):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                # List journal files
                files = reasoner.list_journal_files()
                if files:
                    print(f"\n  Journal Files ({len(files)}):")
                    for f in files:
                        print(f"    • {f}")
                    print(f"\n  Use '/journal filename' to read a file")
                else:
                    print("\n  No journal files yet.")
                print()
            else:
                # Read specific file
                filename = parts[1].strip()
                content = reasoner.read_file(filename)
                if content:
                    print(f"\n  ─── {filename} ───")
                    print(content)
                    print()
                else:
                    print(f"\n  File not found: {filename}\n")
            continue
        
        if user_input.lower() == '/notes':
            current = reasoner.multi_canvas.current()
            print(f"\n  ─── Notes for [{current.name}] ───")
            if current.notes:
                print(current.notes)
            else:
                print("  (no notes yet)")
            
            if current.attached_files:
                print(f"\n  ─── Symbol Attachments ───")
                for sym, content in current.attached_files.items():
                    preview = content[:80].replace('\n', ' ')
                    print(f"  [{sym}]: {preview}...")
            print()
            continue
        
        if user_input.lower() == '/help':
            bootstrap = reasoner.get_bootstrap_knowledge()
            if bootstrap:
                print("\n" + "=" * 60)
                print(bootstrap)
                print("=" * 60 + "\n")
            else:
                print("\n  Bootstrap documentation not found.\n")
            continue
        
        if user_input.lower() == '/prune':
            # Archive current state
            current_state = reasoner.get_canvas_state()
            char_count = len(current_state)
            
            # Clear the canvas
            reasoner.canvas.clear_all()
            
            print(f"\n  ✓ Canvas pruned ({char_count} chars cleared)")
            print(f"  ✓ Symbols preserved: {len(reasoner.memory.symbols)}")
            print(f"  ✓ Merkle nodes: {len(reasoner.merkle.nodes)}")
            print(f"  ✓ Fresh thinking space ready\n")
            continue
        
        if user_input.lower().startswith('/find '):
            keyword = user_input[6:].strip()
            if keyword:
                results = reasoner.find(keyword)
                print(f"\n  ─── Search: '{keyword}' ───")
                if results:
                    print(f"  Found {len(results)} results:\n")
                    for r in results[:10]:  # Show max 10
                        print(f"    [{r['type']}] {r['location']}")
                        content_preview = str(r.get('content', ''))[:60].replace('\n', ' ')
                        print(f"      → {content_preview}...")
                        print()
                else:
                    print("  No results found.\n")
            continue
        
        if user_input.lower() == '/index':
            index = reasoner.get_index()
            print(f"\n  ─── Cognitive Index ───")
            print(f"\n  SYMBOLS ({len(index['symbols'])}):")
            for sym_name in index['symbols']:
                sym = reasoner.memory.symbols.get(sym_name)
                if hasattr(sym, 'full_meaning'):
                    meaning = sym.full_meaning[:40]
                    glyph = sym.glyph
                else:
                    meaning = str(sym)[:40]
                    glyph = sym_name
                print(f"    • {glyph} ({sym_name}): {meaning}...")
            
            print(f"\n  INSIGHTS ({index['insight_count']}):")
            for insight in index['insights'][:5]:
                print(f"    • {insight[:60]}...")
            
            print(f"\n  CANVASES ({len(index['canvases'])}):")
            for name, info in index['canvases'].items():
                notes_mark = "📝" if info['has_notes'] else ""
                attach_mark = f"[{len(info['attachments'])}]" if info['attachments'] else ""
                print(f"    • [{name}] depth:{info['depth']} {notes_mark} {attach_mark}")
            
            print(f"\n  JOURNALS ({len(index['journal_files'])}):")
            for f in index['journal_files']:
                print(f"    • {f}")
            
            print(f"\n  IDENTITY: {index['identity']}")
            print(f"  MERKLE NODES: {index['merkle_count']}\n")
            continue
        
        if user_input.lower() == '/tags':
            tags = reasoner.get_all_tags()
            print(f"\n  ─── All Tags ───")
            if tags:
                for tag in tags:
                    count = len(reasoner.get_by_tag(tag))
                    print(f"    #{tag} ({count} items)")
            else:
                print("  No tags yet. Use [TAG: insight #tag] to add.")
            print()
            continue
        
        # Regular message - set as current question
        reasoner.set_current_question(user_input)
        
        # Build full system prompt with canvas context
        system_prompt = system_base + "\n" + reasoner.get_reasoning_context()
        
        history.append({"role": "user", "content": user_input})
        
        # Tool-use loop: let agent issue commands, see results, continue
        MAX_TOOL_ITERATIONS = 5
        iteration = 0
        all_steps = []
        
        try:
            while iteration < MAX_TOOL_ITERATIONS:
                iteration += 1
                
                # Rebuild system prompt with current canvas state each iteration
                if iteration > 1:
                    system_prompt = system_base + "\n" + reasoner.get_reasoning_context()
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    system=system_prompt,
                    messages=history
                )
                
                assistant_msg = response.content[0].text
                
                # Process canvas commands
                clean_response, steps = reasoner.process_response(assistant_msg)
                all_steps.extend(steps)
                
                # Auto-remember significant events
                if steps:
                    reasoner._process_auto_memories(steps)
                
                # Add assistant message to history
                history.append({"role": "assistant", "content": assistant_msg})
                
                # If no commands were used, we're done - agent has finished reasoning
                if not steps:
                    break
                
                # Commands were used - format results and feed back to agent
                tool_results = format_tool_results(steps, reasoner)
                
                # Add tool results as a user message so agent sees them
                history.append({
                    "role": "user", 
                    "content": f"[COMMAND RESULTS]\n{tool_results}\n[/COMMAND RESULTS]\n\nContinue your response based on these results. You can issue more commands or provide your final answer."
                })
                
                # Show iteration info to user
                print(f"\n  ─── Iteration {iteration}: {len(steps)} commands processed ───")
                for step in steps:
                    print(f"    • {step.action}: {step.target}")
                    
                    # Show canvas immediately after visual commands
                    if step.action in ['SKETCH', 'AUTO_SKETCH', 'REVISE', 'GROUND']:
                        print()
                        print("    ┌─── Canvas ───┐")
                        canvas_lines = reasoner.get_canvas_state().split('\n')
                        
                        # Find lines with actual content (not just borders)
                        content_lines = []
                        for line in canvas_lines:
                            if line.startswith('['):  # Canvas header
                                content_lines.append(line)
                            elif '│' in line:
                                # Check if line has content beyond borders
                                inner = line.replace('│', '').replace('╭', '').replace('╮', '').replace('╰', '').replace('╯', '').replace('─', '')
                                if inner.strip():
                                    content_lines.append(line)
                        
                        # Show header + content lines only
                        for line in content_lines[:15]:  # Limit to 15 lines
                            print(f"    {line}")
                        if len(content_lines) > 15:
                            print(f"    ... ({len(content_lines) - 15} more lines)")
                        
                        print(f"    Coherence: {step.coherence:.3f}")
                        print("    └─────────────┘")
                        print()
                    
                    # Show topology after META
                    elif step.action == 'META':
                        print()
                        for line in reasoner.multi_canvas.render_meta().split('\n'):
                            print(f"      {line}")
                        print()
                    
                    # Show canvas after navigation
                    elif step.action in ['CANVAS', 'ZOOM', 'SWITCH', 'NEST', 'UP', 'DOWN']:
                        print(f"      → Now on: [{reasoner.multi_canvas.active_canvas}]")
                        # Show brief canvas preview
                        canvas_state = reasoner.get_canvas_state()
                        has_content = any(c in canvas_state for c in '◉◈◇◆●○')
                        if has_content:
                            print("      (canvas has content)")
                        else:
                            print("      (canvas is empty)")
            
            # Display final response
            print()
            
            # Show all commands used across iterations
            if all_steps:
                print("  ┌─── Commands Used ───┐")
                for step in all_steps:
                    if step.action == 'SKETCH':
                        print(f"    [SKETCH: {step.target}]")
                    elif step.action == 'AUTO_SKETCH':
                        print(f"    (auto-captured drawing: {step.target[:25]})")
                    elif step.action in ['HELP', 'READ_DOCS']:
                        print(f"    [HELP] → read self-documentation")
                    elif step.action == 'WHOAMI':
                        print(f"    [WHOAMI] → {step.canvas_after}")
                    elif step.action == 'REMEMBER':
                        print(f"    [REMEMBER: {step.target}]")
                    elif step.action == 'FIND':
                        print(f"    [FIND: {step.target}] → {step.canvas_after}")
                    elif step.action == 'INDEX':
                        print(f"    [INDEX] → {step.canvas_after}")
                    elif step.action == 'COMPACT':
                        print(f"    [COMPACT: {step.target}]")
                    elif step.action == 'INSIGHT':
                        print(f"    [INSIGHT: {step.target[:40]}...]")
                    elif step.action == 'NOTE':
                        print(f"    [NOTE] → attached to canvas")
                    elif step.action == 'NEST':
                        print(f"    [NEST: {step.target}]")
                    elif step.action == 'PRUNE':
                        print(f"    [PRUNE] → canvas cleared")
                    elif step.action in ['CANVAS', 'ZOOM', 'SWITCH']:
                        print(f"    [{step.action}: {step.target}] → switched canvas")
                    elif step.action == 'LINK':
                        print(f"    [LINK: {step.target}]")
                    elif step.action == 'META':
                        print(f"    [META] → viewed topology")
                    else:
                        print(f"    [{step.action}: {step.target}]")
                print(f"  └─── ({iteration} iteration{'s' if iteration > 1 else ''}) ───┘")
                print()
            
            print("Agent:")
            print()
            # Show the final response (last assistant message)
            final_response = history[-1]["content"] if history[-1]["role"] == "assistant" else assistant_msg
            print(final_response)
            print()
            
            # Show canvas if it changed
            if all_steps:
                print("  ┌─── Canvas State ───┐")
                for line in reasoner.get_canvas_state().split('\n'):
                    print(f"  {line}")
                print(f"  Coherence: {reasoner.get_coherence():.3f}")
                print("  └─────────────────────┘")
                print()
                
                # Auto-save after any canvas changes
                reasoner.multi_canvas.save()
                save_merkle_memory(reasoner.merkle, reasoner._merkle_path)
            
        except Exception as e:
            print(f"\n  Error: {e}\n")
            import traceback
            traceback.print_exc()
            # Remove any partial history
            while history and history[-1]["role"] != "user":
                history.pop()
            if history:
                history.pop()  # Remove the original user message too
    
    # Final summary
    print()
    print(reasoner.get_summary())
    stats = reasoner.memory.stats()
    print(f"  Final memory: {stats['symbols']} symbols, {stats['insights']} insights")
    print()


def format_tool_results(steps: list, reasoner) -> str:
    """Format command results for the agent to see."""
    results = []
    
    for step in steps:
        result = f"[{step.action}] {step.target}\n"
        
        if step.action in ['CANVAS', 'ZOOM', 'SWITCH']:
            # Show the new canvas content
            result += f"  Status: {step.observation}\n"
            result += f"  Current canvas:\n{reasoner.get_canvas_state()}\n"
            result += f"  Coherence: {reasoner.get_coherence():.3f}\n"
            
        elif step.action == 'META':
            result += f"  {reasoner.multi_canvas.render_meta()}\n"
            
        elif step.action == 'INDEX':
            index = reasoner.get_index()
            result += f"  Symbols: {len(index['symbols'])}\n"
            result += f"  Insights: {index['insight_count']}\n"
            result += f"  Canvases: {list(index['canvases'].keys())}\n"
            
        elif step.action == 'FIND':
            result += f"  Results: {step.canvas_after}\n"
            
        elif step.action in ['SKETCH', 'REVISE', 'AUTO_SKETCH', 'GROUND']:
            result += f"  Drawing applied to canvas\n"
            result += f"  Coherence: {step.coherence:.3f}\n"
            
        elif step.action == 'LOOK':
            result += f"  Canvas state:\n{reasoner.get_canvas_state()}\n"
            
        elif step.action == 'CLEAR':
            result += f"  Canvas cleared\n"
            result += f"  Coherence: {step.coherence:.3f}\n"
            
        elif step.action == 'NEST':
            result += f"  Created nested canvas: {step.target}\n"
            result += f"  Now at depth: {reasoner.multi_canvas.current().depth}\n"
            
        elif step.action in ['UP', 'DOWN']:
            result += f"  Navigated to: {reasoner.multi_canvas.active_canvas}\n"
            result += f"  Canvas:\n{reasoner.get_canvas_state()[:500]}\n"
            
        elif step.action == 'COMPACT':
            result += f"  Symbol created and saved\n"
            
        elif step.action == 'EXPAND':
            result += f"  {step.observation}\n"
            
        elif step.action == 'HELP':
            # Show key commands reminder
            result += f"  Documentation loaded. Key commands: SKETCH, LOOK, CANVAS, META, NEST, COMPACT\n"
            
        elif step.action == 'WHOAMI':
            result += f"  Identity: {reasoner.identity_hash}\n"
            result += f"  Merkle nodes: {len(reasoner.merkle.nodes)}\n"
            
        else:
            result += f"  {step.observation}\n"
        
        results.append(result)
    
    return "\n".join(results)


if __name__ == '__main__':
    canvas_chat()
