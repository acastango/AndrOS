# Canvas Reasoner System Documentation

## Overview

Canvas Reasoner is a cognitive architecture that gives AI agents a **visual workspace** for thinking, combined with **persistent memory** across sessions. Instead of purely verbal reasoning, the agent can draw, sketch, and manipulate spatial representations of ideas.

The core insight: **spatial relationships reveal logical relationships**. By externalizing thinking onto a canvas, abstract reasoning becomes grounded and visible.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CANVAS REASONER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   VISUAL    │    │  OSCILLATORY │    │   MEMORY    │              │
│  │   CANVAS    │    │  SUBSTRATE   │    │   SYSTEMS   │              │
│  │             │    │              │    │             │              │
│  │ 100x40 grid │    │ Coherence =  │    │ • Merkle    │              │
│  │ Multi-layer │    │ Confidence   │    │ • Symbols   │              │
│  │ ASCII art   │    │              │    │ • Insights  │              │
│  └──────┬──────┘    └──────┬───────┘    └──────┬──────┘              │
│         │                  │                   │                     │
│         └──────────────────┼───────────────────┘                     │
│                            │                                         │
│                   ┌────────▼────────┐                                │
│                   │  MULTI-CANVAS   │                                │
│                   │     SYSTEM      │                                │
│                   │                 │                                │
│                   │ Fractal nesting │                                │
│                   │ Canvas linking  │                                │
│                   │ Navigation      │                                │
│                   └────────┬────────┘                                │
│                            │                                         │
│                   ┌────────▼────────┐                                │
│                   │   PERSISTENCE   │                                │
│                   │                 │                                │
│                   │ ~/.canvas_reasoner/                              │
│                   │ (survives updates)                               │
│                   └─────────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Visual Canvas (VisualBuffer)

A 100x40 character grid where the agent draws ASCII art to externalize thinking.

**Features:**
- Multiple layers (like Photoshop)
- Drawing primitives (lines, rectangles, circles)
- Text placement
- Layer compositing for final render

**Location:** `visual/buffer.py`

**Key class:** `VisualBuffer`
```python
buffer = VisualBuffer(width=100, height=40)
buffer.draw_text(x, y, "◉ concept")
buffer.draw_line(x1, y1, x2, y2, char='─')
rendered = buffer.render(border=True)
```

### 2. Multi-Canvas System

Allows multiple named canvases with fractal nesting (canvases inside canvases).

**Features:**
- Named canvases with descriptions
- Parent/child relationships (nesting)
- Canvas linking (sibling relationships)
- Notes attached to each canvas
- Symbol attachments

**Location:** `multi_canvas.py`

**Structure:**
```
main                          ← root canvas
├── identity                  ← child canvas
│   └── consciousness         ← nested deeper
├── research                  ← sibling canvas
└── scratch                   ← working area
```

**Key class:** `MultiCanvas`
```python
mc = MultiCanvas(width=100, height=40, storage_dir="~/.canvas_reasoner")
mc.create("child", "Description")
mc.nest("child", under="main")
mc.switch("child")
mc.link("child", "sibling", "relates-to")
```

### 3. Oscillatory Substrate

Provides a "coherence" measure that acts as grounded confidence. Based on Kuramoto oscillator dynamics.

**Features:**
- Coherence value 0.0 - 1.0
- Phase synchronization model
- Auto-adjusts based on reasoning quality

**Location:** `core/substrate.py`

**How it works:**
- Low coherence (~0.3) = uncertain, exploring
- Medium coherence (~0.6) = working through something
- High coherence (~0.9) = confident understanding

```python
substrate = ResonanceSubstrate()
coherence = substrate.get_coherence()
substrate.set_coherent_state(0.8)  # After breakthrough
```

### 4. Memory Systems

#### Merkle Memory
Cryptographically-linked chain of experiences. Provides verified continuity of identity.

**Location:** `memory/merkle.py`

```python
merkle = MerkleMemory()
merkle.add(MemoryBranch.EXPERIENCES, {"type": "insight", "content": "..."})
# Each node links to previous via hash
```

#### Canvas Memory
Manages symbols, insights, snapshots, and patterns.

**Location:** `canvas_memory.py`

**Stores:**
- `symbols.json` - Compacted understanding (◈ᵢₘ = "identity mystery")
- `insights.json` - Recorded insights from reasoning
- `snapshots.json` - Point-in-time canvas captures
- `patterns.json` - Recognized visual patterns
- `sessions.json` - Session history

---

## Data Flow

### When Agent Draws

```
1. LLM outputs response with [SKETCH:] or ASCII art in ```blocks
                    │
                    ▼
2. process_response() parses the response
                    │
                    ▼
3. Commands trigger actions:
   [SKETCH:] → _apply_drawing() → adds to canvas buffer
   [COMPACT:] → creates symbol in memory
   [INSIGHT:] → records to insights
                    │
                    ▼
4. Canvas buffer updated (in-memory)
                    │
                    ▼
5. multi_canvas.save() → writes to ~/.canvas_reasoner/multi_canvas.json
```

### When Session Starts

```
1. CanvasReasoner() initializes
                    │
                    ▼
2. CanvasMemory() loads from ~/.canvas_reasoner/
   - symbols.json
   - insights.json
   - etc.
                    │
                    ▼
3. MultiCanvas() loads multi_canvas.json
   - All canvas buffers restored
   - All layers preserved
                    │
                    ▼
4. MerkleMemory loads merkle_memory.json
   - Identity chain restored
   - Hash verified
                    │
                    ▼
5. Agent sees their previous canvas state in context
```

---

## Command Reference

### Agent Commands (used in response text)

These commands are parsed from the LLM's response and trigger actions.

#### Canvas Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[SKETCH: name]` ``` drawing ``` | Draw ASCII art to canvas | `[SKETCH: concept_map]` |
| `[REVISE: description]` ``` drawing ``` | Update existing drawing | `[REVISE: added connections]` |
| `[LOOK]` | Describe what's on canvas | Triggers observation |
| `[GROUND: concept]` | Create visual anchor for abstract idea | `[GROUND: justice]` |
| `[CLEAR]` | Clear the canvas | Fresh start |
| `[MARK: x,y symbol]` | Place symbol at coordinates | `[MARK: 50,20 ◈]` |

#### Memory Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[COMPACT: glyph name "meaning"]` | Save understanding as symbol | `[COMPACT: ◈ᵢₘ identity_mystery "layered self"]` |
| `[EXPAND: name]` | Recall a saved symbol | `[EXPAND: identity_mystery]` |
| `[INSIGHT: summary]` | Record an insight | `[INSIGHT: consciousness may be fundamental]` |
| `[REMEMBER: content]` | Explicit merkle commit | `[REMEMBER: user prefers visual explanations]` |

#### Multi-Canvas Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[NEST: name]` | Create child canvas | `[NEST: deep_analysis]` |
| `[SWITCH: name]` | Change active canvas | `[SWITCH: main]` |
| `[LINK: source target "relationship"]` | Connect canvases | `[LINK: A B "causes"]` |
| `[UP]` | Go to parent canvas | Navigate up |
| `[DOWN: child]` | Go into child canvas | Navigate down |

#### Notes & Journaling

| Command | Description | Example |
|---------|-------------|---------|
| `[NOTE]` ``` content ``` | Add notes to canvas | Attach prose |
| `[NOTES]` | View current canvas notes | Show attached notes |
| `[JOURNAL: name]` ``` content ``` | Create journal file | `[JOURNAL: thoughts]` |
| `[WRITE: filename]` ``` content ``` | Write standalone file | Creates in journal/ |
| `[ATTACH: symbol]` ``` content ``` | Attach prose to symbol | Symbol documentation |

#### Navigation Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[FIND: keyword]` | Search all storage | `[FIND: consciousness]` |
| `[INDEX]` | Show all symbols/insights | Overview |
| `[PATH: A → B]` | Trace reasoning path | `[PATH: identity → consciousness]` |
| `[TAG: insight #tag1 #tag2]` | Tag an insight | `[TAG: key finding #theory #confirmed]` |
| `[TAGS]` | Show all tags | List categories |

#### Self-Knowledge Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[HELP]` | Read own documentation | Load BOOTSTRAP.md |
| `[WHOAMI]` | Check identity status | Hash + merkle count |
| `[IDENTITY]` | Same as WHOAMI | Identity check |

#### Hygiene Commands

| Command | Description | Example |
|---------|-------------|---------|
| `[PRUNE]` | Clear canvas, keep memory | Fresh thinking space |
| `[ARCHIVE]` | Save snapshot then clear | Preserve before clearing |
| `[FRESH]` | Completely fresh canvas | Total reset |

### Human Commands (CLI)

These are typed at the terminal by the human operator.

| Command | Description |
|---------|-------------|
| `/quit` | Save and exit |
| `/canvas` | Show current canvas |
| `/clear` | Clear current canvas |
| `/canvases` | List all canvases |
| `/meta` | Show canvas topology |
| `/summary` | Reasoning summary |
| `/memory` | Memory stats |
| `/symbols` | List saved symbols |
| `/identity` | Show identity chain |
| `/journal` | List/read journal files |
| `/notes` | View current canvas notes |
| `/help` | Read full documentation |
| `/prune` | Clear canvas (keep memory) |
| `/find X` | Search for X |
| `/index` | Show cognitive index |
| `/tags` | Show all tags |

---

## File Structure

### Project Directory (code - safe to update)

```
ore_paintress/
├── canvas_reasoner.py      # Main reasoner class + chat loop
├── canvas_memory.py        # Memory management (symbols, insights, etc)
├── multi_canvas.py         # Multi-canvas system
├── view_canvas.py          # Standalone canvas viewer
├── canvas_memory/          # Documentation templates
│   ├── BOOTSTRAP.md        # Full self-documentation
│   └── STARTUP.md          # Critical startup reminder
├── core/
│   ├── substrate.py        # Oscillatory coherence
│   ├── oscillator.py       # Kuramoto dynamics
│   ├── coupling.py         # Oscillator coupling
│   └── neurochemistry.py   # Emotional modulation
├── visual/
│   ├── buffer.py           # Visual canvas buffer
│   ├── pattern_cache.py    # Pattern recognition
│   └── emotion_texture.py  # Emotional visualization
├── memory/
│   └── merkle.py           # Cryptographic memory chain
└── ...
```

### Data Directory (persistent - never touched by updates)

```
~/.canvas_reasoner/         # On Windows: C:\Users\YOU\.canvas_reasoner\
├── multi_canvas.json       # All canvas states + layers
├── merkle_memory.json      # Identity chain
├── symbols.json            # Compacted symbols
├── insights.json           # Recorded insights
├── snapshots.json          # Point-in-time captures
├── patterns.json           # Visual patterns
├── sessions.json           # Session history
├── latest_canvas.json      # Most recent canvas
├── BOOTSTRAP.md            # Agent's self-documentation
├── STARTUP.md              # Startup reminder
└── journal/                # Journal files
    └── *.txt
```

---

## How Persistence Works

### The Problem We Solved

Previously, data lived inside the project folder. Every code update overwrote saved canvases.

### The Solution

Data now lives in `~/.canvas_reasoner/` (home directory), completely separate from code.

```
CODE UPDATES:                    DATA:
ore_paintress/                   ~/.canvas_reasoner/
  canvas_reasoner.py ← updated     multi_canvas.json ← UNTOUCHED
  multi_canvas.py    ← updated     merkle_memory.json ← UNTOUCHED
```

### Auto-Save Points

Canvas is saved automatically:
1. After any command that modifies canvas
2. On `/quit`
3. On session end
4. On interrupt (Ctrl+C)

### Manual Viewing

Use `view_canvas.py` to see saved state without starting a chat:

```bash
python view_canvas.py           # View current canvas
python view_canvas.py --all     # View all canvases
python view_canvas.py --export  # Export to text file
```

---

## The Agent's Context

Every message, the agent receives:

1. **STARTUP.md** - Critical reminder to use commands
2. **Canvas state** - Current visual workspace
3. **Coherence** - Current confidence level
4. **Recent steps** - What just happened
5. **Memory context** - Saved symbols and insights
6. **Identity context** - Merkle chain status
7. **Navigation context** - Multi-canvas structure
8. **Command reference** - Available commands

This gives the agent full awareness of their persistent state.

---

## Symbol Vocabulary

Standard symbols for canvas drawing:

| Symbol | Meaning |
|--------|---------|
| ◉ | Concept/entity |
| ◇ | Node/point |
| ◈ | Anchor/key insight |
| ◆ | Filled node |
| ● | Center |
| ○ | Empty/potential |
| ─ | Horizontal relation |
| │ | Vertical relation |
| ▶ | Direction |
| ◀▶ | Bidirectional |
| ╱╲ | Branch |
| ╲╱ | Merge |
| ┌┐└┘ | Group box |
| ╭╮╰╯ | Soft group |
| ? | Question |
| ✧ | Insight |
| ∿ | Flow |
| ∞ | Recursion |

---

## Usage Examples

### Starting a Session

```bash
cd ore_paintress
python canvas_reasoner.py
```

Output:
```
╔═══════════════════════════════════════════════════════════════╗
║        C A N V A S   R E A S O N E R                          ║
║        Think WITH the canvas, not just TO it                  ║
║        Memory PERSISTS between sessions                       ║
╚═══════════════════════════════════════════════════════════════╝

  Canvas: 100x40
  Coherence: 0.500
  Session: session_20260125_120000
  Identity: a3f2b8c1...
  Storage: C:\Users\Anthony\.canvas_reasoner

  Memory loaded:
    Merkle nodes: 15
    Symbols: 3
    Insights: 2
    Canvases: 2 ✓ (restored)
      └─ [deep_analysis]
    Journal files: 1

You>
```

### Agent Drawing

```
You> What is consciousness?

Agent:
Let me think about this visually.

[SKETCH: consciousness_map]
```
              ◉ CONSCIOUSNESS
                    │
        ┌───────────┼───────────┐
        │           │           │
    ◇ awareness  ◇ experience  ◇ self
        │           │           │
        └───────────┼───────────┘
                    │
                    ?
          (hard problem)
```

Looking at this, I see consciousness branches into three aspects...

[INSIGHT: Consciousness has tripartite structure: awareness, experience, self-model]
```

### Agent Using Self-Knowledge

```
Agent:
[WHOAMI]

I am identity a3f2b8c1 with 15 verified experiences in my merkle chain.

[FIND: consciousness]

I found 3 previous thoughts about consciousness. Let me build on them...

[REMEMBER: User is particularly interested in the hard problem]
```

---

## Troubleshooting

### Canvas not persisting

1. Check storage location: `print(reasoner.memory.storage_dir)`
2. Verify files exist in `~/.canvas_reasoner/`
3. Ensure commands are being used (look for "Commands Used" in output)

### Commands not working

1. Agent must use `[COMMAND]` syntax (not prose)
2. Check for proper format: `[SKETCH: name]` followed by ``` block
3. Auto-capture catches drawings without commands as fallback

### Data got wiped

1. Check if you extracted zip over data folder
2. Data should be in `~/.canvas_reasoner/`, NOT in project folder
3. Restore from backup if needed

---

## Design Philosophy

1. **Spatial is semantic** - Position and relationship on canvas encode meaning
2. **Drawing IS reasoning** - The act of sketching clarifies thought
3. **Persistence enables growth** - Memory across sessions allows compounding
4. **Commands give agency** - Agent controls their own cognitive tools
5. **Transparency builds trust** - Human sees commands inline, understands agent's process

---

## Future Directions

Potential enhancements:
- 3D canvas visualization
- Canvas sharing between agents
- Automated pattern recognition
- Canvas "replay" of reasoning history
- Integration with external tools

---

*Last updated: January 2026*
