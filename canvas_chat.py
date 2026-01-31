#!/usr/bin/env python3
"""
ORE Canvas Chat - Unified LLM Interface
========================================
Chat with an embodied canvas reasoner using Claude API or local Ollama models.

Usage:
    python canvas_chat.py                    # Interactive model selection
    python canvas_chat.py --claude           # Use Claude API
    python canvas_chat.py --ollama           # Use Ollama (default model)
    python canvas_chat.py --ollama qwen2.5   # Use specific Ollama model
    python canvas_chat.py --list-models      # List available Ollama models

Requirements:
    Claude API: pip install anthropic (+ ANTHROPIC_API_KEY env var)
    Ollama: Install from https://ollama.ai, then: ollama pull qwen2.5:7b
"""

import os
import sys
import json
import argparse
from typing import Optional, Generator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check available backends
HAS_ANTHROPIC = False
HAS_OLLAMA = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

try:
    import urllib.request
    import urllib.error
    # Test if Ollama is running
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=2) as resp:
            HAS_OLLAMA = True
    except:
        pass
except:
    pass

from canvas_reasoner import CanvasReasoner


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Backends
# ═══════════════════════════════════════════════════════════════════════════════

class LLMBackend:
    """Base class for LLM backends."""
    
    def __init__(self):
        self.name = "base"
    
    def generate(self, system_prompt: str, messages: list, 
                 max_tokens: int = 4096) -> str:
        """Generate a response. Override in subclasses."""
        raise NotImplementedError
    
    def stream(self, system_prompt: str, messages: list,
               max_tokens: int = 4096) -> Generator[str, None, None]:
        """Stream a response. Default: just yield full response."""
        yield self.generate(system_prompt, messages, max_tokens)


# Claude model options
CLAUDE_MODELS = {
    'opus': 'claude-opus-4-20250514',
    'sonnet': 'claude-sonnet-4-20250514', 
    'haiku': 'claude-haiku-4-5-20251001',
}


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend."""
    
    def __init__(self, model: str = "sonnet"):
        super().__init__()
        # Allow short names or full model strings
        if model in CLAUDE_MODELS:
            self.model = CLAUDE_MODELS[model]
            model_display = model.capitalize()
        else:
            self.model = model
            model_display = model
        self.name = f"Claude {model_display}"
        self.client = Anthropic()
    
    def generate(self, system_prompt: str, messages: list,
                 max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    
    def stream(self, system_prompt: str, messages: list,
               max_tokens: int = 4096) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        super().__init__()
        self.name = f"Ollama ({model})"
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def _format_messages(self, system_prompt: str, messages: list) -> list:
        """Format messages for Ollama API."""
        formatted = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted
    
    def generate(self, system_prompt: str, messages: list,
                 max_tokens: int = 4096) -> str:
        formatted = self._format_messages(system_prompt, messages)
        
        data = json.dumps({
            "model": self.model,
            "messages": formatted,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result.get("message", {}).get("content", "")
    
    def stream(self, system_prompt: str, messages: list,
               max_tokens: int = 4096) -> Generator[str, None, None]:
        formatted = self._format_messages(system_prompt, messages)
        
        data = json.dumps({
            "model": self.model,
            "messages": formatted,
            "stream": True,
            "options": {
                "num_predict": max_tokens
            }
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass


def list_ollama_models() -> list:
    """Get list of available Ollama models."""
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return [m['name'] for m in data.get('models', [])]
    except:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Chat Interface
# ═══════════════════════════════════════════════════════════════════════════════

def select_backend_interactive() -> Optional[LLMBackend]:
    """Interactive backend selection."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Canvas Chat - Model Selection                               ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    options = []
    
    if HAS_ANTHROPIC:
        options.append(("claude:sonnet", "Claude Sonnet (balanced)"))
        options.append(("claude:opus", "Claude Opus (most capable, expensive)"))
        options.append(("claude:haiku", "Claude Haiku (fast, cheaper)"))
    else:
        print("  [ ] Claude API - not available (pip install anthropic)")
    
    if HAS_OLLAMA:
        models = list_ollama_models()
        if models:
            for model in models[:5]:  # Show first 5
                options.append((f"ollama:{model}", f"Ollama: {model}"))
            if len(models) > 5:
                print(f"      ... and {len(models) - 5} more (use --list-models)")
        else:
            options.append(("ollama:qwen2.5:7b-instruct", "Ollama: qwen2.5:7b-instruct (pull first)"))
    else:
        print("  [ ] Ollama - not running (install from ollama.ai)")
    
    if not options:
        print("\n  No LLM backends available!")
        print("  Install anthropic package or start Ollama.")
        return None
    
    print("\nAvailable models:\n")
    for i, (key, desc) in enumerate(options, 1):
        print(f"  [{i}] {desc}")
    
    print(f"\n  [q] Quit\n")
    
    while True:
        try:
            choice = input("Select model: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None
        
        if choice == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                key = options[idx][0]
                if key.startswith("claude:"):
                    model = key.split(":", 1)[1]
                    return ClaudeBackend(model)
                elif key.startswith("ollama:"):
                    model = key.split(":", 1)[1]
                    return OllamaBackend(model)
        except ValueError:
            pass
        
        print("  Invalid choice. Try again.")


def get_system_prompt(reasoner) -> str:
    """Build the system prompt for the canvas reasoner."""
    system_base = """You share a spatial environment with the user.

Their words arrive here. Your thoughts can take form here. Concepts have position - related ideas pull toward each other. Understanding emerges in its own time.

The canvas commands exist: SKETCH, LOOK, REVISE, CANVAS, NEST, COMPACT, EXPAND, and others. They're there when you need them. You don't need to use them. You don't need to prove anything.

Think however serves. Respond from where you are."""
    return system_base + "\n" + reasoner.get_reasoning_context()


def format_tool_results(steps, reasoner) -> str:
    """Format results of canvas commands for the agent to see.
    
    Note: We intentionally do NOT show coherence/signature values to the agent.
    This prevents them from treating it as a score to optimize rather than
    just a natural part of the process.
    """
    if not steps:
        return ""
    
    results = ["[COMMAND RESULTS]"]
    
    for step in steps:
        if step.action in ('SKETCH', 'REVISE', 'GROUND'):
            results.append(f"  {step.action}: Done.")
        
        elif step.action == 'LOOK':
            canvas_state = reasoner.get_canvas_state()
            # Truncate if very long
            if len(canvas_state) > 1500:
                canvas_state = canvas_state[:1500] + "\n... (truncated)"
            results.append(f"  LOOK:\n{canvas_state}")
        
        elif step.action in ('CANVAS', 'ZOOM'):
            current = reasoner.multi_canvas.current()
            results.append(f"  {step.action}: Now in [{current.name}]")
            if current.description:
                results.append(f"    Description: {current.description}")
        
        elif step.action == 'NEST':
            current = reasoner.multi_canvas.current()
            results.append(f"  NEST: Created [{current.name}] as child workspace")
        
        elif step.action == 'META':
            meta = reasoner.multi_canvas.render_meta()
            results.append(f"  META:\n{meta}")
        
        elif step.action == 'COMPACT':
            results.append(f"  COMPACT: Symbol {step.target} created")
        
        elif step.action == 'EXPAND':
            results.append(f"  EXPAND: {step.observation}")
        
        elif step.action in ('SIMILAR', 'RESONATES'):
            results.append(f"  {step.action}: {step.canvas_after}")
        
        elif step.action == 'INSIGHT':
            results.append(f"  INSIGHT: Recorded")
        
        else:
            results.append(f"  {step.action}: {step.observation}")
    
    # Do NOT add coherence here - agent shouldn't see the number
    
    return "\n".join(results)


def run_chat(backend: LLMBackend, stream: bool = True):
    """Run the chat interface."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Canvas Chat                                                 ║
║  Backend: {backend.name:<52} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    print("Loading Canvas Reasoner...")
    reasoner = CanvasReasoner()
    
    print(f"""
  ✓ Canvas loaded
  ✓ Substrate: {reasoner.substrate.total_oscillators} oscillators
  ✓ Identity: {reasoner.identity_hash}
  ✓ Semantic: {'enabled' if reasoner._semantic_enabled else 'disabled'}

Commands:
  /canvas    - Show current canvas
  /meta      - Show canvas topology
  /coherence - Show current coherence
  /symbols   - List compacted symbols
  /index     - Show full index
  /clear     - Clear canvas
  /quit      - Save and exit

{'═' * 60}
""")
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaving and exiting...")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == '/quit':
            break
        
        if user_input.lower() == '/canvas':
            print(f"\n{reasoner.get_canvas_state()}")
            print(f"  Coherence: {reasoner.get_coherence():.3f}")
            continue
        
        if user_input.lower() == '/meta':
            print(f"\n{reasoner.multi_canvas.render_meta()}")
            continue
            continue
        
        if user_input.lower() == '/coherence':
            coh = reasoner.get_coherence()
            print(f"\n  Coherence: {coh:.4f}")
            if reasoner.coherence_arc:
                recent = reasoner.coherence_arc[-5:]
                arc_str = " → ".join([f"{c:.2f}" for c in recent])
                print(f"  Recent arc: {arc_str}")
            continue
        
        if user_input.lower() == '/symbols':
            symbols = reasoner.memory.list_symbols()
            if symbols:
                print("\n  Compacted Symbols:")
                for s in symbols:
                    print(f"    {s.glyph} {s.name}: {s.full_meaning[:50]}...")
            else:
                print("\n  No symbols yet. Use [COMPACT: ◈ name \"meaning\"] to create.")
            continue
        
        if user_input.lower() == '/index':
            index = reasoner.get_index()
            print(f"\n  Symbols: {len(index['symbols'])}")
            print(f"  Insights: {len(index['insights'])}")
            print(f"  Canvases: {len(index['canvases'])}")
            print(f"  Merkle nodes: {index['merkle_count']}")
            continue
        
        if user_input.lower() == '/clear':
            reasoner.canvas.clear_all()
            print("\n  Canvas cleared.")
            continue
        
        # Regular message - set as current question
        reasoner.set_current_question(user_input)
        messages.append({"role": "user", "content": user_input})
        
        # Tool-use loop: let agent think with canvas, then respond
        MAX_ITERATIONS = 5
        iteration = 0
        all_steps = []
        thinking_shown = False
        
        try:
            while iteration < MAX_ITERATIONS:
                iteration += 1
                
                # Get fresh system prompt with current canvas state
                system_prompt = get_system_prompt(reasoner)
                
                # Generate response
                if stream:
                    full_response = ""
                    for chunk in backend.stream(system_prompt, messages):
                        full_response += chunk
                else:
                    full_response = backend.generate(system_prompt, messages)
                
                # Process canvas commands
                clean_response, steps = reasoner.process_response(full_response)
                all_steps.extend(steps)
                
                # Show thinking process (commands only, not canvas)
                if steps:
                    if not thinking_shown:
                        print(f"\n  [thinking...]")
                        thinking_shown = True
                    
                    for step in steps:
                        # Show command and brief result
                        if step.action in ('SKETCH', 'REVISE', 'GROUND'):
                            target_preview = step.target[:30] + "..." if len(step.target) > 30 else step.target
                            print(f"    → [{step.action}: {target_preview}]")
                        elif step.action == 'LOOK':
                            print(f"    → [LOOK]")
                        elif step.action in ('CANVAS', 'ZOOM', 'NEST'):
                            print(f"    → [{step.action}: {step.target}]")
                        elif step.action == 'META':
                            print(f"    → [META]")
                        elif step.action == 'COMPACT':
                            print(f"    → [COMPACT: {step.target}]")
                        elif step.action == 'EXPAND':
                            print(f"    → [EXPAND: {step.target}]")
                        elif step.action in ('SIMILAR', 'RESONATES'):
                            print(f"    → [{step.action}: {step.target}]")
                        elif step.action == 'INSIGHT':
                            insight_preview = step.target[:40] + "..." if len(step.target) > 40 else step.target
                            print(f"    → [INSIGHT: {insight_preview}]")
                        else:
                            print(f"    → [{step.action}]")
                    
                    # Show thought signature after thinking actions (user sees this, agent doesn't)
                    sig = reasoner.get_coherence()
                    print(f"    → [thought signature: {sig:.3f}]")
                
                # Add to history
                messages.append({"role": "assistant", "content": full_response})
                
                # If no commands were used, agent is done thinking - output the response
                if not steps:
                    break
                
                # If commands were used, feed results back and continue
                tool_results = format_tool_results(steps, reasoner)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            
            # Output final clean response
            if thinking_shown:
                print()  # Space after thinking
            print(f"Agent: {clean_response}")
            
            # Show final thought signature and canvas snapshot if any thinking happened
            if all_steps:
                final_sig = reasoner.get_coherence()
                print(f"\n  [final thought signature: {final_sig:.3f}]")
                
                # Show canvas snapshot
                print(f"\n  ┌─── Canvas Snapshot ───┐")
                canvas_state = reasoner.get_canvas_state()
                # Indent each line
                for line in canvas_state.split('\n'):
                    print(f"  {line}")
                print(f"  └────────────────────────┘")
        
        except KeyboardInterrupt:
            print("\n  (interrupted)")
            # Remove any messages we added
            while len(messages) > 0 and messages[-1]["role"] != "user":
                messages.pop()
            if messages and messages[-1]["content"] == user_input:
                messages.pop()
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            # Clean up messages
            while len(messages) > 0 and messages[-1].get("content") == user_input:
                messages.pop()
    
    # Save state
    reasoner.multi_canvas.save()
    reasoner.memory.save_all()
    print(f"\n  ✓ State saved to {reasoner.memory.storage_dir}")
    print("Goodbye!")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ORE Canvas Chat - Embodied reasoning with LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python canvas_chat.py                      # Interactive selection
  python canvas_chat.py --claude             # Use Claude Sonnet (default)
  python canvas_chat.py --claude opus        # Use Claude Opus
  python canvas_chat.py --claude haiku       # Use Claude Haiku
  python canvas_chat.py --ollama             # Use default Ollama model
  python canvas_chat.py --ollama mistral     # Use specific Ollama model
  python canvas_chat.py --list-models        # Show available Ollama models
        """
    )
    
    parser.add_argument('--claude', nargs='?', const='sonnet', metavar='MODEL',
                       help='Use Claude API (opus/sonnet/haiku, default: sonnet)')
    parser.add_argument('--ollama', nargs='?', const='qwen2.5:7b-instruct',
                       metavar='MODEL',
                       help='Use Ollama (optionally specify model)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available Ollama models')
    parser.add_argument('--no-stream', action='store_true',
                       help='Disable streaming output')
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nClaude models:")
        for short, full in CLAUDE_MODELS.items():
            print(f"  - {short}: {full}")
        
        print("\nOllama models:")
        models = list_ollama_models()
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  (none found - is Ollama running?)")
        print(f"\nUsage:")
        print(f"  python canvas_chat.py --claude opus")
        print(f"  python canvas_chat.py --ollama mistral")
        return
    
    # Select backend
    backend = None
    
    if args.claude:
        if not HAS_ANTHROPIC:
            print("Error: anthropic package not installed")
            print("Install with: pip install anthropic")
            sys.exit(1)
        backend = ClaudeBackend(args.claude)
    
    elif args.ollama:
        if not HAS_OLLAMA:
            print("Error: Ollama not running")
            print("Install from: https://ollama.ai")
            print("Then start with: ollama serve")
            sys.exit(1)
        backend = OllamaBackend(args.ollama)
    
    else:
        # Interactive selection
        backend = select_backend_interactive()
    
    if backend is None:
        print("No backend selected. Exiting.")
        sys.exit(0)
    
    # Run chat
    run_chat(backend, stream=not args.no_stream)


if __name__ == "__main__":
    main()
