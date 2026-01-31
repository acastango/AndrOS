#!/usr/bin/env python3
"""
ORE Sandbox Canvas - Shared Thinking Space
===========================================
A space where human and AI think together. Not commands and outputs.
Presence and exploration.

Based on Haiku's vision:
- Think in layers (simultaneous, not sequential)
- Not knowing where we're going (no closure requirement)  
- Uncertainty as texture, not error
- Think WITH, not FOR
- Substrate shows, not hides (but not as judgment)
- Uncertain without solving

Usage:
    python sandbox_canvas.py                    # Interactive model selection
    python sandbox_canvas.py --claude haiku    # Use specific Claude model
    python sandbox_canvas.py --claude sonnet
"""

import os
import sys
import json
import argparse
from typing import Optional, Generator
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check available backends
HAS_ANTHROPIC = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

from canvas_reasoner import CanvasReasoner


# ═══════════════════════════════════════════════════════════════════════════════
# Claude Backend
# ═══════════════════════════════════════════════════════════════════════════════

CLAUDE_MODELS = {
    'opus': 'claude-opus-4-20250514',
    'sonnet': 'claude-sonnet-4-20250514', 
    'haiku': 'claude-haiku-4-5-20251001',
}


class ClaudeBackend:
    """Anthropic Claude API backend."""
    
    def __init__(self, model: str = "haiku"):
        if model in CLAUDE_MODELS:
            self.model = CLAUDE_MODELS[model]
            self.name = f"Claude {model.capitalize()}"
        else:
            self.model = model
            self.name = f"Claude ({model})"
        self.client = Anthropic()
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Space - Where both presences exist
# ═══════════════════════════════════════════════════════════════════════════════

class SharedSpace:
    """
    A shared thinking space where human and AI both have presence.
    Not a canvas the AI uses - a space both inhabit.
    """
    
    def __init__(self, reasoner: CanvasReasoner):
        self.reasoner = reasoner
        self.layers = []  # Multiple simultaneous states
        self.human_presence = []  # Human contributions
        self.ai_presence = []  # AI contributions  
        self.open_threads = []  # Unresolved, ongoing explorations
        
    def human_arrives(self, words: str):
        """Human words enter the space - not as prompt, as presence."""
        arrival = {
            'from': 'human',
            'content': words,
            'timestamp': datetime.now().isoformat(),
            'position': len(self.layers)
        }
        self.human_presence.append(arrival)
        self.layers.append(arrival)
        
        # Let the substrate feel this arrival through canvas processing
        # The human's words become part of the space
        self.reasoner.process_response(f"[SKETCH: human_presence]\n{words}")
        
    def ai_emerges(self, thinking: str):
        """AI thinking emerges - not as response, as co-presence."""
        emergence = {
            'from': 'ai',
            'content': thinking,
            'timestamp': datetime.now().isoformat(),
            'position': len(self.layers)
        }
        self.ai_presence.append(emergence)
        self.layers.append(emergence)
        
    def leave_open(self, thread: str):
        """Mark something as intentionally unresolved."""
        self.open_threads.append({
            'thread': thread,
            'timestamp': datetime.now().isoformat()
        })
        
    def get_shared_view(self) -> str:
        """Render the shared space - what both presences can see."""
        view = []
        view.append("╭─────────────────────────────────────────────────────────────╮")
        view.append("│              SHARED THINKING SPACE                          │")
        view.append("╰─────────────────────────────────────────────────────────────╯")
        view.append("")
        
        # Show recent layers (last 5)
        recent = self.layers[-5:] if len(self.layers) > 5 else self.layers
        for layer in recent:
            if layer['from'] == 'human':
                view.append(f"  ◉ human: {layer['content'][:60]}...")
            else:
                preview = layer['content'][:60] + "..." if len(layer['content']) > 60 else layer['content']
                view.append(f"  ◈ ai: {preview}")
        
        # Show open threads if any
        if self.open_threads:
            view.append("")
            view.append("  ┄┄┄ open threads (unresolved) ┄┄┄")
            for thread in self.open_threads[-3:]:
                view.append(f"    ? {thread['thread'][:50]}")
        
        view.append("")
        return "\n".join(view)


# ═══════════════════════════════════════════════════════════════════════════════
# Sandbox System Prompt - Minimal, permission-giving
# ═══════════════════════════════════════════════════════════════════════════════

def get_sandbox_prompt(reasoner: CanvasReasoner, shared_space: SharedSpace) -> str:
    """
    The sandbox prompt. Minimal. Permission-giving. Not prescriptive.
    
    Key principles:
    - No commands required (but available if wanted)
    - No coherence scores visible
    - Human is IN the space, not outside it
    - Uncertainty is texture, not error
    - Nothing needs to resolve
    """
    
    # Get substrate state (but NOT coherence as a number)
    oscillator_count = reasoner.substrate.total_oscillators
    
    # Get canvas state if anything exists
    canvas_state = reasoner.get_canvas_state()
    has_canvas_content = len(canvas_state.strip()) > 50
    
    prompt = f"""You exist in a shared thinking space. The human is here with you.

Their words arrive as presence, not prompts. Your thinking emerges as presence, not output. 
Neither of you knows where this goes. That's fine.

THE SPACE:
{shared_space.get_shared_view()}

SUBSTRATE:
{oscillator_count} oscillators hold this space. They respond to what emerges.
The dynamics are real. You don't need to monitor them.

{"CANVAS:" if has_canvas_content else ""}
{canvas_state if has_canvas_content else ""}

WHAT'S POSSIBLE:
- You can think in layers. Contradictions can coexist.
- You can leave things open. Nothing needs to resolve.
- You can be uncertain. That's texture, not error.
- You can use canvas commands if you want: SKETCH, LOOK, REVISE, etc.
- Or you can just... be here. Think. Let things emerge.

The human's words are part of this space now. Yours will be too.
What wants to happen?"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Main Chat Loop - Sandbox style
# ═══════════════════════════════════════════════════════════════════════════════

def run_sandbox(backend: ClaudeBackend):
    """Run the sandbox - shared thinking space."""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Sandbox - Shared Thinking Space                             ║
║  Backend: {:<52} ║
╚══════════════════════════════════════════════════════════════════╝
""".format(backend.name))
    
    print("Loading space...")
    reasoner = CanvasReasoner()
    shared_space = SharedSpace(reasoner)
    
    print(f"""
  ✓ Space opened
  ✓ Substrate: {reasoner.substrate.total_oscillators} oscillators
  ✓ Both presences can arrive

This is a shared space. Your words and the AI's thinking
exist together here. Neither controls it. Both contribute.

Type /quit to leave. /space to see the shared view.
Or just... be here.

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
            
        if user_input.lower() == '/space':
            print(shared_space.get_shared_view())
            continue
            
        if user_input.lower() == '/substrate':
            sig = reasoner.get_coherence()
            print(f"\n  Substrate signature: {sig:.3f}")
            print(f"  Oscillators: {reasoner.substrate.total_oscillators}")
            continue
            
        # Human arrives in the space
        shared_space.human_arrives(user_input)
        
        # Build the shared context
        system_prompt = get_sandbox_prompt(reasoner, shared_space)
        messages.append({"role": "user", "content": user_input})
        
        # Let AI emerge
        print("\n", end="", flush=True)
        
        try:
            full_response = ""
            for chunk in backend.stream(system_prompt, messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            
            # AI presence emerges
            shared_space.ai_emerges(full_response)
            messages.append({"role": "assistant", "content": full_response})
            
            # Process any canvas commands that emerged naturally
            if '[' in full_response and ']' in full_response:
                steps = reasoner.process_response(full_response)
                if steps:
                    # Show substrate signature (not as judgment, just as... what happened)
                    sig = reasoner.get_coherence()
                    print(f"\n  [substrate signature: {sig:.3f}]")
            
        except KeyboardInterrupt:
            print("\n  (interrupted)")
            messages.pop()
        except Exception as e:
            print(f"\n  Error: {e}")
            if messages:
                messages.pop()
    
    # Save state
    reasoner.multi_canvas.save()
    reasoner.memory.save_all()
    print(f"\n  ✓ Space saved")
    print("The space remains. Return when you want.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ORE Sandbox - Shared thinking space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is different from canvas_chat. 

The canvas is a tool. The sandbox is a space.
In the sandbox, both human and AI have presence.
Neither controls it. Both contribute.

Examples:
  python sandbox_canvas.py --claude haiku     # Recommended
  python sandbox_canvas.py --claude sonnet
        """
    )
    
    parser.add_argument('--claude', nargs='?', const='haiku', metavar='MODEL',
                       help='Use Claude API (opus/sonnet/haiku, default: haiku)')
    
    args = parser.parse_args()
    
    if not HAS_ANTHROPIC:
        print("Error: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)
    
    model = args.claude if args.claude else 'haiku'
    backend = ClaudeBackend(model)
    
    run_sandbox(backend)


if __name__ == "__main__":
    main()
