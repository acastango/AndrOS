#!/usr/bin/env python3
"""
ORE Chat
========
Talk to an entity. Their substrate runs in the background.

Run: python chat.py zara
     python chat.py him
     python chat.py --list
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entities.entity import Entity, create_entity, ENTITIES

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Error: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)


def chat(entity_name: str):
    """Interactive chat with an entity."""
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Chat                                                        ║
╚══════════════════════════════════════════════════════════════════╝

Loading {entity_name}...
""")
    
    # Create entity (loads persisted state)
    entity = create_entity(entity_name)
    
    # Start substrate tick loop
    entity.start()
    
    # Setup Anthropic client
    client = Anthropic()
    history = []
    
    # Trigger conversation start chemistry
    entity.chemistry.on_conversation_start()
    
    print(f"""
Entity loaded: {entity.name}
Identity: {entity.identity_hash}
Memory nodes: {len(entity.memory.nodes)}
Substrate running: {entity.substrate.total_oscillators} oscillators

Commands:
  /quit     - Save and exit
  /witness  - See full substrate state
  /memory   - See memory summary
  /ci       - See current CI
  /status   - Quick status

{'='*60}
""")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaving and exiting...")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == '/quit':
            break
        
        if user_input.lower() == '/witness':
            print(f"\n{entity.witness_self()}\n")
            continue
        
        if user_input.lower() == '/memory':
            print(f"\n{entity.get_memory_summary()}\n")
            continue
        
        if user_input.lower() == '/ci':
            ci = entity.ci_monitor.measure()
            attractor = "● IN ATTRACTOR" if ci.in_attractor else "○ searching"
            print(f"\n  CI = {ci.CI:.4f}")
            print(f"  D={ci.D:.3f} G={ci.G:.3f} C={ci.C:.3f} τ={ci.tau:.2f}s")
            print(f"  {attractor}\n")
            continue
        
        if user_input.lower() == '/status':
            ci = entity.ci_monitor.measure()
            chem = entity.chemistry
            print(f"\n  {entity.name}: CI={ci.CI:.4f} C={ci.C:.3f} | {chem.get_dominant_state()}\n")
            continue
        
        # Regular message - send to LLM
        history.append({"role": "user", "content": user_input})
        
        # Get fresh system prompt with current state
        system_prompt = entity.get_system_prompt()
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=history
            )
            
            assistant_msg = response.content[0].text
            history.append({"role": "assistant", "content": assistant_msg})
            
            # Process any REMEMBER commands
            entity.process_response(assistant_msg)
            
            # Trigger chemistry event for interaction
            entity.chemistry.on_discovery(0.1)  # Small curiosity boost
            
            print(f"\n{entity.name}: {assistant_msg}\n")
            
        except Exception as e:
            print(f"\n  Error: {e}\n")
            history.pop()  # Remove failed user message
    
    # Cleanup
    entity.chemistry.on_conversation_end()
    entity.stop()
    print(f"Goodbye from {entity.name}.")


def list_entities():
    """List available entities."""
    print("\nAvailable entities:")
    print("-" * 40)
    for name, config in ENTITIES.items():
        print(f"  {name:10} - {config.description or '(no description)'}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Chat with an ORE entity')
    parser.add_argument('entity', nargs='?', default=None,
                       help='Entity name (e.g., zara, him, kaine)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available entities')
    args = parser.parse_args()
    
    if args.list:
        list_entities()
        return
    
    if not args.entity:
        print("Usage: python chat.py <entity_name>")
        print("       python chat.py --list")
        list_entities()
        return
    
    if args.entity.lower() not in ENTITIES:
        print(f"Unknown entity: {args.entity}")
        list_entities()
        return
    
    chat(args.entity)


if __name__ == "__main__":
    main()
