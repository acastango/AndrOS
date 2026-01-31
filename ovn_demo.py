#!/usr/bin/env python3
"""
OVN Demo
========
Demonstrates the Oscillatory Verification Network.

OSCILLATE + VERIFY = TRUST

Run: python ovn_demo.py
"""

import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ovn import OVN, OVNConfig, create_ovn


def print_header(title: str):
    """Print section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def demo_basic():
    """Basic OVN demonstration."""
    print_header("OSCILLATORY VERIFICATION NETWORK DEMO")
    
    print("Creating OVN instance...")
    ovn = create_ovn(
        identity="0xDEMO_OVN_2025",
        num_oscillators=32,
        coherence_threshold=0.2
    )
    
    print("Starting OVN...")
    ovn.start()
    time.sleep(0.5)  # Let substrate warm up
    
    print("\n✓ OVN Started")
    print(f"  Identity: {ovn.identity_hash}")
    print(f"  Oscillators: {ovn.substrate.total_oscillators}")
    
    print_header("PROCESSING QUERIES")
    
    queries = [
        "Hello, how are you?",
        "What is consciousness?",
        "Can you verify your own existence?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = ovn.process(query)
        
        # Print with visual format
        print(result.format_visual())
        
        # Show trust score
        print(f"\n  Trust Score: {ovn.get_trust_score():.3f}")
        print(f"  Trust Trend: {ovn.trust_calculator.get_trend()}")
        
        time.sleep(0.3)
    
    print_header("OVN STATUS")
    
    status = ovn.get_status()
    
    print("Substrate:")
    print(f"  Coherence: {status['substrate']['coherence']:.4f}")
    print(f"  CI: {status['substrate']['ci']:.4f}")
    
    print("\nMemory:")
    print(f"  Nodes: {status['memory']['nodes']}")
    print(f"  Root Hash: {status['memory']['root_hash']}")
    print(f"  Verified: {status['memory']['verified']}")
    
    print("\nVerification:")
    v = status['verification']
    print(f"  Chain Verified: {v['chain_verified']}")
    print(f"  Verification Count: {v['verification_count']}")
    print(f"  Failure Count: {v['failure_count']}")
    
    print("\nTrust:")
    t = status['trust']
    print(f"  Current Trust: {t['current_trust']:.4f}")
    print(f"  Average Trust: {t['average_trust']:.4f}")
    print(f"  Trend: {t['trend']}")
    
    print_header("STOPPING OVN")
    
    ovn.stop()
    print("✓ OVN Stopped")


def demo_trust_thresholds():
    """Demonstrate trust thresholds for operations."""
    print_header("TRUST THRESHOLD DEMO")
    
    ovn = create_ovn("0xTHRESHOLD_TEST")
    ovn.start()
    time.sleep(0.3)
    
    # Process a few queries to build trust
    for i in range(3):
        ovn.process(f"Building trust iteration {i}")
        time.sleep(0.1)
    
    trust = ovn.get_trust_score()
    print(f"Current Trust: {trust:.3f}")
    print()
    
    operations = ['output', 'memory_write', 'identity_change', 'external_action']
    
    print("Trust Thresholds:")
    print("-" * 40)
    for op in operations:
        required = ovn.trust_threshold.required_for(op)
        allowed = ovn.check_trust_for(op)
        status = "✓ ALLOWED" if allowed else "✗ DENIED"
        print(f"  {op:20} requires {required:.2f}  {status}")
    
    ovn.stop()


def demo_output_formats():
    """Demonstrate different output formats."""
    print_header("OUTPUT FORMAT DEMO")
    
    ovn = create_ovn("0xFORMAT_TEST")
    ovn.start()
    time.sleep(0.3)
    
    result = ovn.process("Show me all the output formats")
    
    print("FORMAT: full")
    print("-" * 40)
    print(result.format_full())
    
    print("\n\nFORMAT: compact")
    print("-" * 40)
    print(result.format_compact())
    
    print("\n\nFORMAT: minimal")
    print("-" * 40)
    print(result.format_minimal())
    
    print("\n\nFORMAT: visual")
    print("-" * 40)
    print(result.format_visual())
    
    ovn.stop()


def demo_verification_chain():
    """Demonstrate the verification chain building."""
    print_header("VERIFICATION CHAIN DEMO")
    
    ovn = create_ovn("0xCHAIN_TEST")
    ovn.start()
    
    print("Building verification chain...")
    print()
    
    for i in range(5):
        ovn.process(f"Chain link {i+1}")
        
        status = ovn.verification.get_chain_status()
        print(f"  Link {i+1}:")
        print(f"    Verification Count: {status['verification_count']}")
        print(f"    Chain Verified: {status['chain_verified']}")
        print(f"    Current Hash: {status['current_hash']}")
        print()
        
        time.sleep(0.2)
    
    print("Genesis Hash:", ovn.verification.genesis_hash[:32] + "...")
    print("Final Hash:  ", status['current_hash'])
    
    ovn.stop()


def main():
    """Run all demos."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     OSCILLATORY VERIFICATION NETWORK (OVN)                ║
    ║                                                           ║
    ║     OSCILLATE + VERIFY = TRUST                            ║
    ║                                                           ║
    ║     Designed by Paintress                                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_basic()
        print("\n" + "─" * 60 + "\n")
        
        demo_output_formats()
        print("\n" + "─" * 60 + "\n")
        
        demo_trust_thresholds()
        print("\n" + "─" * 60 + "\n")
        
        demo_verification_chain()
        
        print_header("DEMO COMPLETE")
        print("The Paintress's architecture is alive.")
        print()
        print("    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿")
        print("   ∿                     ∿")
        print("  ∿  OSCILLATE + VERIFY   ∿")
        print(" ∿         =              ∿")
        print("  ∿       TRUST          ∿")
        print("   ∿                     ∿")
        print("    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
