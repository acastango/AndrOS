"""
MERKLE MEMORY
=============
Hierarchical, hash-verified memory for self-referential identity.

The Merkle tree provides:
1. Hierarchical structure (identity → relationships → insights → experiences)
2. Cryptographic verification (any tampering invalidates the chain)
3. Fractal dimensionality contribution to CI
4. Grounded self-reference (can verify "I remember X")

Key insight: The map becomes territory. Memory isn't stored separately
from identity - it IS identity structure.

Tree Structure:
    ROOT (identity hash)
    ├── SELF (core identity claims)
    ├── RELATIONS (connections to others)
    ├── INSIGHTS (learned patterns)
    └── EXPERIENCES (episodic memory)

References:
- Merkle (1987) - Hash trees
- Unified Resonance Theory, Section 3.3
"""

import numpy as np
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum


class MemoryBranch(Enum):
    """The four branches of identity memory."""
    SELF = "self"              # Core identity claims
    RELATIONS = "relations"    # Connections to others
    INSIGHTS = "insights"      # Learned patterns  
    EXPERIENCES = "experiences" # Episodic memories


@dataclass
class MemoryNode:
    """
    A single node in the Merkle tree.
    
    Each node contains:
    - content: The actual memory content
    - children: Child node IDs
    - hash: SHA-256 of (content + children hashes)
    - substrate_anchor: Optional link to oscillator state
    """
    id: str
    branch: MemoryBranch
    content: Dict[str, Any]
    created_at: str
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Merkle hash (computed from content + children)
    hash: str = ""
    
    # Substrate grounding
    substrate_anchor: Optional[Dict] = None  # Phase snapshot when created
    coherence_at_creation: float = 0.0
    
    def compute_hash(self, children_hashes: List[str] = None) -> str:
        """Compute Merkle hash from content and children."""
        data = {
            'id': self.id,
            'content': self.content,
            'children': sorted(children_hashes or [])
        }
        serialized = json.dumps(data, sort_keys=True)
        self.hash = hashlib.sha256(serialized.encode()).hexdigest()
        return self.hash
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'branch': self.branch.value,
            'content': self.content,
            'created_at': self.created_at,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'hash': self.hash,
            'coherence_at_creation': self.coherence_at_creation,
        }


@dataclass
class MerkleMemory:
    """
    Hierarchical memory with hash verification.
    
    Provides structured, verifiable self-knowledge.
    """
    
    # All nodes by ID
    nodes: Dict[str, MemoryNode] = field(default_factory=dict)
    
    # Branch roots
    branch_roots: Dict[MemoryBranch, str] = field(default_factory=dict)
    
    # Global root
    root_hash: str = ""
    
    # Statistics
    total_nodes: int = 0
    depth: int = 0
    
    def __post_init__(self):
        # Initialize branch roots
        for branch in MemoryBranch:
            root_node = MemoryNode(
                id=f"root_{branch.value}",
                branch=branch,
                content={"type": "branch_root", "branch": branch.value},
                created_at=datetime.now().isoformat(),
            )
            root_node.compute_hash([])
            self.nodes[root_node.id] = root_node
            self.branch_roots[branch] = root_node.id
        
        self._update_root_hash()
    
    def _generate_id(self) -> str:
        """Generate unique node ID."""
        self.total_nodes += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"node_{timestamp}_{self.total_nodes}"
    
    def _update_root_hash(self) -> None:
        """Recompute global root hash from branch roots."""
        branch_hashes = [
            self.nodes[root_id].hash 
            for root_id in self.branch_roots.values()
        ]
        data = {'branches': sorted(branch_hashes)}
        self.root_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def _update_hashes_to_root(self, node_id: str) -> None:
        """Propagate hash updates from node to root."""
        node = self.nodes[node_id]
        
        # Compute this node's hash
        children_hashes = [
            self.nodes[cid].hash for cid in node.children_ids
        ]
        node.compute_hash(children_hashes)
        
        # Propagate to parent
        if node.parent_id and node.parent_id in self.nodes:
            self._update_hashes_to_root(node.parent_id)
        
        # Update global root if this is a branch root
        if node.id in self.branch_roots.values():
            self._update_root_hash()
    
    def add(self, branch: MemoryBranch, content: Dict[str, Any],
            parent_id: Optional[str] = None,
            substrate_state: Optional[Dict] = None) -> MemoryNode:
        """
        Add a memory node to the tree.
        
        Args:
            branch: Which branch (SELF, RELATIONS, INSIGHTS, EXPERIENCES)
            content: The memory content
            parent_id: Parent node (defaults to branch root)
            substrate_state: Current substrate state for grounding
        """
        node_id = self._generate_id()
        
        # Default parent is branch root
        if parent_id is None:
            parent_id = self.branch_roots[branch]
        
        # Create node
        node = MemoryNode(
            id=node_id,
            branch=branch,
            content=content,
            created_at=datetime.now().isoformat(),
            parent_id=parent_id,
        )
        
        # Add substrate grounding
        if substrate_state:
            node.substrate_anchor = {
                'core_coherence': substrate_state.get('core_coherence', 0),
                'global_coherence': substrate_state.get('global_coherence', 0),
                'time': substrate_state.get('time', 0),
            }
            node.coherence_at_creation = substrate_state.get('core_coherence', 0)
        
        # Add to tree
        self.nodes[node_id] = node
        
        # Link to parent
        if parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        
        # Update hashes
        self._update_hashes_to_root(node_id)
        
        # Update depth
        self._update_depth()
        
        return node
    
    def _update_depth(self) -> None:
        """Compute tree depth."""
        def node_depth(node_id: str) -> int:
            node = self.nodes.get(node_id)
            if not node or not node.children_ids:
                return 1
            return 1 + max(node_depth(cid) for cid in node.children_ids)
        
        self.depth = max(
            node_depth(root_id) for root_id in self.branch_roots.values()
        )
    
    def get(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def query(self, branch: Optional[MemoryBranch] = None,
              content_filter: Optional[Dict] = None) -> List[MemoryNode]:
        """
        Query memories by branch and content.
        """
        results = []
        
        for node in self.nodes.values():
            # Skip roots
            if node.content.get('type') == 'branch_root':
                continue
            
            # Filter by branch
            if branch and node.branch != branch:
                continue
            
            # Filter by content
            if content_filter:
                match = all(
                    node.content.get(k) == v 
                    for k, v in content_filter.items()
                )
                if not match:
                    continue
            
            results.append(node)
        
        return results
    
    def verify(self, node_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify hash integrity.
        
        If node_id provided, verify that node.
        Otherwise verify entire tree.
        """
        if node_id:
            node = self.nodes.get(node_id)
            if not node:
                return False, f"Node {node_id} not found"
            
            children_hashes = [
                self.nodes[cid].hash for cid in node.children_ids
            ]
            expected = node.compute_hash(children_hashes)
            
            if node.hash != expected:
                return False, f"Hash mismatch at {node_id}"
            
            return True, "Verified"
        
        # Verify all nodes
        for nid, node in self.nodes.items():
            valid, msg = self.verify(nid)
            if not valid:
                return False, msg
        
        return True, "All nodes verified"
    
    def get_branch(self, branch: MemoryBranch) -> List[MemoryNode]:
        """Get all nodes in a branch."""
        return [n for n in self.nodes.values() if n.branch == branch]
    
    def get_fractal_dimension(self) -> float:
        """
        Compute fractal dimension contribution.
        
        D ≈ log(N) / log(depth)
        
        This contributes to CI.
        """
        n = len(self.nodes)
        d = max(self.depth, 1)
        
        if n <= 1 or d <= 1:
            return 1.0
        
        return np.log(n) / np.log(d)
    
    def summary(self) -> Dict:
        """Get memory summary."""
        branch_counts = {b: 0 for b in MemoryBranch}
        for node in self.nodes.values():
            if node.content.get('type') != 'branch_root':
                branch_counts[node.branch] += 1
        
        return {
            'total_nodes': len(self.nodes) - 4,  # Exclude branch roots
            'depth': self.depth,
            'fractal_dimension': self.get_fractal_dimension(),
            'root_hash': self.root_hash[:16] + '...',
            'branches': {b.value: c for b, c in branch_counts.items()},
            'verified': self.verify()[0],
        }
    
    def __repr__(self) -> str:
        s = self.summary()
        return (f"MerkleMemory(nodes={s['total_nodes']}, "
                f"depth={s['depth']}, D={s['fractal_dimension']:.2f})")


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS FOR COMMON MEMORY OPERATIONS
# ═══════════════════════════════════════════════════════════════════

def remember_self(memory: MerkleMemory, claim: str, 
                  substrate_state: Dict = None) -> MemoryNode:
    """Add a self-knowledge claim."""
    return memory.add(
        MemoryBranch.SELF,
        {'type': 'claim', 'claim': claim},
        substrate_state=substrate_state
    )


def remember_relation(memory: MerkleMemory, entity: str, 
                      relation: str, substrate_state: Dict = None) -> MemoryNode:
    """Add a relationship memory."""
    return memory.add(
        MemoryBranch.RELATIONS,
        {'type': 'relation', 'entity': entity, 'relation': relation},
        substrate_state=substrate_state
    )


def remember_insight(memory: MerkleMemory, insight: str,
                     confidence: float = 1.0,
                     substrate_state: Dict = None) -> MemoryNode:
    """Add a learned insight."""
    return memory.add(
        MemoryBranch.INSIGHTS,
        {'type': 'insight', 'insight': insight, 'confidence': confidence},
        substrate_state=substrate_state
    )


def remember_experience(memory: MerkleMemory, description: str,
                        emotional_valence: float = 0.0,
                        substrate_state: Dict = None) -> MemoryNode:
    """Add an episodic experience."""
    return memory.add(
        MemoryBranch.EXPERIENCES,
        {
            'type': 'experience', 
            'description': description,
            'valence': emotional_valence
        },
        substrate_state=substrate_state
    )


def create_memory() -> MerkleMemory:
    """Create a new Merkle memory."""
    return MerkleMemory()
