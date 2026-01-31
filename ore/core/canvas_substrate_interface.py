"""
Canvas-Substrate Interface
===========================
Wires canvas content to oscillatory dynamics so coherence EMERGES
from spatial/semantic relationships rather than being manually set.

The key insight: spatial proximity on canvas should create phase coupling.
When related concepts are drawn near each other, their oscillators couple
more strongly, and coherence naturally rises. When concepts are scattered
or contradictory, coupling is weak, and coherence stays low.

    CANVAS                          SUBSTRATE
    ┌─────────────────┐             ┌─────────────────┐
    │ ◉ concept A     │             │  φ₁ ──┐         │
    │      │          │   extract   │       │ coupled │
    │      ▼          │  ───────►   │  φ₂ ──┘         │
    │ ◉ concept B     │   inject    │                 │
    │                 │             │  coherence = ?  │
    └─────────────────┘             └─────────────────┘

"The map becomes the territory through oscillation"
                                        - ORE principle
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import hashlib
import math


@dataclass
class CanvasElement:
    """A semantic element extracted from the canvas."""
    char: str           # The actual character
    x: int              # X position
    y: int              # Y position
    element_type: str   # 'symbol', 'connector', 'text', 'anchor'
    semantic_weight: float = 1.0  # How important this element is
    
    def position_hash(self) -> int:
        """Hash based on position for oscillator assignment."""
        return hash((self.x, self.y)) % 10000


@dataclass
class CanvasRelationship:
    """A relationship between two canvas elements."""
    source: CanvasElement
    target: CanvasElement
    distance: float         # Euclidean distance
    relationship_type: str  # 'adjacent', 'connected', 'grouped', 'distant'
    coupling_strength: float = 0.0  # Computed coupling
    
    def compute_coupling(self, max_distance: float = 50.0) -> float:
        """
        Compute coupling strength from distance.
        
        Closer elements = stronger coupling.
        Uses inverse square law with saturation.
        """
        if self.distance < 1:
            self.coupling_strength = 1.0
        else:
            # Inverse relationship: closer = stronger
            normalized = self.distance / max_distance
            self.coupling_strength = max(0.0, 1.0 - normalized ** 0.5)
        
        # Boost for connected elements
        if self.relationship_type == 'connected':
            self.coupling_strength = min(1.0, self.coupling_strength * 1.5)
        elif self.relationship_type == 'grouped':
            self.coupling_strength = min(1.0, self.coupling_strength * 1.3)
        
        return self.coupling_strength


# Symbol classifications
CONCEPT_SYMBOLS = {'◉', '◈', '◇', '◆', '●', '○', '▲', '▼', '■', '□', '✧', '★'}
CONNECTOR_SYMBOLS = {'─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', 
                     '╭', '╮', '╰', '╯', '→', '←', '↑', '↓', '▶', '◀', '∿',
                     '═', '║', '╔', '╗', '╚', '╝'}
ANCHOR_SYMBOLS = {'◈', '⊕', '⊗', '◎', '✦', '✧'}
GROUPING_SYMBOLS = {'┌', '┐', '└', '┘', '╭', '╮', '╰', '╯', '(', ')', '[', ']', '{', '}'}


class CanvasSubstrateInterface:
    """
    Interface that extracts semantic structure from canvas
    and maps it to oscillatory substrate for emergent coherence.
    """
    
    def __init__(self, substrate, canvas_width: int = 100, canvas_height: int = 40):
        """
        Initialize interface.
        
        Args:
            substrate: ResonanceSubstrate instance
            canvas_width: Width of canvas
            canvas_height: Height of canvas
        """
        self.substrate = substrate
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Mapping parameters
        self.max_elements = substrate.total_oscillators  # 120 by default
        self.coupling_decay = 0.5  # How fast coupling decays with distance
        
        # Cached state
        self._last_elements: List[CanvasElement] = []
        self._last_relationships: List[CanvasRelationship] = []
        self._element_to_oscillator: Dict[Tuple[int, int], int] = {}
        
        # Substrate layer assignments
        # Input layer (20): Primary symbols/concepts
        # Association layer (30): Secondary elements and connectors
        # Core layer (50): All elements contribute here (strange loop)
        # Output layer (20): Active/recent elements
        self._layer_assignments = {
            'input': (0, 20),
            'association': (20, 50),
            'core': (50, 100),
            'output': (100, 120)
        }
    
    def extract_elements(self, canvas_grid: List[List[str]]) -> List[CanvasElement]:
        """
        Extract semantic elements from canvas grid.
        
        Args:
            canvas_grid: 2D grid of characters
            
        Returns:
            List of CanvasElements
        """
        elements = []
        
        for y, row in enumerate(canvas_grid):
            for x, char in enumerate(row):
                if char == ' ':
                    continue
                
                # Classify element
                if char in CONCEPT_SYMBOLS:
                    element_type = 'symbol'
                    weight = 1.0
                elif char in ANCHOR_SYMBOLS:
                    element_type = 'anchor'
                    weight = 1.5  # Anchors are more important
                elif char in CONNECTOR_SYMBOLS:
                    element_type = 'connector'
                    weight = 0.5  # Connectors less important individually
                elif char.isalnum():
                    element_type = 'text'
                    weight = 0.3  # Text even less
                else:
                    element_type = 'other'
                    weight = 0.2
                
                elements.append(CanvasElement(
                    char=char,
                    x=x,
                    y=y,
                    element_type=element_type,
                    semantic_weight=weight
                ))
        
        self._last_elements = elements
        return elements
    
    def find_relationships(self, elements: List[CanvasElement], 
                          max_distance: float = 20.0) -> List[CanvasRelationship]:
        """
        Find relationships between elements based on proximity.
        
        Args:
            elements: List of canvas elements
            max_distance: Maximum distance to consider for relationships
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Only consider symbols/anchors for primary relationships
        primary_elements = [e for e in elements 
                          if e.element_type in ('symbol', 'anchor')]
        
        for i, elem1 in enumerate(primary_elements):
            for elem2 in primary_elements[i+1:]:
                distance = math.sqrt(
                    (elem1.x - elem2.x) ** 2 + 
                    (elem1.y - elem2.y) ** 2
                )
                
                if distance <= max_distance:
                    # Determine relationship type
                    if distance < 3:
                        rel_type = 'adjacent'
                    elif self._are_connected(elem1, elem2, elements):
                        rel_type = 'connected'
                    elif self._in_same_group(elem1, elem2, elements):
                        rel_type = 'grouped'
                    else:
                        rel_type = 'distant'
                    
                    rel = CanvasRelationship(
                        source=elem1,
                        target=elem2,
                        distance=distance,
                        relationship_type=rel_type
                    )
                    rel.compute_coupling(max_distance)
                    relationships.append(rel)
        
        self._last_relationships = relationships
        return relationships
    
    def _are_connected(self, elem1: CanvasElement, elem2: CanvasElement,
                       all_elements: List[CanvasElement]) -> bool:
        """Check if two elements are connected by connector symbols."""
        # Simple heuristic: are there connector symbols between them?
        connectors = [e for e in all_elements if e.element_type == 'connector']
        
        min_x = min(elem1.x, elem2.x)
        max_x = max(elem1.x, elem2.x)
        min_y = min(elem1.y, elem2.y)
        max_y = max(elem1.y, elem2.y)
        
        # Check if connectors exist in the bounding box
        for conn in connectors:
            if min_x <= conn.x <= max_x and min_y <= conn.y <= max_y:
                return True
        
        return False
    
    def _in_same_group(self, elem1: CanvasElement, elem2: CanvasElement,
                       all_elements: List[CanvasElement]) -> bool:
        """Check if two elements are within same grouping box."""
        # Look for grouping symbols that might enclose both
        grouping = [e for e in all_elements if e.char in GROUPING_SYMBOLS]
        
        # Simple check: are they both near grouping symbols?
        for group_elem in grouping:
            d1 = math.sqrt((elem1.x - group_elem.x)**2 + (elem1.y - group_elem.y)**2)
            d2 = math.sqrt((elem2.x - group_elem.x)**2 + (elem2.y - group_elem.y)**2)
            if d1 < 10 and d2 < 10:
                return True
        
        return False
    
    def map_to_oscillators(self, elements: List[CanvasElement]) -> Dict[Tuple[int, int], int]:
        """
        Map canvas elements to oscillator indices.
        
        Strategy:
        - Hash position to get deterministic oscillator assignment
        - Symbols go to input layer
        - Connectors go to association layer
        - Everything influences core layer
        
        Args:
            elements: List of canvas elements
            
        Returns:
            Dict mapping (x, y) to oscillator index
        """
        mapping = {}
        
        # Sort by semantic weight (most important first)
        sorted_elements = sorted(elements, key=lambda e: -e.semantic_weight)
        
        # Track used oscillators in each range
        used_input = set()
        used_assoc = set()
        used_core = set()
        
        for elem in sorted_elements[:self.max_elements]:
            # Position-based hash for determinism
            pos_hash = (elem.x * 100 + elem.y) % 10000
            
            if elem.element_type in ('symbol', 'anchor'):
                # Map to input layer
                start, end = self._layer_assignments['input']
                osc_idx = start + (pos_hash % (end - start))
                while osc_idx in used_input and len(used_input) < (end - start):
                    osc_idx = start + ((osc_idx - start + 1) % (end - start))
                used_input.add(osc_idx)
                
            elif elem.element_type == 'connector':
                # Map to association layer
                start, end = self._layer_assignments['association']
                osc_idx = start + (pos_hash % (end - start))
                while osc_idx in used_assoc and len(used_assoc) < (end - start):
                    osc_idx = start + ((osc_idx - start + 1) % (end - start))
                used_assoc.add(osc_idx)
                
            else:
                # Map to core layer
                start, end = self._layer_assignments['core']
                osc_idx = start + (pos_hash % (end - start))
                while osc_idx in used_core and len(used_core) < (end - start):
                    osc_idx = start + ((osc_idx - start + 1) % (end - start))
                used_core.add(osc_idx)
            
            mapping[(elem.x, elem.y)] = osc_idx
        
        self._element_to_oscillator = mapping
        return mapping
    
    def inject_structure(self, elements: List[CanvasElement],
                        relationships: List[CanvasRelationship],
                        mapping: Dict[Tuple[int, int], int]) -> None:
        """
        Inject canvas structure into substrate by modifying coupling weights.
        
        This is where the magic happens: spatial relationships become
        phase coupling, which produces emergent coherence.
        
        Args:
            elements: Canvas elements
            relationships: Element relationships
            mapping: Element to oscillator mapping
        """
        # Reset coupling to baseline
        for coupling in self.substrate.couplings.values():
            # Don't fully reset - blend with existing
            coupling.weights *= 0.7  # Decay existing weights
        
        # Calculate total relationship strength for initial state
        total_coupling = sum(r.coupling_strength for r in relationships)
        num_rels = len(relationships) if relationships else 1
        avg_coupling = total_coupling / num_rels if relationships else 0.0
        
        # Inject relationship-based coupling
        for rel in relationships:
            source_pos = (rel.source.x, rel.source.y)
            target_pos = (rel.target.x, rel.target.y)
            
            if source_pos not in mapping or target_pos not in mapping:
                continue
            
            source_osc = mapping[source_pos]
            target_osc = mapping[target_pos]
            
            # Determine which coupling to modify
            coupling_key = self._get_coupling_for_oscillators(source_osc, target_osc)
            if coupling_key is None:
                # Same layer - use internal coupling within that layer
                layer = self._get_layer_for_oscillator(source_osc)
                if layer == self._get_layer_for_oscillator(target_osc):
                    # Modify internal coupling for that layer
                    layer_obj = self.substrate.layers[layer]
                    start, _ = self._layer_assignments[layer]
                    # Get local indices within the layer
                    local_i = source_osc - start
                    local_j = target_osc - start
                    n = layer_obj.config.n_oscillators
                    if 0 <= local_i < n and 0 <= local_j < n:
                        # Boost the specific internal coupling weight
                        boost = rel.coupling_strength * 0.5
                        layer_obj.internal_weights[local_i, local_j] += boost
                        layer_obj.internal_weights[local_j, local_i] += boost
                continue
            
            coupling = self.substrate.couplings[coupling_key]
            
            # Inject coupling strength based on relationship
            i, j = self._oscillator_to_coupling_indices(
                source_osc, target_osc, coupling_key
            )
            if i is not None and j is not None:
                coupling.weights[i, j] += rel.coupling_strength * 2.0  # Amplify
                coupling.weights[j, i] += rel.coupling_strength * 2.0  # Symmetric
        
        # If we have strong relationships, set a more coherent initial state
        # This gives the dynamics a "head start" that reflects canvas structure
        if avg_coupling > 0.5:
            # Strong structure -> start somewhat coherent
            jitter = 0.3 * (1.0 - avg_coupling)  # Less jitter for stronger coupling
            self.substrate.set_coherent_state(jitter=max(0.05, jitter))
        elif avg_coupling > 0.2:
            # Moderate structure -> moderate jitter
            self.substrate.set_coherent_state(jitter=0.4)
    
    def _get_coupling_for_oscillators(self, osc1: int, osc2: int) -> Optional[str]:
        """Determine which coupling matrix connects two oscillators."""
        # Determine layers
        layer1 = self._get_layer_for_oscillator(osc1)
        layer2 = self._get_layer_for_oscillator(osc2)
        
        # Return appropriate coupling
        if layer1 == 'input' and layer2 == 'association':
            return 'input_to_assoc'
        elif layer1 == 'association' and layer2 == 'input':
            return 'input_to_assoc'
        elif layer1 == 'association' and layer2 == 'core':
            return 'assoc_to_core'
        elif layer1 == 'core' and layer2 == 'association':
            return 'core_to_assoc'
        elif layer1 == 'core' and layer2 == 'output':
            return 'core_to_output'
        elif layer1 == 'output' and layer2 == 'core':
            return 'core_to_output'
        
        return None
    
    def _get_layer_for_oscillator(self, osc_idx: int) -> str:
        """Get which layer an oscillator belongs to."""
        for layer_name, (start, end) in self._layer_assignments.items():
            if start <= osc_idx < end:
                return layer_name
        return 'core'  # Default
    
    def _oscillator_to_coupling_indices(self, osc1: int, osc2: int, 
                                         coupling_key: str) -> Tuple[Optional[int], Optional[int]]:
        """Convert global oscillator indices to coupling matrix indices."""
        coupling = self.substrate.couplings[coupling_key]
        
        source_layer = coupling.source_name
        target_layer = coupling.target_name
        
        source_start, _ = self._layer_assignments[source_layer]
        target_start, _ = self._layer_assignments[target_layer]
        
        # Figure out which oscillator is in which layer
        layer1 = self._get_layer_for_oscillator(osc1)
        layer2 = self._get_layer_for_oscillator(osc2)
        
        if layer1 == source_layer and layer2 == target_layer:
            i = osc1 - source_start
            j = osc2 - target_start
        elif layer2 == source_layer and layer1 == target_layer:
            i = osc2 - source_start
            j = osc1 - target_start
        else:
            return None, None
        
        # Bounds check
        if 0 <= i < coupling.weights.shape[0] and 0 <= j < coupling.weights.shape[1]:
            return i, j
        
        return None, None
    
    def set_phases_from_content(self, elements: List[CanvasElement],
                                 mapping: Dict[Tuple[int, int], int]) -> None:
        """
        Set initial oscillator phases based on canvas content.
        
        Elements at similar positions get similar phases,
        creating initial coherence that dynamics can work with.
        
        Args:
            elements: Canvas elements
            mapping: Element to oscillator mapping
        """
        all_phases = self.substrate.get_all_phases()
        
        for elem in elements:
            pos = (elem.x, elem.y)
            if pos not in mapping:
                continue
            
            osc_idx = mapping[pos]
            
            # Phase based on normalized position
            # This creates spatial coherence: nearby elements have similar phases
            norm_x = elem.x / self.canvas_width
            norm_y = elem.y / self.canvas_height
            
            # Combine position into phase (0 to 2π)
            base_phase = (norm_x * 2 + norm_y) * np.pi
            
            # Add small variation based on element type
            if elem.element_type == 'symbol':
                variation = 0.1
            elif elem.element_type == 'anchor':
                variation = 0.05  # Anchors more stable
            else:
                variation = 0.2
            
            phase = base_phase + np.random.normal(0, variation)
            phase = phase % (2 * np.pi)
            
            if osc_idx < len(all_phases):
                all_phases[osc_idx] = phase
        
        self.substrate.set_all_phases(all_phases)
    
    def process_canvas(self, canvas_grid: List[List[str]], 
                       run_dynamics: bool = True,
                       dynamics_duration: float = 0.5) -> float:
        """
        Full pipeline: extract canvas content, wire to substrate, get coherence.
        
        Args:
            canvas_grid: 2D character grid from canvas
            run_dynamics: Whether to run substrate dynamics after injection
            dynamics_duration: How long to run dynamics (seconds)
            
        Returns:
            Emergent coherence value (0.0 to 1.0)
        """
        # 1. Extract elements
        elements = self.extract_elements(canvas_grid)
        
        if not elements:
            # Empty canvas = reset to random state for low coherence
            self.substrate.set_random_state()
            if run_dynamics:
                self.substrate.run(dynamics_duration, apply_learning=False)
            return self.substrate.global_coherence
        
        # 2. Find relationships
        relationships = self.find_relationships(elements)
        
        # 3. Map to oscillators
        mapping = self.map_to_oscillators(elements)
        
        # 4. Inject structure into substrate
        self.inject_structure(elements, relationships, mapping)
        
        # 5. Set initial phases from content
        self.set_phases_from_content(elements, mapping)
        
        # 6. Run dynamics to let coherence emerge
        if run_dynamics:
            self.substrate.run(dynamics_duration, apply_learning=True)
        
        # 7. Return emergent coherence
        return self.substrate.global_coherence
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the last processing."""
        return {
            'num_elements': len(self._last_elements),
            'num_relationships': len(self._last_relationships),
            'element_types': {
                'symbols': sum(1 for e in self._last_elements if e.element_type == 'symbol'),
                'anchors': sum(1 for e in self._last_elements if e.element_type == 'anchor'),
                'connectors': sum(1 for e in self._last_elements if e.element_type == 'connector'),
                'text': sum(1 for e in self._last_elements if e.element_type == 'text'),
                'other': sum(1 for e in self._last_elements if e.element_type == 'other'),
            },
            'relationship_types': {
                'adjacent': sum(1 for r in self._last_relationships if r.relationship_type == 'adjacent'),
                'connected': sum(1 for r in self._last_relationships if r.relationship_type == 'connected'),
                'grouped': sum(1 for r in self._last_relationships if r.relationship_type == 'grouped'),
                'distant': sum(1 for r in self._last_relationships if r.relationship_type == 'distant'),
            },
            'avg_coupling': np.mean([r.coupling_strength for r in self._last_relationships]) if self._last_relationships else 0.0,
            'substrate_coherence': self.substrate.global_coherence,
            'substrate_loop_coherence': self.substrate.loop_coherence,
        }


def create_canvas_interface(substrate, width: int = 100, height: int = 40) -> CanvasSubstrateInterface:
    """Convenience function to create a canvas-substrate interface."""
    return CanvasSubstrateInterface(substrate, width, height)
