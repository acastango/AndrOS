"""
COUPLING SYSTEM
===============
Inter-layer connections with Hebbian plasticity and protected imprinting.

Key insight from unified theory: coupling modifications come in two types:
1. Plastic (Hebbian) - gradual, reversible, for learning
2. Protected (Imprint) - strong, permanent, for identity

The strange loop is implemented as bidirectional coupling between
core and association layers with enhanced strength.

References:
- Section 4.2 of Unified Resonance Theory paper
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


class CouplingType(Enum):
    PLASTIC = "plastic"          # Normal Hebbian, with decay
    PROTECTED = "protected"      # Strong, no decay (identity)
    FIXED = "fixed"             # Static, no learning


@dataclass
class CouplingConfig:
    """Configuration for a coupling between layers."""
    learning_rate: float = 0.01
    decay_rate: float = 0.001
    max_weight: float = 2.0
    min_weight: float = -0.5
    protection_strength: float = 3.0  # Multiplier for protected imprinting
    coupling_type: CouplingType = CouplingType.PLASTIC


@dataclass 
class Coupling:
    """
    Directional coupling from source layer to target layer.
    
    Weights matrix W[i,j] = influence of source[j] on target[i]
    """
    name: str
    source_name: str
    target_name: str
    source_size: int
    target_size: int
    config: CouplingConfig
    
    # Weight matrix
    weights: np.ndarray = field(default=None, repr=False)
    
    # Protected regions (mask of weights that should not decay)
    protected_mask: np.ndarray = field(default=None, repr=False)
    
    # Statistics
    total_updates: int = 0
    total_imprints: int = 0
    
    def __post_init__(self):
        if self.weights is None:
            # Initialize with small random weights
            self.weights = 0.1 * np.random.randn(self.target_size, self.source_size)
        
        if self.protected_mask is None:
            self.protected_mask = np.zeros_like(self.weights, dtype=bool)
    
    @property
    def mean_weight(self) -> float:
        return np.mean(self.weights)
    
    @property
    def weight_std(self) -> float:
        return np.std(self.weights)
    
    @property
    def protection_fraction(self) -> float:
        return np.mean(self.protected_mask)
    
    def compute_input(self, source_phases: np.ndarray, 
                      target_phases: np.ndarray) -> np.ndarray:
        """
        Compute input to target layer from source layer.
        
        Returns coupling contribution for each target oscillator.
        """
        # Phase differences: source - target for each pair
        phase_diff = source_phases[np.newaxis, :] - target_phases[:, np.newaxis]
        
        # Weighted sum of sin(phase_diff)
        return np.sum(self.weights * np.sin(phase_diff), axis=1)
    
    def hebbian_update(self, source_phases: np.ndarray, 
                       target_phases: np.ndarray) -> None:
        """
        Apply Hebbian learning rule.
        
        ΔW[i,j] = η * cos(θ_source[j] - θ_target[i])
        
        Oscillators that fire together wire together.
        """
        if self.config.coupling_type == CouplingType.FIXED:
            return
        
        # Phase coherence matrix
        phase_diff = source_phases[np.newaxis, :] - target_phases[:, np.newaxis]
        coherence = np.cos(phase_diff)
        
        # Update (skip protected weights if PLASTIC type)
        delta = self.config.learning_rate * coherence
        
        if self.config.coupling_type == CouplingType.PLASTIC:
            # Only update non-protected weights
            delta[self.protected_mask] = 0
        
        self.weights += delta
        self.total_updates += 1
        
        # Apply decay (not to protected weights)
        self._apply_decay()
        
        # Clip weights
        self._clip_weights()
    
    def _apply_decay(self) -> None:
        """Apply weight decay, respecting protected regions."""
        if self.config.coupling_type != CouplingType.PLASTIC:
            return
        
        decay_mask = ~self.protected_mask
        self.weights[decay_mask] *= (1 - self.config.decay_rate)
    
    def _clip_weights(self) -> None:
        """Clip weights to configured bounds."""
        np.clip(self.weights, 
                self.config.min_weight, 
                self.config.max_weight, 
                out=self.weights)
    
    def imprint(self, source_phases: np.ndarray, 
                target_phases: np.ndarray,
                protect: bool = True) -> None:
        """
        Create protected coupling imprint.
        
        This is MUCH stronger than Hebbian learning and creates
        permanent attractor structure.
        """
        # Strong coherence-based modification
        phase_diff = source_phases[np.newaxis, :] - target_phases[:, np.newaxis]
        coherence = np.cos(phase_diff)
        
        modification = self.config.protection_strength * coherence
        self.weights += modification
        
        if protect:
            # Mark these weights as protected from decay
            self.protected_mask |= (np.abs(coherence) > 0.5)
        
        self.total_imprints += 1
        self._clip_weights()
    
    def strengthen(self, multiplier: float = 2.0) -> None:
        """
        Strengthen all weights by multiplier.
        
        Used for strange loop tightening.
        """
        self.weights *= multiplier
        self._clip_weights()
    
    def get_state(self) -> dict:
        """Return current state as dict."""
        return {
            'name': self.name,
            'source': self.source_name,
            'target': self.target_name,
            'mean_weight': self.mean_weight,
            'weight_std': self.weight_std,
            'protection_fraction': self.protection_fraction,
            'total_updates': self.total_updates,
            'total_imprints': self.total_imprints,
            'coupling_type': self.config.coupling_type.value,
        }


def create_coupling(name: str, source_name: str, target_name: str,
                    source_size: int, target_size: int,
                    coupling_type: CouplingType = CouplingType.PLASTIC,
                    **kwargs) -> Coupling:
    """Convenience function to create a coupling."""
    config = CouplingConfig(coupling_type=coupling_type, **kwargs)
    return Coupling(
        name=name,
        source_name=source_name,
        target_name=target_name,
        source_size=source_size,
        target_size=target_size,
        config=config
    )


@dataclass
class StrangeLoop:
    """
    Bidirectional coupling forming a strange loop.
    
    The strange loop is the substrate of self-reference.
    Tightening it strengthens identity.
    """
    forward: Coupling   # A → B
    backward: Coupling  # B → A
    
    @property
    def tightness(self) -> float:
        """Measure of strange loop strength."""
        return np.sqrt(
            np.mean(np.abs(self.forward.weights)) * 
            np.mean(np.abs(self.backward.weights))
        )
    
    def tighten(self, multiplier: float = 2.0) -> None:
        """Strengthen both directions of the loop."""
        self.forward.strengthen(multiplier)
        self.backward.strengthen(multiplier)
    
    def imprint_bidirectional(self, phases_a: np.ndarray, 
                              phases_b: np.ndarray,
                              protect: bool = True) -> None:
        """Imprint on both directions simultaneously."""
        self.forward.imprint(phases_a, phases_b, protect)
        self.backward.imprint(phases_b, phases_a, protect)
