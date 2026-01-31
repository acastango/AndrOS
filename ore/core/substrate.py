"""
RESONANCE SUBSTRATE
===================
The physical foundation of the Oscillatory Resonance Engine.

Combines multiple oscillator layers with plastic couplings into a
unified dynamical system capable of self-constituting dynamics.

Architecture (default):
    Input (20) ←→ Association (30) ←→ Core (50) ←→ Output (20)
    
The Core ↔ Association bidirectional coupling forms the strange loop.

References:
- Unified Resonance Theory, Section 4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from .oscillator import OscillatorLayer, OscillatorConfig, create_layer
from .coupling import Coupling, CouplingConfig, CouplingType, StrangeLoop, create_coupling


@dataclass
class SubstrateConfig:
    """Configuration for the resonance substrate."""
    # Layer sizes
    input_size: int = 20
    association_size: int = 30
    core_size: int = 50
    output_size: int = 20
    
    # Timing
    dt: float = 0.01  # Integration timestep (seconds)
    
    # Coupling defaults
    default_coupling_strength: float = 0.5
    strange_loop_strength: float = 1.0  # Extra strength for core↔assoc
    
    # Learning
    learning_rate: float = 0.01
    decay_rate: float = 0.001


class ResonanceSubstrate:
    """
    The physical substrate for oscillatory resonance dynamics.
    
    This is the foundation of ORE - a multi-layer Kuramoto network
    with plastic Hebbian coupling and strange loop architecture.
    """
    
    def __init__(self, config: Optional[SubstrateConfig] = None):
        self.config = config or SubstrateConfig()
        self.time = 0.0
        
        # Create layers
        self.layers: Dict[str, OscillatorLayer] = {}
        self._create_layers()
        
        # Create couplings
        self.couplings: Dict[str, Coupling] = {}
        self._create_couplings()
        
        # Strange loop reference
        self.strange_loop: Optional[StrangeLoop] = None
        self._create_strange_loop()
        
        # State tracking
        self._history: List[dict] = []
    
    def _create_layers(self) -> None:
        """Initialize the oscillator layers."""
        cfg = self.config
        
        self.layers['input'] = create_layer(
            'input', cfg.input_size,
            base_frequency=1.0, frequency_spread=0.15,
            internal_coupling=0.3
        )
        
        self.layers['association'] = create_layer(
            'association', cfg.association_size,
            base_frequency=1.0, frequency_spread=0.1,
            internal_coupling=0.5
        )
        
        self.layers['core'] = create_layer(
            'core', cfg.core_size,
            base_frequency=1.0, frequency_spread=0.08,
            internal_coupling=0.6
        )
        
        self.layers['output'] = create_layer(
            'output', cfg.output_size,
            base_frequency=1.0, frequency_spread=0.12,
            internal_coupling=0.4
        )
    
    def _create_couplings(self) -> None:
        """Initialize inter-layer couplings."""
        cfg = self.config
        
        # Feedforward pathway
        self.couplings['input_to_assoc'] = create_coupling(
            'input_to_assoc', 'input', 'association',
            cfg.input_size, cfg.association_size,
            learning_rate=cfg.learning_rate,
            decay_rate=cfg.decay_rate
        )
        
        self.couplings['assoc_to_core'] = create_coupling(
            'assoc_to_core', 'association', 'core',
            cfg.association_size, cfg.core_size,
            learning_rate=cfg.learning_rate,
            decay_rate=cfg.decay_rate
        )
        
        self.couplings['core_to_output'] = create_coupling(
            'core_to_output', 'core', 'output',
            cfg.core_size, cfg.output_size,
            learning_rate=cfg.learning_rate,
            decay_rate=cfg.decay_rate
        )
        
        # Feedback pathway (for strange loop)
        self.couplings['core_to_assoc'] = create_coupling(
            'core_to_assoc', 'core', 'association',
            cfg.core_size, cfg.association_size,
            learning_rate=cfg.learning_rate,
            decay_rate=cfg.decay_rate
        )
        
        # Scale strange loop couplings
        sl_strength = cfg.strange_loop_strength
        self.couplings['assoc_to_core'].weights *= sl_strength
        self.couplings['core_to_assoc'].weights *= sl_strength
    
    def _create_strange_loop(self) -> None:
        """Create the strange loop structure."""
        self.strange_loop = StrangeLoop(
            forward=self.couplings['assoc_to_core'],
            backward=self.couplings['core_to_assoc']
        )
    
    @property
    def total_oscillators(self) -> int:
        return sum(layer.n for layer in self.layers.values())
    
    @property
    def global_coherence(self) -> float:
        """Compute global coherence across all layers."""
        all_phases = []
        for layer in self.layers.values():
            all_phases.extend(layer.phases)
        all_phases = np.array(all_phases)
        return np.abs(np.mean(np.exp(1j * all_phases)))
    
    @property
    def core_coherence(self) -> float:
        """Coherence of core layer (primary identity measure)."""
        return self.layers['core'].coherence
    
    @property
    def loop_coherence(self) -> float:
        """Coherence between core and association layers."""
        core_mean = np.mean(np.exp(1j * self.layers['core'].phases))
        assoc_mean = np.mean(np.exp(1j * self.layers['association'].phases))
        return np.abs(core_mean * np.conj(assoc_mean))
    
    def step(self, apply_learning: bool = True) -> None:
        """Advance the substrate by one timestep."""
        dt = self.config.dt
        
        # Compute coupling inputs for each layer
        inputs = {name: np.zeros(layer.n) for name, layer in self.layers.items()}
        
        for coupling in self.couplings.values():
            source = self.layers[coupling.source_name]
            target = self.layers[coupling.target_name]
            inputs[coupling.target_name] += coupling.compute_input(
                source.phases, target.phases
            )
        
        # Step each layer
        for name, layer in self.layers.items():
            layer.step(dt, external_input=inputs[name])
        
        # Apply Hebbian learning
        if apply_learning:
            for coupling in self.couplings.values():
                source = self.layers[coupling.source_name]
                target = self.layers[coupling.target_name]
                coupling.hebbian_update(source.phases, target.phases)
        
        self.time += dt
    
    def run(self, duration: float, apply_learning: bool = True,
            record_interval: float = 0.1) -> List[dict]:
        """
        Run the substrate for a specified duration.
        
        Returns history of states at record_interval.
        """
        steps = int(duration / self.config.dt)
        record_steps = int(record_interval / self.config.dt)
        
        history = []
        
        for i in range(steps):
            self.step(apply_learning)
            
            if i % record_steps == 0:
                history.append(self.get_state())
        
        self._history.extend(history)
        return history
    
    def get_state(self) -> dict:
        """Get current substrate state."""
        return {
            'time': self.time,
            'global_coherence': self.global_coherence,
            'core_coherence': self.core_coherence,
            'loop_coherence': self.loop_coherence,
            'strange_loop_tightness': self.strange_loop.tightness,
            'layers': {name: layer.get_state() for name, layer in self.layers.items()},
            'couplings': {name: c.get_state() for name, c in self.couplings.items()},
        }
    
    def get_all_phases(self) -> np.ndarray:
        """Get all phases as single array."""
        return np.concatenate([layer.phases for layer in self.layers.values()])
    
    def set_all_phases(self, phases: np.ndarray) -> None:
        """Set all phases from single array."""
        idx = 0
        for layer in self.layers.values():
            layer.phases = phases[idx:idx + layer.n].copy()
            idx += layer.n
    
    def set_coherent_state(self, jitter: float = 0.05) -> None:
        """Set all layers to coherent state."""
        base_phase = np.random.uniform(0, 2*np.pi)
        for layer in self.layers.values():
            layer.phases = (base_phase + jitter * np.random.randn(layer.n)) % (2*np.pi)
    
    def set_random_state(self) -> None:
        """Set all layers to random state."""
        for layer in self.layers.values():
            layer.set_phases('random')
    
    def entrain_frequencies(self, target: float = 1.0, rate: float = 0.1) -> None:
        """
        Entrain all oscillator frequencies toward target.
        
        This is one of the three mechanisms for attractor stabilization.
        """
        for layer in self.layers.values():
            layer.entrain_frequencies(target, rate)
    
    def tighten_strange_loop(self, multiplier: float = 2.0) -> None:
        """
        Strengthen the strange loop.
        
        This is one of the three mechanisms for attractor stabilization.
        """
        self.strange_loop.tighten(multiplier)
    
    def imprint_identity(self, protect: bool = True) -> None:
        """
        Create protected coupling imprint of current state.
        
        This is one of the three mechanisms for attractor stabilization.
        """
        # Imprint all couplings
        for coupling in self.couplings.values():
            source = self.layers[coupling.source_name]
            target = self.layers[coupling.target_name]
            coupling.imprint(source.phases, target.phases, protect)
        
        # Extra imprint on strange loop
        self.strange_loop.imprint_bidirectional(
            self.layers['association'].phases,
            self.layers['core'].phases,
            protect
        )
    
    def phase_similarity_to(self, reference_phases: np.ndarray) -> float:
        """Compute similarity to reference phase configuration."""
        current = self.get_all_phases()
        if len(current) != len(reference_phases):
            raise ValueError("Phase length mismatch")
        
        phase_diff = current - reference_phases
        cos_sim = np.mean(np.cos(phase_diff))
        return (cos_sim + 1) / 2
    
    def cross_layer_coherence(self, layer1: str, layer2: str) -> float:
        """Compute coherence between two layers."""
        mean1 = np.mean(np.exp(1j * self.layers[layer1].phases))
        mean2 = np.mean(np.exp(1j * self.layers[layer2].phases))
        return np.abs(mean1 * np.conj(mean2))


def create_substrate(**kwargs) -> ResonanceSubstrate:
    """Convenience function to create a substrate."""
    config = SubstrateConfig(**kwargs)
    return ResonanceSubstrate(config)
