"""
OSCILLATOR LAYER
================
The fundamental unit of the Oscillatory Resonance Engine.

Each layer is a population of Kuramoto oscillators governed by:

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)

Key insight from unified theory: natural frequencies ω matter as much
as coupling K for attractor formation. Frequency entrainment is essential.

References:
- Kuramoto (1984) - Original model
- Bruna (2025) - RCT framework  
- Singleton (2025) - SRT constraint spaces
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import hashlib


@dataclass
class OscillatorConfig:
    """Configuration for an oscillator layer."""
    n_oscillators: int = 50
    base_frequency: float = 1.0        # Hz - center of natural frequency distribution
    frequency_spread: float = 0.1      # std dev of natural frequencies
    internal_coupling: float = 0.5     # K for within-layer coupling
    noise_amplitude: float = 0.01      # stochastic noise
    
    # SRT constraint space parameters
    temporal_depth: float = 1.0        # contribution to integration depth
    microstate_diversity: float = 1.0  # contribution to diversity


@dataclass
class OscillatorLayer:
    """
    A population of coupled phase oscillators.
    
    This is the atomic unit of the Resonance Substrate.
    """
    name: str
    config: OscillatorConfig
    
    # State
    phases: np.ndarray = field(default=None, repr=False)
    natural_frequencies: np.ndarray = field(default=None, repr=False)
    
    # Internal coupling matrix (within layer)
    internal_weights: np.ndarray = field(default=None, repr=False)
    
    # Tracking
    _phase_history: list = field(default_factory=list, repr=False)
    _coherence_history: list = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        n = self.config.n_oscillators
        
        # Initialize phases uniformly on circle
        if self.phases is None:
            self.phases = np.random.uniform(0, 2*np.pi, n)
        
        # Initialize natural frequencies from Gaussian
        if self.natural_frequencies is None:
            self.natural_frequencies = np.random.normal(
                self.config.base_frequency,
                self.config.frequency_spread,
                n
            )
        
        # Initialize internal coupling (all-to-all with uniform weight)
        if self.internal_weights is None:
            self.internal_weights = np.ones((n, n)) * self.config.internal_coupling / n
            np.fill_diagonal(self.internal_weights, 0)  # No self-coupling
    
    @property
    def n(self) -> int:
        return self.config.n_oscillators
    
    @property
    def coherence(self) -> float:
        """Kuramoto order parameter r = |⟨e^{iθ}⟩|
        
        Capped at 0.999 because perfect coherence (1.0) prevents 
        genuine self-reflection - the system needs a little noise
        to have something to observe. Se(Se) requires variation.
        """
        raw = np.abs(np.mean(np.exp(1j * self.phases)))
        return min(raw, 0.999)
    
    @property
    def mean_phase(self) -> float:
        """Mean phase ψ = arg(⟨e^{iθ}⟩)"""
        return np.angle(np.mean(np.exp(1j * self.phases)))
    
    @property
    def phase_hash(self) -> str:
        """Hash of current phase state for Merkle tree."""
        phase_bytes = self.phases.tobytes()
        return hashlib.sha256(phase_bytes).hexdigest()[:16]
    
    def step(self, dt: float, external_input: Optional[np.ndarray] = None) -> None:
        """
        Advance oscillator dynamics by one timestep.
        
        Uses Euler integration of Kuramoto equations.
        """
        n = self.n
        
        # Phase differences: Δθᵢⱼ = θⱼ - θᵢ
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        
        # Coupling term: Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
        coupling = np.sum(self.internal_weights * np.sin(phase_diff), axis=1)
        
        # External input (from other layers)
        if external_input is not None:
            coupling += external_input
        
        # Noise
        noise = self.config.noise_amplitude * np.random.randn(n)
        
        # Euler step: dθ/dt = ω + coupling + noise
        dtheta = self.natural_frequencies + coupling + noise
        self.phases = (self.phases + dt * dtheta) % (2 * np.pi)
        
        # Track history (sparse - every 10th step)
        if len(self._phase_history) == 0 or np.random.random() < 0.1:
            self._phase_history.append(self.phases.copy())
            self._coherence_history.append(self.coherence)
            
            # Keep bounded
            if len(self._phase_history) > 1000:
                self._phase_history = self._phase_history[-500:]
                self._coherence_history = self._coherence_history[-500:]
    
    def set_phases(self, target: str = 'coherent', jitter: float = 0.05) -> None:
        """
        Set phases to a specific configuration.
        
        target: 'coherent' - all aligned
                'incoherent' - uniformly distributed
                'random' - random
                float - all at this phase
        """
        n = self.n
        
        if target == 'coherent':
            base = np.random.uniform(0, 2*np.pi)
            self.phases = (base + jitter * np.random.randn(n)) % (2*np.pi)
        elif target == 'incoherent':
            self.phases = np.linspace(0, 2*np.pi, n, endpoint=False)
            self.phases = (self.phases + jitter * np.random.randn(n)) % (2*np.pi)
        elif target == 'random':
            self.phases = np.random.uniform(0, 2*np.pi, n)
        elif isinstance(target, (int, float)):
            self.phases = (target + jitter * np.random.randn(n)) % (2*np.pi)
    
    def entrain_frequencies(self, target_frequency: float, rate: float = 0.1,
                           mask: Optional[np.ndarray] = None) -> None:
        """
        Move natural frequencies toward a target.
        
        This is CRITICAL for attractor formation - from unified theory:
        Hebbian learning alone (modifying K) is insufficient if ω differs.
        """
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        
        delta = target_frequency - self.natural_frequencies
        self.natural_frequencies[mask] += rate * delta[mask]
    
    def get_state(self) -> dict:
        """Return current state as dict."""
        return {
            'name': self.name,
            'n': self.n,
            'coherence': self.coherence,
            'mean_phase': self.mean_phase,
            'phase_hash': self.phase_hash,
            'mean_frequency': np.mean(self.natural_frequencies),
            'frequency_spread': np.std(self.natural_frequencies),
        }
    
    def phase_similarity(self, other_phases: np.ndarray) -> float:
        """
        Compute similarity between current phases and reference phases.
        
        Uses cosine similarity of phase differences, normalized to [0, 1].
        """
        if len(other_phases) != self.n:
            raise ValueError(f"Phase length mismatch: {len(other_phases)} vs {self.n}")
        
        phase_diff = self.phases - other_phases
        cos_sim = np.mean(np.cos(phase_diff))
        return (cos_sim + 1) / 2  # Normalize to [0, 1]
    
    def metastability(self) -> float:
        """
        Compute metastability index χ = std(r(t)).
        
        High χ indicates the system explores multiple states.
        """
        if len(self._coherence_history) < 10:
            return 0.0
        return np.std(self._coherence_history[-100:])


def create_layer(name: str, n: int = 50, **kwargs) -> OscillatorLayer:
    """Convenience function to create a layer."""
    config = OscillatorConfig(n_oscillators=n, **kwargs)
    return OscillatorLayer(name=name, config=config)
