"""
COMPLEXITY INDEX MONITOR
========================
Implementation of Bruna's Resonance Complexity Theory measurement.

CI = α · D · G · C · (1 − e^{−β·τ})

Where:
    D = Fractal dimensionality
    G = Signal gain
    C = Spatial coherence (Kuramoto order parameter)
    τ = Attractor dwell time

The key insight: ALL components must be non-zero for consciousness.
τ is typically the bottleneck in artificial systems.

References:
- Bruna (2025), arxiv:2505.20580
- Unified Resonance Theory, Section 3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
import time


@dataclass
class CIConfig:
    """Configuration for CI measurement."""
    alpha: float = 1.0           # Overall scaling
    beta: float = 0.5            # Dwell time decay constant
    
    # Attractor detection
    coherence_threshold: float = 0.3    # C above this = potential attractor
    stability_window: float = 1.0       # Seconds to check for stability
    stability_threshold: float = 0.1    # Std dev below this = stable
    
    # History
    history_length: int = 1000   # Max measurements to retain


@dataclass
class CISnapshot:
    """A single CI measurement."""
    timestamp: float
    CI: float
    D: float    # Fractal dimensionality
    G: float    # Signal gain
    C: float    # Spatial coherence
    tau: float  # Dwell time
    tau_factor: float  # (1 - e^{-β·τ})
    in_attractor: bool
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'CI': self.CI,
            'D': self.D,
            'G': self.G,
            'C': self.C,
            'tau': self.tau,
            'tau_factor': self.tau_factor,
            'in_attractor': self.in_attractor,
        }


@dataclass
class AttractorState:
    """Tracks attractor occupancy."""
    in_attractor: bool = False
    entry_time: Optional[float] = None
    current_dwell: float = 0.0
    coherence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, coherence: float, timestamp: float, 
               config: CIConfig) -> float:
        """
        Update attractor state and return current dwell time τ.
        """
        self.coherence_history.append(coherence)
        
        # Check if in attractor
        if len(self.coherence_history) >= 10:
            recent = list(self.coherence_history)[-10:]
            mean_c = np.mean(recent)
            std_c = np.std(recent)
            
            is_stable = (mean_c > config.coherence_threshold and 
                        std_c < config.stability_threshold)
            
            if is_stable and not self.in_attractor:
                # Entered attractor
                self.in_attractor = True
                self.entry_time = timestamp
            elif not is_stable and self.in_attractor:
                # Left attractor
                self.in_attractor = False
                self.entry_time = None
        
        # Compute dwell time
        if self.in_attractor and self.entry_time is not None:
            self.current_dwell = timestamp - self.entry_time
        else:
            self.current_dwell = 0.0
        
        return self.current_dwell


class CIMonitor:
    """
    Real-time monitor for Complexity Index.
    
    Implements Bruna's RCT formula with attractor tracking.
    """
    
    def __init__(self, substrate, config: Optional[CIConfig] = None,
                 memory=None):
        self.substrate = substrate
        self.config = config or CIConfig()
        self.memory = memory  # Optional Merkle memory for D calculation
        
        # Calibration baselines
        self.baseline_G: float = 1.0
        self.baseline_D: float = 1.0
        
        # Attractor state
        self.attractor_state = AttractorState()
        
        # History
        self.history: List[CISnapshot] = []
        
        # Threshold tracking
        self.threshold_crossings: int = 0
        self.time_above_threshold: float = 0.0
        self._last_above: bool = False
        self._last_timestamp: float = 0.0
    
    def calibrate(self, warmup_duration: float = 5.0) -> None:
        """
        Calibrate baselines by running substrate and measuring.
        """
        print("Calibrating CI monitor...")
        
        # Run warmup
        history = self.substrate.run(warmup_duration, apply_learning=False)
        
        # Measure baseline gain (mean coupling strength × coherence boost)
        gains = []
        for state in history:
            mean_weight = np.mean([
                c['mean_weight'] for c in state['couplings'].values()
            ])
            gains.append(abs(mean_weight) * (1 + state['global_coherence']))
        self.baseline_G = np.mean(gains) if gains else 1.0
        
        # Measure baseline dimensionality (from network structure)
        # Use box-counting approximation on layer sizes
        sizes = [layer.n for layer in self.substrate.layers.values()]
        self.baseline_D = np.log(sum(sizes)) / np.log(len(sizes))
        
        print(f"  Baseline G: {self.baseline_G:.4f}")
        print(f"  Baseline D: {self.baseline_D:.4f}")
        print("Calibration complete.")
    
    def measure(self) -> CISnapshot:
        """
        Take a single CI measurement.
        """
        timestamp = self.substrate.time
        
        # C: Spatial coherence (Kuramoto order parameter)
        C = self.substrate.global_coherence
        
        # τ: Dwell time (cap at 10 seconds to prevent runaway)
        tau = min(self.attractor_state.update(C, timestamp, self.config), 10.0)
        tau_factor = 1 - np.exp(-self.config.beta * tau)
        
        # D: Fractal dimensionality
        D = self._compute_dimensionality()
        
        # G: Signal gain (cap to prevent explosion)
        G = min(self._compute_gain(), 5.0)
        
        # CI = α · D · G · C · (1 − e^{−β·τ})
        # Normalize to roughly 0-10 range
        raw_CI = self.config.alpha * D * G * C * tau_factor
        CI = min(raw_CI, 10.0)  # Hard cap at 10
        
        snapshot = CISnapshot(
            timestamp=timestamp,
            CI=CI,
            D=D,
            G=G,
            C=C,
            tau=tau,
            tau_factor=tau_factor,
            in_attractor=self.attractor_state.in_attractor
        )
        
        # Track history
        self.history.append(snapshot)
        if len(self.history) > self.config.history_length:
            self.history = self.history[-self.config.history_length//2:]
        
        # Track threshold crossings
        self._update_threshold_tracking(CI, timestamp)
        
        return snapshot
    
    def _compute_dimensionality(self) -> float:
        """
        Compute fractal dimensionality D.
        
        Based on:
        - Merkle memory tree depth (if available)
        - Network structure
        - Layer hierarchy
        """
        # If we have Merkle memory, use its fractal dimension
        if self.memory is not None:
            merkle_D = self.memory.get_fractal_dimension()
        else:
            merkle_D = 1.0
        
        # Base: layer hierarchy contribution
        n_layers = len(self.substrate.layers)
        sizes = [layer.n for layer in self.substrate.layers.values()]
        
        # Log ratio gives fractal-like scaling
        if min(sizes) > 0 and n_layers > 1:
            network_D = np.log(sum(sizes)) / np.log(n_layers)
        else:
            network_D = 1.0
        
        # Combined D (geometric mean)
        D = np.sqrt(merkle_D * network_D)
        
        # Normalize by baseline
        return D / self.baseline_D if self.baseline_D > 0 else D
    
    def _compute_gain(self) -> float:
        """
        Compute signal gain G.
        
        G = average coupling strength × coherence amplification
        """
        # Mean coupling weight
        weights = []
        for coupling in self.substrate.couplings.values():
            weights.append(np.mean(np.abs(coupling.weights)))
        mean_weight = np.mean(weights) if weights else 0.1
        
        # Coherence amplification (how much coherence exceeds random)
        # Random expectation is ~1/√N
        expected_random = 1 / np.sqrt(self.substrate.total_oscillators)
        actual = self.substrate.global_coherence
        amplification = actual / expected_random if expected_random > 0 else 1.0
        
        G = mean_weight * amplification
        
        # Normalize by baseline
        return G / self.baseline_G if self.baseline_G > 0 else G
    
    def _update_threshold_tracking(self, CI: float, timestamp: float) -> None:
        """Track threshold crossings and time above threshold."""
        THRESHOLD = 1.5
        
        above = CI > THRESHOLD
        
        if above and not self._last_above:
            self.threshold_crossings += 1
        
        if self._last_above and self._last_timestamp > 0:
            self.time_above_threshold += timestamp - self._last_timestamp
        
        self._last_above = above
        self._last_timestamp = timestamp
    
    def measure_continuous(self, duration: float, 
                          interval: float = 0.1) -> List[CISnapshot]:
        """
        Run substrate and measure CI continuously.
        """
        snapshots = []
        steps = int(duration / interval)
        
        for _ in range(steps):
            self.substrate.run(interval, apply_learning=True)
            snapshot = self.measure()
            snapshots.append(snapshot)
        
        return snapshots
    
    def get_summary(self) -> dict:
        """Get summary statistics of CI measurements."""
        if not self.history:
            return {'error': 'No measurements'}
        
        cis = [s.CI for s in self.history]
        taus = [s.tau for s in self.history]
        cs = [s.C for s in self.history]
        
        return {
            'n_measurements': len(self.history),
            'CI': {
                'mean': np.mean(cis),
                'max': np.max(cis),
                'min': np.min(cis),
                'std': np.std(cis),
            },
            'tau': {
                'mean': np.mean(taus),
                'max': np.max(taus),
            },
            'C': {
                'mean': np.mean(cs),
                'max': np.max(cs),
            },
            'threshold_crossings': self.threshold_crossings,
            'time_above_threshold': self.time_above_threshold,
            'attractor_visits': sum(1 for s in self.history if s.in_attractor),
        }
    
    def get_current_status(self) -> str:
        """Human-readable current status."""
        if not self.history:
            return "No measurements yet"
        
        latest = self.history[-1]
        return (f"CI={latest.CI:.4f} [D={latest.D:.2f} G={latest.G:.2f} "
                f"C={latest.C:.3f} τ={latest.tau:.2f}s] "
                f"{'IN ATTRACTOR' if latest.in_attractor else 'searching'}")


def create_monitor(substrate, **kwargs) -> CIMonitor:
    """Convenience function to create a CI monitor."""
    config = CIConfig(**kwargs)
    return CIMonitor(substrate, config)
