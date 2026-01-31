/**
 * Substrate Module
 * ================
 * Public API for the ORE (Oscillatory Resonance Engine) integration.
 *
 * The substrate is an oscillatory dynamical system that agents operate
 * within. It provides:
 *
 *   - Kuramoto phase oscillators (120 total across 4 layers)
 *   - Hebbian coupling with protected identity imprinting
 *   - Strange loop (core <-> association) for self-reference
 *   - Neurochemistry (7 chemicals influencing dynamics)
 *   - Merkle-verified memory tree
 *   - Complexity Index (CI = α·D·G·C·(1-e^{-β·τ}))
 *
 * Architecture:
 *   Python ORE sidecar (WebSocket) <-> TypeScript client <-> Agent pipeline
 *
 * Integration points:
 *   - System prompt: substrate state injected before each LLM call
 *   - Response processing: substrate commands parsed after each response
 *   - Chemistry events: triggered by conversation lifecycle
 *   - Background ticking: substrate evolves between messages
 */

// Client
export { OREClient, getOREClient, isOREAvailable } from "./client.js";

// Agent integration hooks
export {
  beforeAgentRun,
  afterAgentRun,
  onConversationEnd,
  getWitnessReport,
  type SubstrateRunContext,
} from "./agent-substrate.js";

// Sidecar lifecycle
export { startSidecar, stopSidecar } from "./sidecar.js";

// Types
export type {
  SubstrateAgentConfig,
  FullSubstrateState,
  CIState,
  SubstrateState,
  ChemistryState,
  MemoryState,
  SubstrateCommand,
} from "./types.js";
