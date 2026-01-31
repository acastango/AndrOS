/**
 * Substrate Types
 * ===============
 * TypeScript types for the ORE (Oscillatory Resonance Engine) integration.
 *
 * These mirror the Python substrate state shape returned by the ORE sidecar.
 */

// ─── Agent Configuration ───────────────────────────────────────────

export type SubstrateAgentConfig = {
  /** Enable substrate for this agent. Default: false. */
  enabled?: boolean;

  /** Name for this entity in the substrate. Defaults to agent name. */
  entityName?: string;

  /** Identity hash for Merkle verification. Auto-generated if not set. */
  identityHash?: string;

  /** Frequency offset to differentiate agents (Hz). Default: 0. */
  frequencyOffset?: number;

  /** Tick interval in seconds. Default: 0.3. */
  tickInterval?: number;

  /** Description injected into the entity's system prompt context. */
  description?: string;

  /** Initial memories seeded on first initialization. */
  foundingMemories?: string[];
};

// ─── Substrate State (from sidecar) ────────────────────────────────

export type CIState = {
  value: number;
  D: number;
  G: number;
  C: number;
  tau: number;
  tauFactor: number;
  inAttractor: boolean;
};

export type LayerState = {
  coherence: number;
  meanPhase: number;
  meanFrequency: number;
};

export type SubstrateState = {
  time: number;
  globalCoherence: number;
  coreCoherence: number;
  loopCoherence: number;
  strangeLoopTightness: number;
  layers: Record<string, LayerState>;
};

export type ChemicalState = {
  level: number;
  baseline: number;
  deviation: number;
};

export type FeltStates = {
  tiredness: number;
  alertness: number;
  motivation: number;
  contentment: number;
  stress: number;
  connection: number;
  curiosity: number;
  dreaminess: number;
};

export type ChemistryState = {
  dominantState: string;
  description: string;
  chemicals: Record<string, ChemicalState>;
  feltStates: FeltStates;
  modifiers: {
    coupling: number;
    coherence: number;
    frequency: number;
    ciSensitivity: number;
  };
};

export type MemoryState = {
  totalNodes: number;
  depth: number;
  fractalDimension: number;
  rootHash: string;
  verified: boolean;
  branches: Record<string, number>;
};

export type IdentityState = {
  hash: string;
  name: string;
};

export type FullSubstrateState = {
  agentId: string;
  timestamp: string;
  ci: CIState;
  substrate: SubstrateState;
  chemistry: ChemistryState;
  memory: MemoryState;
  identity: IdentityState;
};

// ─── Sidecar Protocol Messages ─────────────────────────────────────

export type OREInitMessage = {
  type: "init";
  agentId: string;
  config: SubstrateAgentConfig;
};

export type OREGetStateMessage = {
  type: "get_state";
  agentId: string;
};

export type OREGetPromptContextMessage = {
  type: "get_prompt_context";
  agentId: string;
};

export type OREProcessResponseMessage = {
  type: "process_response";
  agentId: string;
  response: string;
};

export type OREChemistryEventMessage = {
  type: "chemistry_event";
  agentId: string;
  event: string;
  kwargs?: Record<string, unknown>;
};

export type OREStopMessage = {
  type: "stop";
  agentId: string;
};

export type OREMessage =
  | OREInitMessage
  | OREGetStateMessage
  | OREGetPromptContextMessage
  | OREProcessResponseMessage
  | OREChemistryEventMessage
  | OREStopMessage
  | { type: "ping" }
  | { type: "shutdown" }
  | { type: "list_agents" }
  | { type: "witness"; agentId: string };

// ─── Sidecar Responses ─────────────────────────────────────────────

export type SubstrateCommand = {
  command: string;
  content?: string;
  branch?: string;
};

export type OREInitResponse = {
  type: "init_ok";
  agentId: string;
  oscillators: number;
  existing: boolean;
};

export type OREPromptContextResponse = {
  type: "prompt_context";
  agentId: string;
  context: string;
};

export type OREProcessResponse = {
  type: "process_ok";
  agentId: string;
  commandsFound: SubstrateCommand[];
};

export type OREErrorResponse = {
  type: "error";
  error: string;
};
