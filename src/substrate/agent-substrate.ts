/**
 * Agent Substrate Integration
 * ===========================
 * Wires the ORE substrate into the OpenClaw agent execution lifecycle.
 *
 * Provides hooks that the agent runner calls:
 *  - beforeAgentRun()  → init substrate, inject context into system prompt
 *  - afterAgentRun()   → process response for commands, trigger chemistry
 *  - onConversationStart() → chemistry event
 *  - onConversationEnd()   → chemistry event, persist state
 */

import { getOREClient, OREClient } from "./client.js";
import type { SubstrateAgentConfig, SubstrateCommand } from "./types.js";

export type SubstrateRunContext = {
  agentId: string;
  client: OREClient;
  initialized: boolean;
  promptContext: string | null;
};

/**
 * Prepare substrate before an agent run.
 *
 * 1. Ensures the agent's substrate is initialized on the sidecar
 * 2. Fetches current substrate state as formatted prompt context
 * 3. Returns context to be prepended to the system prompt
 *
 * If the sidecar is unavailable, returns null (substrate is opt-in and
 * should not block agent execution).
 */
export async function beforeAgentRun(
  agentId: string,
  config: SubstrateAgentConfig,
): Promise<SubstrateRunContext | null> {
  if (!config.enabled) return null;

  const client = getOREClient();
  try {
    const connected = await client.ensureConnected();
    if (!connected) {
      return null;
    }

    // Initialize (idempotent)
    await client.initAgent(agentId, config);

    // Trigger conversation chemistry
    await client.chemistryEvent(agentId, "on_conversation_start");

    // Get prompt context
    const promptContext = await client.getPromptContext(agentId);

    return {
      agentId,
      client,
      initialized: true,
      promptContext,
    };
  } catch (err) {
    // Substrate failure should not prevent agent from running
    return null;
  }
}

/**
 * Process results after an agent run completes.
 *
 * 1. Sends the agent's response to the sidecar for command processing
 * 2. Handles any substrate commands found (REMEMBER, WITNESS, etc.)
 * 3. Triggers appropriate chemistry events
 *
 * Returns the list of substrate commands that were executed.
 */
export async function afterAgentRun(
  ctx: SubstrateRunContext | null,
  assistantText: string,
): Promise<SubstrateCommand[]> {
  if (!ctx?.initialized) return [];

  try {
    // Process the response for substrate commands
    const commands = await ctx.client.processResponse(ctx.agentId, assistantText);

    // Trigger activity chemistry (the agent did work)
    await ctx.client.chemistryEvent(ctx.agentId, "on_activity", {
      intensity: Math.min(1.0, assistantText.length / 2000),
    });

    return commands;
  } catch {
    return [];
  }
}

/**
 * Notify substrate that a conversation has ended.
 * Triggers chemistry wind-down.
 */
export async function onConversationEnd(
  ctx: SubstrateRunContext | null,
): Promise<void> {
  if (!ctx?.initialized) return;

  try {
    await ctx.client.chemistryEvent(ctx.agentId, "on_conversation_end");
  } catch {
    // non-critical
  }
}

/**
 * Get a witness report for an agent's substrate.
 * Used when the agent issues [WITNESS_SELF].
 */
export async function getWitnessReport(
  agentId: string,
): Promise<string | null> {
  const client = getOREClient();
  try {
    if (!client.isConnected) return null;
    return await client.witness(agentId);
  } catch {
    return null;
  }
}
