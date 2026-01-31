/**
 * ORE Sidecar Client
 * ==================
 * WebSocket client for communicating with the ORE Python sidecar.
 *
 * Manages connection lifecycle, reconnection, and provides typed
 * request/response methods for substrate operations.
 */

import WebSocket from "ws";
import type {
  SubstrateAgentConfig,
  FullSubstrateState,
  SubstrateCommand,
  OREMessage,
  OREInitResponse,
  OREPromptContextResponse,
  OREProcessResponse,
} from "./types.js";

const DEFAULT_ORE_PORT = 9780;
const CONNECT_TIMEOUT_MS = 5_000;
const REQUEST_TIMEOUT_MS = 10_000;
const RECONNECT_DELAY_MS = 2_000;
const MAX_RECONNECT_ATTEMPTS = 5;

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  timer: ReturnType<typeof setTimeout>;
};

export class OREClient {
  private ws: WebSocket | null = null;
  private url: string;
  private connected = false;
  private reconnectAttempts = 0;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private requestSeq = 0;
  private messageBuffer: Array<{ msg: OREMessage; id: string }> = [];
  private onDisconnect?: () => void;

  constructor(port: number = DEFAULT_ORE_PORT, host: string = "127.0.0.1") {
    this.url = `ws://${host}:${port}`;
  }

  // ─── Connection ────────────────────────────────────────────────

  async connect(): Promise<boolean> {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      return true;
    }

    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        resolve(false);
      }, CONNECT_TIMEOUT_MS);

      try {
        this.ws = new WebSocket(this.url);

        this.ws.on("open", () => {
          clearTimeout(timer);
          this.connected = true;
          this.reconnectAttempts = 0;
          this._flushBuffer();
          resolve(true);
        });

        this.ws.on("message", (data) => {
          this._handleMessage(data.toString());
        });

        this.ws.on("close", () => {
          this.connected = false;
          this.onDisconnect?.();
        });

        this.ws.on("error", () => {
          clearTimeout(timer);
          this.connected = false;
          resolve(false);
        });
      } catch {
        clearTimeout(timer);
        resolve(false);
      }
    });
  }

  async ensureConnected(): Promise<boolean> {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      return true;
    }

    for (let i = 0; i < MAX_RECONNECT_ATTEMPTS; i++) {
      const ok = await this.connect();
      if (ok) return true;
      await sleep(RECONNECT_DELAY_MS * (i + 1));
    }

    return false;
  }

  disconnect(): void {
    this.connected = false;
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }

    // Reject all pending requests
    for (const [, pending] of this.pendingRequests) {
      clearTimeout(pending.timer);
      pending.reject(new Error("disconnected"));
    }
    this.pendingRequests.clear();
  }

  get isConnected(): boolean {
    return this.connected && this.ws?.readyState === WebSocket.OPEN;
  }

  // ─── Request/Response ──────────────────────────────────────────

  private _send(msg: OREMessage, requestId?: string): void {
    const payload = requestId ? { ...msg, _reqId: requestId } : msg;
    const raw = JSON.stringify(payload);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(raw);
    } else if (requestId) {
      // Buffer for when connection is restored
      this.messageBuffer.push({ msg, id: requestId });
    }
  }

  private _flushBuffer(): void {
    const buffered = [...this.messageBuffer];
    this.messageBuffer = [];
    for (const { msg, id } of buffered) {
      this._send(msg, id);
    }
  }

  private _handleMessage(raw: string): void {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return;
    }

    // Route to pending request if it has a _reqId
    const reqId = parsed._reqId as string | undefined;
    if (reqId && this.pendingRequests.has(reqId)) {
      const pending = this.pendingRequests.get(reqId)!;
      clearTimeout(pending.timer);
      this.pendingRequests.delete(reqId);
      pending.resolve(parsed);
      return;
    }

    // For responses without _reqId, resolve the oldest pending request
    // of the matching type
    const type = parsed.type as string;
    for (const [id, pending] of this.pendingRequests) {
      clearTimeout(pending.timer);
      this.pendingRequests.delete(id);
      pending.resolve(parsed);
      return;
    }
  }

  private async _request<T>(msg: OREMessage, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
    const reqId = `req_${++this.requestSeq}`;

    if (!(await this.ensureConnected())) {
      throw new Error("ORE sidecar not available");
    }

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(reqId);
        reject(new Error(`ORE request timed out: ${msg.type}`));
      }, timeoutMs);

      this.pendingRequests.set(reqId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer,
      });

      this._send(msg, reqId);
    });
  }

  // ─── Public API ────────────────────────────────────────────────

  /**
   * Initialize a substrate for an agent.
   * Idempotent — returns existing substrate if already initialized.
   */
  async initAgent(agentId: string, config: SubstrateAgentConfig): Promise<OREInitResponse> {
    return this._request<OREInitResponse>({
      type: "init",
      agentId,
      config,
    });
  }

  /**
   * Get the full substrate state for an agent.
   */
  async getState(agentId: string): Promise<FullSubstrateState> {
    return this._request<FullSubstrateState>({
      type: "get_state",
      agentId,
    });
  }

  /**
   * Get formatted prompt context string for system prompt injection.
   * This is the primary integration point — call this before each LLM turn.
   */
  async getPromptContext(agentId: string): Promise<string> {
    const resp = await this._request<OREPromptContextResponse>({
      type: "get_prompt_context",
      agentId,
    });
    return resp.context;
  }

  /**
   * Process an agent response for substrate commands.
   * Call this after each LLM response to handle REMEMBER, WITNESS, etc.
   */
  async processResponse(agentId: string, response: string): Promise<SubstrateCommand[]> {
    const resp = await this._request<OREProcessResponse>({
      type: "process_response",
      agentId,
      response,
    });
    return resp.commandsFound;
  }

  /**
   * Trigger a chemistry event (e.g. on_conversation_start, on_discovery).
   */
  async chemistryEvent(agentId: string, event: string, kwargs?: Record<string, unknown>): Promise<void> {
    await this._request({
      type: "chemistry_event",
      agentId,
      event,
      kwargs,
    });
  }

  /**
   * Get a full witness report for the agent's substrate.
   */
  async witness(agentId: string): Promise<string> {
    const resp = await this._request<{ type: "witness"; report: string }>({
      type: "witness",
      agentId,
    });
    return resp.report;
  }

  /**
   * Stop an agent's substrate and persist state.
   */
  async stopAgent(agentId: string): Promise<void> {
    await this._request({ type: "stop", agentId });
  }

  /**
   * Ping the sidecar to check if it's alive.
   */
  async ping(): Promise<boolean> {
    try {
      await this._request<{ type: "pong" }>({ type: "ping" }, 3_000);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * List all active agent substrates.
   */
  async listAgents(): Promise<string[]> {
    const resp = await this._request<{ type: "agents"; agents: string[] }>({
      type: "list_agents",
    });
    return resp.agents;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ─── Singleton ───────────────────────────────────────────────────

let globalClient: OREClient | null = null;

/**
 * Get or create the global ORE client singleton.
 */
export function getOREClient(port?: number): OREClient {
  if (!globalClient) {
    globalClient = new OREClient(port);
  }
  return globalClient;
}

/**
 * Check if the ORE sidecar is available.
 */
export async function isOREAvailable(port?: number): Promise<boolean> {
  const client = getOREClient(port);
  const ok = await client.connect();
  if (!ok) return false;
  return client.ping();
}
