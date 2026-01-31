/**
 * ORE Sidecar Lifecycle
 * =====================
 * Manages spawning and stopping the Python ORE sidecar process.
 *
 * The sidecar is started when the gateway boots (if substrate is
 * enabled in config) and stopped when the gateway shuts down.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { resolve as pathResolve } from "node:path";
import { existsSync } from "node:fs";
import { getOREClient } from "./client.js";

const DEFAULT_PORT = 9780;
const STARTUP_TIMEOUT_MS = 15_000;
const HEALTH_CHECK_INTERVAL_MS = 2_000;

let sidecarProcess: ChildProcess | null = null;
let sidecarPort = DEFAULT_PORT;

/**
 * Resolve the path to the ORE server module.
 */
function resolveOREPath(): string | null {
  // Look relative to the project root
  const candidates = [
    pathResolve(process.cwd(), "ore", "server.py"),
    pathResolve(__dirname, "..", "..", "ore", "server.py"),
    pathResolve(__dirname, "..", "..", "..", "ore", "server.py"),
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

/**
 * Find a suitable Python binary.
 */
function findPython(): string {
  // Prefer python3, fall back to python
  return "python3";
}

/**
 * Start the ORE sidecar process.
 *
 * Returns true if the sidecar started and is responding to pings.
 */
export async function startSidecar(opts?: {
  port?: number;
  stateDir?: string;
  log?: (...args: unknown[]) => void;
}): Promise<boolean> {
  const log = opts?.log ?? console.log;
  const port = opts?.port ?? DEFAULT_PORT;
  sidecarPort = port;

  // Check if already running
  const client = getOREClient(port);
  const alreadyRunning = await client.connect();
  if (alreadyRunning) {
    const alive = await client.ping();
    if (alive) {
      log("[ore] sidecar already running on port", port);
      return true;
    }
  }

  // Find ORE server
  const serverPath = resolveOREPath();
  if (!serverPath) {
    log("[ore] ore/server.py not found — substrate disabled");
    return false;
  }

  // Build args
  const args = ["-m", "ore.server", "--port", String(port)];
  if (opts?.stateDir) {
    args.push("--state-dir", opts.stateDir);
  }

  const python = findPython();
  const cwd = pathResolve(serverPath, "..", "..");

  log(`[ore] starting sidecar: ${python} ${args.join(" ")}`);
  log(`[ore] cwd: ${cwd}`);

  try {
    sidecarProcess = spawn(python, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    });

    // Log output
    sidecarProcess.stdout?.on("data", (data) => {
      const line = data.toString().trim();
      if (line) log(`[ore] ${line}`);
    });

    sidecarProcess.stderr?.on("data", (data) => {
      const line = data.toString().trim();
      if (line) log(`[ore:err] ${line}`);
    });

    sidecarProcess.on("exit", (code) => {
      log(`[ore] sidecar exited with code ${code}`);
      sidecarProcess = null;
    });

    // Wait for sidecar to be ready
    const ready = await waitForSidecar(port, STARTUP_TIMEOUT_MS);
    if (ready) {
      log(`[ore] sidecar ready on port ${port}`);
      return true;
    } else {
      log("[ore] sidecar failed to start within timeout");
      stopSidecar();
      return false;
    }
  } catch (err) {
    log(`[ore] failed to spawn sidecar: ${err}`);
    return false;
  }
}

/**
 * Stop the ORE sidecar process.
 */
export function stopSidecar(): void {
  if (sidecarProcess) {
    try {
      // Send shutdown message first for clean exit
      const client = getOREClient(sidecarPort);
      if (client.isConnected) {
        // Fire and forget
        client.disconnect();
      }
      sidecarProcess.kill("SIGTERM");
    } catch {
      // ignore
    }
    sidecarProcess = null;
  }
}

/**
 * Wait for the sidecar to become responsive.
 */
async function waitForSidecar(port: number, timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  const client = getOREClient(port);

  while (Date.now() - start < timeoutMs) {
    try {
      const connected = await client.connect();
      if (connected) {
        const alive = await client.ping();
        if (alive) return true;
      }
    } catch {
      // not ready yet
    }
    await sleep(HEALTH_CHECK_INTERVAL_MS);
  }

  return false;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
