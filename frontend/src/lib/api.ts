import type {
  ClearFaultResponse,
  Component,
  FleetEntry,
  InjectFaultResponse,
  KnowledgeStatus,
  Machine,
  MachineHealth,
  PdmPrediction,
  Reading,
  ResetSimulatorResponse,
  SearchRequest,
  SearchResponse,
  Sensor,
  SensorHistory,
  Session,
  SimulatorState,
} from "@/lib/types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TENANT_ID = import.meta.env.VITE_TENANT_ID ?? "demo";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "X-Tenant-Id": TENANT_ID,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // response had no JSON body
    }
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(`${res.status} ${path}: ${detail}`, res.status, body);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export function getMachines(): Promise<Machine[]> {
  return request("/machines");
}

export function getMachine(id: string): Promise<Machine> {
  return request(`/machines/${encodeURIComponent(id)}`);
}

export function getComponents(id: string): Promise<Component[]> {
  return request(`/machines/${encodeURIComponent(id)}/components`);
}

export function getSensors(id: string): Promise<Sensor[]> {
  return request(`/machines/${encodeURIComponent(id)}/sensors`);
}

export function getLatestReadings(machineId?: string): Promise<Reading[]> {
  return request(`/sensors/latest${query({ machine_id: machineId })}`);
}

export function getSensorHistory(sensorId: string, minutes: number): Promise<SensorHistory> {
  return request(`/sensors/${encodeURIComponent(sensorId)}/history${query({ minutes })}`);
}

export function getMachineHealth(id: string): Promise<MachineHealth> {
  return request(`/machines/${encodeURIComponent(id)}/health`);
}

export function getFleetPredictions(): Promise<FleetEntry[]> {
  return request("/pdm/fleet");
}

export function getPrediction(id: string): Promise<PdmPrediction> {
  return request(`/pdm/${encodeURIComponent(id)}/prediction`);
}

export function getSimulatorState(): Promise<SimulatorState> {
  return request("/simulator/state");
}

export function injectFault(
  machineId: string,
  faultType: string,
  severity: number,
): Promise<InjectFaultResponse> {
  return request("/simulator/inject-fault", {
    method: "POST",
    body: JSON.stringify({ machine_id: machineId, fault_type: faultType, severity }),
  });
}

export function clearFault(machineId: string, faultType: string): Promise<ClearFaultResponse> {
  return request("/simulator/clear-fault", {
    method: "POST",
    body: JSON.stringify({ machine_id: machineId, fault_type: faultType }),
  });
}

export function resetSimulator(): Promise<ResetSimulatorResponse> {
  return request("/simulator/reset", { method: "POST" });
}

export function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  return request("/knowledge/status");
}

export function searchKnowledge(body: SearchRequest): Promise<SearchResponse> {
  return request("/knowledge/search", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getSession(id: string): Promise<Session> {
  return request(`/chat/sessions/${encodeURIComponent(id)}`);
}
