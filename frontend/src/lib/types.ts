/**
 * Mirrors the FastAPI backend contract (app/schemas/*, app/models/*).
 * Datetimes are ISO 8601 strings, as serialized by Pydantic.
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------
export type MachineStatus = "running" | "stopped" | "maintenance" | "fault";

export type SensorType = "temperature" | "vibration" | "pressure" | "rpm" | "power";

export type ComponentType =
  | "motor"
  | "bearing"
  | "pump"
  | "spindle"
  | "gearbox"
  | "belt"
  | "valve"
  | "cylinder"
  | "roller"
  | "controller"
  | "sensor"
  | "frame"
  | "other";

export type ReadingQuality = "GOOD" | "SUSPECT" | "BAD";
export type ReadingSource = "SIMULATOR" | "OPCUA" | "MQTT" | "DATASET";
export type SensorStatus = "NORMAL" | "WARNING" | "CRITICAL" | "UNKNOWN";
export type TrendDirection = "IMPROVING" | "STABLE" | "DEGRADING";

export type ModuleName =
  "RESOLVER" | "RAG" | "PDM" | "RCA" | "MAINTENANCE" | "INVENTORY" | "SAFETY" | "PRODUCTION";

export type Intent =
  "REPORT_FAULT" | "ASK_QUESTION" | "CHECK_STATUS" | "REQUEST_PROCEDURE" | "ASSESS_IMPACT";

export type Urgency = "LOW" | "NORMAL" | "HIGH" | "CRITICAL";
export type UserRole = "TECHNICIAN" | "ENGINEER" | "MANAGER" | "SAFETY_OFFICER";
export type ModuleStatus = "OK" | "PARTIAL" | "UNAVAILABLE" | "SKIPPED" | "REUSED";
export type OrchestrationStatus = "COMPLETE" | "PARTIAL" | "CLARIFICATION_NEEDED" | "NOT_FOUND";
export type NarrativeSource = "LLM" | "TEMPLATE";
export type ResolutionStatus = "RESOLVED" | "AMBIGUOUS" | "NOT_FOUND";

export type FaultType =
  | "BEARING_WEAR"
  | "MOTOR_OVERHEAT"
  | "LUBRICATION_LOSS"
  | "BELT_MISALIGNMENT"
  | "SEAL_LEAK"
  | "TOOL_WEAR";

export type DocType =
  | "MANUAL"
  | "SOP"
  | "MAINTENANCE_GUIDE"
  | "REPAIR_HISTORY"
  | "INCIDENT_REPORT"
  | "SPEC_SHEET"
  | "TROUBLESHOOTING";

export type DocumentStatus = "PENDING" | "PROCESSING" | "INDEXED" | "FAILED";

// ---------------------------------------------------------------------------
// Machine hierarchy (app/schemas/machine.py)
// ---------------------------------------------------------------------------
export interface Machine {
  machine_id: string;
  name: string;
  model: string;
  manufacturer: string;
  site_id: string;
  line_id: string;
  position_in_line: number;
  criticality: number;
  status: MachineStatus;
  aliases: string[];
  installed_at: string | null;
  last_maintenance_at: string | null;
}

export interface Component {
  component_id: string;
  machine_id: string;
  name: string;
  type: ComponentType;
  part_number: string | null;
  parent_component_id: string | null;
}

export interface Sensor {
  sensor_id: string;
  machine_id: string;
  component_id: string | null;
  type: SensorType;
  unit: string;
  normal_min: number;
  normal_max: number;
  warning_threshold: number;
  critical_threshold: number;
}

// ---------------------------------------------------------------------------
// Readings & health (app/schemas/sensor.py)
// ---------------------------------------------------------------------------
export interface Reading {
  sensor_id: string;
  machine_id: string;
  component_id: string | null;
  sensor_type: SensorType;
  value: number;
  unit: string;
  timestamp: string;
  quality: ReadingQuality;
  source: ReadingSource;
}

export interface SensorHealth {
  sensor_id: string;
  sensor_type: SensorType;
  value: number | null;
  unit: string;
  timestamp: string | null;
  status: SensorStatus;
  score: number;
  normal_min: number;
  normal_max: number;
  warning_threshold: number;
  critical_threshold: number;
}

export interface MachineHealth {
  machine_id: string;
  name: string;
  status: SensorStatus;
  health_score: number;
  sensor_count: number;
  stale: boolean;
  sensors: SensorHealth[];
  last_updated: string | null;
}

export interface HistoryPoint {
  timestamp: string;
  value: number;
  quality: ReadingQuality;
}

export interface SensorHistory {
  sensor_id: string;
  machine_id: string | null;
  sensor_type: SensorType | null;
  unit: string | null;
  minutes: number;
  count: number;
  points: HistoryPoint[];
}

// ---------------------------------------------------------------------------
// Predictive maintenance (app/schemas/pdm.py)
// ---------------------------------------------------------------------------
export interface ContributingFeature {
  name: string;
  value: number;
  importance: number;
}

export interface PdmPrediction {
  machine_id: string;
  failure_probability: number;
  remaining_useful_life_hours: number;
  health_score: number;
  predicted_failure_time: string | null;
  predicted_failure_mode: string | null;
  confidence: number;
  contributing_features: ContributingFeature[];
  trend_direction: TrendDirection;
  readings_used: number;
  channels_present: string[];
  generated_at: string;
}

export interface FleetEntry {
  machine_id: string;
  name: string;
  line_id: string;
  prediction: PdmPrediction | null;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Simulator (app/sensors/simulator.py, app/schemas/sensor.py)
// ---------------------------------------------------------------------------
export interface ActiveFault {
  fault_type: FaultType;
  severity: number;
  progression_rate: number;
  onset_runtime_hours: number;
  target_failure_hours: number;
  description: string;
}

export interface MachineSimState {
  machine_id: string;
  health: number;
  runtime_hours: number;
  load_factor: number;
  sim_seconds: number;
  active_faults: ActiveFault[];
}

export interface SimulatorState {
  source: string;
  time_scale: number;
  interval_seconds: number;
  machines: MachineSimState[];
}

export interface InjectFaultResponse {
  injected: ActiveFault;
  machine: MachineSimState;
}

export interface ClearFaultResponse {
  cleared: boolean;
  fault_type: string;
  machine: MachineSimState;
}

export interface ResetSimulatorResponse {
  reset: boolean;
  machines: MachineSimState[];
}

// ---------------------------------------------------------------------------
// Knowledge base (app/schemas/knowledge.py)
// ---------------------------------------------------------------------------
export interface KnowledgeStatus {
  backend: string;
  backend_reason: string;
  embedding_model: string;
  embedding_dimension: number;
  chunk_count: number;
  document_count: number;
  tenant_id: string;
}

export interface SearchRequest {
  query: string;
  machine_id?: string | null;
  doc_types?: DocType[] | null;
  top_k?: number | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  document_title: string | null;
  text: string;
  score: number;
  vector_score: number;
  keyword_score: number;
  page_number: number | null;
  section_title: string | null;
  doc_type: string | null;
  machine_ids: string[];
  machine_models: string[];
  is_table: boolean;
  matched_terms: string[];
}

export interface SearchResponse {
  query: string;
  chunks: RetrievedChunk[];
  backend_used: string;
  total_candidates: number;
  machine_filter_applied: boolean;
  machine_id: string | null;
  machine_model: string | null;
  doc_types: string[] | null;
  embedding_model: string | null;
  reason: string | null;
}

// ---------------------------------------------------------------------------
// Domain agent outputs (app/schemas/agents.py)
// ---------------------------------------------------------------------------
export type ProcedureSource = "DOCUMENTED" | "DERIVED";
export type SkillLevel = "BASIC" | "INTERMEDIATE" | "SPECIALIST";

export interface ProcedureCitation {
  document_id: string | null;
  page_number: number | null;
}

export interface ProcedureStep {
  order: number;
  instruction: string;
  component_id: string | null;
  tools_required: string[];
  estimated_minutes: number;
  caution: string | null;
  citation: ProcedureCitation | null;
}

export interface RequiredPart {
  part_number: string;
  description: string;
  quantity: number;
  component_id: string | null;
}

export interface MaintenancePlan {
  procedure_steps: ProcedureStep[];
  required_parts: RequiredPart[];
  required_tools: string[];
  total_estimated_minutes: number;
  skill_level: SkillLevel;
  procedure_source: ProcedureSource;
}

export type StockStatus = "IN_STOCK" | "LOW_STOCK" | "OUT_OF_STOCK";

export interface InventoryItem {
  part_number: string;
  description: string;
  required_qty: number;
  available_qty: number;
  status: StockStatus;
  location: string | null;
  alternatives: string[];
  lead_time_days: number;
}

export interface InventoryStatus {
  items: InventoryItem[];
  all_parts_available: boolean;
  blocking_parts: string[];
  earliest_full_availability_days: number;
}

export type HazardSeverity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type EnergySourceType =
  "ELECTRICAL" | "HYDRAULIC" | "PNEUMATIC" | "THERMAL" | "MECHANICAL" | "CHEMICAL";
export type SafetySource = "DOCUMENTED" | "GENERIC";

export interface Hazard {
  hazard_type: string;
  description: string;
  severity: HazardSeverity;
  source_component_id: string | null;
}

export interface LotoStep {
  order: number;
  instruction: string;
  verification: string;
}

export interface EnergySource {
  type: EnergySourceType;
  location: string;
  isolation_method: string;
}

export interface SafetyBriefing {
  hazards: Hazard[];
  required_ppe: string[];
  lockout_tagout_steps: LotoStep[];
  energy_sources_to_isolate: EnergySource[];
  permits_required: string[];
  blocking_conditions: string[];
  standard_preconditions: string[];
  citations: Record<string, unknown>[];
  source: SafetySource;
}

export type DowntimeRecommendation = "REPAIR_NOW" | "SCHEDULE_NEXT_WINDOW" | "MONITOR";

export interface DowntimeEstimate {
  repair_time: number;
  total_including_parts_wait: number;
}

export interface CostEstimate {
  downtime_cost: number;
  parts_cost: number;
  total: number;
  currency: string;
}

export interface ProductionImpact {
  downtime_estimate_minutes: DowntimeEstimate;
  units_lost_estimate: number;
  is_bottleneck: boolean;
  downstream_machines_affected: string[];
  cost_estimate: CostEstimate;
  recommendation: DowntimeRecommendation;
  recommendation_rationale: string;
  assumptions: string[];
}

// ---------------------------------------------------------------------------
// Root-cause analysis (app/schemas/rca.py)
// ---------------------------------------------------------------------------
export type EvidenceSource = "SENSOR" | "PDM_MODEL" | "HISTORY" | "DOCUMENT" | "THRESHOLD";
export type EvidenceStrength = "WEAK" | "MODERATE" | "STRONG";

export interface RcaCitation {
  document_id: string;
  page_number: number | null;
  section_title: string | null;
}

export interface Evidence {
  evidence_id: string;
  source: EvidenceSource;
  description: string;
  strength: EvidenceStrength;
  value: string | number | null;
  citation: RcaCitation | null;
}

export interface CausalHypothesis {
  cause_id: string;
  description: string;
  component_id: string | null;
  fault_mode: string;
  probability: number;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
}

export interface CausalStep {
  order: number;
  description: string;
  mechanism: string;
  evidence_ids: string[];
  sensor_signals: string[];
}

export interface RCAResult {
  machine_id: string;
  primary_cause: CausalHypothesis | null;
  alternative_causes: CausalHypothesis[];
  causal_chain: CausalStep[];
  evidence: Evidence[];
  confidence: number;
  confidence_basis: string;
  analysis_timestamp: string;
  insufficient_data: boolean;
  missing_data: string[];
  narrative_generated: boolean;
  tenant_id: string | null;
}

// ---------------------------------------------------------------------------
// Orchestration result (app/schemas/orchestration.py)
// ---------------------------------------------------------------------------
export interface ModuleRun {
  name: ModuleName;
  status: ModuleStatus;
  elapsed_ms: number;
  reason: string | null;
  error_detail: string | null;
  degraded_inputs: string[];
  reused: boolean;
}

export type RoutingSource =
  | "LLM"
  | "FALLBACK_NO_CLIENT"
  | "FALLBACK_TIMEOUT"
  | "FALLBACK_INVALID"
  | "FALLBACK_ERROR"
  | "EXPLICIT";

export interface RoutingDecision {
  selected_modules: ModuleName[];
  reasoning: string;
  source: RoutingSource;
  intent: Intent;
  urgency: Urgency;
  machine_reference: string | null;
  dropped_modules: string[];
}

export interface MachineSummary {
  machine_id: string;
  name: string;
  model: string;
  line_id: string;
  status: string;
}

export interface ClarificationCandidate {
  machine_id: string;
  name: string;
  model: string;
  line_id: string;
  status: string;
  confidence: number;
  matched_by: string | null;
}

export interface Clarification {
  question: string;
  candidates: ClarificationCandidate[];
}

/** Shared by OrchestrationResult.citations and the SSE `citation` event. */
export interface Citation {
  document_id: string;
  title: string | null;
  page_number: number | null;
  section_title: string | null;
}

export interface BlockedStep {
  order: number;
  instruction: string;
  blocked_by_parts: string[];
}

export interface OrchestrationResult {
  request_id: string;
  tenant_id: string;
  user_role: UserRole;
  intent: Intent;
  urgency: Urgency;
  status: OrchestrationStatus;
  machine: MachineSummary | null;
  clarification: Clarification | null;
  narrative: string;
  narrative_source: NarrativeSource;
  detected_language: string;
  language_fallback: boolean;
  modules_run: ModuleRun[];
  routing_decision: RoutingDecision;
  rca: RCAResult | null;
  pdm: PdmPrediction | null;
  maintenance: MaintenancePlan | null;
  inventory: InventoryStatus | null;
  safety: SafetyBriefing | null;
  production: ProductionImpact | null;
  citations: Citation[];
  conflicts_surfaced: string[];
  total_elapsed_ms: number;
  session_id: string | null;
  safety_critical: boolean;
  safety_clearance_required: boolean;
  provisional: boolean;
  blocked_steps: BlockedStep[];
  omitted_for_role: string[];
  generated_at: string;
}

// ---------------------------------------------------------------------------
// Sessions (app/schemas/orchestration.py)
// ---------------------------------------------------------------------------
export interface SessionTurn {
  request_id: string;
  message: string;
  user_role: UserRole;
  machine_id: string | null;
  status: OrchestrationStatus;
  narrative: string;
  modules_run: string[];
  at: string;
}

export interface Session {
  session_id: string;
  tenant_id: string;
  last_machine_id: string | null;
  turns: SessionTurn[];
  created_at: string | null;
  updated_at: string | null;
  cached_modules: string[];
  cache_age_seconds: number | null;
}

// ---------------------------------------------------------------------------
// Chat / SSE (app/schemas/stream.py)
// ---------------------------------------------------------------------------
export interface ChatRequest {
  message: string;
  session_id?: string;
  user_role: UserRole;
  machine_id?: string;
  language?: string;
}

export interface SessionEventData {
  session_id: string;
  request_id: string;
}

export interface RoutingEventData {
  selected_modules: ModuleName[];
  intent: Intent;
  urgency: Urgency;
  reasoning: string;
}

/** The confirmed machine, flattened to what a header needs. */
export interface ResolutionMachine {
  machine_id: string;
  name: string;
  model: string;
}

/** One machine the operator might have meant, when the reference is ambiguous. */
export interface ResolutionCandidate {
  machine_id: string;
  name: string;
  model: string;
  line_id: string | null;
  status: string | null;
  confidence: number;
  matched_by: string | null;
}

export interface ResolutionEventData {
  status: ResolutionStatus;
  machine: ResolutionMachine | null;
  candidates: ResolutionCandidate[];
  clarification_question: string | null;
}

export interface ModuleStartEventData {
  module: ModuleName;
  level: number;
}

export interface ModuleFinishEventData {
  module: ModuleName;
  status: ModuleStatus;
  elapsed_ms: number;
  reason: string | null;
  summary: string;
}

export interface NarrativeDeltaEventData {
  text: string;
}

export interface ConflictEventData {
  description: string;
}

export interface ErrorEventData {
  message: string;
  recoverable: boolean;
}

export interface DoneEventData {
  total_elapsed_ms: number;
}

/** One SSE frame from POST /chat/stream, discriminated on `type`. */
export type StreamEvent =
  | { type: "session"; data: SessionEventData }
  | { type: "routing"; data: RoutingEventData }
  | { type: "resolution"; data: ResolutionEventData }
  | { type: "module_start"; data: ModuleStartEventData }
  | { type: "module_finish"; data: ModuleFinishEventData }
  | { type: "narrative_delta"; data: NarrativeDeltaEventData }
  | { type: "citation"; data: Citation }
  | { type: "conflict"; data: ConflictEventData }
  | { type: "result"; data: OrchestrationResult }
  | { type: "error"; data: ErrorEventData }
  | { type: "done"; data: DoneEventData };

export type StreamEventType = StreamEvent["type"];
