"""Physics-informed machine simulator.

Not random noise. Every reading is composed from named physical contributions:

    value = baseline + diurnal_drift + load_variation + noise + degradation

The part that matters is ``degradation``. Faults are *mechanisms*, not per-sensor
scripts: one fault drives several sensors through coefficients and lags, so the
signals move together the way they do on a real machine. Bearing wear raises
vibration first and only later shows as heat, because the friction has to
develop before it can warm anything. Temperature never rises on its own.

Determinism
-----------
The whole model is a pure function of *simulated* time. Noise is drawn from a
generator seeded on ``(seed, sensor_id, sim_seconds)``, so two simulators
advanced to the same simulated instant produce byte-identical readings. That is
what makes ``time_scale`` honest: it changes only how fast simulated time
advances per wall-clock second, never what the readings are when you get there.
A three-week bearing failure replayed in two minutes passes through exactly the
same values as the real-time run.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence

from app.models.machine import ComponentType, SensorType
from app.models.reading import ReadingQuality, ReadingSource, SensorReading, utcnow
from app.sensors.base import SensorSource

SECONDS_PER_HOUR = 3600.0


# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------
class FaultType(str, Enum):
    """Failure mechanisms the simulator can develop."""

    bearing_wear = "BEARING_WEAR"
    motor_overheat = "MOTOR_OVERHEAT"
    lubrication_loss = "LUBRICATION_LOSS"
    belt_misalignment = "BELT_MISALIGNMENT"
    seal_leak = "SEAL_LEAK"
    tool_wear = "TOOL_WEAR"


@dataclass(frozen=True)
class SensorEffect:
    """How one fault moves one sensor type.

    ``coefficient``
        Signed, expressed in multiples of the sensor's normal span, applied at
        severity 1.0. Negative means the reading falls (rpm under load, pressure
        on a leak).
    ``lag``
        Severity below which this sensor does not respond at all. This is what
        encodes causal ordering: temperature with ``lag=0.35`` physically cannot
        move until the vibration that generates the friction is well developed.
    ``curve``
        ``"exponential"`` for mechanisms that accelerate (crack growth, wear
        debris), ``"linear"`` for those that creep.
    ``instability``
        Extra noise, in multiples of span, that scales with severity — a
        misaligned belt makes rpm hunt rather than simply drop.
    """

    coefficient: float
    lag: float = 0.0
    curve: str = "linear"
    instability: float = 0.0

    def contribution(self, severity: float) -> float:
        """Fraction of ``coefficient`` active at ``severity`` (0.0 before lag)."""
        if severity <= self.lag:
            return 0.0
        local = (severity - self.lag) / max(1e-9, 1.0 - self.lag)
        local = min(1.0, max(0.0, local))
        if self.curve == "exponential":
            # Normalised e^kx so the curve still reaches exactly 1.0 at severity 1.
            k = 3.0
            return (math.exp(k * local) - 1.0) / (math.exp(k) - 1.0)
        return local


@dataclass(frozen=True)
class FaultModel:
    """A failure mechanism: what it affects, how fast, and in what order."""

    fault_type: FaultType
    description: str
    component_types: tuple[ComponentType, ...]
    effects: Mapping[SensorType, SensorEffect]
    #: Runtime hours from onset to failure at ``progression_rate == 1.0``.
    target_failure_hours: float
    #: Health consumed at severity 1.0. Not every mechanism is equally fatal —
    #: a seized bearing stops the machine (1.0), a dull tool is a consumable
    #: that degrades the part before it threatens the spindle (0.6).
    health_impact: float = 1.0


#: The fault library, keyed by the component types that suffer each mechanism.
#: Coefficients are tuned so a fully developed fault drives its primary sensor
#: past the critical threshold on every machine in the seed fleet.
FAULT_MODELS: dict[FaultType, FaultModel] = {
    FaultType.bearing_wear: FaultModel(
        fault_type=FaultType.bearing_wear,
        description=(
            "Rolling-element surface fatigue. Vibration rises first and "
            "accelerates as spalling spreads; friction then heats the housing; "
            "drag pulls rpm down slightly and raises draw."
        ),
        component_types=(ComponentType.bearing, ComponentType.roller),
        effects={
            SensorType.vibration: SensorEffect(2.6, lag=0.00, curve="exponential"),
            SensorType.temperature: SensorEffect(1.3, lag=0.35, curve="linear"),
            SensorType.power: SensorEffect(0.35, lag=0.45, curve="linear"),
            SensorType.rpm: SensorEffect(-0.12, lag=0.50, curve="linear"),
        },
        target_failure_hours=504.0,  # three weeks
        health_impact=1.0,
    ),
    FaultType.motor_overheat: FaultModel(
        fault_type=FaultType.motor_overheat,
        description=(
            "Winding/cooling degradation. Temperature climbs, resistive losses "
            "push power up with it, and the motor loses speed under load."
        ),
        component_types=(ComponentType.motor,),
        effects={
            SensorType.temperature: SensorEffect(1.5, lag=0.00, curve="exponential"),
            SensorType.power: SensorEffect(0.70, lag=0.10, curve="linear"),
            SensorType.rpm: SensorEffect(-0.25, lag=0.40, curve="linear"),
            SensorType.vibration: SensorEffect(0.40, lag=0.60, curve="linear"),
        },
        target_failure_hours=72.0,
        health_impact=1.0,
    ),
    FaultType.lubrication_loss: FaultModel(
        fault_type=FaultType.lubrication_loss,
        description=(
            "Oil film breakdown. Metal-to-metal friction raises heat and "
            "vibration together — the two rise almost in step, which is what "
            "distinguishes it from bearing wear."
        ),
        component_types=(
            ComponentType.bearing,
            ComponentType.gearbox,
            ComponentType.pump,
            ComponentType.spindle,
        ),
        effects={
            SensorType.temperature: SensorEffect(1.40, lag=0.05, curve="exponential"),
            SensorType.vibration: SensorEffect(1.80, lag=0.15, curve="exponential"),
            SensorType.power: SensorEffect(0.50, lag=0.30, curve="linear"),
        },
        target_failure_hours=240.0,
        health_impact=0.95,
    ),
    FaultType.belt_misalignment: FaultModel(
        fault_type=FaultType.belt_misalignment,
        description=(
            "Belt running off-track. Vibration rises steadily, drag costs a "
            "little power, and speed becomes unstable rather than merely low."
        ),
        component_types=(ComponentType.belt, ComponentType.roller),
        effects={
            SensorType.vibration: SensorEffect(2.20, lag=0.00, curve="linear"),
            SensorType.power: SensorEffect(0.45, lag=0.20, curve="linear"),
            SensorType.rpm: SensorEffect(-0.08, lag=0.10, curve="linear", instability=0.06),
        },
        target_failure_hours=336.0,
        health_impact=0.70,
    ),
    FaultType.seal_leak: FaultModel(
        fault_type=FaultType.seal_leak,
        description=(
            "Seal wear on a pressurised circuit. Pressure bleeds off gradually "
            "while the pump works harder to hold setpoint, warming the fluid."
        ),
        component_types=(ComponentType.cylinder, ComponentType.valve, ComponentType.pump),
        effects={
            SensorType.pressure: SensorEffect(-0.85, lag=0.00, curve="linear"),
            SensorType.power: SensorEffect(0.55, lag=0.25, curve="linear"),
            SensorType.temperature: SensorEffect(0.50, lag=0.40, curve="linear"),
        },
        target_failure_hours=168.0,
        health_impact=0.85,
    ),
    FaultType.tool_wear: FaultModel(
        fault_type=FaultType.tool_wear,
        description=(
            "Cutting edge dulling. Chatter raises vibration, cutting force "
            "raises spindle power, and both degrade surface quality — the "
            "vibration/power ratio is the usable proxy for finish."
        ),
        component_types=(ComponentType.spindle, ComponentType.other),
        effects={
            SensorType.vibration: SensorEffect(1.90, lag=0.00, curve="exponential"),
            SensorType.power: SensorEffect(0.80, lag=0.10, curve="linear"),
            SensorType.temperature: SensorEffect(0.60, lag=0.45, curve="linear"),
            SensorType.rpm: SensorEffect(-0.10, lag=0.50, curve="linear", instability=0.03),
        },
        target_failure_hours=48.0,
        health_impact=0.60,
    ),
}


#: Which mechanisms a given component type can develop.
FAULTS_BY_COMPONENT_TYPE: dict[ComponentType, tuple[FaultType, ...]] = {}
for _model in FAULT_MODELS.values():
    for _component_type in _model.component_types:
        FAULTS_BY_COMPONENT_TYPE.setdefault(_component_type, ())
        FAULTS_BY_COMPONENT_TYPE[_component_type] += (_model.fault_type,)


# ---------------------------------------------------------------------------
# Healthy-machine behaviour, per sensor type
# ---------------------------------------------------------------------------
#: Amplitude of the 24-hour cycle, in multiples of span. Temperature follows
#: ambient; a spindle's speed does not care what time it is.
DIURNAL_AMPLITUDE: dict[SensorType, float] = {
    SensorType.temperature: 0.10,
    SensorType.vibration: 0.02,
    SensorType.pressure: 0.02,
    SensorType.rpm: 0.01,
    SensorType.power: 0.02,
}

#: How strongly production load moves each sensor, in multiples of span.
LOAD_SENSITIVITY: dict[SensorType, float] = {
    SensorType.power: 0.55,
    SensorType.temperature: 0.30,
    SensorType.pressure: 0.20,
    SensorType.vibration: 0.15,
    SensorType.rpm: 0.10,
}

#: Measurement noise (1 sigma), in multiples of span.
NOISE_SIGMA: dict[SensorType, float] = {
    SensorType.vibration: 0.030,
    SensorType.pressure: 0.015,
    SensorType.temperature: 0.012,
    SensorType.power: 0.015,
    SensorType.rpm: 0.010,
}

#: Generic wear applied via ``set_health`` when no fault explains the damage.
HEALTH_WEAR_COEFFICIENT: dict[SensorType, float] = {
    SensorType.vibration: 0.55,
    SensorType.temperature: 0.40,
    SensorType.power: 0.25,
    SensorType.pressure: -0.15,
    SensorType.rpm: -0.10,
}

#: Sensors that cannot physically read below zero.
_NON_NEGATIVE = frozenset(
    {SensorType.vibration, SensorType.rpm, SensorType.power, SensorType.pressure}
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class ActiveFault:
    """A fault developing on a machine."""

    fault_type: FaultType
    severity: float = 0.1
    progression_rate: float = 1.0
    onset_runtime_hours: float = 0.0

    @property
    def model(self) -> FaultModel:
        return FAULT_MODELS[self.fault_type]

    def advance(self, delta_hours: float) -> None:
        """Progress toward failure. Severity saturates at 1.0 (failed)."""
        if delta_hours <= 0:
            return
        rate = self.progression_rate / max(1e-9, self.model.target_failure_hours)
        self.severity = min(1.0, self.severity + delta_hours * rate)

    def to_dict(self) -> dict:
        return {
            "fault_type": self.fault_type.value,
            "severity": round(self.severity, 4),
            "progression_rate": self.progression_rate,
            "onset_runtime_hours": round(self.onset_runtime_hours, 3),
            "target_failure_hours": self.model.target_failure_hours,
            "description": self.model.description,
        }


@dataclass
class MachineState:
    """Everything the simulator remembers about one machine."""

    machine_id: str
    health: float = 1.0
    runtime_hours: float = 0.0
    load_factor: float = 0.5
    sim_seconds: float = 0.0
    faults: dict[FaultType, ActiveFault] = field(default_factory=dict)
    #: Health set explicitly via ``set_health``, independent of faults.
    manual_health: Optional[float] = None

    def recompute_health(self) -> None:
        """Health falls as faults develop; an explicit override wins."""
        if self.manual_health is not None:
            self.health = self.manual_health
            return
        # Multiplicative, not additive: independent mechanisms each consume a
        # share of what health remains. Two half-developed faults leave the
        # machine impaired (0.25), not dead — additive damage would call that
        # a total failure, which is wrong and would mask the real thing.
        remaining = 1.0
        for fault in self.faults.values():
            remaining *= 1.0 - min(1.0, fault.severity * fault.model.health_impact)
        self.health = round(max(0.0, remaining), 4)

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "health": round(self.health, 4),
            "runtime_hours": round(self.runtime_hours, 3),
            "load_factor": round(self.load_factor, 4),
            "sim_seconds": round(self.sim_seconds, 3),
            "active_faults": [f.to_dict() for f in self.faults.values()],
        }


@dataclass(frozen=True)
class SensorSpec:
    """The sensor definition the simulator needs, decoupled from Mongo."""

    sensor_id: str
    machine_id: str
    sensor_type: SensorType
    unit: str
    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float
    component_id: Optional[str] = None

    @classmethod
    def from_document(cls, doc: Mapping) -> "SensorSpec":
        """Build from a seeded ``Sensor`` document or model dump."""
        return cls(
            sensor_id=str(doc["sensor_id"]),
            machine_id=str(doc["machine_id"]),
            sensor_type=SensorType(doc["type"]),
            unit=str(doc["unit"]),
            normal_min=float(doc["normal_min"]),
            normal_max=float(doc["normal_max"]),
            warning_threshold=float(doc["warning_threshold"]),
            critical_threshold=float(doc["critical_threshold"]),
            component_id=doc.get("component_id"),
        )

    @property
    def span(self) -> float:
        """Normal operating band width; never zero, so it is safe to divide by."""
        return max(abs(self.normal_max - self.normal_min), 1e-6)

    @property
    def baseline(self) -> float:
        """Midpoint of the normal band — where a healthy sensor sits."""
        return (self.normal_min + self.normal_max) / 2.0


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class MachineSimulator(SensorSource):
    """A stateful, physics-informed fleet simulator.

    Pass ``sensors`` explicitly (tests, demos) or leave it ``None`` to load the
    sensor catalogue from MongoDB on ``connect()``.
    """

    source_type = ReadingSource.simulator

    def __init__(
        self,
        tenant_id: str,
        sensors: Optional[Iterable[Mapping]] = None,
        interval_seconds: float = 2.0,
        time_scale: float = 1.0,
        seed: int = 1337,
    ) -> None:
        from app.db import normalize_tenant_id

        self._tenant_id = normalize_tenant_id(tenant_id)
        self._specs: dict[str, SensorSpec] = {}
        self._by_machine: dict[str, list[SensorSpec]] = {}
        self._states: dict[str, MachineState] = {}
        self._connected = False
        self._interval = max(0.001, float(interval_seconds))
        self._time_scale = max(1e-6, float(time_scale))
        self._seed = seed

        if sensors is not None:
            self._load_specs(sensors)

    # -- catalogue ---------------------------------------------------------
    def _load_specs(self, sensors: Iterable[Mapping]) -> None:
        """Index the sensor catalogue and create a state per machine."""
        self._specs.clear()
        self._by_machine.clear()
        for doc in sensors:
            spec = doc if isinstance(doc, SensorSpec) else SensorSpec.from_document(doc)
            self._specs[spec.sensor_id] = spec
            self._by_machine.setdefault(spec.machine_id, []).append(spec)
        for machine_id in self._by_machine:
            self._states.setdefault(machine_id, MachineState(machine_id=machine_id))

    async def _load_specs_from_db(self) -> None:
        """Read the sensor catalogue from Mongo (lazy import keeps tests dry)."""
        from app.db import get_tenant_scope
        from app.schemas.machine import COLLECTIONS

        scope = get_tenant_scope(self._tenant_id)
        docs = [doc async for doc in scope[COLLECTIONS.sensors].find({})]
        self._load_specs(docs)

    # -- SensorSource interface -------------------------------------------
    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def name(self) -> str:
        return "simulator"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        if not self._specs:
            await self._load_specs_from_db()
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def stream(self) -> AsyncIterator[SensorReading]:
        """Emit every sensor of every machine each interval, advancing state."""
        self._require_connected()
        while True:
            for reading in self.tick(self._interval):
                yield reading
            await asyncio.sleep(self._interval)

    async def read_once(self, sensor_id: str) -> SensorReading:
        """Current value of one sensor. Does not advance simulated time."""
        self._require_connected()
        spec = self._specs.get(sensor_id)
        if spec is None:
            raise KeyError(f"Unknown sensor '{sensor_id}'")
        return self._reading_for(spec, self._states[spec.machine_id])

    # -- time & state ------------------------------------------------------
    @property
    def time_scale(self) -> float:
        """Simulated seconds per wall-clock second."""
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale = max(1e-6, float(value))

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def advance(self, wall_seconds: float) -> None:
        """Advance every machine by ``wall_seconds`` of *wall-clock* time.

        Simulated time moves ``wall_seconds * time_scale``; that multiplier is
        the only thing ``time_scale`` touches.
        """
        sim_seconds = wall_seconds * self._time_scale
        delta_hours = sim_seconds / SECONDS_PER_HOUR
        for state in self._states.values():
            state.sim_seconds += sim_seconds
            state.runtime_hours += delta_hours
            for fault in state.faults.values():
                fault.advance(delta_hours)
            state.load_factor = self._load_factor(state.sim_seconds)
            state.recompute_health()

    def tick(self, wall_seconds: Optional[float] = None) -> list[SensorReading]:
        """Advance time and return one reading for every sensor in the fleet."""
        self.advance(self._interval if wall_seconds is None else wall_seconds)
        return [
            self._reading_for(spec, self._states[spec.machine_id])
            for spec in self._specs.values()
        ]

    def readings_for_machine(self, machine_id: str) -> list[SensorReading]:
        """Current readings for one machine, without advancing time."""
        state = self._states.get(machine_id)
        if state is None:
            raise KeyError(f"Unknown machine '{machine_id}'")
        return [self._reading_for(s, state) for s in self._by_machine[machine_id]]

    # -- control surface ---------------------------------------------------
    def inject_fault(
        self,
        machine_id: str,
        fault_type: FaultType | str,
        severity: float = 0.1,
        progression_rate: float = 1.0,
    ) -> ActiveFault:
        """Start (or re-arm) a fault on a machine.

        Re-injecting an existing fault resets its severity and rate rather than
        stacking a second copy of the same mechanism.
        """
        state = self._state_or_raise(machine_id)
        ftype = FaultType(fault_type)
        fault = ActiveFault(
            fault_type=ftype,
            severity=min(1.0, max(0.0, float(severity))),
            progression_rate=max(0.0, float(progression_rate)),
            onset_runtime_hours=state.runtime_hours,
        )
        state.faults[ftype] = fault
        state.recompute_health()
        return fault

    def clear_fault(self, machine_id: str, fault_type: FaultType | str) -> bool:
        """Remove a fault (a repair). Returns False if it was not present."""
        state = self._state_or_raise(machine_id)
        removed = state.faults.pop(FaultType(fault_type), None) is not None
        state.recompute_health()
        return removed

    def set_health(self, machine_id: str, health: float) -> MachineState:
        """Force a machine's health, independent of any modelled fault.

        This is a coarse override for demos and fixtures: it applies generic
        wear to every sensor. Faults remain the physically-modelled path — pass
        ``None`` to hand control back to them.
        """
        state = self._state_or_raise(machine_id)
        state.manual_health = (
            None if health is None else min(1.0, max(0.0, float(health)))
        )
        state.recompute_health()
        return state

    def get_machine_state(self, machine_id: str) -> MachineState:
        return self._state_or_raise(machine_id)

    def get_all_states(self) -> dict[str, MachineState]:
        return dict(self._states)

    def reset(self) -> None:
        """Return every machine to hour zero, full health, no faults."""
        for machine_id in self._states:
            self._states[machine_id] = MachineState(machine_id=machine_id)

    def _state_or_raise(self, machine_id: str) -> MachineState:
        state = self._states.get(machine_id)
        if state is None:
            known = ", ".join(sorted(self._states)) or "<none loaded>"
            raise KeyError(f"Unknown machine '{machine_id}'. Known machines: {known}")
        return state

    # -- the model ---------------------------------------------------------
    @staticmethod
    def _load_factor(sim_seconds: float) -> float:
        """Production load in 0..1, from superposed cycles of simulated time.

        A fast cycle stands in for part-to-part work and a slow one for shift
        rhythm. Deterministic in simulated time so replays are reproducible.
        """
        fast = math.sin(2.0 * math.pi * sim_seconds / 900.0)
        slow = math.sin(2.0 * math.pi * sim_seconds / 13500.0)
        return min(1.0, max(0.0, 0.55 + 0.25 * fast + 0.12 * slow))

    @staticmethod
    def _diurnal(sim_seconds: float) -> float:
        """-1..1 over a 24-hour simulated day, coldest at hour 0."""
        return -math.cos(2.0 * math.pi * sim_seconds / 86400.0)

    def _noise(self, spec: SensorSpec, state: MachineState, sigma: float) -> float:
        """Gaussian noise seeded on (sensor, simulated instant) — reproducible."""
        if sigma <= 0:
            return 0.0
        rng = random.Random(f"{self._seed}|{spec.sensor_id}|{state.sim_seconds:.3f}")
        return rng.gauss(0.0, sigma)

    def _degradation(self, spec: SensorSpec, state: MachineState) -> float:
        """Combined contribution of every active fault, in engineering units.

        Faults sum: a machine with both a leaking seal and a hot motor draws
        more power than either alone, which is the physically right answer.
        """
        total = 0.0
        for fault in state.faults.values():
            effect = fault.model.effects.get(spec.sensor_type)
            if effect is None:
                continue
            active = effect.contribution(fault.severity)
            total += effect.coefficient * spec.span * active
            if effect.instability:
                jitter = effect.instability * spec.span * active
                total += self._noise(spec, state, jitter)
        return total

    def _generic_wear(self, spec: SensorSpec, state: MachineState) -> float:
        """Wear implied by an explicitly set health, with no fault to explain it."""
        if state.manual_health is None:
            return 0.0
        damage = 1.0 - state.health
        if damage <= 0:
            return 0.0
        coefficient = HEALTH_WEAR_COEFFICIENT.get(spec.sensor_type, 0.0)
        return coefficient * spec.span * damage

    def compute_value(self, spec: SensorSpec, state: MachineState) -> float:
        """The full sensor model for one sensor at the machine's current state."""
        span = spec.span
        stype = spec.sensor_type

        diurnal = DIURNAL_AMPLITUDE.get(stype, 0.0) * span * self._diurnal(
            state.sim_seconds
        )
        load = LOAD_SENSITIVITY.get(stype, 0.0) * span * (state.load_factor - 0.5)
        noise = self._noise(spec, state, NOISE_SIGMA.get(stype, 0.01) * span)

        value = (
            spec.baseline
            + diurnal
            + load
            + noise
            + self._degradation(spec, state)
            + self._generic_wear(spec, state)
        )

        if stype in _NON_NEGATIVE:
            value = max(0.0, value)
        return round(value, 4)

    @staticmethod
    def _quality(spec: SensorSpec, value: float) -> ReadingQuality:
        """Flag readings outside the physically plausible envelope.

        An over-critical value is still GOOD quality — the machine is in trouble,
        the sensor is not. SUSPECT is reserved for values no working transducer
        on this range should report.
        """
        span = spec.span
        if value < spec.normal_min - 2.0 * span:
            return ReadingQuality.suspect
        if value > spec.critical_threshold + 1.5 * span:
            return ReadingQuality.suspect
        return ReadingQuality.good

    def _reading_for(self, spec: SensorSpec, state: MachineState) -> SensorReading:
        value = self.compute_value(spec, state)
        return SensorReading(
            tenant_id=self._tenant_id,
            sensor_id=spec.sensor_id,
            machine_id=spec.machine_id,
            component_id=spec.component_id,
            sensor_type=spec.sensor_type,
            value=value,
            unit=spec.unit,
            timestamp=utcnow(),
            quality=self._quality(spec, value),
            source=ReadingSource.simulator,
        )

    # -- introspection -----------------------------------------------------
    @property
    def sensor_specs(self) -> Sequence[SensorSpec]:
        return list(self._specs.values())

    def spec(self, sensor_id: str) -> SensorSpec:
        return self._specs[sensor_id]
